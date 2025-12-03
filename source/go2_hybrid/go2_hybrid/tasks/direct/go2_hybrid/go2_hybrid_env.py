# go2_hybrid_env.py
from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from .visualisation import VelArrowsVisualizer
from isaaclab.utils.math import quat_apply, quat_conjugate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import omni.timeline
import math

from .go2_hybrid_env_cfg import Go2HybridEnvCfg
from .rewards import compute_all_rewards
from .terminations import illegal_contact, out_of_bounds, time_out

class Go2HybridEnv(DirectRLEnv):
    
    cfg: Go2HybridEnvCfg

    def __init__(self, cfg: Go2HybridEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._step_counter =0
        self._marker= None

        # LiDAR configuration
        self._lidar_range = 70.0  # metres
        self._lidar_buffer = None
        self._lidar_buffer_size = 2  # number of temporal frames to stack

        # ---- NEW: control temporal spacing between LiDAR frames ----
        # Desired time between stored LiDAR frames (in seconds)
        self._lidar_stack_interval_s = 0.35  # e.g. 0.35 s between frames (t and t+0.35)
        # Convert to integer steps based on env dt
        self._lidar_stack_interval_steps = max(
            1, int(round(self._lidar_stack_interval_s / self.step_dt))
        )
        # Per-env counters (how many steps since last buffer update)
        self._lidar_stack_counters = torch.full(
            (self.num_envs,),
            self._lidar_stack_interval_steps,
            dtype=torch.int32,
            device=self.device,
        )

        self._ROI_offset =  (0.0, 0.0, 0.0) #from bf
        self._ROI_box_length = 2.0    # metres forward (x direction)
        self._ROI_box_width = 1.0      # metres sideways (y direction)
        self._ROI_box_height = 0.5     # metres up (z direction)

        # Timers for command resampling
        self._cmd_timer = torch.zeros(self.num_envs, device=self.device)
        self._cmd_interval = torch.full((self.num_envs,), 10.0, device=self.device)  # seconds


        # action buffers (derived from Gym space for robustness)
        dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._stand_height_ref = torch.zeros(self.num_envs, device=self.device)

        # X/Y linear velocity (body frame) + yaw rate commands
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)

        
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_penalty",
                "ang_vel_xy_penalty",
                "joint_torque_penalty",
                "joint_acc_penalty",
                "action_rate_penalty",
                "feet_air_time",
                "undesired_contacts",
                # "forward_progress",
                "flat_orientation",
                "joint_pos_limit",
                # "energy_penalty",
                "feet_slide_penalty",
                "base_height_penalty",
                "foot_clearance_reward",
                "track_heading_reward",
                "stand_still_joint_deviation_l1",
                "foot_lateral_separation_penalty",
            ]
        }

        # base: if your base is named "base" (or try "trunk" as fallback)
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # feet: explicit four feet
        self._feet_ids, _ = self._contact_sensor.find_bodies(['FL_foot','FR_foot', 'RL_foot', 'RR_foot'])

        # thighs (undesired contacts): explicit four thighs
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(['FL_thigh','FR_thigh', 'RL_thigh', 'RR_thigh'])




    def _setup_scene(self):
        # Ground plane (flat)
        #spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # Spawn robot from cfg
        self._robot = Articulation(self.cfg.robot_cfg)   # note: cfg attribute name is robot_cfg in your direct cfg
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._lidar_scanner=RayCaster(self.cfg.lidar_scanner)
        self.scene.sensors["lidar_scanner"]=self._lidar_scanner
        self._lidar_buffer= None
        self._lidar_buffer_size = 2

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

       
        # ------- ROI Debug Markers (set up once) ------- #
        self._ROI_debug_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/DebugROI",
            markers={
                "roi_corner": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # green
                ),
                # "base_frame": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                #     scale=(0.3, 0.3, 0.3),
                # ),
            },
        )       

        _origin_debug_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/OriginMarker",
            markers={
                "origin_box": sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )

        _lidar_origin_debug_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/LiDAROriginMarker",
            markers={
                "lidar_origin_box": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
            },
        )

        _vel_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/VelMarkers",
            markers={
                "cmd_arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "output_arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },            
        )


        _origin_debug_marker = VisualizationMarkers(_origin_debug_marker_cfg)
        translations = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32)  # shape (1,3)
        _origin_debug_marker.visualize(translations=translations)

        self._roi_debug_markers = VisualizationMarkers(self._ROI_debug_marker_cfg)
        self._roi_marker_type = list(self._ROI_debug_marker_cfg.markers.keys())  # ['roi_corner', 'base_frame']
        self._roi_marker_indices = torch.tensor([0, 0, 0, 0, 1], device=self.device)  # 4 corners + 1 base frame marker

        self._lidar_origin_debug_marker = VisualizationMarkers(_lidar_origin_debug_marker_cfg)
        self._lidar_origin_marker_type = list(_lidar_origin_debug_marker_cfg.markers.keys())  # ['lidar_origin_box']
        self._lidar_origin_marker_indices = torch.tensor([0], device=self.device)  # 1 marker

        self._vel_markers = VisualizationMarkers(_vel_marker_cfg)

        #----------------------------------------------#


        # Clone & replicate envs
        self.scene.clone_environments(copy_from_source=False)

        # CPU collision filtering (same as reference)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        if actions is not None:
            self._actions = actions.clone()

        # Advance command timer and resample as needed
        self._cmd_timer += self.step_dt
        need_resample = self._cmd_timer >= self._cmd_interval
        if torch.any(need_resample):
            self.resample_commands(need_resample.nonzero(as_tuple=False).squeeze(-1))
            self._cmd_timer[need_resample] = 0.0
            self._cmd_interval[need_resample] = torch.empty_like(
                self._cmd_interval[need_resample]
            ).uniform_(8.0, 12.0)

        action_scale = getattr(self.cfg, "action_scale", 0.15)
        self._processed_actions = action_scale * self._actions + self._robot.data.default_joint_pos
        


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # Actor observations (realistic)
        lidar_obs = self.get_stacked_hits()
        obs_policy = torch.cat(
            [
                self._robot.data.root_lin_vel_b,                  # (N,3) → vx, vy, vz
                self._robot.data.root_ang_vel_b,                  # (N,3) → ωx, ωy, ωz
                self._robot.data.projected_gravity_b,             # (N,3) → gx, gy, gz
                self._commands,                                   # (N,4) → cmd_vx, cmd_vy, cmd_yaw_rate, heading
                self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,ndof) joint pos error
                self._robot.data.joint_vel,                       # (N,ndof) joint velocities
                self._actions,                                    # (N,ndof) previous actions
                lidar_obs,                                        # (N, ...) lidar hits
            ],
            dim=-1,
        )

        # Critic observations (privileged)
        privileged = torch.cat(
            [
                obs_policy,
                self._robot.data.root_pos_w,            # (N, 3)
                self._robot.data.root_quat_w,           # (N, 4)
                self._robot.data.applied_torque,        # (N, ndof)
                self._contact_sensor.data.net_forces_w.reshape(self.num_envs, -1), # contacts
                self._contact_sensor.data.last_air_time.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
        self._step_counter += 1

        #-----DEBUGGER for hits (base/world)--------#

        #hits_b = self.get_bf_hits_normalised()
        #print(f"[DEBUG] Step {self._step_counter} — lidar_obs shape {lidar_obs.shape}")
        #print("Current base frame hits:", hits_b[0])
        # print(f"[DEBUG] Frame_t-2 hits: {lidar_obs[0].cpu().numpy()}")
        # print("------------------NEXT STEP------------------")

        #print("obs dim:", obs_policy.shape[-1], "state dim:", privileged.shape[-1])
        # self._visualize_roi_box()
        #self._visualize_lidar_origin()
        self._visualize_velocity_arrows()

        # if self._step_counter % 50 == 0:  # every 100 steps
        #     #self.plot_lidar_3d(env_id=0, show_history=False)
        #     hits_b = self.get_bf_hits(torch.tensor([0], device=self.device))[0]
        #     hits_ds = self.get_hits_downsampled(hits_b)
        #     hits = self.get_hits_norm(hits_ds)
        #     print("Current hits:", hits)
        # print("obs shape:", obs_policy.shape, "state shape:", privileged.shape)
        return {
            "policy": obs_policy,      # for actor network
            "critic": privileged,      # for critic network
        }

    def _get_rewards(self) -> torch.Tensor:
        total, terms = compute_all_rewards(self)
        # accumulate episodic sums for logging (same keys as terms)
        for k, v in terms.items():
            self._episode_sums[k] += v
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute episode termination signals (orientation, contact, bounds, timeout)."""

        # --- Individual terminations ---
        time_outs = time_out(self)
        base_contact = illegal_contact(self, threshold=5.0, body_names=["base"])
        oob = out_of_bounds(self, margin=0.5)

        # --- Combine ---
        terminated = base_contact | oob

        # --- Optional Debug ---
        # if torch.any(terminated):
        #     num_contact = torch.count_nonzero(base_contact).item()
        #     num_oob = torch.count_nonzero(oob).item()
        #     num_timeout = torch.count_nonzero(time_outs).item()

        #     print(
        #         f"[STEP {self._step_counter:05d}] Terminated={torch.count_nonzero(terminated).item()} | "
        #         f"BaseContact={num_contact}, OOB={num_oob}, Timeout={num_timeout}"
        #     )


        return terminated, time_outs


   
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # stagger resets to avoid spikes
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # clear actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # # sample new commands in [-1, 1] (same as AnymalC)
        # self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-1.0, 1.0)  # vx
        # self._commands[env_ids, 1] = torch.zeros_like(self._commands[env_ids, 1]).uniform_(-1.0, 1.0)  # vy
        # self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 2]).uniform_(-1.0, 1.0)  # yaw rate
        # self._commands[env_ids, 3] = torch.zeros_like(self._commands[env_ids, 3]).uniform_(-math.pi, math.pi)  # heading


        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        # Robot spawn position from terrain
        base_origin = self._terrain.env_origins[env_ids].clone()

        # move a little backward from the first step (assuming stairs go +X)
        base_origin[:, 0] -= 2.5
        # lift robot slightly so it’s not intersecting the mesh
        base_origin[:, 2] += 0.4


        # Add per-env origin if available (scene may expose env_origins)
        origins = getattr(self.scene, "env_origins", None)
        if origins is None:
            origins = torch.zeros(self.num_envs, 3, device=self.device)
        default_root_state[:, :3] += origins[env_ids]


        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = base_origin
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reference standing height per env (z of default pose + env origin z)
        self._stand_height_ref[env_ids] = (self._robot.data.default_root_state[env_ids][:, 2] + origins[env_ids][:, 2])

        # initialisation or clearing of lidar buffer for reset envs
        if self._lidar_buffer is None or self._lidar_buffer.shape[0] != self.num_envs:
            # get number of rays by sampling current hits
            hits0 = self._lidar_scanner.data.ray_hits_w[env_ids]  # shape (envs, R, 3)
            num_rays = hits0.shape[1]
            # buffer shape: (num_envs, buffer_size, num_rays, 3)
            self._lidar_buffer = torch.zeros(
                (self.num_envs, self._lidar_buffer_size, num_rays, 3),
                dtype=hits0.dtype,
                device=self.device,
            )
        else:
            # zero‐out the buffer entries for reset envs
            self._lidar_buffer[env_ids] = 0.0

        # ---- NEW: reset LiDAR temporal counters for these envs ----
        # Set to interval so that first call to get_stacked_hits() refreshes immediately
        self._lidar_stack_counters[env_ids] = self._lidar_stack_interval_steps

        ################### Debug snippet to put into your environment class (e.g., in go2_hybrid_env.py)####################
        # if self._step_counter == 0:
        #     # 1) Print joint names → indices
        #     joint_names = self._robot.data.joint_names  # tensor of strings or list
        #     print("=== Joint index mapping ===")
        #     for i, name in enumerate(joint_names):
        #         print(f"joint index {i} -> name {name}")

        #     # 2) Print first few values of observation vector for env_id 0
        #     obs_full = self._get_observations()["policy"][0].cpu().numpy()
        #     print("\n=== Observation vector (first 20 entries) ===")
        #     for i in range(min(20, obs_full.shape[0])):
        #         print(f"obs index {i} = {obs_full[i]:.4f}")

        #     # 3) Apply known small lateral (y-direction) perturbation
        #     # Set a command with vy != 0 forcing lateral motion
        #     self._commands[0, :2] = torch.tensor([0.0, 0.5], device=self.device)  # vx=0, vy=0.5
        #     self._commands[0, 2:] = torch.tensor([0.0, 0.0], device=self.device)   # yaw_rate=0
        #     # Step environment for one step (you might call step once)
        #     # (Assumes external step call; if inside env, step then print)
        #     print("\n-- After lateral command (vy=0.5) --")
        #     obs2 = self._get_observations()["policy"][0].cpu().numpy()
        #     for i in range(min(20, obs2.shape[0])):
        #         if abs(obs2[i] - obs_full[i]) > 1e-3:
        #             print(f"obs index {i} changed from {obs_full[i]:.4f} → {obs2[i]:.4f}")

        #     # 4) Apply small yaw rate command
        #     self._commands[0, :2] = torch.tensor([0.0, 0.0], device=self.device)
        #     self._commands[0, 2]  = 0.3  # yaw_rate
        #     print("\n-- After yaw_rate command (yaw_rate=0.3) --")
        #     obs3 = self._get_observations()["policy"][0].cpu().numpy()
        #     for i in range(min(20, obs3.shape[0])):
        #         if abs(obs3[i] - obs2[i]) > 1e-3:
        #             print(f"obs index {i} changed from {obs2[i]:.4f} → {obs3[i]:.4f}")

        ################################################


        # --- Immediately sample a new command at episode start ---
        self.resample_commands(env_ids)

        # Reset command timers so resampling happens after ~10s, not before
        self._cmd_timer[env_ids] = 0.0
        self._cmd_interval[env_ids] = torch.empty_like(self._cmd_interval[env_ids]).uniform_(8.0, 12.0)


        # episode logs (averaged over the just-reset envs)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s #to get per-second normalization - "how big each reward/penalty was per episode"
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def resample_commands(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)

        # sample random heading
        heading = torch.empty(num_envs, device=self.device).uniform_(-math.pi, math.pi)
        self._commands[env_ids, 3] = heading

        # sample linear speed and direction relative to heading
        speed = torch.empty(num_envs, device=self.device).uniform_(0.0, 1.0)
        direction_offset = torch.empty(num_envs, device=self.device).uniform_(-math.pi/6, math.pi/6)  # ±30° cone

        # world-frame velocities aligned with heading
        vx_world = speed * torch.cos(heading + direction_offset)
        vy_world = speed * torch.sin(heading + direction_offset)
        yaw_rate = torch.empty(num_envs, device=self.device).uniform_(-0.5, 0.5)

        self._commands[env_ids, 0] = vx_world
        self._commands[env_ids, 1] = vy_world
        self._commands[env_ids, 2] = yaw_rate


    def get_bf_hits(self, env_ids=None):
        """Return all LiDAR hits in BASE frame. Also replaces NaNs with max-range."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Hits in WORLD frame
        hits_w = self._lidar_scanner.data.ray_hits_w[env_ids]      # [N, R, 3]

        # Base pose in WORLD frame
        base_pos_w  = self._robot.data.root_pos_w[env_ids]         # [N, 3]
        base_quat_w = self._robot.data.root_quat_w[env_ids]        # [N, 4]
        base_quat_inv = quat_conjugate(base_quat_w)                # [N, 4]

        # Shift into base origin
        hits_shifted = hits_w - base_pos_w.unsqueeze(1)            # [N, R, 3]

        # Correct quaternion expansion
        base_quat_exp = base_quat_inv.unsqueeze(1).expand(-1, hits_shifted.shape[1], -1)
        # shape = [N, R, 4]

        # Rotate into base frame
        hits_b = quat_apply(base_quat_exp, hits_shifted)           # [N, R, 3]

        # Replace NaNs
        hits_b = torch.nan_to_num(hits_b, nan=self._lidar_range)

        return hits_b


    def get_hits_downsampled(self, hits_b: torch.Tensor):
        """
        Downsample LiDAR points by retaining only those within a box region in front of the robot.
        - Points outside the ROI are discarded (not zeroed).
        - NaN points (no hit) are replaced with max lidar range.
        """
        # --- Replace NaNs with max range first ---
        hits_b = torch.nan_to_num(hits_b, nan=self._lidar_range)

        # --- Define ROI bounds (in base frame) ---
        x_off, y_off, z_off = self._ROI_offset
        x_min, x_max = x_off, x_off + self._ROI_box_length
        y_min, y_max = -self._ROI_box_width / 2 + y_off, self._ROI_box_width / 2 + y_off
        # We’re ignoring z-axis limits intentionally for your use case

        # --- Create mask for points inside ROI ---
        mask = (
            (hits_b[..., 0] >= x_min)
            & (hits_b[..., 0] <= x_max)
            & (hits_b[..., 1] >= y_min)
            & (hits_b[..., 1] <= y_max)
        )

        # Replace outside-ROI points with max lidar range
        hits_filtered = hits_b.clone()
        hits_filtered[~mask] = self._lidar_range

        return hits_filtered  


    def get_hits_norm(self, hits_ds: torch.Tensor):
        """Normalize to 70 m range after downsampling."""

        # hits_b = self.get_bf_hits(env_ids)
        # hits_b_ds = self.get_hits_downsampled(hits_b)

        hits_b_ds_norm = hits_ds / self._lidar_range
        return hits_b_ds_norm
    

    def get_stacked_hits(self, env_ids=None):
        """Temporal stack of normalized, downsampled LiDAR hits."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # --- Proper call sequence ---
        hits_b = self.get_bf_hits(env_ids)
        # hits_ds = self.get_hits_downsampled(hits_b)
        # current_hits_b = self.get_hits_norm(hits_ds)
        current_hits_b = self.get_hits_norm(hits_b)

        # --- Buffer stacking ---
        buff = self._lidar_buffer  # shape: [num_envs, buffer_size, R, 3]
        buff[env_ids, :-1, :, :] = buff[env_ids, 1:, :, :]
        buff[env_ids, -1, :, :] = current_hits_b

        N = current_hits_b.shape[0]
        B = self._lidar_buffer_size
        R = current_hits_b.shape[1]
        stacked = buff[env_ids].reshape(N, B * R * 3)
        return stacked     

    def plot_lidar_3d(self, env_id=0, frame="world", show_history=True):
        """
        Visualize the 3D LiDAR point cloud for a given environment.
        Can plot either a single frame or the full temporal stack from get_stacked_hits().

        Args:
            env_id (int): Which environment to visualize.
            frame (str): "world" or "base" (base recommended since your LiDAR is base-frame aligned).
            show_history (bool): If True, use temporally stacked hits; else use current hits only.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import omni.timeline

        tl = omni.timeline.get_timeline_interface()
        tl.pause()

        if show_history:
            # Retrieve stacked LiDAR buffer (flattened)
            stacked_hits = self.get_stacked_hits(torch.tensor([env_id], device=self.device))  # shape [1, B*R*3]
            B = self._lidar_buffer_size
            R = self._lidar_scanner.cfg.pattern_cfg.num_rays
            hits = stacked_hits.view(B, R, 3).detach().cpu().numpy()  # [B, R, 3]

            # Combine or colorize frames
            colors = plt.cm.plasma(np.linspace(0, 1, B))  # color gradient for temporal frames
        else:
            # Single current frame (normalized)
            hits = self.get_hits_norm(torch.tensor([env_id], device=self.device))[0].detach().cpu().numpy()
            B = 1
            colors = [plt.cm.plasma(0.5)]

        # Remove invalid hits (NaNs or infs)
        mask = np.isfinite(hits).all(axis=-1)
        hits = np.where(mask[..., None], hits, np.nan)

        # Create 3D scatter
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # --- ensure hits has shape [B, R, 3] ---
        if hits.ndim == 2 and hits.shape[1] == 3:
            # single frame -> add a temporal dimension
            hits = hits[None, ...]        # [1, R, 3]
            B = 1
        elif hits.ndim == 1:
            # completely flattened vector -> reshape to (-1, 3)
            hits = hits.reshape(1, -1, 3)
            B = 1
        elif hits.ndim == 3:
            # already correct shape [B, R, 3]
            B = hits.shape[0]
        else:
            raise ValueError(f"Unexpected LiDAR hits shape: {hits.shape}")
        
        hits = hits * self._lidar_range  # Uncomment to un-normalize for plotting

        # Plot each temporal frame
        for b in range(B):

            valid = np.isfinite(hits[b]).all(axis=1)
            if np.sum(valid) == 0:
                continue
            ax.scatter(
                hits[b, valid, 0],  # x coords of valid hits in frame b
                hits[b, valid, 1],  # y coords of valid hits in frame b
                hits[b, valid, 2],  # z coords of valid hits in frame b
                s=2,
                c=[colors[b]] if B > 1 else hits[b, valid, 2],
                cmap=None if B > 1 else "viridis",
                label=f"frame {b}" if B > 1 else None,
            )

        
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Y (normalized)")
        ax.set_zlabel("Z (normalized)")
        title = f"LiDAR hits (stacked={show_history}) – Env {env_id}, Step {self._step_counter}"
        ax.set_title(title)

        # Optional equal aspect ratio
        all_hits = hits.reshape(-1, 3)
        finite = np.isfinite(all_hits).all(axis=1)
        if np.sum(finite) > 0:
            max_range = np.ptp(all_hits[finite], axis=0).max() / 2.0
            mid = np.mean(all_hits[finite], axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        if B > 1:
            ax.legend(loc="upper right", fontsize="x-small")

        plt.tight_layout()
        plt.show(block=True)
        plt.close()
        tl.play()

    def _visualize_roi_box(self, env_id=0):
        """Visualize ROI corners relative to the robot base frame."""
        base_pos = self._robot.data.root_pos_w[env_id]
        base_quat = self._robot.data.root_quat_w[env_id]

        # --- Compute ROI corners in base frame ---
        x_off, y_off, _ = self._ROI_offset
        x_min, x_max = x_off, x_off + self._ROI_box_length
        y_min, y_max = -self._ROI_box_width / 2 + y_off, self._ROI_box_width / 2 + y_off
        roi_corners_b = torch.tensor([
            [x_min, y_min, 0.0],
            [x_min, y_max, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
        ], dtype=torch.float32, device=self.device)

        # --- Transform corners to world frame ---
        roi_corners_w = base_pos.unsqueeze(0) + quat_apply(base_quat.unsqueeze(0), roi_corners_b)
        translations = torch.cat([roi_corners_w, base_pos.unsqueeze(0)], dim=0)

        # Visualize ROI + base frame marker
        self._roi_debug_markers.visualize(translations=translations, marker_indices=self._roi_marker_indices)

    def _visualize_lidar_origin(self, env_id=0):
        """Visualize a small sphere where the RayCaster (LiDAR) is attached."""
        # Retrieve the LiDAR sensor
        lidar = self._lidar_scanner

        # Get the LiDAR origin pose in world frame
        offset_tensor = torch.tensor(self.cfg.lidar_scanner.offset.pos, device=self.device)
        lidar_pos_w = lidar.data.pos_w[env_id] + offset_tensor  # [3]
        #lidar_quat_w = lidar.data.quat_w[env_id] + self.cfg.lidar_scanner.offset.quat  # [4]

        # Convert to tensor of shape [1, 3]
        translations = lidar_pos_w.unsqueeze(0)

        # Visualize (position only; orientation ignored for sphere)
        self._lidar_origin_debug_marker.visualize(translations=translations, marker_indices=self._lidar_origin_marker_indices)

    def _visualize_velocity_arrows(
        self,
        env_ids=None,
        base_marker_scale=(0.5, 0.5, 0.5),
        scale_mult=3.0,
        height_offset=0.3,
    ):
        import torch
        import numpy as np
        from isaaclab.utils.math import quat_mul, quat_from_euler_xyz

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        M = env_ids.shape[0]

        base_pos_w = self._robot.data.root_pos_w[env_ids].clone()
        base_quat_w = self._robot.data.root_quat_w[env_ids]
        base_pos_w[:, 2] += height_offset

        default_scale = torch.tensor(base_marker_scale, device=self.device).unsqueeze(0).repeat(M, 1)
        
        # ================= Command (green) arrow =================
        # Rotate commanded (vx, vy) by heading to match body-frame intent
        heading = self._commands[env_ids, 3]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)

        vx = self._commands[env_ids, 0]
        vy = self._commands[env_ids, 1]
        vx_rot = cos_h * vx + sin_h * vy
        vy_rot = -sin_h * vx + cos_h * vy
        cmd_body = torch.stack((vx_rot, vy_rot), dim=1)

        cmd_speed = torch.linalg.norm(cmd_body, dim=1)
        arrow_scale_cmd = default_scale.clone()
        arrow_scale_cmd[:, 0] *= cmd_speed * scale_mult

        heading_cmd = torch.atan2(cmd_body[:, 1], cmd_body[:, 0])
        zeros = torch.zeros_like(heading_cmd)
        arrow_quat_local_cmd = quat_from_euler_xyz(zeros, zeros, heading_cmd)
        arrow_quat_cmd = quat_mul(base_quat_w, arrow_quat_local_cmd)


        # ================= Output (blue) arrow =================
        vel_body_xy = self._robot.data.root_lin_vel_b[env_ids, :2]
        vel_speed = torch.linalg.norm(vel_body_xy, dim=1)
        arrow_scale_out = default_scale.clone()
        arrow_scale_out[:, 0] *= vel_speed * scale_mult

        heading_out = torch.atan2(vel_body_xy[:, 1], vel_body_xy[:, 0])
        arrow_quat_local_out = quat_from_euler_xyz(zeros, zeros, heading_out)
        arrow_quat_out = quat_mul(base_quat_w, arrow_quat_local_out)

        # ================= Merge for single visualize() call =================
        translations = torch.cat([base_pos_w, base_pos_w], dim=0)
        orientations = torch.cat([arrow_quat_cmd, arrow_quat_out], dim=0)
        scales       = torch.cat([arrow_scale_cmd, arrow_scale_out], dim=0)

        marker_indices = torch.cat([
            torch.zeros(M, dtype=torch.int32, device=self.device),  # prototype 0 = cmd_arrow
            torch.ones(M,  dtype=torch.int32, device=self.device),  # prototype 1 = output_arrow
        ], dim=0)

        # One unified call
        self._vel_markers.visualize(
            translations=translations.cpu().numpy(),
            orientations=orientations.cpu().numpy(),
            scales=scales.cpu().numpy(),
            marker_indices=marker_indices.cpu().numpy(),
        )




