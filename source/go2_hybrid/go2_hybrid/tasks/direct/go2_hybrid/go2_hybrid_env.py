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
        self._lidar_buffer = None
        self._lidar_range = 70.0  # metres

        self._ROI_offset =  (0.0, 0.0, 0.0) #from bf
        self._ROI_box_length = 2.0    # metres forward (x direction)
        self._ROI_box_width = 1.0      # metres sideways (y direction)
        self._ROI_box_height = 0.5     # metres up (z direction)

        # Timers for command resampling
        self._cmd_timer = torch.zeros(self.num_envs, device=self.device)
        self._cmd_interval = torch.full((self.num_envs,), 15.0, device=self.device)  # seconds

        #Phase 1: Stairs Terrain - Goal -> top of stairs
        self._goal_pos = torch.tensor([0.0, 0.0, 1.1], device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)

        # action buffers (derived from Gym space for robustness)
        dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._stand_height_ref = torch.zeros(self.num_envs, device=self.device)

        # X/Y linear velocity (body frame) + yaw rate commands
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)

        
        # Episode reward tracking (matches compute_all_rewards in rewards.py)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # Tracking rewards
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "track_heading_reward",
                
                # Core locomotion penalties
                "lin_vel_z_penalty",
                "ang_vel_xy_penalty",
                "action_rate_penalty",
                "joint_torque_penalty",
                "joint_acc_penalty",
                
                # Foot contact rewards
                "feet_air_time",
                "feet_slide_penalty",
                "undesired_contacts",
                
                # Joint limits
                "joint_pos_limit",
                
                # Stairs-specific rewards
                "thigh_lift",
                "step_detection",
                "hind_push",
                "front_placement",
                "body_height_progress",
                "pitch_stability",
                
                # **NEW: Critical penalties**
                "base_pitch_penalty",
                "base_height_maintenance",
                "front_foot_separation",
                "rear_foot_separation",
                
                # Goal progress
                "goal_progress",
            ]
        }

        # base: if your base is named "base" (or try "trunk" as fallback)
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # feet: explicit four feet
        self._feet_ids, _ = self._contact_sensor.find_bodies(['FL_foot','FR_foot', 'RL_foot', 'RR_foot'])

        # thighs (undesired contacts): explicit four thighs
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(['FL_thigh','FR_thigh', 'RL_thigh', 'RR_thigh'])

        # Define hind feet explicitly
        self._hind_feet_ids = torch.tensor([self._feet_ids[2], self._feet_ids[3]], device=self.device)
        self._hind_last_contact_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._hind_was_in_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)




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

        # New Lookahead Marker
        _lookahead_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/LookaheadMarker",
            markers={
                "lookahead_box": sim_utils.CuboidCfg(
                    size=(0.05, 0.8, 0.3),  # thin box showing lookahead region
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0),  # yellow
                        opacity=0.5  # semi-transparent
                    ),
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

        self._lookahead_marker = VisualizationMarkers(_lookahead_marker_cfg)
        self._lookahead_marker_type = list(_lookahead_marker_cfg.markers.keys())  # ['lookahead_box']
        self._lookahead_marker_indices = torch.tensor([0], device=self.device)  # 1 marker

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

        #-----VISUALIZATIONS--------#
        # self._visualize_roi_box()
        #self._visualize_lidar_origin()
        self._visualize_velocity_arrows()
        #self._visualize_step_detection_zone()


        # if self._step_counter % 100 == 0:  # every 100 steps
        #     self.plot_lidar_3d()
            # hits_b = self.get_bf_hits(torch.tensor([0], device=self.device))[0]
            # hits_ds = self.get_hits_downsampled(hits_b)
            # hits = self.get_hits_norm(hits_ds)
            # print("Current hits:", hits)
        return {
            "policy": obs_policy,      # for actor network
            "critic": privileged,      # for critic network
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using centralized weight dictionary."""
        total, terms = compute_all_rewards(self)
        
        # Accumulate episodic sums for logging
        for k, v in terms.items():
            if k not in self._episode_sums:
                self._episode_sums[k] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[k] += v
        
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute episode termination signals (orientation, contact, bounds, timeout)."""

        # --- Individual terminations ---
        time_outs = time_out(self)
        base_contact = illegal_contact(self, threshold=5.0, body_names=["base", "Head_upper", "Head_lower"])
        oob = out_of_bounds(self, margin=0.5)
    

        # --- Combine ---
        terminated = base_contact | oob
        # terminated = base_contact
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

        # Reset robot internal buffers
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # stagger resets to avoid identical trajectories
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length)
            )

        # Clear actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # ===============================
        # 1. Compute spawn pose (Phase 1)
        # ===============================
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Terrain origin (but your USD stairs are the same for all envs)
        base_origin = self._terrain.env_origins[env_ids].clone()

        # Move robot backward from first step and lift slightly
        base_origin[:, 0] -= 2.5
        base_origin[:, 2] += 0.4


        # Apply spawn pose to default root state
        default_root_state[:, :3] = base_origin

        # Write pose/velocity to simulator
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ===============================
        # 2. Initialize goal-tracking distance
        # ===============================
        # IMPORTANT: must use base_origin (actual spawn), not default_root_state
        base_pos = base_origin  # shape (N,3)
        dist0 = torch.norm(self._goal_pos - base_pos, dim=1)
        self._prev_goal_dist[env_ids] = dist0

        # ===============================
        # 3. Reference standing height
        # ===============================
        self._stand_height_ref[env_ids] = default_root_state[:, 2]

        # ===============================
        # Initialize reward tracking buffers
        # ===============================
        
        # Body height progress tracking
        if not hasattr(self, '_prev_body_height'):
            self._prev_body_height = torch.zeros(self.num_envs, device=self.device)
        self._prev_body_height[env_ids] = default_root_state[:, 2]
        
        # (Keep your existing _prev_foot_z, _hind_last_contact_pos, etc.)
        
        # --- Initialize previous foot heights for stair stepping reward ---
        foot_z = self._robot.data.body_pos_w[env_ids][:, self._feet_ids, 2]  # shape: (N,4)
        if not hasattr(self, "_prev_foot_z"):
            # allocate tensor for all envs
            self._prev_foot_z = torch.zeros(self.num_envs, 4, device=self.device)
        # update only reset envs
        self._prev_foot_z[env_ids] = foot_z.clone()

        hind_pos = self._robot.data.body_pos_w[env_ids][:, self._hind_feet_ids, :]  # (n,2,3)
        self._hind_last_contact_pos[env_ids] = hind_pos
        self._hind_was_in_contact[env_ids] = False

        # ===============================
        # 4. Reset LiDAR buffer
        # ===============================
        if self._lidar_buffer is None or self._lidar_buffer.shape[0] != self.num_envs:
            hits0 = self._lidar_scanner.data.ray_hits_w[env_ids]
            num_rays = hits0.shape[1]
            self._lidar_buffer = torch.zeros(
                (self.num_envs, self._lidar_buffer_size, num_rays, 3),
                dtype=hits0.dtype, device=self.device
            )
        else:
            self._lidar_buffer[env_ids] = 0.0

        # ===============================
        # 5. Commands — Phase 1 (forward only)
        # ===============================
        self.resample_commands(env_ids)  # but restrict inside resample_commands()
        self._cmd_timer[env_ids] = 0.0
        self._cmd_interval[env_ids] = torch.empty_like(
            self._cmd_interval[env_ids]
        ).uniform_(8.0, 12.0)

        # ===============================
        # 6. Logging
        # ===============================
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/base_contact"] = \
            torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = \
            torch.count_nonzero(self.reset_time_outs[env_ids]).item()


    def resample_commands(self, env_ids: torch.Tensor):
        """Slower, forward-only commands for stairs climbing."""
        num_envs = len(env_ids)
        
        # Forward speed only (stairs curriculum)
        vx = torch.empty(num_envs, device=self.device).uniform_(0.0, 1.0)
        vy = torch.zeros(num_envs, device=self.device)  # no lateral movement
        yaw_rate = torch.empty(num_envs, device=self.device).uniform_(-0.2, 0.2)
        heading = torch.zeros(num_envs, device=self.device)  # face stairs
        
        self._commands[env_ids, 0] = vx
        self._commands[env_ids, 1] = vy
        self._commands[env_ids, 2] = yaw_rate
        self._commands[env_ids, 3] = heading
        

    def get_bf_hits(self, env_ids=None):
        """Return all raw LiDAR hit points in base frame (metres). Also replaces NaNs with max range (70m)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        hits_w = self._lidar_scanner.data.ray_hits_w[env_ids]
        base_pos_w = self._robot.data.root_pos_w[env_ids]
        base_quat_w = self._robot.data.root_quat_w[env_ids]

        base_quat_inv = quat_conjugate(base_quat_w)
        hits_shifted = hits_w - base_pos_w.unsqueeze(1)
        N, R, _ = hits_shifted.shape
        base_quat_exp = base_quat_inv.unsqueeze(1).expand(-1, R, -1)
        hits_b = quat_apply(base_quat_exp, hits_shifted)

        # replace NaNs with max-range value
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

    def plot_lidar_3d(self, env_id=0):
        """
        Visualize the 3D LiDAR point cloud for the current frame only.

        Args:
            env_id (int): Which environment to visualize.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import omni.timeline

        tl = omni.timeline.get_timeline_interface()
        tl.pause()

        # Get current frame hits in base frame
        hits_b = self.get_bf_hits(torch.tensor([env_id], device=self.device))  # [1, R, 3]
        hits_norm = self.get_hits_norm(hits_b).detach().cpu().numpy()  # [1, R, 3]
        hits = hits_norm.squeeze(0)  # [R, 3]
        
        # Denormalize for visualization
        hits_denorm = hits * self._lidar_range

        # Remove invalid hits
        valid = np.isfinite(hits_denorm).all(axis=1)
        if np.sum(valid) == 0:
            print(f"[WARNING] No valid LiDAR hits at step {self._step_counter}")
            tl.play()
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 6))
        
        # --- LEFT: 3D scatter ---
        ax1 = fig.add_subplot(121, projection="3d")
        
        scatter = ax1.scatter(
            hits_denorm[valid, 0],
            hits_denorm[valid, 1],
            hits_denorm[valid, 2],
            c=hits_denorm[valid, 2],  # color by height
            cmap='viridis',
            s=3,
            edgecolors='none',
        )
        
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title(f"3D LiDAR Point Cloud – Env {env_id}, Step {self._step_counter}")
        
        # Equal aspect ratio
        max_range = np.ptp(hits_denorm[valid], axis=0).max() / 2.0
        mid = np.mean(hits_denorm[valid], axis=0)
        ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.colorbar(scatter, ax=ax1, label='Height (m)', shrink=0.6)

        # --- RIGHT: Top-down (X-Y) view ---
        ax2 = fig.add_subplot(122)
        
        scatter2 = ax2.scatter(
            hits_denorm[valid, 0],
            hits_denorm[valid, 1],
            c=hits_denorm[valid, 2],  # color by height
            cmap='viridis',
            s=9,
            edgecolors='black',
            linewidths=0.3,
        )
        
        # Robot origin marker
        ax2.scatter(0, 0, s=100, c='red', marker='x', linewidths=2, label='Robot Base', zorder=10)
        
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("Top-Down View (X-Y Plane)")
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize="small")
        
        plt.colorbar(scatter2, ax=ax2, label='Height (m)', shrink=0.8)

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

    def _visualize_step_detection_zone(self, env_id=0, lookahead_distance=0.35):
        """
        Visualize the LiDAR detection zone used in thigh_lift_reward.
        
        NOTE: The detection zone is defined in the robot's BASE FRAME (not LiDAR frame),
        because get_bf_hits() returns points transformed to base_link coordinates.
        
        Args:
            env_id: Which environment to visualize (default 0)
            lookahead_distance: Center of detection window in base frame X (matches reward)
        """
        base_pos = self._robot.data.root_pos_w[env_id]
        base_quat = self._robot.data.root_quat_w[env_id]

        # Detection zone definition (matches thigh_lift_reward exactly)
        x_min, x_max = lookahead_distance - 0.1, lookahead_distance + 0.1  # ±10cm around center
        y_min, y_max = -0.2, 0.2  # ±20cm lateral
        z_height = 0.15  # visualize at 15cm height (mid-step)

        # Box center in base frame
        box_center_b = torch.tensor(
            [
                (x_min + x_max) / 2.0,  # center X
                0.0,                     # center Y
                z_height                 # center Z
            ],
            dtype=torch.float32,
            device=self.device
        )

        # Transform to world frame
        box_center_w = base_pos + quat_apply(
            base_quat.unsqueeze(0), 
            box_center_b.unsqueeze(0)
        ).squeeze(0)

        # Compute box dimensions
        box_depth = x_max - x_min   # 0.2m
        box_width = y_max - y_min   # 0.4m  
        box_height = 0.3            # 30cm tall (covers typical step height)

        # Visualize using your existing marker
        translations = box_center_w.unsqueeze(0)  # [1, 3]
        orientations = base_quat.unsqueeze(0)     # [1, 4]
        scales = torch.tensor(
            [[box_depth, box_width, box_height]],
            dtype=torch.float32,
            device=self.device
        )

        self._lookahead_marker.visualize(
            translations=translations.cpu().numpy(),
            orientations=orientations.cpu().numpy(),
            scales=scales.cpu().numpy(),
            marker_indices=self._lookahead_marker_indices.cpu().numpy(),
        )