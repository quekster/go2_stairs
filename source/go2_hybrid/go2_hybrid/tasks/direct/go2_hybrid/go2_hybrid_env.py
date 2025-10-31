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

from .go2_hybrid_env_cfg import Go2HybridEnvCfg
from .rewards import compute_all_rewards
from .terminations import illegal_contact, out_of_bounds, time_out, bad_orientation


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

        # self._roi_debug_markers = None
        # self._roi_marker_type
        # self._roi_marker_indices
        

        # action buffers (derived from Gym space for robustness)
        dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._stand_height_ref = torch.zeros(self.num_envs, device=self.device)

        # X/Y linear velocity (body frame) + yaw rate commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # logging buckets (same keys as AnymalC so your logs look familiar)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "forward_progress",
                "flat_orientation",
                "lin_vel_z_penalty",
                "ang_vel_xy_penalty",
                "joint_torque_penalty",
                "joint_acc_penalty",
                "action_rate_penalty",
                "feet_air_time",
                "undesired_contacts",
                "energy_penalty",
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

        self._height_scanner=RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"]=self._height_scanner
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
        _origin_debug_marker = VisualizationMarkers(_origin_debug_marker_cfg)
        translations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # shape (1,3)
        _origin_debug_marker.visualize(translations=translations)

        self._roi_debug_markers = VisualizationMarkers(self._ROI_debug_marker_cfg)
        self._roi_marker_type = list(self._ROI_debug_marker_cfg.markers.keys())  # ['roi_corner', 'base_frame']
        self._roi_marker_indices = torch.tensor([0, 0, 0, 0, 1], device=self.device)  # 4 corners + 1 base frame marker

        self._lidar_origin_debug_marker = VisualizationMarkers(_lidar_origin_debug_marker_cfg)
        self._lidar_origin_marker_type = list(_lidar_origin_debug_marker_cfg.markers.keys())  # ['lidar_origin_box']
        self._lidar_origin_marker_indices = torch.tensor([0], device=self.device)  # 1 marker

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
        # cache previous actions and compute processed PD targets
        self._previous_actions = self._actions.clone()
        if actions is not None and actions.numel() > 0:
            self._actions = actions.clone()

        action_scale = getattr(self.cfg, "action_scale", 0.5)  # default if not set in cfg
        self._processed_actions = action_scale * self._actions + self._robot.data.default_joint_pos


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        lidar_obs = self.get_stacked_hits()
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,                              # (N,3)
                self._robot.data.root_ang_vel_b,                              # (N,3)
                self._robot.data.projected_gravity_b,                         # (N,3)
                self._commands,                                               # (N,3)
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # (N,ndof)
                self._robot.data.joint_vel,                                   # (N,ndof)
                self._actions,                                                # (N,ndof)

                lidar_obs
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

        # self._visualize_roi_box()
        # self._visualize_lidar_origin()

        # if self._step_counter % 50 == 0:  # every 100 steps
        #     #self.plot_lidar_3d(env_id=0, show_history=False)
        #     hits_b = self.get_bf_hits(torch.tensor([0], device=self.device))[0]
        #     hits_ds = self.get_hits_downsampled(hits_b)
        #     hits = self.get_hits_norm(hits_ds)
        #     print("Current hits:", hits)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total, terms = compute_all_rewards(self)
        # accumulate episodic sums for logging (same keys as terms)
        for k, v in terms.items():
            self._episode_sums[k] += v
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: 
        time_outs = time_out(self)
        #bad_orient = bad_orientation(self, limit_angle=2.2)
        # base_contact = illegal_contact(self, threshold=1.0, body_names=["base","Head_lower", "Head_upper"])
        base_contact = illegal_contact(self, threshold=1.0, body_names=["base"])
        oob = out_of_bounds(self, margin=0.2)
        terminated = base_contact | oob

                # --- Debug prints (only for early steps or when something triggers) ---
        if torch.any(terminated):
            num_contact = torch.count_nonzero(base_contact).item()
            #num_orient  = torch.count_nonzero(bad_orient).item()
            num_timeout = torch.count_nonzero(time_outs).item()
            num_oob = torch.count_nonzero(oob).item()

            if num_oob > 0:
                triggered = torch.nonzero(oob).squeeze(-1).tolist()
                print(f"Out-of-bounds triggered in envs: {triggered}")

            # print(f"[STEP {self._step_counter:04d}] "
            #     f"Terminated: {torch.count_nonzero(terminated).item()} | "
            #     f"BaseContact={num_contact}, BadOrient={num_orient}, Timeout={num_timeout}")


            # print(f"[STEP {self._step_counter:04d}] "
            #     f"Terminated: {torch.count_nonzero(terminated).item()} | "
            #     f"BaseContact={num_contact}, Timeout={num_timeout}")

            # Optional: print which envs specifically triggered each
            # if num_contact > 0:
            #     triggered = torch.nonzero(base_contact).squeeze(-1).tolist()
            #     print(f"Base contact triggered in envs: {triggered}")
            # if num_orient > 0:
            #     triggered = torch.nonzero(bad_orient).squeeze(-1).tolist()
            #     print(f"Bad orientation triggered in envs: {triggered}")
            # if num_timeout > 0:
            #     triggered = torch.nonzero(time_outs).squeeze(-1).tolist()
            #     print(f"Timeout triggered in envs: {triggered}")

        return time_outs, terminated
   
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

        # sample new commands in [-1, 1] (same as AnymalC)
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

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
            hits0 = self._height_scanner.data.ray_hits_w[env_ids]  # shape (envs, R, 3)
            num_rays = hits0.shape[1]
            # buffer shape: (num_envs, buffer_size, num_rays, 3)
            self._lidar_buffer = torch.zeros(
                (self.num_envs, self._lidar_buffer_size, num_rays, 3),
                dtype=hits0.dtype, device=self.device
            )
        else:
            # zero‐out the buffer entries for reset envs
            self._lidar_buffer[env_ids] = 0.0


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


    def get_bf_hits(self, env_ids=None):
        """Return all raw LiDAR hit points in base frame (metres). Also replaces NaNs with max range (70m)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        hits_w = self._height_scanner.data.ray_hits_w[env_ids]
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
            R = self._height_scanner.cfg.pattern_cfg.num_rays
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
        lidar = self._height_scanner

        # Get the LiDAR origin pose in world frame
        offset_tensor = torch.tensor(self.cfg.height_scanner.offset.pos, device=self.device)
        lidar_pos_w = lidar.data.pos_w[env_id] + offset_tensor  # [3]
        #lidar_quat_w = lidar.data.quat_w[env_id] + self.cfg.height_scanner.offset.quat  # [4]

        # Convert to tensor of shape [1, 3]
        translations = lidar_pos_w.unsqueeze(0)

        # Visualize (position only; orientation ignored for sphere)
        self._lidar_origin_debug_marker.visualize(translations=translations, marker_indices=self._lidar_origin_marker_indices)


