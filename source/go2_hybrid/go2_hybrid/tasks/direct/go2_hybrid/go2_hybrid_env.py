# go2_hybrid_env.py
from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from .visualisation import VelArrowsVisualizer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import omni.timeline
import math

from .go2_hybrid_env_cfg import Go2HybridEnvCfg
from .rewards_UD_p5 import compute_all_rewards
from .terminations import illegal_contact, out_of_bounds, time_out, flipped_over, stuck, end_point_termination

class Go2HybridEnv(DirectRLEnv):
    
    cfg: Go2HybridEnvCfg

    def __init__(self, cfg: Go2HybridEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._step_counter =0
        self._marker= None
        self._base_body_id = 0
        
        self._lidar_range = self.cfg.lidar_range

        # Timers for command resampling
        self._cmd_timer = torch.zeros(self.num_envs, device=self.device)
        self._cmd_interval = torch.full((self.num_envs,), 10.0, device=self.device)  # seconds
        self._stuck_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)


        # action buffers (derived from Gym space for robustness)
        dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._previous_previous_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._stand_height_ref = torch.zeros(self.num_envs, device=self.device)

        # X/Y linear velocity (body frame) + yaw rate commands
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_root_x = torch.zeros(self.num_envs, device=self.device)

        self._episode_sums = {}

        # base: if your base is named "base" (or try "trunk" as fallback)
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # feet: explicit four feet
        self._feet_ids, _ = self._contact_sensor.find_bodies(['FL_foot','FR_foot', 'RL_foot', 'RR_foot'])


        # thighs (undesired contacts): explicit four thighs
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(['FL_thigh','FR_thigh', 'RL_thigh', 'RR_thigh', 'Head_lower', 'FL_calf','FR_calf', 'RL_calf', 'RR_calf'])



    def _setup_scene(self):

        # Spawn robot from cfg
        self._robot = Articulation(self.cfg.robot_cfg)   # note: cfg attribute name is robot_cfg in your direct cfg
        self.scene.articulations["robot"] = self._robot

        # Cache base body index for external-force visualisation and disturbance-aware rewards.
        base_body_ids, _ = self._robot.find_bodies("base")
        if len(base_body_ids) == 0:
            base_body_ids, _ = self._robot.find_bodies("trunk")
        if len(base_body_ids) > 0:
            self._base_body_id = int(base_body_ids[0])

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner=RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"]=self._height_scanner 

        self._lidar_scanner=RayCaster(self.cfg.lidar_scanner)
        self.scene.sensors["lidar_scanner"]=self._lidar_scanner
        self._lidar_buffer= None
        self._lidar_buffer_size = 2

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        _end_point_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/EndPointMarker",
            markers={
                "origin_box": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
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
        _force_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/ForceMarkers",
            markers={
                "force_arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                ),
            },
        )

        _end_point_marker = VisualizationMarkers(_end_point_marker_cfg)
        translations = torch.tensor([[self.cfg.end_point_pos, 0.0, 0.0]], dtype=torch.float32)  # icra map
        _end_point_marker.visualize(translations=translations)
        # _origin_debug_marker = VisualizationMarkers(_origin_debug_marker_cfg)
        # translations = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32)  # shape (1,3)
        # _origin_debug_marker.visualize(translations=translations)

        self._lidar_origin_debug_marker = VisualizationMarkers(_lidar_origin_debug_marker_cfg)
        self._lidar_origin_marker_type = list(_lidar_origin_debug_marker_cfg.markers.keys())  # ['lidar_origin_box']
        self._lidar_origin_marker_indices = torch.tensor([0], device=self.device)  # 1 marker

        self._vel_markers = VisualizationMarkers(_vel_marker_cfg)
        self._force_markers = VisualizationMarkers(_force_marker_cfg)


        # Clone & replicate envs
        self.scene.clone_environments(copy_from_source=False)


        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._robot.data.prev_body_lin_vel_w = self._robot.data.body_lin_vel_w.clone()

        self._previous_actions = self._actions.clone()
        self._previous_previous_actions = self._previous_actions.clone()
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
        # lidar_obs = self.get_stacked_hits()
        height_obs = (self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5).clip(-1.0, 1.0) # -0.5 is an empirical centering offset introduced so that the height-observation distribution is centered around 0 for flat terrain
        lidar_obs = self.get_single_lidar_obs()
        obs_policy = torch.cat(
            [
                self._robot.data.root_lin_vel_b,                  # (N,3) → vx, vy, vz
                self._robot.data.root_ang_vel_b,                  # (N,3) → ωx, ωy, ωz
                self._robot.data.projected_gravity_b,             # (N,3) → gx, gy, gz
                self._commands,                                   # (N,4) → cmd_vx, cmd_vy, cmd_yaw_rate, heading
                self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,ndof) joint pos error
                self._robot.data.joint_vel,                       # (N,ndof) joint velocities
                self._actions,                                    # (N,12) previous actions
                lidar_obs,                                        # (N, 135) lidar hits
            ],
            dim=-1,
        )

        # Critic observations (privileged)
        privileged = torch.cat(
            [
                obs_policy, #(N, 184)
                self._robot.data.root_pos_w,            # (N, 3)
                self._robot.data.root_quat_w,           # (N, 4)
                self._robot.data.applied_torque,        # (N, 12)
                self._contact_sensor.data.net_forces_w.reshape(self.num_envs, -1), #(N, 57)
                self._contact_sensor.data.last_air_time.reshape(self.num_envs, -1), #(N, 19)
                height_obs,                             # (N, 35)
            ],
            dim=-1,
        )
        self._step_counter += 1


        # print("obs dim:", obs_policy.shape[-1], "state dim:", privileged.shape[-1])
        # print("net_contact_forces_w dim:", self._contact_sensor.data.net_forces_w.reshape(self.num_envs, -1).shape, "last_air_time dim:",self._contact_sensor.data.last_air_time.reshape(self.num_envs, -1).shape)
        # print("height_obs dim:", height_obs.shape, "lidar_obs dim:", lidar_obs.shape, "actions dim:", self._actions.shape, "commands dim:", self._commands.shape)
        # print("lidar_obs[0] as list:", lidar_obs[0])
        self._visualize_lidar_origin()
        self._visualize_velocity_arrows()
        self._visualize_external_force_arrows()

        ###### Used for forward_progress_position
        # 1. Read current x position
        x_now = self._robot.data.root_pos_w[:, 0]
        # 3. Update _prev_root_x AFTER computing dx
        self._prev_root_x = x_now.clone()

        return {
            "policy": obs_policy,      # for actor network
            "critic": privileged,      # for critic network
        }

    def _get_rewards(self) -> torch.Tensor:
        total, terms = compute_all_rewards(self)
        if not self._episode_sums:
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in terms.keys()
            }
        # accumulate episodic sums for logging (same keys as terms)
        for k, v in terms.items():
            self._episode_sums[k] += v
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute episode termination signals (orientation, contact, bounds, timeout)."""

        # Termination terms for this single-phase branch.
        time_outs = time_out(self)
        base_contact = illegal_contact(self, threshold=5.0, body_names=["base"])
        oob = out_of_bounds(self, margin=0.5)

        # Task-specific termination terms:
        flipped = flipped_over(self, threshold=-0.2)
        stuck_term = stuck(self)
        end_term = end_point_termination(self)

        terminated = base_contact | oob | flipped | stuck_term | end_term


        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        # DirectRLEnv applies reset-mode EventManager terms here (including DR terms from cfg.events).
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # stagger resets to avoid spikes
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # clear actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0

        # reset stuck counters
        self._stuck_counter[env_ids] = 0

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]


        # Robot spawn position from terrain
        base_origin = self._terrain.env_origins[env_ids].clone()

        # move a little backward from the first step (assuming stairs go +X)
        base_origin[:, 0] -= self.cfg.base_x_offset
        # lift robot slightly so it’s not intersecting the mesh
        base_origin[:, 2] += self.cfg.base_z_offset


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

        self._robot.data.prev_body_lin_vel_w = self._robot.data.body_lin_vel_w.clone()

        # Reset optional reward trackers
        if hasattr(self, "_stagnation_buffer"):
            # Fill rolling buffer with current position so stagnation doesn't trigger immediately
            current_x = self._robot.data.root_pos_w[env_ids, 0:1]  # [len(env_ids), 1]
            self._stagnation_buffer[env_ids, :] = current_x.expand(-1, self._stagnation_buffer.shape[1])


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
        """Command resampling policy for this branch."""
        num_envs = len(env_ids)

        # -------------------------
        # Stairs / ICRA behavior:
        # forward speed only, fixed heading
        # -------------------------
        heading = torch.zeros(num_envs, device=self.device)
        self._commands[env_ids, 3] = heading

        p_stop = 0.20  # 20% exact zero-speed commands
        speed = torch.empty(num_envs, device=self.device).uniform_(0.4, 1.0)

        stop_mask = torch.rand(num_envs, device=self.device) < p_stop
        speed[stop_mask] = 0.0

        self._commands[env_ids, 0] = speed
        self._commands[env_ids, 1] = 0.0
        self._commands[env_ids, 2] = 0.0 





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


    def get_hits_norm(self, hits_ds: torch.Tensor):
        """Normalize to 70 m range after downsampling."""
        hits_b_ds_norm = hits_ds / self._lidar_range
        return hits_b_ds_norm
     
    
    def get_single_lidar_obs(self, env_ids=None):
        """
        Return current frame LiDAR hits:
        - base frame
        - normalized to [0, 1]
        - flattened (N, R*3)
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 1. Convert world → base frame
        hits_b = self.get_bf_hits(env_ids)             # [N, R, 3]

        # 2. Normalize (divide by max lidar range)
        hits_norm = self.get_hits_norm(hits_b)         # [N, R, 3]

        # 3. Flatten because policy expects (N, ?)
        hits_flat = hits_norm.reshape(self.num_envs, -1)

        return hits_flat      

    def _visualize_lidar_origin(self, env_id=0):
        """Visualize a small sphere where the RayCaster (LiDAR) is attached."""
        # Retrieve the LiDAR sensor
        lidar = self._lidar_scanner

        # Get the LiDAR origin pose in world frame
        offset_tensor = torch.tensor(self.cfg.lidar_scanner.offset.pos, device=self.device)
        lidar_pos_w = lidar.data.pos_w[env_id] + offset_tensor  # [3]
        #lidar_quat_w = lidar.data.quat_w[env_id] + self.cfg.height_scanner.offset.quat  # [4]

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

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        M = env_ids.shape[0]

        base_pos_w = self._robot.data.root_pos_w[env_ids].clone()
        base_quat_w = self._robot.data.root_quat_w[env_ids]
        base_pos_w[:, 2] += height_offset

        default_scale = torch.tensor(base_marker_scale, device=self.device).unsqueeze(0).repeat(M, 1)
        
        # ================= Command (green) arrow =================
        # World-frame sampling path: rotate commanded (vx, vy) by heading to body intent.
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

    def _visualize_external_force_arrows(
        self,
        env_ids=None,
        base_marker_scale=(0.5, 0.5, 0.5),
        scale_mult=0.15,
        height_offset=0.5,
        min_force_to_draw=0.05,
    ):
        """Visualize currently applied base external force (reset/interval randomization)."""
        if self._force_markers is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        M = env_ids.shape[0]
        base_body_id = int(getattr(self, "_base_body_id", 0))

        base_pos_w = self._robot.data.root_pos_w[env_ids].clone()
        base_quat_w = self._robot.data.root_quat_w[env_ids]
        base_pos_w[:, 2] += height_offset

        # IsaacLab stores applied external wrench in these per-body buffers.
        force_buffer = getattr(self._robot, "_external_force_b", None)
        if force_buffer is None:
            force_buffer = getattr(self._robot.data, "external_force_b", None)

        if force_buffer is not None and force_buffer.ndim == 3 and force_buffer.shape[1] > base_body_id:
            force_b = force_buffer[env_ids, base_body_id, :]
        else:
            force_b = torch.zeros((M, 3), dtype=base_pos_w.dtype, device=self.device)

        # Use full 3D body-frame force so interval-sampled vertical pushes are also visible.
        force_mag = torch.linalg.norm(force_b, dim=1)

        default_scale = torch.tensor(base_marker_scale, device=self.device).unsqueeze(0).repeat(M, 1)
        arrow_scale = default_scale.clone()
        arrow_scale[:, 0] *= force_mag * scale_mult

        # Build local yaw/pitch so +X arrow aligns with 3D force direction in base frame.
        heading = torch.atan2(force_b[:, 1], force_b[:, 0])
        xy_norm = torch.linalg.norm(force_b[:, :2], dim=1)
        pitch = -torch.atan2(force_b[:, 2], xy_norm + 1.0e-8)
        zeros = torch.zeros_like(heading)
        arrow_quat_local = quat_from_euler_xyz(zeros, pitch, heading)
        arrow_quat = quat_mul(base_quat_w, arrow_quat_local)

        # Hide tiny forces for cleaner visualization.
        tiny_mask = force_mag < min_force_to_draw
        arrow_scale[tiny_mask] = 0.0

        self._force_markers.visualize(
            translations=base_pos_w.cpu().numpy(),
            orientations=arrow_quat.cpu().numpy(),
            scales=arrow_scale.cpu().numpy(),
        )
