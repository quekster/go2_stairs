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
from isaaclab.utils.math import quat_from_angle_axis, yaw_quat, quat_apply  # used later

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import omni.timeline

from .go2_hybrid_env_cfg import Go2HybridEnvCfg
from .rewards import total_reward_and_terms


class Go2HybridEnv(DirectRLEnv):
    
    cfg: Go2HybridEnvCfg

    def __init__(self, cfg: Go2HybridEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._step_counter =0
        self._marker= None

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
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                # "base_height_gauss",
                # "base_height_below",
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


        # # Contact sensor (cover entire robot; history for air-time)
        # contact_cfg = ContactSensorCfg(
        #     prim_path="/World/envs/env_.*/Robot/.*",
        #     history_length=3,
        #     update_period=0.005,
        #     track_air_time=True,
        # )

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner=RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"]=self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

       #-------drawing a visual debug cube at world origin-------#
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/DebugMarkers", 
            markers={
                "origin_box": sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(1.0, 1.0, 1.0),
                ),
            }
        )
        self._marker = VisualizationMarkers(marker_cfg)
        translations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # shape (1,3)
        self._marker.visualize(translations=translations)



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
        #lidar_scan = torch.clamp(self._height_scanner.data.ray_hits_)
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,                              # (N,3)
                self._robot.data.root_ang_vel_b,                              # (N,3)
                self._robot.data.projected_gravity_b,                         # (N,3)
                self._commands,                                               # (N,3)
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # (N,ndof)
                self._robot.data.joint_vel,                                   # (N,ndof)
                self._actions,                                                # (N,ndof)
            ],
            dim=-1,
        )
        self._step_counter += 1
        if self._step_counter % 100 == 0:  # every 100 steps
            print("Step", self._step_counter, 
                "Front ray hits (world frame):",
                self._height_scanner.data.ray_hits_w[0, 85:95])
            self.plot_lidar_3d(env_id=0)    



        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total, terms = total_reward_and_terms(self)
        # accumulate episodic sums for logging (same keys as terms)
        for k, v in terms.items():
            self._episode_sums[k] += v
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: 
        time_out = self.episode_length_buf >= self.max_episode_length - 1 
        net_contact_forces = self._contact_sensor.data.net_forces_w_history 
        # terminate if strong contact on base link 
        died = torch.any( torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1 ) 
        return died, time_out
   
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
        base_origin[:, 0] -= 2.0
        # lift robot slightly so it’s not intersecting the mesh
        base_origin[:, 2] += 0.5


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

    def plot_lidar_3d(self, env_id = 0):
        tl = omni.timeline.get_timeline_interface()
        tl.pause()

        lidar = self.scene.sensors["height_scanner"]
        hits = lidar.data.ray_hits_w[env_id].detach().cpu().numpy()

        mask = np.isfinite(hits).all(axis=1)
        hits = hits[mask]

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(hits[:, 0], hits[:, 1], hits[:, 2], s=2, c=hits[:, 2], cmap='viridis')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Lidar hits (world frame) – Env {env_id}, Step {self._step_counter}')

        # Optional: set equal aspect ratio for better spatial perception
        max_range = np.ptp(hits, axis=0).max() / 2.0
        mid = np.mean(hits, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()
        plt.show(block=True)
        plt.close()
        tl.play()
