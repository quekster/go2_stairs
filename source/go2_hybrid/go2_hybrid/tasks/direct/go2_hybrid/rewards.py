# rewards.py
import torch
from typing import Dict, Tuple
import math

def nan_check(reward: torch.Tensor, obs_type: str):
    """Check if the reward contains NaN values."""
    if torch.any(torch.isnan(reward)):
        print(f"Reward {obs_type} contains NaN values.")
    if torch.any(torch.isinf(reward)):
        print(f"Reward {obs_type} contains Inf values.")

def track_lin_vel_xy_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of commanded linear velocity (x,y) in body frame."""
    cmd_body = get_heading_rotated_commands(env)
    lin_vel_err = torch.sum(torch.square(cmd_body[:, :2] - env._robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_err / std2)

def track_ang_vel_z_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of commanded yaw rate."""
    err = torch.square(env._commands[:, 2] - env._robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-err / std2)

def track_heading_reward(env, std2: float = 0.5) -> torch.Tensor:
    """Reward facing toward commanded heading angle."""
    # Get robot yaw from its quaternion
    quat = env._robot.data.root_quat_w
    # yaw = atan2(2*(wz + xy), 1 - 2*(y^2 + z^2))
    yaw = torch.atan2(
        2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
    )
    yaw_err = torch.square(torch.atan2(torch.sin(yaw - env._commands[:, 3]),
                                       torch.cos(yaw - env._commands[:, 3])))
    return torch.exp(-yaw_err / std2)

def lin_vel_z_penalty(env) -> torch.Tensor:
    return torch.square(env._robot.data.root_lin_vel_b[:, 2])

def ang_vel_xy_penalty(env) -> torch.Tensor:
    return torch.sum(torch.square(env._robot.data.root_ang_vel_b[:, :2]), dim=1)

def joint_torque_penalty(env) -> torch.Tensor:
    return torch.sum(torch.square(env._robot.data.applied_torque), dim=1)

def joint_acc_penalty(env) -> torch.Tensor:
    return torch.sum(torch.square(env._robot.data.joint_acc), dim=1)

def action_rate_penalty(env) -> torch.Tensor:
    delta = env._actions - env._previous_actions
    return torch.sum(delta ** 2, dim=1) / env._actions.shape[1]

# def feet_air_time(env, threshold: float = 0.5, min_cmd_xy: float = 0.1) -> torch.Tensor:
#     """Encourage regular stepping by rewarding longer swing time upon touchdown."""
#     first_contact = env._contact_sensor.compute_first_contact(env.step_dt)[:, env._feet_ids]
#     last_air_time = env._contact_sensor.data.last_air_time[:, env._feet_ids]
#     r = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     moving = (torch.norm(env._commands[:, :2], dim=1) > min_cmd_xy)
#     return r * moving


# def feet_slide(env) -> torch.Tensor:
#     """Penalize foot sliding when in contact with the ground."""
#     contact = env._contact_sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0] > 1.0
#     foot_ids = env._feet_ids
#     foot_vel_xy = env._robot.data.body_lin_vel_w[:, foot_ids, :2]
#     return torch.sum(torch.norm(foot_vel_xy, dim=-1) * contact[:, foot_ids], dim=1)

def base_height_l2_lidar(env, target_height: float) -> torch.Tensor:
    """
    Base-height L2 penalty using only the FIRST CHANNEL of a LiDAR RayCaster,
    with ray-hits converted to the robot's base frame.

    target_height:
        Desired base height ABOVE the detected terrain (in base-frame meters).
        Example: 0.33 for Go2 standing height.

    Returns:
        (N_envs,) tensor of squared error.
    """

    terrain_height_b = get_height_lidar(env)    # [N]

    # 5) Compute adjusted target height:
    #
    #   desired_base_height = target_height ABOVE terrain
    #
    #   base_frame_height_error = 0 - (terrain_height + target_height)
    #
    # Since base-frame z of robot base = 0.0,
    # the desired height above terrain is simply:
    adjusted_target = terrain_height_b + target_height

    # 6) L2 squared penalty = (base_z_b - adjusted_target)^2
    # But base_z_b = 0 → simplify:
    # penalty = adjusted_target * adjusted_target
    penalty = penalty = torch.zeros_like(terrain_height_b)

    return penalty

# def base_height_l2(
#     env: ManagerBasedRLEnv,
#     target_height: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     sensor_cfg: SceneEntityCfg | None = None,
# ) -> torch.Tensor:
#     """Penalize asset height from its target using L2 squared kernel.

#     Note:
#         For flat terrain, target height is in the world frame. For rough terrain,
#         sensor readings can adjust the target height to account for the terrain.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     if sensor_cfg is not None:
#         sensor: RayCaster = env.scene[sensor_cfg.name]
#         # Adjust the target height using the sensor data
#         adjusted_target_height = target_height + torch.mean(torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=2.0, posinf=2.0, neginf=-2.0), dim=1)
#     else:
#         # Use the provided target height directly for flat terrain
#         adjusted_target_height = target_height
#     # Compute the L2 squared penalty
#     return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)

def smoothness_penalty(env):
    """Penalize second derivative (jerk) of action like IsaacLab SmoothnessReward."""
    # a_t, a_{t-1}, a_{t-2}
    a_t   = env._actions
    a_t1  = env._previous_actions
    a_t2  = env._previous_previous_actions

    reward = torch.square(a_t - 2.0 * a_t1 + a_t2)
    reward *= a_t1 != 0.0
    reward *= a_t2 != 0.0

    # squared magnitude per env → negative because jerk is bad
    return torch.sum(reward, dim=1)

def undesired_contacts(env, threshold: float = 1.0) -> torch.Tensor:
    f = env._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(f[:, :, env._undesired_contact_body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def flat_orientation(env) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel."""
    result = torch.sum(torch.square(env._robot.data.projected_gravity_b[:, :2]), dim=1)
    return result

# def joint_pos_limits(env) -> torch.Tensor:
#     """Penalize joint positions outside soft limits."""
#     lower_limits = env._robot.data.soft_joint_pos_limits[:,:,0]
#     upper_limits = env._robot.data.soft_joint_pos_limits[:,:,1]
#     joint_pos = env._robot.data.joint_pos

#     below_lower = (lower_limits - joint_pos).clamp(min=0.0)
#     above_upper = (joint_pos - upper_limits).clamp(min=0.0)

#     out_of_limits = below_lower + above_upper
#     return torch.sum(out_of_limits, dim=1)

def energy_penalty(env):
    """ Controlling the Solo12 quadruped robot with deep reinforcement learning https://www.nature.com/articles/s41598-023-38259-7""" 
    # approximate energy consumption
    torque = env._robot.data.applied_torque
    joint_vel = env._robot.data.joint_vel
    return torch.sum(torch.abs(torque * joint_vel), dim=1)


def foot_clearance_reward(env, target_height: float = 0.10, std: float = 0.05, tanh_mult: float = 2.0) -> torch.Tensor:
    "Reward swinging feet for clearing specified height."
    foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]
    foot_z_error = torch.square(foot_z - target_height)    
    foot_vel_xy = torch.norm(env._robot.data.body_lin_vel_w[:, env._feet_ids, :2], dim=2)
    foot_vel_tanh = torch.tanh(tanh_mult*foot_vel_xy) #tanh function scales smoothly between 0 and 1
    reward = foot_z_error*foot_vel_tanh
    return torch.exp(-torch.sum(reward, dim=1) / (2 * std**2))

def stand_still_joint_deviation_l1(env, command_threshold: float = 0.06) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    commands = env._commands # [num_envs, 4]: [vx, vy, yaw_rate, heading]
    joint_dev = torch.sum(torch.abs(env._robot.data.joint_pos - env._robot.data.default_joint_pos), dim=1) # L1 deviation per environment (sum over all joints)
    cmd_mag = torch.norm(commands[:, :2], dim=1) # magnitude of (vx, vy) command
    return joint_dev * (cmd_mag < command_threshold)




def compute_all_rewards(env) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    raw: Dict[str, torch.Tensor] = {
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp(env),
        "track_ang_vel_z_exp": track_ang_vel_z_exp(env),
        "lin_vel_z_penalty": lin_vel_z_penalty(env),
        "ang_vel_xy_penalty": ang_vel_xy_penalty(env),
        "joint_torque_penalty": joint_torque_penalty(env),
        "joint_acc_penalty": joint_acc_penalty(env),
        "action_rate_penalty": action_rate_penalty(env),
        "undesired_contacts": undesired_contacts(env),
        "flat_orientation": flat_orientation(env),
        # "joint_pos_limit": joint_pos_limits(env),  # <-- NEW
        "energy_penalty": energy_penalty(env),
        # "feet_slide_penalty": feet_slide(env),
        # "base_height_penalty": base_height_penalty(env),
        "foot_clearance_reward": foot_clearance_reward(env, target_height=0.10),
        "track_heading_reward": track_heading_reward(env),
        # "stand_still_joint_deviation_l1": stand_still_joint_deviation_l1(env),
        "smoothness_penalty": smoothness_penalty(env),
        "base_height_l2_lidar": base_height_l2_lidar(env, target_height=0.33),

    }

    # --- Scales: tuned for flat-ground learning ---
    w = {
        "track_lin_vel_xy_exp": 2.0,
        "track_ang_vel_z_exp": 0.7,
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -2.0e-5,
        "joint_acc_penalty": -2.0e-7,
        "action_rate_penalty": -0.2,
        "undesired_contacts": -0.5,
        "flat_orientation": -0.2,
        "lin_vel_z_penalty": -2.0,
        "energy_penalty": -1.0e-6,
        # "feet_slide_penalty": -0.1,
        # "base_height_penalty": -6.5,
        "foot_clearance_reward": 2.5,
        "track_heading_reward": 0.1,
        # "joint_pos_limit": -0.05,
        # "stand_still_joint_deviation_l1": -0.01,
        "smoothness_penalty": -0.01,
        "base_height_l2_lidar": -1.0,
    }

    dt = env.step_dt
    scaled: Dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        scaled[key] = val * w[key] * dt

    total_reward = torch.sum(torch.stack(list(scaled.values())), dim=0)
    return total_reward, scaled

def get_heading_rotated_commands(env) -> torch.Tensor:
    """
    Rotate commanded (x, y) velocities from world-heading frame into body frame
    using the commanded heading angle.
    Returns tensor [num_envs, 3] (vx_body, vy_body, yaw_rate)
    """
    # commanded heading
    heading = env._commands[:, 3]
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    vx = env._commands[:, 0]
    vy = env._commands[:, 1]

    # rotation from world heading to body frame
    vx_rot = cos_h * vx + sin_h * vy
    vy_rot = -sin_h * vx + cos_h * vy

    yaw_rate = env._commands[:, 2]
    return torch.stack((vx_rot, vy_rot, yaw_rate), dim=1)


def get_height_lidar(env, channel: int = None) -> torch.Tensor:
    """
    Returns the estimated terrain height (in BASE FRAME) using LiDAR ray hits.
    This extracts rays from a selected vertical channel and computes the mean
    hit height (z-value) for each environment.

    Args:
        env: the IsaacLab RL environment with:
             - env.get_bf_hits(): returns LiDAR hits in base frame [N, R, 3]
             - env._lidar_scanner.cfg.pattern_cfg.channels: number of vertical channels

        channel: optional integer specifying which channel to use.
                 If None → automatically selects the LOWEST (most downward) channel.

    Returns:
        terrain_height_b: (N_envs,) tensor containing terrain height estimate
                          relative to base frame (z-value).

                          Example values:
                            -0.32 → ground 32 cm below base
                            -0.10 → step is near base height
                             0.00 → LiDAR rays hitting base plane
    """

    # Retrieve hits in base frame
    hits_b = env.get_bf_hits() # hits_b: [N_envs, total_rays, 3]

    # if env._step_counter % 50 == 0: 
    print("Number of rays:", hits_b.shape[1])
    print("LiDAR hits (base frame):", hits_b)
    print("-----------")
    # ----------------------------------------------
    # 2) Determine channel layout
    # ----------------------------------------------
    num_channels = env._lidar_scanner.cfg.pattern_cfg.channels
    total_rays   = hits_b.shape[1]
    rays_per_ch  = total_rays // num_channels

    # Default channel: lowest / most downward-facing
    if channel is None:
        channel = num_channels - 1

    # ----------------------------------------------
    # 3) Slice rays belonging to this channel
    # ----------------------------------------------
    start = channel * rays_per_ch
    end   = (channel + 1) * rays_per_ch
    ch_hits = hits_b[:, start:end, :]     # shape: [N_envs, rays_per_ch, 3]

    # ----------------------------------------------
    # 4) Extract z-values (height in base frame)
    # ----------------------------------------------
    z_vals = ch_hits[..., 2]              # [N_envs, rays_per_ch]

    # ----------------------------------------------
    # 5) Mean terrain height estimate per environment
    # ----------------------------------------------
    terrain_height_b = torch.mean(z_vals, dim=1)     # [N_envs]

    return terrain_height_b