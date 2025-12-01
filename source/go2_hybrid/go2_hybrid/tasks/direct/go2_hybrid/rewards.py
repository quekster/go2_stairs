# rewards.py
import torch
from typing import Dict, Tuple
import math
from isaaclab.utils.math import quat_apply, quat_conjugate

def nan_check(reward: torch.Tensor, obs_type: str):
    """Check if the reward contains NaN values."""
    if torch.any(torch.isnan(reward)):
        print(f"Reward {obs_type} contains NaN values.")
    if torch.any(torch.isinf(reward)):
        print(f"Reward {obs_type} contains Inf values.")

def track_lin_vel_xy_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of commanded linear velocity (x,y) in body frame."""
    # cmd_body = get_heading_rotated_commands(env)
    # lin_vel_err = torch.sum(torch.square(cmd_body[:, :2] - env._robot.data.root_lin_vel_b[:, :2]), dim=1)
    lin_vel_error = torch.sum(torch.square(env._commands[:, :2] - env._robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std2)

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


def feet_slide(env) -> torch.Tensor:
    """Penalize foot sliding when in contact with the ground."""
    contact = env._contact_sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0] > 1.0
    foot_ids = env._feet_ids
    foot_vel_xy = env._robot.data.body_lin_vel_w[:, foot_ids, :2]
    return torch.sum(torch.norm(foot_vel_xy, dim=-1) * contact[:, foot_ids], dim=1)

def base_height_l2_lidar(env, height_safety_margin: float = 0.03, target_height: float = 0.40,) -> torch.Tensor:
    """Penalize deviation of base height from target using LiDAR-based terrain estimate."""
    terrain_height_b = get_height_lidar(env, channel=0) + height_safety_margin  # [N]
    height_error = (-terrain_height_b) - target_height
    penalty = torch.square(height_error)
    return penalty


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

def joint_pos_limits(env) -> torch.Tensor:
    """Penalize joint positions outside soft limits."""
    lower_limits = env._robot.data.soft_joint_pos_limits[:,:,0]
    upper_limits = env._robot.data.soft_joint_pos_limits[:,:,1]
    joint_pos = env._robot.data.joint_pos

    below_lower = (lower_limits - joint_pos).clamp(min=0.0)
    above_upper = (joint_pos - upper_limits).clamp(min=0.0)

    out_of_limits = below_lower + above_upper
    return torch.sum(out_of_limits, dim=1)

def energy_penalty(env):
    """ Controlling the Solo12 quadruped robot with deep reinforcement learning https://www.nature.com/articles/s41598-023-38259-7""" 
    # approximate energy consumption
    torque = env._robot.data.applied_torque
    joint_vel = env._robot.data.joint_vel
    return torch.sum(torch.abs(torque * joint_vel), dim=1)


def foot_clearance_reward(
    env,
    desired_clearance: float = 0.20,
    safety_margin: float = 0.10,
    channels=None
) -> torch.Tensor:
    """
    Encourage each swinging foot to clear the terrain height predicted by LiDAR.
    Uses BASE FRAME for all height computations.
    """

    # -------------------------------------------------------------
    # 1) Terrain height from LiDAR (in base frame)
    # -------------------------------------------------------------
    terrain_height_b = get_height_lidar(env, channel=0)   # [N]
    terrain_height_b = terrain_height_b + safety_margin           # lift terrain a bit


    # -------------------------------------------------------------
    # 2) Foot positions in WORLD frame
    # -------------------------------------------------------------
    foot_pos_w = env._robot.data.body_pos_w[:, env._feet_ids, :]   # [N, 4, 3]


    # -------------------------------------------------------------
    # 3) Convert foot positions WORLD → BASE frame
    #     (using the SAME CORRECT METHOD as your LiDAR conversion)
    # -------------------------------------------------------------
    base_pos_w  = env._robot.data.root_pos_w                       # [N, 3]
    base_quat_w = env._robot.data.root_quat_w                      # [N, 4]
    base_quat_inv = quat_conjugate(base_quat_w)                    # [N, 4]

    # Shift into base origin
    foot_shifted = foot_pos_w - base_pos_w.unsqueeze(1)            # [N, 4, 3]

    # Expand quaternion to match foot count (exact correct pattern)
    base_quat_exp = base_quat_inv.unsqueeze(1).expand(-1, foot_shifted.shape[1], -1)
    # shape = [N, 4, 4]

    # Rotate into base frame
    foot_pos_b = quat_apply(base_quat_exp, foot_shifted)           # [N, 4, 3]
    foot_z_b = foot_pos_b[..., 2]                                  # [N, 4]


    # -------------------------------------------------------------
    # 4) Compute clearance relative to terrain height
    # -------------------------------------------------------------
    # clearance_i = foot_z - terrain_z
    clearance = foot_z_b - terrain_height_b.unsqueeze(1)           # [N, 4]


    # -------------------------------------------------------------
    # 5) Clearance error: want clearance ≥ desired_clearance
    # -------------------------------------------------------------
    # clearance_error = desired_clearance - clearance                # [N, 4]
    # clearance_penalty = torch.square(clearance_error)

    # # -------------------------------------------------------------
    # # 6) Weight penalty by swing activity (stance legs ignored)
    # # -------------------------------------------------------------
    # foot_vel_xy = torch.norm(
    #     env._robot.data.body_lin_vel_w[:, env._feet_ids, :2], dim=2
    # )   # [N, 4]

    # swing_weight = torch.tanh(2.0 * foot_vel_xy)                  # smooth gating

    # weighted_penalty = clearance_penalty * swing_weight           # [N, 4]


    # # -------------------------------------------------------------
    # # 7) Sum over 4 legs
    # # -------------------------------------------------------------
    # reward = torch.sum(weighted_penalty, dim=1)                    # [N]

    #testing new:
    per_foot_reward = torch.clamp(clearance, min=0.0, max=desired_clearance)
    reward = torch.sum(per_foot_reward, dim=1)

    nan_check(reward, "foot_clearance_reward")
    return reward


def stand_still_joint_deviation_l1(env, command_threshold: float = 0.06) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    commands = env._commands # [num_envs, 4]: [vx, vy, yaw_rate, heading]
    joint_dev = torch.sum(torch.abs(env._robot.data.joint_pos - env._robot.data.default_joint_pos), dim=1) # L1 deviation per environment (sum over all joints)
    cmd_mag = torch.norm(commands[:, :2], dim=1) # magnitude of (vx, vy) command
    return joint_dev * (cmd_mag < command_threshold)

def foot_vertical_accel_reward(env, scale=0.5):
    """
    Reward upward vertical acceleration for ANY stuck foot (front or hind).
    'Stuck' means the foot is in contact and horizontal velocity is near zero.
    """

    # Current step foot vertical velocity
    vel_now = env._robot.data.body_lin_vel_w[:, env._feet_ids, 2]   # [N,4]

    # Previous step vertical velocity
    # IMPORTANT: if your env does not store this, I can show you how to add it.
    vel_prev = env._robot.data.prev_body_lin_vel_w[:, env._feet_ids, 2]  # [N,4]

    # Vertical acceleration (finite difference)
    acc = vel_now - vel_prev    # [N,4]

    # Only upward acceleration is rewarded
    acc_up = torch.clamp(acc, min=0.0)

    # Identify stuck feet (front AND hind)
    stuck = foot_stuck_mask(env)            # [N,4], already includes all legs

    # Reward only for stuck legs
    reward = acc_up * stuck                 # [N,4]

    return torch.sum(reward, dim=1) * scale




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
        "joint_pos_limit": joint_pos_limits(env),  # <-- NEW
        "energy_penalty": energy_penalty(env),
        "feet_slide_penalty": feet_slide(env),
        "foot_clearance_reward": foot_clearance_reward(env),
        # "track_heading_reward": track_heading_reward(env),
        # "stand_still_joint_deviation_l1": stand_still_joint_deviation_l1(env),
        "smoothness_penalty": smoothness_penalty(env),
        "base_height_l2_lidar": base_height_l2_lidar(env, target_height=0.33),
        "foot_vertical_accel_reward": foot_vertical_accel_reward(env),


    }

    # --- Scales: tuned for flat-ground learning ---
    w = {
        "track_lin_vel_xy_exp": 5.0,
        "track_ang_vel_z_exp": 1.0,
         "lin_vel_z_penalty": -0.5,       
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -2.0e-5,
        "joint_acc_penalty": -2.0e-7,
        "action_rate_penalty": -0.2,
        "undesired_contacts": -1.0,
        "flat_orientation": -0.2,
        "energy_penalty": -1.0e-6,
        "feet_slide_penalty": -0.5,
        "foot_clearance_reward": 2.5,
        # "track_heading_reward": 0.1,
        "joint_pos_limit": -0.2,
        # "stand_still_joint_deviation_l1": -0.01,
        "smoothness_penalty": -0.01,
        "base_height_l2_lidar": -0.02,
        "foot_vertical_accel_reward": 0.5,
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
    Estimate terrain height in BASE FRAME from LiDAR hits.
    
    IMPORTANT:
        Verified channel ordering from your data:
            Channel 0  → lowest elevation angle (most downward)
            Channel N-1 → highest elevation (closest to horizontal)
        Therefore channel 0 is best for terrain height estimation.
    """

    # 1) Get LiDAR hits in base frame
    hits_b = env.get_bf_hits()          # [N, total_rays, 3]
    N, total_rays, _ = hits_b.shape

    # 2) Get LiDAR vertical channel layout
    num_channels = env._lidar_scanner.cfg.pattern_cfg.channels
    rays_per_ch  = total_rays // num_channels

    # 3) Default: use lowest elevation channel (closest to ground)
    if channel is None:
        channel = 0    # <-- IMPORTANT: lowest elevation

    # 4) Slice ray block belonging to this channel
    start = channel * rays_per_ch
    end   = (channel + 1) * rays_per_ch
    ch_hits = hits_b[:, start:end, :]   # [N, rays_per_ch, 3]

    # 5) Extract z-values in BASE frame
    z_vals = ch_hits[..., 2]            # negative = below robot

    # 6) Mean terrain height estimate
    #terrain_height_b = torch.mean(z_vals, dim=1)
    
    #testing to take max instead:
    terrain_height_b = torch.max(z_vals, dim=1).values

    # Debug print
    # if env._step_counter % 50 == 0:  # every 100 steps
    # print("Number of rays:", hits_b.shape[1]) 
    # print("LiDAR hits (base frame):", hits_b) 
    # print("Estimated terrain height (base frame):", terrain_height_b)
    # print("-----------")

    

    return terrain_height_b

def foot_stuck_mask(env, vel_thresh=0.03, force_thresh=10.0):
    """
    Returns mask [N, 4] of feet considered 'stuck':
    - In contact (strong force)
    - Almost no horizontal movement
    """
    forces = env._contact_sensor.data.net_forces_w[:, env._feet_ids]  # [N,4,3]
    in_contact = (forces.norm(dim=-1) > force_thresh)                  # [N,4]

    foot_vel_w = env._robot.data.body_lin_vel_w[:, env._feet_ids, :]  # [N,4,3]
    horiz_vel = torch.norm(foot_vel_w[..., :2], dim=-1)                # [N,4]

    stuck = in_contact & (horiz_vel < vel_thresh)
    return stuck.float()  # [N,4]
