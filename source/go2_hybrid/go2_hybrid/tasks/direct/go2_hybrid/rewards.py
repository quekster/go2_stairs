# rewards.py
import torch
from typing import Dict, Tuple
import math

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

def feet_air_time(env, threshold: float = 0.5, min_cmd_xy: float = 0.1) -> torch.Tensor:
    """Encourage regular stepping by rewarding longer swing time upon touchdown."""
    first_contact = env._contact_sensor.compute_first_contact(env.step_dt)[:, env._feet_ids]
    last_air_time = env._contact_sensor.data.last_air_time[:, env._feet_ids]
    r = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    moving = (torch.norm(env._commands[:, :2], dim=1) > min_cmd_xy)
    return r * moving


def feet_slide(env) -> torch.Tensor:
    """Penalize foot sliding when in contact with the ground."""
    contact = env._contact_sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0] > 1.0
    foot_ids = env._feet_ids
    foot_vel_xy = env._robot.data.body_lin_vel_w[:, foot_ids, :2]
    return torch.sum(torch.norm(foot_vel_xy, dim=-1) * contact[:, foot_ids], dim=1)


def undesired_contacts(env, threshold: float = 1.0) -> torch.Tensor:
    f = env._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(f[:, :, env._undesired_contact_body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

def forward_progress(env) -> torch.Tensor:
    """Encourage forward body velocity in x (in body frame)."""
    return env._robot.data.root_lin_vel_b[:, 0].clip(min=0.0)

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

# def body_height_penalty(env, std: float = 0.05, stair_threshold: float = 0.15) -> torch.Tensor:
#     """
#     Penalize deviation of base height from nominal standing height,
#     but automatically disable the penalty when climbing stairs
#     (i.e., if any foot is significantly elevated above ground level).
#     """

#     # Base COM height
#     base_z = env._robot.data.root_pos_w[:, 2]
#     target_z = 0.3

#     # Raw height deviation
#     err_sq = torch.square(base_z - target_z)

#     # --- STAIR-AWARE ACTIVE MASK ---
#     # 1. Foot positions in world frame
#     foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]  # [N, 4]
#     # 2. Ground level under robot (approximate as min foot height)
#     ground_z = torch.min(foot_z, dim=1).values                 # [N]
#     # 3. Detect stair condition: any foot higher than ground_z + threshold
#     stair_condition = (foot_z - ground_z.unsqueeze(1) > stair_threshold).any(dim=1).float()
#     # 4. Detect stance: at least one foot in contact
#     contact_forces = env._contact_sensor.data.net_forces_w[:, env._feet_ids, :]
#     contact_mag = torch.norm(contact_forces, dim=-1)
#     contact_any = (contact_mag > 1.0).any(dim=1).float()
#     # 5. Disable penalty if climbing (stair_condition = 1)
#     active_mask = contact_any * (1.0 - stair_condition)

#     # Gaussian-shaped penalty
#     penalty = 1.0 - torch.exp(-err_sq / (2 * std**2))

#     return active_mask * penalty

def base_height_penalty(env, std: float = 0.05) -> torch.Tensor:
    """
    Penalize deviation of the robot's base height from its nominal standing height.

    This is the IsaacLab equivalent of 'reward_base_height' from legged-gym,
    but written as a *penalty* (larger when too low or too high).

    Args:
        env:  Go2HybridEnv (DirectRLEnv subclass).
        std:  scaling factor controlling how sharply deviations are penalized.
    Returns:
        torch.Tensor: per-env penalty values (positive = bad).
    """
    # Base COM height in world frame
    base_z = env._robot.data.root_pos_w[:, 2]

    # Reference standing height (set during env.reset())
    target_z = 0.3

    # Squared deviation
    err_sq = torch.square(base_z - target_z)

    # Optional Gaussian shaping (makes near-target small penalty, large far away)
    # penalty = 1.0 - torch.exp(-err_sq / (2 * std**2))

    return err_sq

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
        "feet_air_time": feet_air_time(env),
        "undesired_contacts": undesired_contacts(env),
        # "forward_progress": forward_progress(env),
        "flat_orientation": flat_orientation(env),
        "joint_pos_limit": joint_pos_limits(env),  # <-- NEW
        # "energy_penalty": energy_penalty(env),
        "feet_slide_penalty": feet_slide(env),
        "base_height_penalty": base_height_penalty(env),
        "foot_clearance_reward": foot_clearance_reward(env, target_height=0.10),
        "track_heading_reward": track_heading_reward(env),
        "stand_still_joint_deviation_l1": stand_still_joint_deviation_l1(env),

    }

    # --- Scales: tuned for flat-ground learning ---
    w = {
        "track_lin_vel_xy_exp": 2.0,
        "track_ang_vel_z_exp": 0.7,
        # "forward_progress": 0.5,
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -2.0e-5,
        "joint_acc_penalty": -2.0e-7,
        "action_rate_penalty": -0.5,
        "feet_air_time": 0.4,
        "undesired_contacts": -1.0,
        "flat_orientation": -4.0,
        "lin_vel_z_penalty": -2.0,
        # "energy_penalty": -1.0e-6,
        "feet_slide_penalty": -0.1,
        "base_height_penalty": -6.5,
        "foot_clearance_reward": 0.5,
        "track_heading_reward": 0.1,
        "joint_pos_limit": -0.4,
        "stand_still_joint_deviation_l1": -0.4,
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
