# rewards.py
import torch
from typing import Dict, Tuple

def track_lin_vel_xy_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of commanded linear velocity (x,y)."""
    lin_vel_err = torch.sum(torch.square(env._commands[:, :2] - env._robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_err / std2)

def track_ang_vel_z_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of commanded yaw rate."""
    err = torch.square(env._commands[:, 2] - env._robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-err / std2)


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
    # gravity_b = env._robot.data.projected_gravity_b  # projected gravity in body frame
    # # Upright gives cos(theta) ≈ 1, upside-down gives cos(theta) ≈ -1
    # uprightness = gravity_b[:, 2]
    result = torch.sum(torch.square(env._robot.data.projected_gravity_b[:, :2]), dim=1)
    return result

def energy_penalty(env):
    """ Controlling the Solo12 quadruped robot with deep reinforcement learning https://www.nature.com/articles/s41598-023-38259-7""" 
    # approximate energy consumption
    torque = env._robot.data.applied_torque
    joint_vel = env._robot.data.joint_vel
    return torch.sum(torch.abs(torque * joint_vel), dim=1)

def body_height_reward(env, std: float = 0.05) -> torch.Tensor:
    """Encourage maintaining base height near reference standing height."""
    base_z = env._robot.data.root_pos_w[:, 2]
    err = torch.square(base_z - env._stand_height_ref)
    return torch.exp(-err / (2 * std**2))

def body_height_penalty(env, min_height: float = 0.20) -> torch.Tensor:
    """Penalize when body COM too close to ground."""
    base_z = env._robot.data.root_pos_w[:, 2]
    return (min_height - base_z).clamp(min=0.0)

def foot_clearance_reward(env, target_height: float = 0.10, std: float = 0.05, tanh_mult: float = 2.0) -> torch.Tensor:
    "Reward swinging feet for clearing specified height."
    foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]
    foot_z_error = torch.square(foot_z - target_height)    
    foot_vel_xy = torch.norm(env._robot.data.body_lin_vel_w[:, env._feet_ids, :2], dim=2)
    foot_vel_tanh = torch.tanh(tanh_mult*foot_vel_xy) #tanh function scales smoothly between 0 and 1
    reward = foot_z_error*foot_vel_tanh
    return torch.exp(-torch.sum(reward, dim=1) / (2 * std**2))

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
        "forward_progress": forward_progress(env),
        "flat_orientation": flat_orientation(env),
        "energy_penalty": energy_penalty(env),
        "feet_slide_penalty": feet_slide(env),
        "body_height_reward": body_height_reward(env),
        "body_height_penalty": body_height_penalty(env),
        "foot_clearance_reward": foot_clearance_reward(env, target_height=0.10),

    }

    # --- Scales: tuned for flat-ground learning ---
    w = {
        "track_lin_vel_xy_exp": 3.0,
        "track_ang_vel_z_exp": 0.5,
        "forward_progress": 0.5,
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -2.5e-5,
        "joint_acc_penalty": -2.5e-7,
        "action_rate_penalty": -0.5,
        "feet_air_time": 0.2,
        "undesired_contacts": -1.0,
        "flat_orientation": -5.0,
        "lin_vel_z_penalty": -2.0,
        "energy_penalty": -0.000001,
        "feet_slide_penalty": -0.1,
        "body_height_reward": 0.2,
        "body_height_penalty": -0.2,
        "foot_clearance_reward": 0.2,
    }

    dt = env.step_dt
    scaled: Dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        scaled[key] = val * w[key] * dt

    total_reward = torch.sum(torch.stack(list(scaled.values())), dim=0)
    return total_reward, scaled