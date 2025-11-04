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

def forward_progress(env) -> torch.Tensor:
    """Encourage forward body velocity in x (in body frame)."""
    return env._robot.data.root_lin_vel_b[:, 0].clip(min=0.0)

def flat_orientation(env) -> torch.Tensor:
    """Reward staying upright (body z-axis aligned with world +z)."""
    # gravity_b = env._robot.data.projected_gravity_b  # projected gravity in body frame
    # # Upright gives cos(theta) ≈ 1, upside-down gives cos(theta) ≈ -1
    # uprightness = gravity_b[:, 2]
    result = torch.sum(torch.square(env._robot.data.projected_gravity_b[:, :2]), dim=1)
    return result


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

def undesired_contacts(env, threshold: float = 1.0) -> torch.Tensor:
    f = env._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(f[:, :, env._undesired_contact_body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

def energy_penalty(env):
    """ Controlling the Solo12 quadruped robot with deep reinforcement learning https://www.nature.com/articles/s41598-023-38259-7""" 
    # approximate energy consumption
    torque = env._robot.data.applied_torque
    joint_vel = env._robot.data.joint_vel
    return torch.sum(torch.abs(torque * joint_vel), dim=1)

def compute_all_rewards(env) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    raw: Dict[str, torch.Tensor] = {
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp(env),
        "track_ang_vel_z_exp": track_ang_vel_z_exp(env),
        "forward_progress": forward_progress(env),
        "flat_orientation": flat_orientation(env),
        "lin_vel_z_penalty": lin_vel_z_penalty(env),
        "ang_vel_xy_penalty": ang_vel_xy_penalty(env),
        "joint_torque_penalty": joint_torque_penalty(env),
        "joint_acc_penalty": joint_acc_penalty(env),
        "action_rate_penalty": action_rate_penalty(env),
        "feet_air_time": feet_air_time(env),
        "undesired_contacts": undesired_contacts(env),
        "energy_penalty": energy_penalty(env),
    }

    # --- Scales: tuned for flat-ground learning ---
    w = {
        "track_lin_vel_xy_exp": 1.0,
        "track_ang_vel_z_exp": 0.5,
        "forward_progress": 0.5,
        "flat_orientation": -5.0,
        "lin_vel_z_penalty": -2.0,
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -2.5e-5,
        "joint_acc_penalty": -2.5e-7,
        "action_rate_penalty": -0.01,
        "feet_air_time": 0.5,
        "undesired_contacts": -1.0,
        "energy_penalty": -0.000001,
    }

    dt = env.step_dt
    scaled: Dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        scaled[key] = val * w[key] * dt

    total_reward = torch.sum(torch.stack(list(scaled.values())), dim=0)
    return total_reward, scaled
