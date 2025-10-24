# rewards.py
import torch
from typing import Dict, Tuple

def track_lin_vel_xy_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    lin_vel_err = torch.sum(torch.square(env._commands[:, :2] - env._robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_err / std2)


def track_ang_vel_z_exp(env, std2: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    err = torch.square(env._commands[:, 2] - env._robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-err / std2)


def lin_vel_z_l2(env) -> torch.Tensor:
    """Penalize vertical linear velocity."""
    return torch.square(env._robot.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env) -> torch.Tensor:
    """Penalize body roll/pitch rates."""
    return torch.sum(torch.square(env._robot.data.root_ang_vel_b[:, :2]), dim=1)


def joint_torques_l2(env) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    return torch.sum(torch.square(env._robot.data.applied_torque), dim=1)


def joint_acc_l2(env) -> torch.Tensor:
    """Penalize jerk (smoothness)."""
    return torch.sum(torch.square(env._robot.data.joint_acc), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env._actions - env._previous_actions), dim=1)


def feet_air_time(env, threshold: float = 0.5, min_cmd_xy: float = 0.1) -> torch.Tensor:
    """Reward longer swing time on touchdown; masked when standing."""
    first_contact = env._contact_sensor.compute_first_contact(env.step_dt)[:, env._feet_ids]
    last_air_time = env._contact_sensor.data.last_air_time[:, env._feet_ids]
    r = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    moving = (torch.norm(env._commands[:, :2], dim=1) > min_cmd_xy)
    return r * moving


def undesired_contacts(env, threshold: float = 1.0) -> torch.Tensor:
    """Count contacts (above threshold) on 'bad' bodies (thighs)."""
    f = env._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(f[:, :, env._undesired_contact_body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    return torch.sum(torch.square(env._robot.data.projected_gravity_b[:, :2]), dim=1)

# def base_height_gaussian(env, std: float = 0.05, fallback_h: float = 0.38) -> torch.Tensor:
#     """
#     Positive reward: bell curve centered at a per-env reference height.
#     Peak 1.0 at ref; decays as |z - ref| grows.
#     """
#     z = env._robot.data.root_pos_w[:, 2]
#     h_ref = getattr(env, "_stand_height_ref", None)
#     if h_ref is None or h_ref.numel() == 0:
#         h_ref = torch.full_like(z, fallback_h)
#     return torch.exp(-((z - h_ref) ** 2) / (std ** 2))

# def base_height_below_hinge(env, margin: float = -0.02):
#     """
#     Penalty (to be used with a negative scale): if base Z is below (h_ref + margin),
#     returns the shortfall amount. Zero when at/above the threshold.
#     """
#     z = env._robot.data.root_pos_w[:, 2]
#     h_ref = getattr(env, "_stand_height_ref", None)
#     if h_ref is None or h_ref.numel() == 0:
#         h_ref = torch.full_like(z, 0.38)
#     h_min = h_ref + margin  # allow small crouch
#     return torch.relu(h_min - z)

def reward_terms(env) -> Dict[str, torch.Tensor]:
    """Compute raw (unscaled) terms."""
    return {
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp(env, std2=0.25),
        "track_ang_vel_z_exp":  track_ang_vel_z_exp(env,  std2=0.25),
        "lin_vel_z_l2":         lin_vel_z_l2(env),
        "ang_vel_xy_l2":        ang_vel_xy_l2(env),
        "dof_torques_l2":       joint_torques_l2(env),
        "dof_acc_l2":           joint_acc_l2(env),
        "action_rate_l2":       action_rate_l2(env),
        "feet_air_time":        feet_air_time(env, threshold=0.5, min_cmd_xy=0.1),
        "undesired_contacts":   undesired_contacts(env, threshold=1.0),
        "flat_orientation_l2":  flat_orientation_l2(env),

        # NEW standing terms:
        # "base_height_gauss":    base_height_gaussian(env, std=0.05),
        # "base_height_below":    base_height_below_hinge(env, margin=-0.02),

        
    }


def total_reward_and_terms(env) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Scale terms by cfg weights and step_dt; then sum."""
    t = reward_terms(env)

    # existing scales
    lin_vel_scale = getattr(env.cfg, "lin_vel_reward_scale", 1.0)
    yaw_rate_scale = getattr(env.cfg, "yaw_rate_reward_scale", 0.5)
    z_vel_scale = getattr(env.cfg, "z_vel_reward_scale", -2.0)
    ang_vel_scale = getattr(env.cfg, "ang_vel_reward_scale", -0.05)
    torque_scale = getattr(env.cfg, "joint_torque_reward_scale", -2.5e-5)
    accel_scale = getattr(env.cfg, "joint_accel_reward_scale", -2.5e-7)
    action_rate_scale = getattr(env.cfg, "action_rate_reward_scale", -0.01)
    feet_air_time_scale = getattr(env.cfg, "feet_air_time_reward_scale", 0.5)
    undesired_contact_scale = getattr(env.cfg, "undesired_contact_reward_scale", -1.0)
    flat_orientation_scale = getattr(env.cfg, "flat_orientation_reward_scale", -5.0)

    # base_height_gauss_scale = getattr(env.cfg, "base_height_reward_scale", 1.0)
    # base_height_below_scale = getattr(env.cfg, "base_height_below_penalty_scale", -5.0)


    # scale & integrate over dt
    dt = env.step_dt
    scaled: Dict[str, torch.Tensor] = {
        "track_lin_vel_xy_exp": t["track_lin_vel_xy_exp"] * lin_vel_scale * dt,
        "track_ang_vel_z_exp":  t["track_ang_vel_z_exp"]  * yaw_rate_scale * dt,
        "lin_vel_z_l2":         t["lin_vel_z_l2"]         * z_vel_scale * dt,
        "ang_vel_xy_l2":        t["ang_vel_xy_l2"]        * ang_vel_scale * dt,
        "dof_torques_l2":       t["dof_torques_l2"]       * torque_scale * dt,
        "dof_acc_l2":           t["dof_acc_l2"]           * accel_scale * dt,
        "action_rate_l2":       t["action_rate_l2"]       * action_rate_scale * dt,
        "feet_air_time":        t["feet_air_time"]        * feet_air_time_scale * dt,
        "undesired_contacts":   t["undesired_contacts"]   * undesired_contact_scale * dt,
        "flat_orientation_l2":  t["flat_orientation_l2"]  * flat_orientation_scale * dt,

        # "base_height_gauss":    t["base_height_gauss"]    * base_height_gauss_scale * dt,
        # "base_height_below":    t["base_height_below"]    * base_height_below_scale * env.step_dt,

    }

    total = torch.sum(torch.stack(list(scaled.values())), dim=0)
    return total, scaled
