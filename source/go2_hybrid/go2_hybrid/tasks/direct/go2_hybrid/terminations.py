import torch
from typing import Dict, Tuple

def time_out(env) -> torch.Tensor:
    """Terminate when max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env, threshold: float = 1.0, body_names: list[str] = ["base"]) -> torch.Tensor:
    """Terminate when the contact force on specific bodies exceeds the force threshold."""
    contact_sensor = env._contact_sensor
    net_contact_forces = contact_sensor.data.net_forces_w_history  # [num_envs, history_len, num_bodies, 3]

    body_ids, _ = contact_sensor.find_bodies(body_names)

    # Compute max contact force magnitude across history
    max_force = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0]

    # Return boolean termination mask [num_envs]
    return torch.any(max_force > threshold, dim=1)


def bad_orientation(env, limit_angle: float = 0.8) -> torch.Tensor:
    """Terminate when body tilts too far from upright orientation."""
    gravity_b = env._robot.data.projected_gravity_b  # [num_envs, 3]
    cos_angle = gravity_b[:, 2]
    return cos_angle < torch.cos(torch.tensor(limit_angle))


import torch

def out_of_bounds(env, margin: float = 0.5, contact_threshold: float = 1.0) -> torch.Tensor:
    """
    Terminate when the robot leaves its assigned terrain tile or falls below ground,
    but only *after* the robot has first touched the terrain (contact sensor detects ground contact).

    Args:
        env: The DirectRLEnv.
        margin: Extra boundary tolerance (metres).
        contact_threshold: Minimum contact force magnitude (N) to consider "landed".
    """
    # --- Base position and terrain info ---
    base_pos = env._robot.data.root_pos_w                 # [N, 3]
    env_origins = env._terrain.env_origins                # [N, 3]
    # rel_xy = base_pos[:, :2] - env_origins[:, :2]


    # # --- Terrain horizontal boundary ---
    # half_extent_x = 7.0 / 2.0
    # half_extent_y = 1.6 / 2.0

    # # --- Compute OOB conditions ---
    # out_x = torch.abs(rel_xy[:, 0]) > (half_extent_x + margin)
    # out_y = torch.abs(rel_xy[:, 1]) > (half_extent_y + margin)
    below_ground = base_pos[:, 2] < (env_origins[:, 2] - 0.5)
    # oob_mask = out_x | out_y | below_ground               # [N]

    # # --- Check out-of-bounds conditions ---
    # out_x = torch.abs(rel_xy[:, 0]) > (half_extent_x + margin)
    # out_y = torch.abs(rel_xy[:, 1]) > (half_extent_y + margin)
    below_ground = base_pos[:, 2] < (env_origins[:, 2] - 0.5)


    out_of_bounds = below_ground
    return out_of_bounds
