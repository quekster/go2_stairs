import torch
from typing import Dict, Tuple


def time_out(env) -> torch.Tensor:
    """Terminate when max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env, threshold: float = 5.0, body_names: list[str] = ["base"]) -> torch.Tensor:
    """
    Terminate when specified bodies experience large contact forces
    (e.g., base collision with ground).
    """
    contact_sensor = env._contact_sensor
    net_forces = contact_sensor.data.net_forces_w_history  # [num_envs, history_len, num_bodies, 3]

    body_ids, _ = contact_sensor.find_bodies(body_names)
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # maximum force magnitude per body across history
    max_force = torch.max(torch.norm(net_forces[:, :, body_ids], dim=-1), dim=1)[0]
    return torch.any(max_force > threshold, dim=1)


def out_of_bounds(env, margin: float = 0.5) -> torch.Tensor:
    """
    Terminate when the robot leaves its assigned area or falls below ground.
    """
    base_pos = env._robot.data.root_pos_w           # [N, 3]
    env_origins = env._terrain.env_origins          # [N, 3]

    # # distance in XY from center
    # rel_xy = base_pos[:, :2] - env_origins[:, :2]
    # out_xy = torch.any(torch.abs(rel_xy) > (env.scene.cfg.env_spacing / 2 + margin), dim=1)

    # below ground level
    below_ground = base_pos[:, 2] < (env_origins[:, 2] - 0.1)

    return below_ground

def flipped_over(env, threshold: float = 0.0) -> torch.Tensor:
    """
    Terminate when the robot is upside-down or significantly flipped.
    
    projected_gravity_b[:, 2] meaning:
        ~ -1.0   → upright
        ~  0.0   → sideways
        ~ +1.0   → upside-down or on back
    
    So if gravity_z > threshold, the robot is not upright enough.
    
    threshold = -0.2 means:
        -1.0 ... -0.2     → OK (upright-ish)
        -0.2 ... +1.0     → TERMINATE (flipped/back/side)
    """
    g_b = env._robot.data.projected_gravity_b[:, 2]
    return g_b > threshold


