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


# def out_of_bounds(env, margin: float = 0.5, ground_contact_threshold: float = 0.1) -> torch.Tensor:
#     """
#     Terminate when out of the allowed region.

#     - Phase 4: terminate on contact with /World/ground.
#     - Other phases: keep legacy below-ground check.
#     """
#     # Keep this argument for compatibility with existing callers.
#     del margin

#     if getattr(env, "phase_id", -1) == 4 and getattr(env, "_ground_contact_sensor", None) is not None:
#         force_hist = env._ground_contact_sensor.data.force_matrix_w_history
#         # Expected shape: [N, T, B, M, 3], where M is the number of filter prims.
#         if force_hist is not None and force_hist.numel() > 0 and force_hist.shape[3] > 0:
#             force_mag = torch.norm(force_hist, dim=-1)
#             max_force = torch.amax(force_mag, dim=(1, 2, 3))
#             return max_force > ground_contact_threshold
#         return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

#     base_pos = env._robot.data.root_pos_w           # [N, 3]
#     env_origins = env._terrain.env_origins          # [N, 3]

#     # below ground level
#     below_ground = base_pos[:, 2] < (env_origins[:, 2] - 0.1)

#     return below_ground

def out_of_bounds(env, margin: float = 0.5) -> torch.Tensor:
    """
    Terminate when the robot leaves its assigned area or falls below ground.
    """
    base_pos = env._robot.data.root_pos_w           # [N, 3]
    env_origins = env._terrain.env_origins          # [N, 3]

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

def stuck(env, vel_thresh: float = 0.03, cmd_thresh: float = 0.2, stuck_time: float = 2.0) -> torch.Tensor:
    """
    Terminate when the robot is commanded to move forward but makes no progress
    for a prolonged period (i.e., stuck on a stair).

    Conditions:
      - Forward command:          cmd_vx > cmd_thresh
      - Actual forward velocity:  |vxb| < vel_thresh
      - Persistence: must remain 'still' for stuck_time seconds.

    Requires env to maintain:
      env._stuck_counter  (int32 tensor [N])

    Returns:
      stuck_mask: BoolTensor[N]
    """

    dt = env.step_dt
    threshold_steps = int(stuck_time / dt)

    # Base-frame forward velocity
    vxb = env._robot.data.root_lin_vel_b[:, 0]    # [N]

    # Forward intention
    cmd_vx = env._commands[:, 0]                  # [N]
    forward_intent = cmd_vx > cmd_thresh

    # Not moving forward
    no_motion = torch.abs(vxb) < vel_thresh

    # Stuck if both conditions are true
    still = forward_intent & no_motion

    # Ensure counter exists
    if not hasattr(env, "_stuck_counter"):
        env._stuck_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # Update counters
    env._stuck_counter[still] += 1
    env._stuck_counter[~still] = 0

    # Terminated if we exceed threshold steps
    stuck_mask = env._stuck_counter >= threshold_steps
    return stuck_mask

def end_point_termination(env) -> torch.Tensor:
    """
    Terminate when the robot reaches the end point.
    """
    # root position in WORLD frame
    x_pos = env._robot.data.root_pos_w[:, 0]   # [N]
    end_point = getattr(env.cfg, "end_point_pos", None)    # float
    return x_pos > end_point
