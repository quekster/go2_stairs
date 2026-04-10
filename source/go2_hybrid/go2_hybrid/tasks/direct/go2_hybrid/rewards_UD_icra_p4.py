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
    lin_vel_error = torch.sum(torch.square(env._commands[:, :2] - env._robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std2)


def track_modified_vel_reward(env, base_std2=0.25, pitch_thresh=0.15):
    """
    Blends base-frame velocity tracking with world-frame forward progression,
    depending on pitch angle. Smooth transition avoids reward conflict.
    """

    # --- base-frame tracking ---
    vel_b = env._robot.data.root_lin_vel_b[:, :2]
    cmd   = env._commands[:, :2]
    base_error = torch.sum((cmd - vel_b)**2, dim=1)
    track_base = torch.exp(-base_error / base_std2)

    # --- world-frame forward progress ---
    track_world = world_aligned_velocity_reward(env)   # from earlier

    # --- compute pitch magnitude ---
    pitch = get_pitch_from_quat(env._robot.data.root_quat_w).abs()

    # pitch-based blending
    weight = torch.sigmoid( 5.0 * (pitch - pitch_thresh) )

    # --- blend ---
    reward = (1 - weight) * track_base + weight * track_world

    return reward

def world_aligned_velocity_reward(env, scale=1.0):
    """
    Reward the robot for producing world-frame forward motion
    even when pitched or rolled.
    This rotates the desired command into BASE frame, so the
    robot learns to compensate for orientation.
    """

    # desired direction in WORLD frame (normalized)
    cmd = env._commands[:, :2]                # (vx, vy)
    cmd_3d = torch.cat([cmd, torch.zeros_like(cmd[:, :1])], dim=1)   # [N,3]
    cmd_norm = cmd_3d / (torch.norm(cmd_3d, dim=1, keepdim=True) + 1e-6)

    # actual velocity in WORLD frame
    vel_w = env._robot.data.root_lin_vel_w[:, :3]                   # [N,3]

    # convert ACTUAL velocity into BASE frame
    base_quat = env._robot.data.root_quat_w                          # [N,4]
    base_quat_inv = quat_conjugate(base_quat)
    vel_b = quat_apply(base_quat_inv, vel_w)                         # [N,3]

    # convert DESIRED world direction into BASE frame
    desired_dir_b = quat_apply(base_quat_inv, cmd_norm)              # [N,3]

    # cosine similarity = alignment
    alignment = torch.sum(vel_b * desired_dir_b, dim=1)

    # keep reward positive
    return scale * torch.relu(alignment)



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

def base_height_l2_lidar(env, target: float = 0.30, std: float = 0.05) -> torch.Tensor:
    """"
    Penalize deviation of the robot's base height from a terrain-relative target height.
    To maintain a stable target above whatever terrain the height_scanner detects.
    """
    base_z = env._robot.data.root_pos_w[:, 2]
    height_hits_z = env._height_scanner.data.ray_hits_w[..., 2]
    terrain_z = torch.mean(torch.nan_to_num(height_hits_z,nan=2.0, posinf=2.0, neginf=-2.0), dim=1)
    adjusted_target_z = terrain_z + target
    err_sq = torch.square(base_z - adjusted_target_z)
    return err_sq


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

#Phase 0: flat walking
def flat_orientation(env) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel."""
    result = torch.sum(torch.square(env._robot.data.projected_gravity_b[:, :2]), dim=1)
    return result

def flat_orientation_roll(env) -> torch.Tensor:
    """
    Penalize ROLL deviation only.
    Pitch is intentionally NOT penalized so the robot can lean forward
    while climbing stairs.
    """
    g_b = env._robot.data.projected_gravity_b   # [N,3]

    roll_component = g_b[:, 0]    # X component → roll
    # pitch_component = g_b[:, 1] # Y component → pitch (ignored)

    return torch.square(roll_component)

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
    desired_clearance: float = 0.10,
    safety_margin: float = 0.05,
    radius: float = 0.12,
    channels=None
    ) -> torch.Tensor:
    """
    Encourage each swinging foot to clear the terrain height predicted by LiDAR.
    Uses BASE FRAME for all height computations.
    """

    # -------------------------------------------------------------
    # 1) Terrain height from LiDAR (in base frame)
    # -------------------------------------------------------------
    # terrain_height_b = get_height_lidar(env, channel=0)   # [N]
    # terrain_height_b = terrain_height_b + safety_margin           # lift terrain a bit
    terrain_height_b = feet_height_scanner(env, radius=radius)   # [N]
    terrain_height_b = terrain_height_b + safety_margin           # lift terrain a bit
    terrain_height_b = torch.nan_to_num(terrain_height_b, nan=0.0, posinf=0.0, neginf=0.0)


    # -------------------------------------------------------------
    # 2) Foot positions in WORLD frame
    # -------------------------------------------------------------
    foot_pos_w = env._robot.data.body_pos_w[:, env._feet_ids, :]   # [N, 4, 3]


    # -------------------------------------------------------------
    # 3) Convert foot positions WORLD → BASE frame
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
    clearance = foot_z_b - terrain_height_b          # [N, 4]


    per_foot_reward = torch.clamp(clearance, min=0.0, max=desired_clearance)
    reward = torch.sum(per_foot_reward, dim=1)

    nan_check(reward, "foot_clearance_reward")
    return reward


def foot_vertical_accel_reward(env, scale=0.5):
    """
    Reward upward vertical acceleration for ANY stuck foot (front or hind).
    'Stuck' means the foot is in contact and horizontal velocity is near zero.
    """

    # Current step foot vertical velocity
    vel_now = env._robot.data.body_lin_vel_w[:, env._feet_ids, 2]   # [N,4]

    # Previous step vertical velocity
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

def backward_vel_penalty(env, vel_thresh: float = 0.02) -> torch.Tensor:
    """
    Penalize backward body-frame velocity when commanded to go forward.

    - Looks at root_lin_vel_b[:, 0]  (forward v_x in base frame).
    - Only applies when commanded forward (cmd_vx > some threshold).
    """
    vxb = env._robot.data.root_lin_vel_b[:, 0]    # [N]
    cmd_vx = env._commands[:, 0]                  # [N]

    # Only when we *intend* to go forward
    forward_intent = cmd_vx > 0.1

    # Positive penalty for going backward (vxb < 0)
    backward_amount = torch.clamp(-vxb, min=0.0)  # 0 if vxb >= 0

    penalty = backward_amount * forward_intent
    return penalty

def feet_air_time_rear(env, threshold: float = 0.5, min_cmd_xy: float = 0.1) -> torch.Tensor:
    """
    Encourage regular stepping *only for the rear legs* by rewarding
    longer swing time upon touchdown.
    """

    # contact for all 4 feet → slice rear feet [2,3]
    first_contact = env._contact_sensor.compute_first_contact(env.step_dt)[:, env._feet_ids]
    first_contact_rear = first_contact[:, 2:]                   # [N, 2]

    # last air time for all 4 feet → slice rear feet
    last_air = env._contact_sensor.data.last_air_time[:, env._feet_ids]
    last_air_rear = last_air[:, 2:]                            # [N, 2]

    # reward for rear feet only
    r = torch.sum((last_air_rear - threshold) * first_contact_rear, dim=1)  # [N]

    # apply only when robot is commanded to move
    moving = torch.norm(env._commands[:, :2], dim=1) > min_cmd_xy

    return r * moving

def forward_progress_world(env, min_speed: float = 0.0) -> torch.Tensor:
    """
    Simple positive reward for forward COM motion in WORLD frame.
    Encourages the robot to actually go up the stairs instead of stalling.
    """
    vx_w = env._robot.data.root_lin_vel_w[:, 0]  # +X is stair direction
    # Reward only forward motion above a small threshold
    return torch.clamp(vx_w - min_speed, min=0.0)

def forward_progress_position(env, cmd_thresh: float = 0.05, delta_pos_scale: float = 1.0) -> torch.Tensor:
    """
    Position-based forward progress reward, but only when the robot
    is commanded to move forward.

    Reward = max( x(t) - x(t-1), 0 ) if command_vx > cmd_thresh
    Reward = 0 otherwise.
    """

    # Current world-frame X position of the base
    x_now = env._robot.data.root_pos_w[:, 0]          # [N]

    # Previous X positions (must be stored in the env)
    x_prev = env._prev_root_x                         # [N]

    # Positional delta
    delta = x_now - x_prev

    # Positive movement only
    delta_pos = torch.clamp(delta, min=0.0) * delta_pos_scale   # [N]

    # Retrieve forward command (body-frame desired vx)
    cmd_vx = env._commands[:, 0]                       # [N]
    
    # Condition mask: only apply reward if commanded to move forward
    moving_mask = (cmd_vx > cmd_thresh).float()        # [N]

    # Final gated reward
    reward = delta_pos * moving_mask

    return reward


def rear_match_front(env, pos_weight=3.0, vel_thresh=0.05, offset_forward=0.0, max_dist=0.3):
    """
    Encourage rear feet to place themselves where the front feet have stabilized.
    Mimics natural cat-like tracking where the hind foot steps into the front foot's position.

    Args:
        pos_weight:      Scales the positional matching reward (stricter = higher value).
        vel_thresh:      Threshold (m/s) for detecting stable front feet.
        offset_forward:  Optional forward bias to encourage stepping slightly ahead of the front foot.
        max_dist:        Maximum distance for reward - no reward if rear feet are further than this.
    """

    # --- Extract foot positions & velocities in world frame ---
    pos_w = env._robot.data.body_pos_w[:, env._feet_ids, :]        # [N,4,3]
    vel_w = env._robot.data.body_lin_vel_w[:, env._feet_ids, :]    # [N,4,3]

    forces = env._contact_sensor.data.net_forces_w[:, env._feet_ids, :]  # [N,4,3]
    contact = (forces.norm(dim=-1) > 5.0).float()               # [N,4]

    # FRONT feet
    FL = pos_w[:, 0, :]
    FR = pos_w[:, 1, :]

    FL_vel = torch.norm(vel_w[:, 0, :2], dim=1)    # horizontal velocity
    FR_vel = torch.norm(vel_w[:, 1, :2], dim=1)

    FL_contact = contact[:, 0]   # 1 if in contact, else 0
    FR_contact = contact[:, 1]

    # REAR feet positions and velocities
    RL = pos_w[:, 2, :]
    RR = pos_w[:, 3, :]
    RL_vel = vel_w[:, 2, :]  # Full 3D velocity for motion check
    RR_vel = vel_w[:, 3, :]

    # --- Condition: front foot must be stable AND in contact ---
    FL_stable = ((FL_vel < vel_thresh) & (FL_contact > 0)).float()   # [N]
    FR_stable = ((FR_vel < vel_thresh) & (FR_contact > 0)).float()

    # --- Target footholds ---
    target_FL = FL.clone()
    target_FR = FR.clone()

    # Optional forward offset in world X
    target_FL[:, 0] += offset_forward
    target_FR[:, 0] += offset_forward

    # --- Errors (distance to target) ---
    err_RL = torch.norm(RL - target_FL, dim=1)   # RL → FL
    err_RR = torch.norm(RR - target_FR, dim=1)   # RR → FR

    # --- Distance cutoff: no reward if too far ---
    close_enough_RL = (err_RL < max_dist).float()
    close_enough_RR = (err_RR < max_dist).float()

    # --- Motion toward target requirement ---
    # Direction vectors from rear to target
    dir_RL = target_FL - RL  # [N, 3]
    dir_RR = target_FR - RR  # [N, 3]

    # Normalize directions
    dir_RL_norm = dir_RL / (torch.norm(dir_RL, dim=1, keepdim=True) + 1e-6)
    dir_RR_norm = dir_RR / (torch.norm(dir_RR, dim=1, keepdim=True) + 1e-6)

    # Dot product: positive means moving toward target
    moving_toward_RL = torch.sum(RL_vel * dir_RL_norm, dim=1)
    moving_toward_RR = torch.sum(RR_vel * dir_RR_norm, dim=1)

    # Gate: reward if moving toward target OR already very close (within 10cm)
    motion_gate_RL = ((moving_toward_RL > 0.0) | (err_RL < 0.1)).float()
    motion_gate_RR = ((moving_toward_RR > 0.0) | (err_RR < 0.1)).float()

    # Apply all gates
    err_RL = err_RL * FL_stable * close_enough_RL * motion_gate_RL
    err_RR = err_RR * FR_stable * close_enough_RR * motion_gate_RR

    # --- Positive reward via exponential decay (now stricter) ---
    # With pos_weight=3.0: reward drops to 0.05 at 0.5m (vs 0.6 with pos_weight=1.0)
    reward = torch.exp(-pos_weight * (err_RL + err_RR))

    # Zero reward if no valid targets
    has_valid_target = ((FL_stable + FR_stable) > 0).float()
    reward = reward * has_valid_target

    return reward

def stagnation_penalty(env, vel_thresh: float = 0.05, position_thresh: float = 0.02, window_steps: int = 50) -> torch.Tensor:
    """
    Penalize lack of progress when commanded to move forward.
    Uses a rolling window to detect if robot is stuck over ~1 second.

    Args:
        vel_thresh: Velocity threshold below which robot is considered stopped (m/s).
        position_thresh: Position change threshold over window (m).
        window_steps: Number of steps for rolling window (~1 second at 50Hz).

    Requires env to maintain:
        env._stagnation_buffer: [N, window_steps] rolling position buffer
        env._stagnation_idx: Current index in buffer
    """
    # Initialize buffers if not exists
    if not hasattr(env, "_stagnation_buffer"):
        env._stagnation_buffer = torch.zeros(
            env.num_envs, window_steps, device=env.device
        )
        env._stagnation_idx = 0

    current_x = env._robot.data.root_pos_w[:, 0]

    # Store current position in rolling buffer
    env._stagnation_buffer[:, env._stagnation_idx] = current_x
    env._stagnation_idx = (env._stagnation_idx + 1) % window_steps

    # Compute position change over window
    oldest_x = env._stagnation_buffer[:, env._stagnation_idx]
    position_change = torch.abs(current_x - oldest_x)

    # Low velocity check
    vx_w = torch.abs(env._robot.data.root_lin_vel_w[:, 0])
    low_velocity = (vx_w < vel_thresh).float()

    # Low position change check
    low_progress = (position_change < position_thresh).float()

    # Forward command check
    cmd_vx = env._commands[:, 0]
    forward_intent = (cmd_vx > 0.1).float()

    # Penalty if all three conditions met: commanded forward, low velocity, no position change
    penalty = low_velocity * low_progress * forward_intent

    return penalty


def foot_lateral_separation_penalty(env, target_width: float = 0.30) -> torch.Tensor:
    """
    Penalize deviation of left/right foot lateral separation from a single target value,
    for both front and rear pairs.

    Args:
        target_width: Desired lateral (Y-axis) separation between L/R feet (meters).
    """
    # Foot positions in WORLD frame: [N, 4, 3]
    feet_w = env._robot.data.body_pos_w[:, env._feet_ids, :]

    # Assuming order: [FL, FR, RL, RR]
    FL_y = feet_w[:, 0, 1]
    FR_y = feet_w[:, 1, 1]
    RL_y = feet_w[:, 2, 1]
    RR_y = feet_w[:, 3, 1]

    # Lateral separations
    front_lat_dist = torch.abs(FL_y - FR_y)  # [N]
    rear_lat_dist  = torch.abs(RL_y - RR_y)  # [N]

    # Squared deviation from target for both pairs
    front_err = (front_lat_dist - target_width).pow(2)
    rear_err  = (rear_lat_dist  - target_width).pow(2)

    penalty = front_err + rear_err  # [N]

    return penalty

def hip_deflection_l2(env) -> torch.Tensor:
    """
    Penalize deviations from neutral hip deflection using an L2 kernel.
    Uses only the 4 hip joints (indices 0–3 in your joint order).
    """
    hip_pos = env._robot.data.joint_pos[:, 0:4]   # [N, 4]
    # L2 penalty (average over 4 hips)
    penalty = torch.sum(hip_pos ** 2, dim=1) / 4.
    return penalty

def track_center_path(env, std: float = 0.15) -> torch.Tensor:
    """
    Reward the robot for staying close to the centerline of the stair path.
    World-frame Y = 0 is assumed to be the ideal straight path.

    A Gaussian reward: exp(-(y^2) / std^2)
    """
    # world-frame Y position of robot base
    y = env._robot.data.root_pos_w[:, 1]  # [N]

    # Gaussian falloff: centered at 0, max reward = 1
    reward = torch.exp(-(y * y) / (std * std))

    return reward

def rear_swing_pitch(
    env,
    pitch_sin_thresh: float = 0.17,   # ~10 deg: sin(10°)=0.173
    min_cmd_x: float = 0.10,
    force_thresh: float = 5.0,
    vz_min: float = 0.05,             # ignore tiny noise
    vz_cap: float = 0.50,             # cap to prevent “kick” exploitation
) -> torch.Tensor:
    """
    Reward rear-foot swing (no contact + upward foot velocity) when the base is pitching forward.

    Pitch detection uses gravity in base frame:
      g_b = R^T * [0,0,-1], and forward pitch typically increases g_b.x in x-forward frames.
    If you find the sign is inverted in your setup, flip the inequality on g_b[:,0].
    """

    # ---------- condition: pitching forward ----------
    q_w = env._robot.data.root_quat_w  # [N,4]
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)  # [N,3]
    g_b = quat_apply(quat_conjugate(q_w), g_w)  # [N,3]

    pitching_fwd = (g_b[:, 0] > pitch_sin_thresh).float()  # if sign wrong, use < -pitch_sin_thresh

    # Gate to only matter when commanded forward motion exists
    moving_fwd = (env._commands[:, 0] > min_cmd_x).float()
    cond = pitching_fwd * moving_fwd  # [N]

    # ---------- rear swing mask ----------
    forces = env._contact_sensor.data.net_forces_w[:, env._feet_ids, :]  # [N,4,3]
    contact = (forces.norm(dim=-1) > force_thresh)  # [N,4]
    rear_swing = (~contact)[:, 2:].float()          # [N,2] (RL, RR)

    # ---------- rear foot upward velocity in base frame ----------
    foot_vel_w = env._robot.data.body_lin_vel_w[:, env._feet_ids, :]     # [N,4,3]
    base_vel_w = env._robot.data.root_lin_vel_w.unsqueeze(1)             # [N,1,3]
    rel_vel_w = foot_vel_w - base_vel_w                                  # [N,4,3]

    q_inv = quat_conjugate(q_w).unsqueeze(1).expand(-1, rel_vel_w.shape[1], -1)  # [N,4,4]
    rel_vel_b = quat_apply(q_inv, rel_vel_w)                                     # [N,4,3]

    rear_vz = rel_vel_b[:, 2:, 2]  # [N,2]
    vz_shaped = torch.clamp(rear_vz - vz_min, min=0.0, max=vz_cap)  # only reward upward swing

    # ---------- final reward ----------
    r = torch.sum(vz_shaped * rear_swing, dim=1) * cond  # [N]
    return r



def stand_still_cmd_penalty(
    env,
    yaw_weight: float = 0.5,
    jiggle_weight: float = 0.25,
    joint_hold_weight: float = 1.0,
    cmd_eps: float = 1.0e-2,
    flat_roll_pitch_sin_thresh: float = 0.12,
    flat_terrain_delta_thresh: float = 0.04,
    terrain_scan_radius: float = 0.12,
) -> torch.Tensor:
    """
    Penalize stop jitter and joint offset from default only when:
      1) velocity/yaw command is near zero, and
      2) robot stance is on flat ground.

    Flat-ground gate uses both:
      - base orientation (projected gravity x/y), and
      - local terrain height spread under the 4 feet.
    """
    cmd = env._commands[:, :3]
    stop_mask = (torch.norm(cmd, dim=1) < cmd_eps)

    g_xy = torch.norm(env._robot.data.projected_gravity_b[:, :2], dim=1)
    orientation_flat = g_xy < flat_roll_pitch_sin_thresh

    terrain_z_b = feet_height_scanner(env, radius=terrain_scan_radius)  # [N,4]
    terrain_step = torch.max(terrain_z_b, dim=1).values - torch.min(terrain_z_b, dim=1).values
    terrain_step = torch.nan_to_num(terrain_step, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    terrain_flat = terrain_step < flat_terrain_delta_thresh

    flat_stop_mask = (stop_mask & orientation_flat & terrain_flat).float()

    lin_xy_sq = torch.sum(env._robot.data.root_lin_vel_b[:, :2] ** 2, dim=1)
    yaw_rate_sq = env._robot.data.root_ang_vel_b[:, 2] ** 2
    action_delta = env._actions - env._previous_actions
    action_jiggle = torch.mean(action_delta ** 2, dim=1)
    joint_default_err = torch.mean(
        (env._robot.data.joint_pos - env._robot.data.default_joint_pos) ** 2, dim=1
    )

    penalty = lin_xy_sq + yaw_weight * yaw_rate_sq + jiggle_weight * action_jiggle + joint_hold_weight * joint_default_err
    return penalty * flat_stop_mask


def compute_all_rewards(env) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    raw: Dict[str, torch.Tensor] = {
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp(env),
        # "track_modified_vel_reward": track_modified_vel_reward(env),
        "track_ang_vel_z_exp": track_ang_vel_z_exp(env),
        "lin_vel_z_penalty": lin_vel_z_penalty(env),
        "ang_vel_xy_penalty": ang_vel_xy_penalty(env),
        "joint_torque_penalty": joint_torque_penalty(env),
        "joint_acc_penalty": joint_acc_penalty(env),
        "action_rate_penalty": action_rate_penalty(env),
        "undesired_contacts": undesired_contacts(env),
        "flat_orientation": flat_orientation(env),
        "flat_orientation_roll": flat_orientation_roll(env),
        "joint_pos_limit": joint_pos_limits(env),  # <-- NEW
        "energy_penalty": energy_penalty(env),
        "feet_slide_penalty": feet_slide(env),
        "foot_clearance_reward": foot_clearance_reward(env),
        "smoothness_penalty": smoothness_penalty(env),
        "base_height_l2_lidar": base_height_l2_lidar(env, target=0.30),
        "foot_vertical_accel_reward": foot_vertical_accel_reward(env),
        "backward_vel_penalty": backward_vel_penalty(env),
        "feet_air_time_rear": feet_air_time_rear(env),
        "forward_progress": forward_progress_position(env, delta_pos_scale=100.0),
        "rear_match_front": rear_match_front(env, pos_weight=1.0, vel_thresh=0.05, offset_forward=0.0),
        "stagnation_penalty": stagnation_penalty(env),
        "foot_lateral_separation_penalty": foot_lateral_separation_penalty(env),
        "hip_deflection_l2": hip_deflection_l2(env),
        "track_center_path": track_center_path(env),
        "rear_swing_pitch": rear_swing_pitch(env),
        "stand_still_cmd_penalty": stand_still_cmd_penalty(env),

    }

    # --- Scales: tuned for flat-ground learning ---
    w = {
        "track_lin_vel_xy_exp": 8.0,
        # "track_modified_vel_reward": 2.0,
        "track_ang_vel_z_exp": 1.0,
         "lin_vel_z_penalty": -0.5,       
        "ang_vel_xy_penalty": -0.5,
        "joint_torque_penalty": -2.0e-5,
        "joint_acc_penalty": -2.0e-7,
        "action_rate_penalty": -0.2,
        "undesired_contacts": -4.0,
        "flat_orientation": -0.8, 
        "flat_orientation_roll": -2.0,
        "energy_penalty": -1.0e-6,
        "feet_slide_penalty": -0.5,
        "foot_clearance_reward": 2.5,
        "joint_pos_limit": -0.6,
        "smoothness_penalty": -0.01,
        "base_height_l2_lidar": -2.0,
        "foot_vertical_accel_reward": 1.4,
        "backward_vel_penalty": -4.0,
        "feet_air_time_rear": 2.0,
        "stagnation_penalty": -3.0,
        "forward_progress": 2.0,
        "rear_match_front": 2.0,
        "foot_lateral_separation_penalty": -4.0,
        "hip_deflection_l2": -5.0,
        "track_center_path": 4.0,
        "rear_swing_pitch": 2.0,
        "stand_still_cmd_penalty": -2.0
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

    
    # terrain_height_b = torch.max(z_vals, dim=1).values
    terrain_height_b = torch.mean(z_vals, dim=1)

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

def get_pitch_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    quat: [N, 4] as (x, y, z, w)
    Returns pitch angle in radians.
    """
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    # formula for pitch (rotation around Y)
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    return pitch

def feet_height_scanner(env, radius: float = 0.12,) -> torch.Tensor:
    """Estimate terrain height under each foot using privileged height-scanner hits.

    Returns:
        terrain_z_b: [N, 4] terrain height under each foot, expressed in BASE frame (z).
    """
    # Height scanner hits in WORLD frame: [N, R, 3]
    hits_w = env._height_scanner.data.ray_hits_w

    # Convert hits to BASE frame
    base_pos_w = env._robot.data.root_pos_w                     # [N,3]
    base_quat_w = env._robot.data.root_quat_w                   # [N,4]
    base_quat_inv = quat_conjugate(base_quat_w)                 # [N,4]

    hits_shifted = hits_w - base_pos_w.unsqueeze(1)             # [N,R,3]
    q_exp = base_quat_inv.unsqueeze(1).expand(-1, hits_shifted.shape[1], -1)  # [N,R,4]
    hits_b = quat_apply(q_exp, hits_shifted)                    # [N,R,3]

    hits_xy = hits_b[..., :2]                                   # [N,R,2]
    hits_z  = hits_b[...,  2]                                   # [N,R]

    # Foot positions in BASE frame
    foot_pos_w = env._robot.data.body_pos_w[:, env._feet_ids, :] # [N,4,3]
    foot_shifted = foot_pos_w - base_pos_w.unsqueeze(1)          # [N,4,3]
    qf_exp = base_quat_inv.unsqueeze(1).expand(-1, foot_shifted.shape[1], -1)  # [N,4,4]
    foot_b = quat_apply(qf_exp, foot_shifted)                    # [N,4,3]
    foot_xy = foot_b[..., :2]                                    # [N,4,2]

    # Build a neighborhood mask: rays whose XY are within `radius` of each foot XY
    # dist2: [N,4,R]
    diff = hits_xy.unsqueeze(1) - foot_xy.unsqueeze(2)           # [N,4,R,2]
    dist2 = (diff ** 2).sum(dim=-1)                              # [N,4,R]
    near = dist2 <= (radius * radius)                            # [N,4,R]

    # Valid hit mask (finite z)
    valid = torch.isfinite(hits_z).unsqueeze(1)                  # [N,1,R]
    mask = near & valid                                          # [N,4,R]

    # Median z of nearby rays (robust); if none nearby, fall back to global median
    nan = torch.tensor(float("nan"), device=hits_z.device, dtype=hits_z.dtype)
    z_sel = torch.where(mask, hits_z.unsqueeze(1), nan)          # [N,4,R]
    terrain_z_b = torch.nanmedian(z_sel, dim=2).values           # [N,4]

    # Fallback for feet with no valid neighbors: global nanmedian over all rays
    global_z = torch.nanmedian(hits_z, dim=1).values             # [N]
    no_data = ~torch.isfinite(terrain_z_b)                       # [N,4]
    terrain_z_b = torch.where(no_data, global_z.unsqueeze(1), terrain_z_b)

    return terrain_z_b
