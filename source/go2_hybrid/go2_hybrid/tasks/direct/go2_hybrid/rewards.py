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

def height_tracking_stairs_bidirectional(env, target_height_gain: float = 0.18) -> torch.Tensor:
    """Reward climbing OR descending based on commanded direction."""
    current_height = env._robot.data.root_pos_w[:, 2]
    
    if not hasattr(env, '_prev_base_height'):
        env._prev_base_height = current_height.clone()
    
    # Absolute height change (no clamping)
    height_change = current_height - env._prev_base_height
    env._prev_base_height = current_height.clone()
    
    # Determine if command wants upward (+vx) or downward (-vx) motion
    cmd_direction = torch.sign(env._commands[:, 0])  # +1 for forward, -1 for backward
    
    # Reward when height change matches commanded direction
    aligned_gain = height_change * cmd_direction
    return (aligned_gain / target_height_gain).clamp(min=0.0)

def foot_height_variance(env) -> torch.Tensor:
    """Encourage staggered foot placement (critical for stairs)."""
    foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]
    return torch.var(foot_z, dim=1)

def forward_progress_stairs_bidirectional(env) -> torch.Tensor:
    """Reward forward/backward motion appropriate to command."""
    vx_body = env._robot.data.root_lin_vel_b[:, 0]
    cmd_vx = env._commands[:, 0]
    
    # Reward when velocity matches commanded sign
    aligned_progress = vx_body * torch.sign(cmd_vx)
    return aligned_progress.clamp(min=0.0)


def base_height_penalty(env, std: float = 0.08, adaptive: bool = True) -> torch.Tensor:
    """Height penalty that adapts to terrain elevation."""
    base_z = env._robot.data.root_pos_w[:, 2]
    
    if adaptive:
        # Use minimum foot height as ground reference
        foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]
        ground_ref = torch.min(foot_z, dim=1).values + 0.28  # nominal standing height
    else:
        ground_ref = env._terrain.env_origins[:, 2] + 0.28
    
    err_sq = torch.square(base_z - ground_ref)
    return err_sq / (2 * std**2)

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

def excessive_tilt_penalty(env, pitch_threshold: float = 0.6, roll_threshold: float = 0.5) -> torch.Tensor:
    """Penalize excessive pitch/roll angles (smooth version of termination)."""
    proj_grav = env._robot.data.projected_gravity_b  # [N, 3]
    
    # Calculate pitch and roll
    pitch = torch.asin(proj_grav[:, 0].clamp(-1.0, 1.0))
    roll = torch.atan2(proj_grav[:, 1], -proj_grav[:, 2])
    
    # Smooth penalty that increases as angles exceed thresholds
    pitch_penalty = (torch.abs(pitch) - pitch_threshold).clamp(min=0.0) ** 2
    roll_penalty = (torch.abs(roll) - roll_threshold).clamp(min=0.0) ** 2
    
    return pitch_penalty + roll_penalty

def goal_progress_reward(env) -> torch.Tensor: #Phase 1 addition!
    """
    Positive reward when the robot moves closer to the goal.
    reward = max(prev_dist - current_dist, 0)
    """
    base_pos = env._robot.data.root_pos_w[:, :3]  # (N,3)
    dist_now = torch.norm(env._goal_pos - base_pos, dim=1)

    # improvement = positive only when getting closer
    improvement = (env._prev_goal_dist - dist_now).clamp(min=0.0)

    # update buffer
    env._prev_goal_dist = dist_now.clone()

    return improvement

def lidar_clearance_reward(
    env, 
    lookhead_distance: float = 1.75,
    base_clearance: float = 0.10,
    clearance_scaling: float = 1.5,
    std: float = 0.08
) -> torch.Tensor:
    """
    Reward foot clearance that adapts based on upcoming terrain height from LiDAR.
    Higher obstacles ahead → encourage higher foot lift.
    """
    # Get LiDAR hits in base frame
    hits_b = env.get_bf_hits()  # [N, num_rays, 3]
    
    # Filter points in front of robot within lookahead distance
    forward_mask = (hits_b[..., 0] > 0) & (hits_b[..., 0] < lookhead_distance)
    
    # Get maximum height of terrain ahead (per environment)
    forward_heights = torch.where(
        forward_mask,
        hits_b[..., 2],
        torch.tensor(-1e6, device=env.device)  # ignore invalid points
    )
    max_terrain_height = torch.max(forward_heights.reshape(env.num_envs, -1), dim=1)[0]
    
    # Adaptive target: higher terrain → higher foot clearance required
    # Clamp to reasonable values
    adaptive_target = (base_clearance + clearance_scaling * max_terrain_height).clamp(0.05, 0.35)
    
    # Get current foot heights
    foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]  # [N, 4]
    base_z = env._robot.data.root_pos_w[:, 2].unsqueeze(1)  # [N, 1]
    foot_clearance = foot_z - base_z  # relative to base
    
    # Detect swing phase (feet with significant XY velocity)
    foot_vel_xy = torch.norm(
        env._robot.data.body_lin_vel_w[:, env._feet_ids, :2], 
        dim=2
    )  # [N, 4]
    is_swinging = foot_vel_xy > 0.3
    
    # Squared error between actual clearance and adaptive target
    clearance_error = torch.square(foot_clearance - adaptive_target.unsqueeze(1))
    
    # Only reward during swing phase
    weighted_error = clearance_error * is_swinging.float()
    
    return torch.exp(-torch.sum(weighted_error, dim=1) / (2 * std**2))

def height_stagnation_penalty(
    env, 
    time_threshold: float = 3.0,
    height_tolerance: float = 0.05
) -> torch.Tensor:
    """
    Penalize staying at the same height for too long.
    Encourages continuous climbing progress.
    
    Args:
        time_threshold: Seconds before penalty kicks in (3.0s = 60 steps at 20Hz)
        height_tolerance: Height difference considered "same level" (5cm)
    """
    current_height = env._robot.data.root_pos_w[:, 2]
    
    # Initialize tracking on first call
    if not hasattr(env, '_height_stagnation_timer'):
        env._height_stagnation_timer = torch.zeros(env.num_envs, device=env.device)
        env._last_reference_height = current_height.clone()
    
    # Check if robot has moved vertically
    height_change = torch.abs(current_height - env._last_reference_height)
    has_progressed = height_change > height_tolerance
    
    # Update timers
    env._height_stagnation_timer += env.step_dt  # Increment all timers
    env._height_stagnation_timer[has_progressed] = 0.0  # Reset for robots that moved
    env._last_reference_height[has_progressed] = current_height[has_progressed]
    
    # Penalty grows quadratically after threshold
    time_over_threshold = (env._height_stagnation_timer - time_threshold).clamp(min=0.0)
    penalty = time_over_threshold ** 2
    
    return penalty

def foot_on_next_step_reward(env, step_height=0.17, tolerance=0.04):
    # current foot heights (N,4)
    foot_z = env._robot.data.body_pos_w[:, env._feet_ids, 2]

    prev = env._prev_foot_z  # (N,4)

    # detect feet that clearly moved up by at least ~one step
    lifted = (foot_z - prev) > (step_height - tolerance)   # (N,4)

    # update buffer
    env._prev_foot_z = foot_z.clone()

    # reward count of upward-stepping feet
    return lifted.float().sum(dim=1)



def hind_leg_drive(env):
    # body-frame forward velocity
    vx = env._robot.data.root_lin_vel_b[:, 0]  # (N,)

    # contact forces for hind feet
    forces = env._contact_sensor.data.net_forces_w[:, env._hind_feet_ids]  # (N,2,3)
    hind_in_contact = forces.norm(dim=-1) > 5.0                            # (N,2)

    # if ANY hind foot has solid contact
    has_hind_support = hind_in_contact.any(dim=1).float()                  # (N,)

    return vx * has_hind_support

def hind_leg_step_forward(env, force_thresh=5.0):
    # current hind foot positions & forces
    hind_pos = env._robot.data.body_pos_w[:, env._hind_feet_ids, :]          # (N,2,3)
    forces   = env._contact_sensor.data.net_forces_w[:, env._hind_feet_ids]  # (N,2,3)

    in_contact = forces.norm(dim=-1) > force_thresh                          # (N,2)
    was_in_contact = env._hind_was_in_contact                                # (N,2)

    # Detect new contact events: now in contact, previously not
    new_contact = in_contact & (~was_in_contact)                             # (N,2)

    # Forward displacement since last *stance* position
    delta = hind_pos - env._hind_last_contact_pos                            # (N,2,3)
    delta_x = delta[..., 0]                                                  # (N,2)

    # Reward only positive forward step at new contacts
    step_forward = torch.clamp(delta_x, min=0.0) * new_contact.float()       # (N,2)

    # Update memory:
    # - for ANY contact (new or continued), refresh last_contact_pos
    env._hind_last_contact_pos[in_contact] = hind_pos[in_contact]
    env._hind_was_in_contact = in_contact

    # Sum over the two hind feet
    return step_forward.sum(dim=1)

def front_foot_lateral_separation_penalty(env, min_width: float = 0.18) -> torch.Tensor:
    """
    Penalize when front left/right feet are too close laterally (prevents crossing).
    
    Args:
        min_width: Minimum lateral (Y-axis) separation between FL and FR feet (meters).
    """
    foot_pos = env._robot.data.body_pos_w[:, env._feet_ids, :]  # [N, 4, 3]
    
    # Assuming foot order is [FL, FR, RL, RR]
    front_lat_dist = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])  # FL vs FR (Y-axis)
    
    # Penalty when separation is below threshold
    penalty = (min_width - front_lat_dist).clamp(min=0.0) ** 2
    
    return penalty


def rear_foot_lateral_separation_penalty(env, min_width: float = 0.18) -> torch.Tensor:
    """
    Penalize when rear left/right feet are too close laterally (prevents crossing).
    
    Args:
        min_width: Minimum lateral (Y-axis) separation between RL and RR feet (meters).
    """
    foot_pos = env._robot.data.body_pos_w[:, env._feet_ids, :]  # [N, 4, 3]
    
    # Assuming foot order is [FL, FR, RL, RR]
    rear_lat_dist = torch.abs(foot_pos[:, 2, 1] - foot_pos[:, 3, 1])   # RL vs RR (Y-axis)
    
    # Penalty when separation is below threshold
    penalty = (min_width - rear_lat_dist).clamp(min=0.0) ** 2
    
    return penalty



def thigh_lift_reward(
    env,
    step_height: float = 0.17,
    lookahead_distance: float = 0.35,
    tolerance: float = 0.08  # increased tolerance
) -> torch.Tensor:
    """
    Reward lifting thigh joints to match detected step height ahead.
    **IMPROVED**: More selective triggering, better swing detection.
    """
    # 1. Detect step height using LiDAR
    hits_b = env.get_bf_hits()  # [N, num_rays, 3]
    
    x_min, x_max = lookahead_distance - 0.15, lookahead_distance + 0.15  # wider window
    y_min, y_max = -0.25, 0.25  # wider lateral range
    
    forward_mask = (
        (hits_b[..., 0] > x_min) & (hits_b[..., 0] < x_max) &
        (hits_b[..., 1] > y_min) & (hits_b[..., 1] < y_max)
    )

    forward_heights = torch.where(
        forward_mask,
        hits_b[..., 2],
        torch.tensor(-1e6, device=env.device)
    )
    max_terrain_z_base = torch.max(
        forward_heights.reshape(env.num_envs, -1), 
        dim=1
    )[0]
    
    # Get ground reference
    foot_z_world = env._robot.data.body_pos_w[:, env._feet_ids, 2]
    ground_level_world = torch.min(foot_z_world, dim=1)[0]
    base_z_world = env._robot.data.root_pos_w[:, 2]
    ground_level_base = ground_level_world - base_z_world
    
    detected_step_height = (max_terrain_z_base - ground_level_base).clamp(0.0, step_height * 1.5)
    
    # **IMPROVED CONDITIONS**
    has_step = detected_step_height > 0.08  # increased from 0.05 (need clearer signal)
    is_moving_forward = env._robot.data.root_lin_vel_b[:, 0] > 0.15  # increased from 0.1
    
    # **NEW: Check base is high enough (not belly crawling)**
    height_above_ground = base_z_world - ground_level_world
    is_upright = height_above_ground > 0.20  # at least 20cm clearance
    
    should_lift = has_step & is_moving_forward & is_upright
    
    # DEBUG
    if env._step_counter % 100 == 0:
        num_hits = forward_mask[0].sum().item()
        print(f"[Thigh Lift] Step {env._step_counter}: "
              f"step_h={detected_step_height[0]:.3f}m | "
              f"vx={env._robot.data.root_lin_vel_b[0, 0]:.2f} | "
              f"upright={is_upright[0].item()} | "
              f"active={should_lift[0].item()}")
    
    # 2. Get thigh joint angles
    thigh_angles = env._robot.data.joint_pos[:, 4:8]
    front_thigh_angles = thigh_angles[:, :2]  # FL, FR
    
    # **IMPROVED TARGET**: More aggressive lift for clear obstacles
    target_thigh_angle = torch.where(
        should_lift,
        (detected_step_height / step_height).clamp(0.7, 1.2) * 1.5,  # 1.05-1.8 rad
        torch.zeros_like(detected_step_height)
    )
    
    # 3. **IMPROVED SWING DETECTION**: Use vertical velocity AND foot clearance
    front_foot_vel_z = env._robot.data.body_lin_vel_w[:, env._feet_ids[:2], 2]
    front_foot_z = env._robot.data.body_pos_w[:, env._feet_ids[:2], 2]
    foot_clearance = front_foot_z - ground_level_world.unsqueeze(1)
    
    is_swinging = (torch.abs(front_foot_vel_z) > 0.2) | (foot_clearance > 0.05)  # either lifting OR airborne
    
    # 4. Reward computation
    angle_error = torch.square(front_thigh_angles - target_thigh_angle.unsqueeze(1))
    weighted_error = angle_error * is_swinging.float() * should_lift.unsqueeze(1).float()
    
    reward = torch.exp(-torch.sum(weighted_error, dim=1) / (2 * tolerance**2))
    reward = torch.where(should_lift, reward, torch.zeros_like(reward))
    
    return reward


def step_detection_lookahead_reward(
    env,
    lookahead_distance: float = 0.4,
    target_detection_height: float = 0.17
) -> torch.Tensor:
    """
    Reward for LiDAR detecting step ahead (encourages forward exploration).
    **FIXED**: Uses ground-relative height measurement.
    """
    hits_b = env.get_bf_hits()  # [N, num_rays, 3]
    
    # Get ground reference (same as thigh_lift)
    foot_z_world = env._robot.data.body_pos_w[:, env._feet_ids, 2]
    ground_level_world = torch.min(foot_z_world, dim=1)[0]
    base_z_world = env._robot.data.root_pos_w[:, 2]
    ground_level_base = ground_level_world - base_z_world
    
    # Check for obstacle ahead
    x_range = (hits_b[..., 0] > 0.2) & (hits_b[..., 0] < lookahead_distance)
    
    # **FIX: Measure height relative to ground**
    relative_heights = hits_b[..., 2] - ground_level_base.unsqueeze(1)
    z_range = relative_heights > (target_detection_height * 0.5)  # at least 8.5cm
    
    detects_step = torch.any(x_range & z_range, dim=1)
    
    # # DEBUG
    # if env._step_counter % 50 == 0:
    #     print(f"[Step Detection] Env 0: detected={detects_step[0].item()} | "
    #           f"ground_base={ground_level_base[0]:.3f}m")
    
    return detects_step.float()


def hind_leg_push_reward(env, force_threshold: float = 30.0) -> torch.Tensor:
    """
    Reward vertical ground reaction forces on hind feet.
    **FIXED**: Only reward when robot is moving forward (not sitting back).
    """
    # Get hind foot forces
    hind_forces = env._contact_sensor.data.net_forces_w[:, env._feet_ids[2:]]  # [N, 2, 3]
    vertical_forces = hind_forces[..., 2].clamp(min=0.0)
    
    # **FIX: Only reward push when moving forward**
    is_moving_forward = env._robot.data.root_lin_vel_b[:, 0] > 0.1
    
    normalized_forces = (vertical_forces / force_threshold).clamp(0.0, 2.0)
    avg_force = torch.sum(normalized_forces, dim=1) / 2.0
    
    # Zero out reward if not moving
    return torch.where(is_moving_forward, avg_force, torch.zeros_like(avg_force))

def front_foot_placement_reward(
    env,
    step_depth: float = 0.18,
    tolerance: float = 0.08
) -> torch.Tensor:
    """
    Reward front feet landing ON the next step (not before/after).
    Uses base X position to estimate expected landing zone.
    """
    base_x = env._robot.data.root_pos_w[:, 0]
    front_feet_x = env._robot.data.body_pos_w[:, env._feet_ids[:2], 0]  # [N, 2]
    
    # Expected landing zone: one step ahead of base
    target_x = base_x.unsqueeze(1) + step_depth
    
    # Error in X placement
    placement_error = torch.abs(front_feet_x - target_x)
    
    # Only reward during contact
    front_contact = env._contact_sensor.data.net_forces_w[:, env._feet_ids[:2]].norm(dim=-1) > 1.0
    
    # Gaussian reward centered at target
    reward_per_foot = torch.exp(-placement_error**2 / (2 * tolerance**2)) * front_contact.float()
    return torch.mean(reward_per_foot, dim=1)


def body_height_progress_reward(env, step_height: float = 0.17) -> torch.Tensor:
    """
    Reward increasing body Z position (simplified height tracking).
    """
    current_height = env._robot.data.root_pos_w[:, 2]
    
    if not hasattr(env, '_prev_body_height'):
        env._prev_body_height = current_height.clone()
        return torch.zeros_like(current_height)
    
    height_gain = (current_height - env._prev_body_height).clamp(min=0.0)
    env._prev_body_height = current_height.clone()
    
    # Normalize to step height
    return (height_gain / (step_height * 0.5)).clamp(0.0, 2.0)


def pitch_stability_stairs(env, target_pitch: float = 0.0, tolerance: float = 0.3) -> torch.Tensor:
    """
    Penalize excessive pitch deviation (but allow some pitch for stairs).
    More lenient than flat_orientation.
    """
    proj_grav = env._robot.data.projected_gravity_b
    pitch = torch.asin(proj_grav[:, 0].clamp(-1.0, 1.0))
    
    # Allow pitch in range [-tolerance, +tolerance]
    pitch_error = (torch.abs(pitch - target_pitch) - tolerance).clamp(min=0.0)
    return pitch_error ** 2

def base_pitch_penalty(env, max_pitch: float = 0.5) -> torch.Tensor:
    """
    Penalize extreme backward pitch (sitting on haunches).
    
    Args:
        max_pitch: Maximum allowed pitch deviation (radians)
    """
    proj_grav = env._robot.data.projected_gravity_b
    pitch = torch.asin(proj_grav[:, 0].clamp(-1.0, 1.0))
    
    # Penalize backward pitch beyond threshold
    backward_pitch = (-pitch - max_pitch).clamp(min=0.0)  # negative pitch = leaning back
    
    return backward_pitch ** 2


def base_height_maintenance_reward(env, min_height: float = 0.25) -> torch.Tensor:
    """
    Penalize when base is too low (prevents belly crawling).
    
    Args:
        min_height: Minimum height above ground level (meters)
    """
    # Get ground reference from feet
    foot_z_world = env._robot.data.body_pos_w[:, env._feet_ids, 2]
    ground_level = torch.min(foot_z_world, dim=1)[0]
    
    # Current base height above ground
    base_z_world = env._robot.data.root_pos_w[:, 2]
    height_above_ground = base_z_world - ground_level
    
    # Penalty when too low
    height_deficit = (min_height - height_above_ground).clamp(min=0.0)
    
    return height_deficit ** 2


def compute_all_rewards(env) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute all reward terms and apply weights."""
    
    # ========== COMPUTE RAW REWARDS ==========
    raw: Dict[str, torch.Tensor] = {
        # Tracking rewards
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp(env),
        "track_ang_vel_z_exp": track_ang_vel_z_exp(env),
        "track_heading_reward": track_heading_reward(env),
        
        # Core locomotion penalties
        "lin_vel_z_penalty": lin_vel_z_penalty(env),
        "ang_vel_xy_penalty": ang_vel_xy_penalty(env),
        "action_rate_penalty": action_rate_penalty(env),
        "joint_torque_penalty": joint_torque_penalty(env),
        "joint_acc_penalty": joint_acc_penalty(env),
        
        # Foot contact rewards
        "feet_air_time": feet_air_time(env),
        "feet_slide_penalty": feet_slide(env),
        "undesired_contacts": undesired_contacts(env),
        
        # Joint limits
        "joint_pos_limit": joint_pos_limits(env),
        
        # **STAIRS-SPECIFIC**
        "thigh_lift": thigh_lift_reward(env, step_height=0.17, lookahead_distance=0.35),
        "step_detection": step_detection_lookahead_reward(env, lookahead_distance=0.4),
        "hind_push": hind_leg_push_reward(env, force_threshold=30.0),
        "front_placement": front_foot_placement_reward(env, step_depth=0.18),
        "body_height_progress": body_height_progress_reward(env, step_height=0.17),
        "pitch_stability": pitch_stability_stairs(env, target_pitch=0.0, tolerance=0.4),
        
        # **NEW: Critical fixes**
        "base_pitch_penalty": base_pitch_penalty(env, max_pitch=0.5),
        "base_height_maintenance": base_height_maintenance_reward(env, min_height=0.25),
        "front_foot_separation": front_foot_lateral_separation_penalty(env, min_width=0.2),
        "rear_foot_separation": rear_foot_lateral_separation_penalty(env, min_width=0.2),
        
        # Goal progress
        "goal_progress": goal_progress_reward(env),
    }

    # ========== REWARD WEIGHTS ==========
    w = {
        # Tracking (FURTHER REDUCED - stairs need careful movement)
        "track_lin_vel_xy_exp": 4.0,      # reduced from 2.0
        "track_ang_vel_z_exp": 0.3,
        "track_heading_reward": 0.3,
        
        # Locomotion penalties
        "lin_vel_z_penalty": -2.0,
        "ang_vel_xy_penalty": -0.05,
        "action_rate_penalty": -0.01,
        "joint_torque_penalty": -0.0001,
        "joint_acc_penalty": -2.5e-7,
        
        # Foot contact
        "feet_air_time": 0.5,
        "feet_slide_penalty": -0.2,
        "undesired_contacts": -1.0,
        
        # Joint limits
        "joint_pos_limit": -1.0,
        
        # **STAIRS (REBALANCED)**
        "thigh_lift": 3.0,                # reduced from 4.0 (was too dominant)
        "step_detection": 1.5,            # INCREASED from 0.5 (need more exploration)
        "hind_push": 1.5,                 # reduced from 2.0
        "front_placement": 2.0,           # INCREASED from 1.0 (critical for stairs)
        "body_height_progress": 6.0,      # INCREASED from 3.0 (main objective!)
        "pitch_stability": -0.5,
        
        # **NEW PENALTIES (HIGH PRIORITY)**
        "base_pitch_penalty": -4.0,       # STRONG penalty for sitting back
        "base_height_maintenance": -5.0,  # VERY STRONG - prevent belly crawling
        "front_foot_separation": -3.0,        # HIGHER priority (critical for stepping)
        "rear_foot_separation": -3.0,         # lower priority (stability support)
        
        # Goal
        "goal_progress": 5.0,
    }

    # ========== APPLY WEIGHTS & TIME SCALING ==========
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