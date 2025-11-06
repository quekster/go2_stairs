# # visualisation.py
# from __future__ import annotations
# import numpy as np
# import torch

# from isaaclab.utils.math import yaw_quat, quat_apply  # rotate body->world with yaw only

# # Isaac Sim debug draw API
# import omni.isaac.debug_draw as debug_draw

# def _to_cpu_np(t: torch.Tensor) -> np.ndarray:
#     return t.detach().cpu().numpy()

# class VelArrowsVisualizer:
#     """
#     Draws command (goal) and measured velocity arrows above the robot base.
#     - Converts body-frame XY velocities to WORLD frame using yaw-only rotation,
#       so the arrows look correct even if the robot pitches/rolls.
#     - Draws only the first N envs for performance.
#     """
#     def __init__(
#         self,
#         robot,
#         num_envs_to_draw: int = 8,
#         z_offset: float = 0.45,
#         length_gain: float = 2.0,
#         draw_goal: bool = True,
#         draw_meas: bool = True,
#         thickness: float = 2.0,
#         visible: bool = True,
#     ):
#         self.robot = robot
#         self.n_draw = num_envs_to_draw
#         self.z_offset = z_offset
#         self.length_gain = length_gain
#         self.draw_goal = draw_goal
#         self.draw_meas = draw_meas
#         self.thickness = thickness
#         self._visible = visible
#         self._dd = debug_draw.acquire_debug_draw_interface()

#         # RGBA
#         self._col_goal = (0.2, 0.9, 0.3, 1.0)   # green
#         self._col_meas = (0.9, 0.9, 0.9, 1.0)   # white

#     def set_visibility(self, flag: bool):
#         self._visible = flag

#     def update(self, cmd_vel_b: torch.Tensor, lin_vel_b: torch.Tensor):
#         """Call every _post_physics_step()."""
#         if not self._visible:
#             return
#         # pull base world pose
#         pos_w = self.robot.data.root_pos_w       # (N,3)
#         quat_w = self.robot.data.root_quat_w     # (N,4)
#         N = min(pos_w.shape[0], self.n_draw)

#         # yaw-only quaternion for stable world arrows
#         yaw_q = yaw_quat(quat_w[:N])             # (N,4)

#         # body-frame XY -> 3D vectors
#         cmd_b = torch.zeros((N, 3), device=cmd_vel_b.device, dtype=cmd_vel_b.dtype)
#         cmd_b[:, :2] = cmd_vel_b[:N, :2]
#         meas_b = torch.zeros_like(cmd_b)
#         meas_b[:, :2] = lin_vel_b[:N, :2]

#         # rotate to world using yaw only
#         cmd_w = quat_apply(yaw_q, cmd_b)         # (N,3)
#         meas_w = quat_apply(yaw_q, meas_b)       # (N,3)

#         # numpy arrays for debug draw
#         p = _to_cpu_np(pos_w[:N])
#         cmd = _to_cpu_np(cmd_w)
#         meas = _to_cpu_np(meas_w)

#         # draw arrows
#         lg = self.length_gain
#         z_off = self.z_offset
#         for i in range(N):
#             start = p[i].copy()
#             start[2] += z_off

#             if self.draw_goal:
#                 end_g = start + lg * cmd[i]
#                 # zero out z component so arrows are horizontal
#                 end_g[2] = start[2]
#                 self._dd.draw_arrow(tuple(start), tuple(end_g), self._col_goal, self.thickness)

#             if self.draw_meas:
#                 end_m = start + lg * meas[i]
#                 end_m[2] = start[2]
#                 self._dd.draw_arrow(tuple(start), tuple(end_m), self._col_meas, self.thickness)
