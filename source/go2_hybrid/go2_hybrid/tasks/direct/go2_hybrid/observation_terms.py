from __future__ import annotations

import torch


def root_ang_vel_b(env) -> torch.Tensor:
    """Base angular velocity in robot body frame."""
    return env._robot.data.root_ang_vel_b


def projected_gravity_b(env) -> torch.Tensor:
    """Gravity vector projected into robot body frame."""
    return env._robot.data.projected_gravity_b


def joint_pos_rel(env) -> torch.Tensor:
    """Joint positions relative to default joint pose."""
    return env._robot.data.joint_pos - env._robot.data.default_joint_pos


def joint_vel(env) -> torch.Tensor:
    """Joint velocities."""
    return env._robot.data.joint_vel

