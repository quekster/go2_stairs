# --- Symmetry augmentation for Go2 (DirectRLEnv, left–right only) ----------------
from __future__ import annotations
import torch
from typing import Optional, Dict, Tuple, Union

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

# Public API (like IsaacLab's example expects)
__all__ = ["compute_symmetric_states"]

@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None, **kwargs):
    if actions is not None:
        ndof = actions.shape[1] 
    else:
        ndof = 12 #default 12 joints for go2
        unwrapped = getattr(env, "unwrapped", None) or getattr(env, "env", None) # try to unwrap the environment (env -> RslRlVecEnvWrapper)
        for key in ("num_actions",):
            val = getattr(unwrapped, key, None) if unwrapped is not None else None
            if isinstance(val, int):
                ndof = val
                break
        else:
            # gym space fallback
            space = getattr(unwrapped, "single_action_space", None) if unwrapped is not None else None
            if hasattr(space, "shape") and space.shape:
                ndof = int(space.shape[-1])

    # container handling (unchanged)
    obs_is_dict = isinstance(obs, dict)
    policy_obs = obs["policy"] if obs_is_dict and obs is not None else (obs if obs is not None else None)

    # augment obs
    if policy_obs is not None:
        B, D = policy_obs.shape
        obs_aug_policy = policy_obs.new_zeros((B * 2, D))
        obs_aug_policy[:B] = policy_obs
        obs_aug_policy[B:] = _transform_policy_obs_left_right(policy_obs, ndof)
        obs_out = {**obs, "policy": obs_aug_policy} if obs_is_dict else obs_aug_policy
    else:
        obs_out = None

    # augment actions
    if actions is not None:
        B, U = actions.shape
        actions_aug = actions.new_zeros((B * 2, U))
        actions_aug[:B] = actions
        actions_aug[B:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_out, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor, ndof: int) -> torch.Tensor:
    x = obs.clone()
    device = x.device

    # Fixed segment boundaries based on my own observation structure
    i = 0
    lin_vel = slice(i, i+3); i += 3
    ang_vel = slice(i, i+3); i += 3
    proj_g  = slice(i, i+3); i += 3
    cmds    = slice(i, i+4); i += 4
    jpos    = slice(i, i+ndof); i += ndof
    jvel    = slice(i, i+ndof); i += ndof
    lacts   = slice(i, i+ndof); i += ndof
    lidar   = slice(i, x.shape[1])

    # sign flips
    x[:, lin_vel] *= torch.tensor([1.0, -1.0,  1.0], device=device)
    x[:, ang_vel] *= torch.tensor([-1.0,  1.0, -1.0], device=device)
    x[:, proj_g ] *= torch.tensor([1.0, -1.0,  1.0], device=device)

    # commands: [vx, vy, yaw_rate, heading] -> vy, yaw, heading flip
    x_cmd = x[:, cmds].clone()
    x_cmd[:, 1] *= -1.0   # vy
    x_cmd[:, 2] *= -1.0   # yaw_rate
    x_cmd[:, 3] *= -1.0   # heading
    x[:, cmds] = x_cmd

    # joints
    x[:, jpos]  = _switch_go2_joints_left_right(x[:, jpos])
    x[:, jvel]  = _switch_go2_joints_left_right(x[:, jvel])
    x[:, lacts] = _switch_go2_joints_left_right(x[:, lacts])

    # LiDAR (flattened [*, 3]) — flip y
    if lidar.stop - lidar.start > 0:
        lidar_flat = x[:, lidar]
        num_pts3 = (lidar_flat.shape[1] // 3) * 3
        if num_pts3 > 0:
            head = lidar_flat[:, :num_pts3].reshape(-1, num_pts3 // 3, 3)
            head[:, :, 1] *= -1.0
            lidar_flat[:, :num_pts3] = head.reshape(lidar_flat.shape[0], num_pts3)
        x[:, lidar] = lidar_flat

    return x


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """
    Swap L↔R legs and flip hip signs.
    """
    y = actions.clone()
    y[:] = _switch_go2_joints_left_right(y[:])
    return y



def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """
    Go2 joint order (12):
      0: FL_hip_joint,   1: FR_hip_joint,   2: RL_hip_joint,   3: RR_hip_joint,
      4: FL_thigh_joint, 5: FR_thigh_joint, 6: RL_thigh_joint, 7: RR_thigh_joint,
      8: FL_calf_joint,  9: FR_calf_joint, 10: RL_calf_joint, 11: RR_calf_joint

    Left–right swap:
      FL <-> FR,   RL <-> RR
    Hip joints (0..3) change sign under mirror.
    """
    out = torch.zeros_like(joint_data)
    # left <-- right
    out[..., [0, 4, 8]]  = joint_data[..., [1, 5, 9]]    # FL <- FR
    out[..., [2, 6, 10]] = joint_data[..., [3, 7, 11]]   # RL <- RR
    # right <-- left
    out[..., [1, 5, 9]]  = joint_data[..., [0, 4, 8]]    # FR <- FL
    out[..., [3, 7, 11]] = joint_data[..., [2, 6, 10]]   # RR <- RL

    # Hip sign flip (ab/adduction) for indices [0,1,2,3]
    out[..., [0, 1, 2, 3]] *= -1.0
    return out
# ----------------------------------------------------------------------



@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    logger = "wandb"
    wandb_project = "go2_stairs"
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "go2_traversal"
    empirical_normalization = False


    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15, #change to 0.15 for phase 1-4, 0.8 for phase 0
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    
    #=========For LSTM:=============
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=0.15, #change to 0.15 for phase 1-4, 0.8 for phase 0
    #     actor_hidden_dims=[256, 256, 128],
    #     critic_hidden_dims=[256, 256, 128],
    #     activation="elu",
    #     rnn_type="lstm",      # LSTM enabled
    #     rnn_hidden_dim=256,   # memory size
    #     rnn_num_layers=1,     # stacked LSTM layers
    # )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=False,
            mirror_loss_coeff=0.5,
            data_augmentation_func=compute_symmetric_states,
        ),
    )
