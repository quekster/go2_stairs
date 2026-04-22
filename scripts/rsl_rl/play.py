# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import signal
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from body_attitude_eval import PlayBodyAttitudeRecorder  # isort: skip
from duty_phase_eval import PlayDutyPhaseRecorder  # isort: skip
from footfall_eval import PlayFootfallRecorder  # isort: skip
from foot_trajectory_eval import PlayFootTrajectoryRecorder  # isort: skip

# Body-attitude pitch reference settings (hard-coded; edit here for different stairs).
BODY_ATTITUDE_PITCH_REF_H = 0.18  # stair rise h (meters)
BODY_ATTITUDE_PITCH_REF_D = 0.18 # stair run d (meters)
BODY_ATTITUDE_PITCH_REF_MODE = "ascent"  # one of: ascent, descent, flat

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--footfall_eval",
    action="store_true",
    default=False,
    help="Enable footfall CSV recording and plotting.",
)
parser.add_argument(
    "--footfall_env_id",
    type=int,
    default=0,
    help="Environment index to record for footfall plotting.",
)
parser.add_argument(
    "--footfall_force_threshold",
    type=float,
    default=5.0,
    help="Contact force threshold (N) used to classify stance for each foot.",
)
parser.add_argument(
    "--duty_phase_eval",
    action="store_true",
    default=False,
    help="Enable duty-factor and phase-offset evaluation.",
)
parser.add_argument(
    "--foot_trajectory_eval",
    "--foot_tip_eval",
    dest="foot_trajectory_eval",
    action="store_true",
    default=False,
    help="Enable foot trajectory evaluation.",
)
parser.add_argument(
    "--foot_trajectory_env_id",
    "--foot_tip_env_id",
    dest="foot_trajectory_env_id",
    type=int,
    default=0,
    help="Environment index to record for foot trajectory plotting.",
)
parser.add_argument(
    "--foot_trajectory_force_threshold",
    "--foot_tip_force_threshold",
    dest="foot_trajectory_force_threshold",
    type=float,
    default=5.0,
    help="Contact force threshold (N) used to classify swing/stance for foot trajectory analysis.",
)
parser.add_argument(
    "--duty_phase_env_id",
    type=int,
    default=0,
    help="Environment index to record for duty-factor/phase-offset plotting.",
)
parser.add_argument(
    "--duty_phase_force_threshold",
    type=float,
    default=5.0,
    help="Contact force threshold (N) used to classify stance for duty/phase metrics.",
)
parser.add_argument(
    "--duty_phase_reference_leg",
    type=str,
    default="RL",
    choices=("FL", "FR", "RL", "RR"),
    help="Reference leg used to define stride cycles and phase offsets.",
)
parser.add_argument(
    "--body_attitude_eval",
    action="store_true",
    default=False,
    help="Enable body-attitude stability evaluation.",
)
parser.add_argument(
    "--body_attitude_env_id",
    type=int,
    default=0,
    help="Environment index to record for body-attitude stability plots.",
)
parser.add_argument(
    "--body_attitude_roll_band_deg",
    type=float,
    default=6.0,
    help="Roll stability threshold band in degrees.",
)
parser.add_argument(
    "--body_attitude_pitch_band_deg",
    type=float,
    default=8.0,
    help="Pitch-error stability threshold band in degrees.",
)
parser.add_argument(
    "--body_attitude_tilt_band_deg",
    type=float,
    default=10.0,
    help="Tilt-magnitude threshold in degrees.",
)
parser.add_argument(
    "--body_attitude_ang_vel_band_deg_s",
    type=float,
    default=35.0,
    help="Body angular-rate magnitude threshold in deg/s.",
)
parser.add_argument(
    "--body_attitude_recovery_window_s",
    type=float,
    default=2.0,
    help="Time window after each external-force event for recovery metrics.",
)
parser.add_argument(
    "--recording_timer",
    type=float,
    default=0.0,
    help="Auto-stop play/recording after this many seconds of simulation time. Disabled if <= 0.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_hybrid.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    footfall_recorder = None
    if args_cli.footfall_eval:
        footfall_output_dir = os.path.join(log_dir, "footfall_eval")
        footfall_recorder = PlayFootfallRecorder(
            output_dir=footfall_output_dir,
            env_id=args_cli.footfall_env_id,
            force_threshold=args_cli.footfall_force_threshold,
        )
        print(f"[INFO] Footfall evaluation enabled. Output directory: {footfall_recorder.output_dir}")

    duty_phase_recorder = None
    if args_cli.duty_phase_eval:
        duty_phase_output_dir = os.path.join(log_dir, "duty_phase_eval")
        duty_phase_recorder = PlayDutyPhaseRecorder(
            output_dir=duty_phase_output_dir,
            env_id=args_cli.duty_phase_env_id,
            force_threshold=args_cli.duty_phase_force_threshold,
            phase_reference_leg=args_cli.duty_phase_reference_leg,
        )
        print(
            "[INFO] Duty/phase evaluation enabled. "
            f"Reference leg: {duty_phase_recorder.phase_reference_leg}. "
            f"Output directory: {duty_phase_recorder.output_dir}"
        )

    foot_trajectory_recorder = None
    if args_cli.foot_trajectory_eval:
        foot_trajectory_output_dir = os.path.join(log_dir, "foot_trajectory_eval")
        foot_trajectory_recorder = PlayFootTrajectoryRecorder(
            output_dir=foot_trajectory_output_dir,
            env_id=args_cli.foot_trajectory_env_id,
            force_threshold=args_cli.foot_trajectory_force_threshold,
        )
        print(
            "[INFO] Foot-trajectory evaluation enabled. "
            f"Output directory: {foot_trajectory_recorder.output_dir}"
        )

    body_attitude_recorder = None
    if args_cli.body_attitude_eval:
        body_attitude_output_dir = os.path.join(log_dir, "body_attitude_eval")
        body_attitude_recorder = PlayBodyAttitudeRecorder(
            output_dir=body_attitude_output_dir,
            env_id=args_cli.body_attitude_env_id,
            roll_band_deg=args_cli.body_attitude_roll_band_deg,
            pitch_band_deg=args_cli.body_attitude_pitch_band_deg,
            tilt_band_deg=args_cli.body_attitude_tilt_band_deg,
            ang_vel_band_deg_s=args_cli.body_attitude_ang_vel_band_deg_s,
            pitch_ref_h=BODY_ATTITUDE_PITCH_REF_H,
            pitch_ref_d=BODY_ATTITUDE_PITCH_REF_D,
            pitch_ref_mode=BODY_ATTITUDE_PITCH_REF_MODE,
            recovery_window_s=args_cli.body_attitude_recovery_window_s,
        )
        if body_attitude_recorder.pitch_ref_mode == "ascent":
            ref_note = (
                "Ascent mode: pitch-ref sign is auto-resolved from observed pitch trace at finalize. "
                f"Initial |pitch_ref|={abs(body_attitude_recorder.pitch_ref_deg):.2f} deg."
            )
        else:
            ref_note = f"pitch_ref={body_attitude_recorder.pitch_ref_deg:.2f} deg."
        print(
            "[INFO] Body-attitude evaluation enabled. "
            f"Pitch ref mode={body_attitude_recorder.pitch_ref_mode}, "
            f"h={body_attitude_recorder.pitch_ref_h:.4f} m, "
            f"d={body_attitude_recorder.pitch_ref_d:.4f} m, "
            f"{ref_note} "
            f"Output directory: {body_attitude_recorder.output_dir}"
        )

    if args_cli.recording_timer > 0.0:
        print(f"[INFO] Recording timer enabled: {args_cli.recording_timer:.2f}s (simulation time).")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    sim_step = 0
    stop_requested = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(_signum, _frame):
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            print("\n[INFO] Ctrl+C received. Stopping play loop and finalizing outputs...")
            return
        print("\n[INFO] Second Ctrl+C received. Forcing shutdown...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        # simulate environment
        while simulation_app.is_running() and not stop_requested:
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)

            sim_step += 1
            sim_time_s = sim_step * dt

            if footfall_recorder is not None:
                footfall_recorder.record_step(env, sim_step=sim_step, sim_time_s=sim_time_s)
            if duty_phase_recorder is not None:
                duty_phase_recorder.record_step(env, sim_step=sim_step, sim_time_s=sim_time_s)
            if foot_trajectory_recorder is not None:
                foot_trajectory_recorder.record_step(env, sim_step=sim_step, sim_time_s=sim_time_s)
            if body_attitude_recorder is not None:
                body_attitude_recorder.record_step(env, sim_step=sim_step, sim_time_s=sim_time_s)

            if args_cli.recording_timer > 0.0 and sim_time_s >= args_cli.recording_timer:
                stop_requested = True
                print(
                    f"[INFO] Recording timer reached ({args_cli.recording_timer:.2f}s). "
                    "Stopping play loop..."
                )

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Closing play loop...")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if footfall_recorder is not None:
            output_paths = footfall_recorder.finalize()
            print(f"[INFO] Footfall session saved to: {output_paths['output_dir']}")
            print(f"[INFO] Footfall CSV saved to: {output_paths['csv_path']}")
            print(f"[INFO] Footfall plot saved to: {output_paths['plot_path']}")
        if duty_phase_recorder is not None:
            output_paths = duty_phase_recorder.finalize()
            print(f"[INFO] Duty/phase session saved to: {output_paths['output_dir']}")
            print(f"[INFO] Duty/phase samples CSV saved to: {output_paths['samples_csv_path']}")
            print(f"[INFO] Duty/phase cycles CSV saved to: {output_paths['cycles_csv_path']}")
            print(f"[INFO] Duty/phase summary CSV saved to: {output_paths['summary_csv_path']}")
            print(f"[INFO] Duty/phase plot saved to: {output_paths['plot_path']}")
        if foot_trajectory_recorder is not None:
            output_paths = foot_trajectory_recorder.finalize()
            print(f"[INFO] Foot-trajectory session saved to: {output_paths['output_dir']}")
            print(f"[INFO] Foot-trajectory samples CSV saved to: {output_paths['samples_csv_path']}")
            print(f"[INFO] Foot-trajectory summary CSV saved to: {output_paths['summary_csv_path']}")
            print(f"[INFO] Foot-trajectory plot saved to: {output_paths['plot_path']}")
        if body_attitude_recorder is not None:
            output_paths = body_attitude_recorder.finalize()
            print(f"[INFO] Body-attitude session saved to: {output_paths['output_dir']}")
            print(f"[INFO] Body-attitude samples CSV saved to: {output_paths['samples_csv_path']}")
            print(f"[INFO] Body-attitude events CSV saved to: {output_paths['events_csv_path']}")
            print(f"[INFO] Body-attitude summary CSV saved to: {output_paths['summary_csv_path']}")
            print(f"[INFO] Body-attitude plot saved to: {output_paths['plot_path']}")
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
