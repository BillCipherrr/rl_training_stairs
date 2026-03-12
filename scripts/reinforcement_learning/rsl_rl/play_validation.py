# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Validation script for stair-climbing policies (standard and PIE).

Runs a trained policy on the fixed-difficulty validation environment and
reports per-episode success rates. A trial counts as **success** when the
robot's maximum height gain during the episode exceeds ``--success_height``
(default 0.20 m, i.e. at least one full 20 cm step).

The runner type is selected automatically:
- Task name contains "PIE"  →  PIEOnPolicyRunner  (148-dim actor input)
- Otherwise                 →  OnPolicyRunner     (45-dim actor input)

Usage examples::

    # Standard policy (non-PIE)
    python scripts/reinforcement_learning/rsl_rl/play_validation.py \\
        --task Stairs-Deeprobotics-Lite3-Validation-v0 \\
        --num_envs 50 --num_eval_episodes 200

    # PIE policy — task must be the PIE variant
    python scripts/reinforcement_learning/rsl_rl/play_validation.py \\
        --task Stairs-Deeprobotics-Lite3-PIE-Validation-v0 \\
        --load_run 2026-03-10_16-27-27 --checkpoint model_18500.pt \\
        --num_envs 50 --num_eval_episodes 200

    # Record video (headless)
    python scripts/reinforcement_learning/rsl_rl/play_validation.py \\
        --task Stairs-Deeprobotics-Lite3-PIE-Validation-v0 \\
        --load_run 2026-03-10_16-27-27 --checkpoint model_18500.pt \\
        --num_envs 16 --headless --video --video_length 500

    # Override forward velocity (default 0.5 m/s)
    python scripts/reinforcement_learning/rsl_rl/play_validation.py \\
        --task Stairs-Deeprobotics-Lite3-PIE-Validation-v0 \\
        --load_run 2026-03-10_16-27-27 --checkpoint model_18500.pt \\
        --num_envs 50 --vel_x 0.4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
from rl_utils import camera_follow

# ---- CLI arguments ----
parser = argparse.ArgumentParser(description="Validate stair-climbing policy (20 cm steps).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during validation.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Stairs-Deeprobotics-Lite3-PIE-Validation-v0",
    help="Gym task ID. Use the PIE variant for PIE checkpoints.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--free-camera", action="store_true", default=False)
# NOTE: --experiment_name is provided by cli_args.add_rsl_rl_args() below.

# Validation-specific arguments
parser.add_argument(
    "--num_eval_episodes",
    type=int,
    default=0,
    help="Stop after this many completed episodes (0 = run indefinitely).",
)
parser.add_argument(
    "--success_height",
    type=float,
    default=0.20,
    help="Height gain (m) above initial z to count as a successful trial. Default: 0.20 (one 20 cm step).",
)
parser.add_argument(
    "--vel_x",
    type=float,
    default=None,
    help="Override the fixed forward velocity command (m/s). Env default is 0.5 m/s.",
)

# PIE-specific arguments (only used when task contains "PIE")
parser.add_argument("--pie_lr", type=float, default=3e-4)
parser.add_argument("--pie_recon_weight", type=float, default=1.0)
parser.add_argument("--pie_est_weight", type=float, default=1.0)
parser.add_argument("--pie_kl_weight", type=float, default=0.01)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rl_training.tasks.manager_based.locomotion.velocity.pie import PIEOnPolicyRunner

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Detect PIE mode from task name so runner selection is automatic
_USE_PIE = "PIE" in args_cli.task


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run validation and report stair-climbing success rate."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ------ Override velocity command if requested ------
    if args_cli.vel_x is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vel_x, args_cli.vel_x)

    # ------ Resolve checkpoint ------
    exp_name = args_cli.experiment_name if args_cli.experiment_name else agent_cfg.experiment_name
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", exp_name))
    print(f"[INFO] Loading experiment from: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Checkpoint: {resume_path}")

    # ------ Create environment ------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "validation"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording validation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------ Load policy (PIE or standard) ------
    if _USE_PIE:
        pie_cfg = {
            "proprio_dim": 45,
            "proprio_history_len": 10,
            "depth_channels": 1,
            "depth_history_len": 2,
            "depth_height": 64,
            "depth_width": 64,
            "hidden_dim": 128,
            "latent_dim": 32,
            "map_dim": 64,
            "map_gt_dim": 345,
            "learning_rate": args_cli.pie_lr,
            "recon_weight": args_cli.pie_recon_weight,
            "est_weight": args_cli.pie_est_weight,
            "kl_weight": args_cli.pie_kl_weight,
            "use_depth": True,
            "depth_augmentation": {},
            "proprio_obs_key": "pie_proprio",
            "gt_vel_key": "pie_gt_vel",
            "gt_clearance_key": "pie_gt_clearance",
            "gt_height_scan_key": "pie_gt_height_scan",
            "gt_next_state_key": "pie_gt_next_state",
        }
        train_cfg = agent_cfg.to_dict()
        train_cfg["pie_cfg"] = pie_cfg
        runner = PIEOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ------ Print validation parameters ------
    vel_x = env_cfg.commands.base_velocity.ranges.lin_vel_x[0]
    print("\n" + "=" * 60)
    print("[VALIDATION] Parameters")
    print(f"  Task              : {args_cli.task}")
    print(f"  Runner            : {'PIEOnPolicyRunner' if _USE_PIE else 'OnPolicyRunner'}")
    print(f"  Checkpoint        : {os.path.basename(resume_path)}")
    print(f"  Num envs          : {env_cfg.scene.num_envs}")
    print(f"  Forward velocity  : {vel_x:.2f} m/s")
    print(f"  Success threshold : {args_cli.success_height:.2f} m height gain")
    print(f"  Target episodes   : {'unlimited' if args_cli.num_eval_episodes == 0 else args_cli.num_eval_episodes}")
    print("=" * 60 + "\n")

    # ------ Tracking state ------
    initial_heights: torch.Tensor | None = None
    max_heights: torch.Tensor | None = None
    episode_count = 0
    success_count = 0

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting validation loop. Press Ctrl+C to stop early.\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _rewards, dones, _infos = env.step(actions)

            # ------ Height tracking ------
            # IMPORTANT: after env.step() returns dones=True, Isaac Lab has already
            # reset those environments. current_z for done envs is therefore the NEW
            # spawn position, NOT the terminal position.
            # Fix: compute max_gain BEFORE updating max_heights with the reset position.
            current_z = env.unwrapped.scene["robot"].data.root_pos_w[:, 2]

            if initial_heights is None:
                initial_heights = current_z.clone()
                max_heights = current_z.clone()
                print(
                    f"[DEBUG] Spawn height (step 1): "
                    f"mean={current_z.mean().item():.3f} m  "
                    f"min={current_z.min().item():.3f} m  "
                    f"max={current_z.max().item():.3f} m"
                )

            # ------ Episode completion (must come before max_heights update) ------
            if dones.any():
                done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                # max_heights[done_idx] here still reflects the peak DURING the episode
                # (current_z for done envs is the reset position, not yet applied)
                max_gain = max_heights[done_idx] - initial_heights[done_idx]

                for gain in max_gain:
                    episode_count += 1
                    if gain.item() >= args_cli.success_height:
                        success_count += 1

                # Re-initialise trackers for completed envs to new spawn position
                initial_heights[done_idx] = current_z[done_idx]
                max_heights[done_idx] = current_z[done_idx]

                rate = 100.0 * success_count / episode_count
                avg_gain = max_gain.mean().item()
                print(
                    f"[EVAL] Episodes: {episode_count:4d} | "
                    f"Success: {success_count:4d} ({rate:5.1f}%) | "
                    f"Batch max-gain avg: {avg_gain:.3f} m"
                )

            # Update max_heights only for still-alive envs (done envs were just reset above)
            alive = ~dones
            if alive.any():
                max_heights[alive] = torch.maximum(max_heights[alive], current_z[alive])

            if args_cli.num_eval_episodes > 0 and episode_count >= args_cli.num_eval_episodes:
                print("\n[INFO] Reached target episode count. Stopping.")
                break

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            if not args_cli.headless and not args_cli.free_camera:
                camera_follow(env)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Validation interrupted by user.")

    # ------ Final summary ------
    print("\n" + "=" * 60)
    print("[VALIDATION] Final Results")
    print(f"  Total episodes    : {episode_count}")
    if episode_count > 0:
        final_rate = 100.0 * success_count / episode_count
        print(f"  Successful        : {success_count} / {episode_count}")
        print(f"  Success rate      : {final_rate:.1f}%")
        if final_rate >= 80:
            print("  Verdict           : PASS (>= 80%)")
        elif final_rate >= 50:
            print("  Verdict           : MARGINAL (50-80%)")
        else:
            print("  Verdict           : FAIL (< 50%)")
    else:
        print("  No episodes completed.")
    print("=" * 60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
