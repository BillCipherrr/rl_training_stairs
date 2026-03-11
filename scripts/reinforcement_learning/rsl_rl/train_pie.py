# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Script to train RL agent with RSL-RL + PIE Estimator.

This training script uses the PIEOnPolicyRunner instead of the standard
OnPolicyRunner. The PIE Runner integrates the dual-level implicit-explicit
estimator into the PPO training loop.

Usage::

    python scripts/reinforcement_learning/rsl_rl/train_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --num_envs 4096

    # With headless mode
    python scripts/reinforcement_learning/rsl_rl/train_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --num_envs 4096 --headless

    # Resume training
    python scripts/reinforcement_learning/rsl_rl/train_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --num_envs 4096 --resume --load_run <run_dir>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL + PIE Estimator.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Stairs-Deeprobotics-Lite3-PIE-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# PIE-specific arguments
parser.add_argument("--pie_lr", type=float, default=3e-4, help="PIE Estimator learning rate.")
parser.add_argument("--pie_recon_weight", type=float, default=1.0, help="PIE reconstruction loss weight.")
parser.add_argument("--pie_est_weight", type=float, default=1.0, help="PIE estimation loss weight.")
parser.add_argument("--pie_kl_weight", type=float, default=0.01, help="PIE KL divergence loss weight.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import PIE Runner
from rl_training.tasks.manager_based.locomotion.velocity.pie import PIEOnPolicyRunner

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL + PIE Estimator agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- PIE Configuration ---
    # Build PIE config from CLI args and defaults
    pie_cfg = {
        "proprio_dim": 45,
        "proprio_history_len": 10,   # PIE paper H1
        "depth_channels": 1,
        "depth_history_len": 2,      # PIE paper H2
        "depth_height": 64,
        "depth_width": 64,
        "hidden_dim": 128,
        "latent_dim": 32,
        "map_dim": 64,
        "map_gt_dim": 345,           # Lite3 height scan: GridPattern(res=0.07, size=[1.6,1.0]) = 23x15
        "learning_rate": args_cli.pie_lr,
        "recon_weight": args_cli.pie_recon_weight,
        "est_weight": args_cli.pie_est_weight,
        "kl_weight": args_cli.pie_kl_weight,
        "use_depth": True,           # Enable depth for sim-to-real deployment
        # --- Depth Domain Randomization (sim-to-real) ---
        "depth_augmentation": {
            "noise_std": 0.02,                # Additive Gaussian noise (metres, ~RealSense D435i)
            "dropout_prob": 0.3,              # Probability of rectangular dropout per env
            "dropout_num_rects_range": [1, 5], # Min/max dropout rectangles
            "dropout_size_range": [0.05, 0.3], # Rectangle size as fraction of image dim
            "latency_prob": 0.2,              # Probability of replacing with stale frame
            "latency_max_frames": 3,          # Max stale frame delay
            "shift_prob": 0.3,                # Probability of spatial shift
            "shift_max_pixels": 4,            # Max pixel shift in each direction
            "scale_range": [0.95, 1.05],      # Multiplicative depth scale perturbation
        },
        # Observation group keys in the TensorDict
        "proprio_obs_key": "pie_proprio",
        "gt_vel_key": "pie_gt_vel",
        "gt_clearance_key": "pie_gt_clearance",
        "gt_height_scan_key": "pie_gt_height_scan",
        "gt_next_state_key": "pie_gt_next_state",
    }

    # Add PIE config to the agent config dict
    train_cfg = agent_cfg.to_dict()
    train_cfg["pie_cfg"] = pie_cfg

    # Create PIE runner (instead of standard OnPolicyRunner)
    runner = PIEOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "pie.yaml"), pie_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
