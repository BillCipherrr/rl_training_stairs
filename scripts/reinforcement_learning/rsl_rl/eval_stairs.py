# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Evaluate a trained policy on stairs of specific heights.

This script tests a trained stair-climbing policy on fixed-height straight
staircases (ascending and descending) and reports stability metrics:
- Survival rate (% of episodes without termination)
- Average episode length (steps)
- Forward distance traveled (m, along commanded direction)
- Lateral deviation from commanded direction (m)

Key design choices:
- Uses STRAIGHT staircases (not pyramid) to avoid terrain-induced lateral drift.
- Robot always spawns facing +x (yaw=0) toward the stairs.
- Stair corridor is 6m wide to allow natural gait variation without wall contact.

Usage:
    conda run -n env_isaaclab_2.3.0 python \\
        scripts/reinforcement_learning/rsl_rl/eval_stairs.py \\
        --task Stairs-Deeprobotics-Lite3-History-v0 \\
        --load_run 2026-02-05_03-22-22 \\
        --headless --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

parser = argparse.ArgumentParser(description="Evaluate stair climbing policy on straight staircases.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during evaluation.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of recorded video (steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None, help="Override total number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--step_heights", type=float, nargs="+", default=[0.14, 0.16, 0.18, 0.20],
    help="Step heights to evaluate (meters).",
)
parser.add_argument("--num_envs_per_height", type=int, default=64,
                    help="Environments per step height (split 50/50 ascending/descending, or all if --direction is set).")
parser.add_argument("--eval_steps", type=int, default=1000, help="Total simulation steps.")
parser.add_argument("--cmd_vel_x", type=float, default=0.5, help="Commanded forward velocity (m/s).")
parser.add_argument(
    "--direction", type=str, default="both", choices=["both", "ascending", "descending"],
    help="Which stair direction to test. 'both' tests ascending+descending, others test only one.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import trimesh
from collections import defaultdict

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Import terrain classes from shared module
# ---------------------------------------------------------------------------

from rl_training.tasks.manager_based.locomotion.velocity.mdp.terrains import (
    StraightAscendingStairsTerrainCfg,
    StraightDescendingStairsTerrainCfg,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_eval_terrain_cfg(step_heights: list[float], direction: str = "both") -> tuple[TerrainGeneratorCfg, list[str]]:
    """Build evaluation terrain with straight staircases at fixed heights.

    Creates sub-terrains based on direction: 'ascending', 'descending', or 'both'.
    Terrain cell size = (8, 8) m. With flat_start=flat_end=1.5m and
    step_width=0.28m, each cell has ~18 steps.

    Args:
        step_heights: Step heights to evaluate (m).
        direction: Which direction(s) to include ('ascending', 'descending', 'both').

    Returns:
        Tuple of (TerrainGeneratorCfg, terrain_names list).
    """
    sub_terrains = {}
    terrain_names = []

    for h in step_heights:
        h_cm = int(round(h * 100))
        if direction in ("both", "ascending"):
            sub_terrains[f"ascending_{h_cm}cm"] = StraightAscendingStairsTerrainCfg(
                proportion=1.0,  # normalized below
                step_height_range=(h, h),
                step_width=0.28,
                stair_width=6.0,
                flat_start_length=1.5,
                flat_end_length=1.5,
            )
            terrain_names.append(f"ascending_{h_cm}cm")
        if direction in ("both", "descending"):
            sub_terrains[f"descending_{h_cm}cm"] = StraightDescendingStairsTerrainCfg(
                proportion=1.0,  # normalized below
                step_height_range=(h, h),
                step_width=0.28,
                stair_width=6.0,
                flat_start_length=1.5,
                flat_end_length=1.5,
            )
            terrain_names.append(f"descending_{h_cm}cm")

    # Normalize proportions so they sum to 1
    proportion = 1.0 / len(sub_terrains)
    for cfg in sub_terrains.values():
        cfg.proportion = proportion

    terrain_cfg = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=5,
        num_cols=len(sub_terrains),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains=sub_terrains,
    )
    return terrain_cfg, terrain_names


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion (w, x, y, z format). Returns shape (...)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def decompose_displacement(
    displacement_xy: torch.Tensor, initial_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose XY displacement into forward and lateral components.

    Args:
        displacement_xy: Shape (N, 2).
        initial_yaw: Shape (N,) in radians.

    Returns:
        (forward_dist, lateral_dev), each shape (N,).
    """
    cos_y = torch.cos(initial_yaw)
    sin_y = torch.sin(initial_yaw)
    dx = displacement_xy[:, 0]
    dy = displacement_xy[:, 1]
    forward = dx * cos_y + dy * sin_y
    lateral = -dx * sin_y + dy * cos_y
    return forward, lateral


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate stair climbing policy on straight staircases."""
    step_heights = args_cli.step_heights
    num_envs_per_height = args_cli.num_envs_per_height
    eval_steps = args_cli.eval_steps
    cmd_vel_x = args_cli.cmd_vel_x

    num_heights = len(step_heights)
    num_terrain_types = num_heights * 2
    total_envs = args_cli.num_envs if args_cli.num_envs is not None else num_envs_per_height * num_heights

    terrain_names = []
    for h in step_heights:
        h_cm = int(round(h * 100))
        terrain_names.append(f"ascending_{h_cm}cm")
        terrain_names.append(f"descending_{h_cm}cm")

    print(f"\n{'='*70}")
    print(f"  Stair Climbing Evaluation (Straight Stairs)")
    print(f"{'='*70}")
    print(f"  Step heights: {[f'{h*100:.0f}cm' for h in step_heights]}")
    print(f"  Total environments: {total_envs} ({total_envs // num_terrain_types} per terrain type)")
    print(f"  Evaluation steps: {eval_steps}  |  cmd_vel: vx={cmd_vel_x} m/s")
    print(f"{'='*70}\n")

    # ----- Environment configuration -----
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = total_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Use straight staircase terrain
    env_cfg.scene.terrain.terrain_generator = build_eval_terrain_cfg(step_heights)
    env_cfg.scene.terrain.max_init_terrain_level = None

    # Disable evaluation-irrelevant settings
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels = None

    # Fix heading to yaw=0 so robots always face +x (toward stairs)
    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

    # Fix velocity commands to constant values
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (cmd_vel_x, cmd_vel_x)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    # ----- Load policy -----
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading from: {log_root_path}")

    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    log_dir = os.path.dirname(resume_path)
    eval_output_dir = os.path.join(log_dir, "eval_stairs")
    os.makedirs(eval_output_dir, exist_ok=True)

    # ----- Create environment -----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_dir = os.path.join(eval_output_dir, "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ----- Terrain type mapping -----
    terrain = env.unwrapped.scene.terrain
    terrain_type_indices = terrain.terrain_types.clone()  # (num_envs,) — column index per env
    device = env.unwrapped.device

    print("\n[INFO] Environment distribution:")
    for col_idx, name in enumerate(terrain_names):
        count = (terrain_type_indices == col_idx).sum().item()
        print(f"  Col {col_idx} ({name}): {count} envs")

    # ----- Initialize tracking -----
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt

    ep_start_pos = robot.data.root_pos_w[:, :2].clone()   # (N, 2)
    ep_start_yaw = get_yaw_from_quat(robot.data.root_quat_w)  # (N,)
    ep_steps = torch.zeros(total_envs, device=device)
    ep_terrain = terrain_type_indices.clone()

    metrics = defaultdict(lambda: {
        "episode_lengths": [],
        "forward_distances": [],
        "lateral_deviations": [],
        "terminated": [],   # True = fell/illegal, False = timeout/survived
    })

    # ----- Evaluation loop -----
    obs = env.get_observations()
    print(f"\n[INFO] Running {eval_steps} steps...")

    for step in range(eval_steps):
        pre_pos = robot.data.root_pos_w[:, :2].clone()  # position BEFORE step

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        ep_steps += 1

        done_mask = dones.bool()
        if done_mask.any():
            time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_indices:
                i = idx.item()
                disp = (pre_pos[i] - ep_start_pos[i]).unsqueeze(0)
                yaw = ep_start_yaw[i].unsqueeze(0)
                fwd, lat = decompose_displacement(disp, yaw)

                is_timeout = bool(time_outs[i].item()) if isinstance(time_outs, torch.Tensor) else False
                t = ep_terrain[i].item()

                metrics[t]["episode_lengths"].append(ep_steps[i].item())
                metrics[t]["forward_distances"].append(fwd.item())
                metrics[t]["lateral_deviations"].append(abs(lat.item()))
                metrics[t]["terminated"].append(not is_timeout)

            # Reset tracking for done envs
            ep_steps[done_mask] = 0
            ep_start_pos[done_mask] = robot.data.root_pos_w[done_mask, :2].clone()
            ep_start_yaw[done_mask] = get_yaw_from_quat(robot.data.root_quat_w[done_mask])
            ep_terrain[done_mask] = terrain_type_indices[done_mask]

        if (step + 1) % 200 == 0:
            print(f"  Step {step + 1}/{eval_steps}")

    # Record remaining (still-alive) episodes as survived
    for i in range(total_envs):
        if ep_steps[i].item() > 0:
            t = ep_terrain[i].item()
            disp = (robot.data.root_pos_w[i, :2] - ep_start_pos[i]).unsqueeze(0)
            yaw = ep_start_yaw[i].unsqueeze(0)
            fwd, lat = decompose_displacement(disp, yaw)
            metrics[t]["episode_lengths"].append(ep_steps[i].item())
            metrics[t]["forward_distances"].append(fwd.item())
            metrics[t]["lateral_deviations"].append(abs(lat.item()))
            metrics[t]["terminated"].append(False)

    # ----- Print report -----
    header = (
        f"{'Terrain':<22} {'Episodes':>8} {'Survival%':>10} "
        f"{'Avg Len':>9} {'Fwd(m)':>8} {'Lat(m)':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Checkpoint: {resume_path}")
    print(f"  cmd: vx={cmd_vel_x} m/s | dt={dt:.4f}s | steps={eval_steps}")
    print(f"{'='*70}\n")
    print(header)
    print(sep)

    height_agg = defaultdict(lambda: {"episode_lengths": [], "forward_distances": [],
                                       "lateral_deviations": [], "terminated": []})

    def _stats(m):
        n = len(m["episode_lengths"])
        if n == 0:
            return None
        surv = (1.0 - sum(m["terminated"]) / n) * 100
        return dict(
            n=n,
            survival=surv,
            avg_len=sum(m["episode_lengths"]) / n,
            avg_fwd=sum(m["forward_distances"]) / n,
            avg_lat=sum(m["lateral_deviations"]) / n,
        )

    for col_idx, name in enumerate(terrain_names):
        m = metrics[col_idx]
        s = _stats(m)
        if s is None:
            print(f"{name:<22} {'N/A':>8}")
        else:
            print(
                f"{name:<22} {s['n']:>8d} {s['survival']:>9.1f}% "
                f"{s['avg_len']:>9.1f} {s['avg_fwd']:>8.3f} {s['avg_lat']:>8.3f}"
            )
        h_cm = name.split("_")[-1]
        for key in ["episode_lengths", "forward_distances", "lateral_deviations", "terminated"]:
            height_agg[h_cm][key].extend(m[key])

    print(f"\n{'='*70}")
    print(f"  SUMMARY BY HEIGHT (ascending + descending combined)")
    print(f"{'='*70}\n")
    print(header)
    print(sep)

    for h in step_heights:
        h_cm = f"{int(round(h * 100))}cm"
        s = _stats(height_agg[h_cm])
        if s is None:
            print(f"{h_cm:<22} {'N/A':>8}")
        else:
            print(
                f"{h_cm:<22} {s['n']:>8d} {s['survival']:>9.1f}% "
                f"{s['avg_len']:>9.1f} {s['avg_fwd']:>8.3f} {s['avg_lat']:>8.3f}"
            )

    # ----- Save report -----
    report_path = os.path.join(eval_output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write("Stair Climbing Evaluation Report (Straight Stairs)\n")
        f.write(f"{'='*70}\n")
        f.write(f"Checkpoint: {resume_path}\n")
        f.write(f"Step heights: {[f'{h*100:.0f}cm' for h in step_heights]}\n")
        f.write(f"cmd: vx={cmd_vel_x} m/s, vy=0, wz=0 | Initial yaw: 0 (fixed)\n")
        f.write(f"Eval steps: {eval_steps} | dt={dt:.4f}s | Total envs: {total_envs}\n")
        f.write(f"{'='*70}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for col_idx, name in enumerate(terrain_names):
            s = _stats(metrics[col_idx])
            if s is None:
                f.write(f"{name:<22} {'N/A':>8}\n")
            else:
                f.write(
                    f"{name:<22} {s['n']:>8d} {s['survival']:>9.1f}% "
                    f"{s['avg_len']:>9.1f} {s['avg_fwd']:>8.3f} {s['avg_lat']:>8.3f}\n"
                )
        f.write(f"\n{'='*70}\n  SUMMARY BY HEIGHT\n{'='*70}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for h in step_heights:
            h_cm = f"{int(round(h * 100))}cm"
            s = _stats(height_agg[h_cm])
            if s is None:
                f.write(f"{h_cm:<22} {'N/A':>8}\n")
            else:
                f.write(
                    f"{h_cm:<22} {s['n']:>8d} {s['survival']:>9.1f}% "
                    f"{s['avg_len']:>9.1f} {s['avg_fwd']:>8.3f} {s['avg_lat']:>8.3f}\n"
                )

    print(f"\n[INFO] Report saved to: {report_path}")
    if args_cli.video:
        print(f"[INFO] Video saved to: {video_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
