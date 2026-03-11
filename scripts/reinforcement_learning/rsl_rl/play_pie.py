# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Script to play a trained PIE agent in Isaac Lab.

This play script uses PIEOnPolicyRunner instead of the standard
OnPolicyRunner, so the PIE history buffer and injection are active
during inference.

Usage::

    # Play with GUI (50 envs, latest checkpoint)
    python scripts/reinforcement_learning/rsl_rl/play_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --num_envs 50

    # Play specific checkpoint
    python scripts/reinforcement_learning/rsl_rl/play_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --num_envs 50 \
        --load_run 2026-02-07_00-31-29 --checkpoint model_18700.pt

    # Record video (headless)
    python scripts/reinforcement_learning/rsl_rl/play_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --num_envs 16 --headless --video --video_length 500

    # Keyboard control
    python scripts/reinforcement_learning/rsl_rl/play_pie.py \
        --task Stairs-Deeprobotics-Lite3-PIE-v0 \
        --keyboard
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Force Vulkan renderer to use the same GPU as CUDA.
# On multi-GPU systems, Vulkan may pick a GPU that is already full,
# causing "Out of GPU memory allocating resource 'Foundation Resource Allocator'".
# Set this BEFORE importing AppLauncher / Omniverse.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
from rl_utils import camera_follow

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained PIE RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Stairs-Deeprobotics-Lite3-PIE-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--free-camera", action="store_true", default=False, help="Use free camera instead of following camera."
)

# PIE-specific arguments (must match train_pie.py defaults)
parser.add_argument("--pie_lr", type=float, default=1e-3, help="PIE Estimator learning rate.")
parser.add_argument("--pie_recon_weight", type=float, default=1.0, help="PIE reconstruction loss weight.")
parser.add_argument("--pie_est_weight", type=float, default=1.0, help="PIE estimation loss weight.")
parser.add_argument("--pie_kl_weight", type=float, default=0.01, help="PIE KL divergence loss weight.")

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
import time
import torch
from torch.export import Dim

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import PIE Runner
from rl_training.tasks.manager_based.locomotion.velocity.pie import PIEOnPolicyRunner

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with PIE agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 16

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --- Play-specific terrain settings ---
    # Spawn robots randomly across terrain grid (ignore curriculum levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # Reduce terrain grid to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 3
        env_cfg.scene.terrain.terrain_generator.num_cols = 3
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        # Reduce sub-terrain size to further save rendering VRAM
        env_cfg.scene.terrain.terrain_generator.size = (6.0, 6.0)

    # Disable observation noise for clean evaluation
    env_cfg.observations.policy.enable_corruption = False
    # Remove random pushing events
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    # Disable curriculum changes during play
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1] / 2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # --- Resolve checkpoint path ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for rsl_rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Build PIE config ---
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
        "depth_augmentation": {},  # No augmentation during inference (empty = defaults, unused in eval)
        "proprio_obs_key": "pie_proprio",
        "gt_vel_key": "pie_gt_vel",
        "gt_clearance_key": "pie_gt_clearance",
        "gt_height_scan_key": "pie_gt_height_scan",
        "gt_next_state_key": "pie_gt_next_state",
    }

    train_cfg = agent_cfg.to_dict()
    train_cfg["pie_cfg"] = pie_cfg

    # --- Create PIE runner and load checkpoint ---
    runner = PIEOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # extract the neural network module
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # export policy (actor only) to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )

    # --- Export PIE Estimator ---
    # Wrapper to convert dict-output forward() into a clean tuple interface
    # suitable for ONNX/JIT export. Inputs: (proprio_history, depth_history, gru_hidden)
    # Outputs: (injection, gru_hidden_out)
    class PIEEstimatorExportWrapper(torch.nn.Module):
        """Export wrapper for PIE Estimator.

        Wraps the estimator forward pass so that:
        - GRU hidden state is an explicit input/output (stateless per call).
        - Output is a tuple (injection, gru_hidden) instead of a dict.
        - injection = cat[est_vel(3), est_clearance(4), map_enc(map_dim), latent_mu(latent_dim)]
        """

        def __init__(self, estimator):
            super().__init__()
            self.estimator = estimator

        def forward(self, proprio_history: torch.Tensor, depth_history: torch.Tensor, gru_hidden: torch.Tensor):
            e = self.estimator
            proprio_feat = e.proprio_encoder(proprio_history)
            depth_feat = e.depth_encoder(depth_history)

            # Bypass TransformerEncoder.forward() to avoid aten::_transformer_encoder_layer_fwd.
            # In PyTorch 2.7, torch.onnx.export internally traces TransformerEncoder and
            # triggers the fused fast-path kernel regardless of enable_nested_tensor.
            # Iterating through TransformerEncoderLayer instances directly calls only
            # standard attention + feedforward ops that ONNX opset 17 supports.
            fusion = e.fusion
            tokens = torch.stack([proprio_feat, depth_feat], dim=1)  # (batch, 2, feat)
            attended = tokens
            for layer in fusion.transformer.layers:
                attended = layer(attended)
            pooled = attended.mean(dim=1)
            new_hidden = fusion.gru(pooled, gru_hidden)
            fused = new_hidden

            est_vel = e.vel_head(fused)
            est_clearance = e.clearance_head(fused)
            latent_mu = e.latent_mu_head(fused)
            map_enc = e.map_enc_head(fused)
            injection = torch.cat([est_vel, est_clearance, map_enc, latent_mu], dim=-1)
            return injection, new_hidden

    class PIEPolicyCombinedWrapper(torch.nn.Module):
        """Combined PIE Estimator + Policy Actor for single-file deployment.

        Packages both networks into one module so that a single ONNX/JIT file
        can be loaded on the robot without managing two separate models.
        The GRU hidden state is an explicit input/output (stateless per call).

        Inputs:  (policy_obs, proprio_history, depth_history, gru_hidden)
        Outputs: (actions, gru_hidden_out)

        Where ``policy_obs`` is the raw environment "policy" observation
        (before PIE augmentation), and ``actions`` are the actor outputs.
        """

        def __init__(self, estimator, policy_actor):
            super().__init__()
            self.estimator = estimator
            self.policy_actor = policy_actor

        def forward(
            self,
            policy_obs: torch.Tensor,
            proprio_history: torch.Tensor,
            depth_history: torch.Tensor,
            gru_hidden: torch.Tensor,
        ):
            e = self.estimator
            proprio_feat = e.proprio_encoder(proprio_history)
            depth_feat = e.depth_encoder(depth_history)

            # Manual transformer iteration (avoid fused aten op, same as PIEEstimatorExportWrapper)
            fusion = e.fusion
            tokens = torch.stack([proprio_feat, depth_feat], dim=1)
            attended = tokens
            for layer in fusion.transformer.layers:
                attended = layer(attended)
            pooled = attended.mean(dim=1)
            new_hidden = fusion.gru(pooled, gru_hidden)
            fused = new_hidden

            est_vel = e.vel_head(fused)
            est_clearance = e.clearance_head(fused)
            latent_mu = e.latent_mu_head(fused)
            map_enc = e.map_enc_head(fused)
            injection = torch.cat([est_vel, est_clearance, map_enc, latent_mu], dim=-1)

            # Concatenate PIE injection with raw policy observation
            augmented_obs = torch.cat([policy_obs, injection], dim=-1)

            # Run policy actor MLP
            actions = self.policy_actor(augmented_obs)
            return actions, new_hidden

    pie_export = PIEEstimatorExportWrapper(runner.pie_estimator).to(env.unwrapped.device)
    pie_export.eval()

    # Dummy inputs for tracing / ONNX export
    num_dummy = 1
    device = env.unwrapped.device
    dummy_proprio = torch.zeros(num_dummy, pie_cfg["proprio_history_len"], pie_cfg["proprio_dim"], device=device)
    dummy_depth = torch.zeros(
        num_dummy, pie_cfg["depth_history_len"], pie_cfg["depth_channels"],
        pie_cfg["depth_height"], pie_cfg["depth_width"], device=device,
    )
    dummy_hidden = torch.zeros(num_dummy, pie_cfg["hidden_dim"], device=device)

    # JIT export (TorchScript trace)
    os.makedirs(export_model_dir, exist_ok=True)
    pie_jit_path = os.path.join(export_model_dir, "pie_estimator.pt")
    with torch.no_grad():
        traced_pie = torch.jit.trace(pie_export, (dummy_proprio, dummy_depth, dummy_hidden))
    traced_pie.save(pie_jit_path)
    print(f"[INFO] Exported PIE Estimator JIT to: {pie_jit_path}")

    # ONNX export
    # Use dynamo=True (torch.export-based exporter) because the legacy JIT-based
    # torch.onnx.export applies _jit_pass_onnx which re-fuses TransformerEncoderLayer
    # ops into aten::_transformer_encoder_layer_fwd — unsupported in any ONNX opset.
    # The dynamo exporter decomposes the graph before ONNX translation and avoids this.
    pie_onnx_path = os.path.join(export_model_dir, "pie_estimator.onnx")
    _batch = Dim("batch")
    with torch.no_grad():
        torch.onnx.export(
            pie_export,
            (dummy_proprio, dummy_depth, dummy_hidden),
            pie_onnx_path,
            dynamo=True,
            input_names=["proprio_history", "depth_history", "gru_hidden"],
            output_names=["injection", "gru_hidden_out"],
            dynamic_shapes={
                "proprio_history": {0: _batch},
                "depth_history":   {0: _batch},
                "gru_hidden":      {0: _batch},
            },
        )
    print(f"[INFO] Exported PIE Estimator ONNX to: {pie_onnx_path}")

    # --- Export Combined PIE + Policy (single-file deployment) ---
    combined_export = PIEPolicyCombinedWrapper(
        runner.pie_estimator, policy_nn.actor,
    ).to(device)
    combined_export.eval()

    # Compute raw policy obs dimension (actor input = raw_policy + pie_injection)
    _actor_input_dim = next(policy_nn.actor.parameters()).shape[1]
    _injection_dim = 3 + 4 + pie_cfg["map_dim"] + pie_cfg["latent_dim"]
    policy_obs_dim = _actor_input_dim - _injection_dim
    dummy_policy_obs = torch.zeros(num_dummy, policy_obs_dim, device=device)

    # JIT export
    combined_jit_path = os.path.join(export_model_dir, "pie_policy_combined.pt")
    with torch.no_grad():
        traced_combined = torch.jit.trace(
            combined_export,
            (dummy_policy_obs, dummy_proprio, dummy_depth, dummy_hidden),
        )
    traced_combined.save(combined_jit_path)
    print(f"[INFO] Exported combined PIE+Policy JIT to: {combined_jit_path}")

    # ONNX export (dynamo-based, same reason as PIE estimator)
    combined_onnx_path = os.path.join(export_model_dir, "pie_policy_combined.onnx")
    _batch_c = Dim("batch")
    with torch.no_grad():
        torch.onnx.export(
            combined_export,
            (dummy_policy_obs, dummy_proprio, dummy_depth, dummy_hidden),
            combined_onnx_path,
            dynamo=True,
            input_names=["policy_obs", "proprio_history", "depth_history", "gru_hidden"],
            output_names=["actions", "gru_hidden_out"],
            dynamic_shapes={
                "policy_obs":      {0: _batch_c},
                "proprio_history": {0: _batch_c},
                "depth_history":   {0: _batch_c},
                "gru_hidden":      {0: _batch_c},
            },
        )
    print(f"[INFO] Exported combined PIE+Policy ONNX to: {combined_onnx_path}")

    # --- Verify ONNX exports against PyTorch reference ---
    try:
        import onnxruntime as ort
        import numpy as np

        print("[VERIFY] Running PyTorch vs ONNX numerical verification...")

        # Prepare random test inputs on CPU
        _test_policy_obs = torch.randn(1, policy_obs_dim, device="cpu")
        _test_proprio = torch.randn(
            1, pie_cfg["proprio_history_len"], pie_cfg["proprio_dim"], device="cpu",
        )
        _test_depth = torch.randn(
            1, pie_cfg["depth_history_len"], pie_cfg["depth_channels"],
            pie_cfg["depth_height"], pie_cfg["depth_width"], device="cpu",
        )
        _test_hidden = torch.zeros(1, pie_cfg["hidden_dim"], device="cpu")

        _atol = 1e-4  # absolute tolerance for FP32

        # 1) PIE Estimator: PyTorch vs ONNX
        pie_export.cpu().eval()
        with torch.no_grad():
            _pt_inj, _pt_h = pie_export(_test_proprio, _test_depth, _test_hidden)
        _sess_pie = ort.InferenceSession(pie_onnx_path, providers=["CPUExecutionProvider"])
        _ort_pie = _sess_pie.run(None, {
            "proprio_history": _test_proprio.numpy(),
            "depth_history":   _test_depth.numpy(),
            "gru_hidden":      _test_hidden.numpy(),
        })
        _pie_inj_diff = float(np.abs(_pt_inj.numpy() - _ort_pie[0]).max())
        _pie_hid_diff = float(np.abs(_pt_h.numpy() - _ort_pie[1]).max())
        print(f"[VERIFY] PIE Estimator  — injection max_diff={_pie_inj_diff:.2e}, "
              f"hidden max_diff={_pie_hid_diff:.2e}")
        assert _pie_inj_diff < _atol, f"PIE injection mismatch: {_pie_inj_diff}"
        assert _pie_hid_diff < _atol, f"PIE hidden mismatch: {_pie_hid_diff}"

        # 2) Combined PIE+Policy: PyTorch vs ONNX
        combined_export.cpu().eval()
        with torch.no_grad():
            _pt_act, _pt_ch = combined_export(
                _test_policy_obs, _test_proprio, _test_depth, _test_hidden,
            )
        _sess_comb = ort.InferenceSession(combined_onnx_path, providers=["CPUExecutionProvider"])
        _ort_comb = _sess_comb.run(None, {
            "policy_obs":      _test_policy_obs.numpy(),
            "proprio_history": _test_proprio.numpy(),
            "depth_history":   _test_depth.numpy(),
            "gru_hidden":      _test_hidden.numpy(),
        })
        _comb_act_diff = float(np.abs(_pt_act.numpy() - _ort_comb[0]).max())
        _comb_hid_diff = float(np.abs(_pt_ch.numpy() - _ort_comb[1]).max())
        print(f"[VERIFY] Combined Model — actions max_diff={_comb_act_diff:.2e}, "
              f"hidden max_diff={_comb_hid_diff:.2e}")
        assert _comb_act_diff < _atol, f"Combined actions mismatch: {_comb_act_diff}"
        assert _comb_hid_diff < _atol, f"Combined hidden mismatch: {_comb_hid_diff}"

        # 3) Combined JIT: PyTorch vs JIT
        _jit_comb = torch.jit.load(combined_jit_path, map_location="cpu")
        with torch.no_grad():
            _jit_act, _jit_h = _jit_comb(_test_policy_obs, _test_proprio, _test_depth, _test_hidden)
        _jit_act_diff = float(np.abs(_pt_act.numpy() - _jit_act.numpy()).max())
        print(f"[VERIFY] Combined JIT   — actions max_diff={_jit_act_diff:.2e}")
        assert _jit_act_diff < _atol, f"JIT actions mismatch: {_jit_act_diff}"

        # Shape sanity check
        print(f"[VERIFY] Output shapes  — actions={list(_ort_comb[0].shape)}, "
              f"gru_hidden={list(_ort_comb[1].shape)}")

        # Move models back to original device for play loop
        pie_export.to(device)
        combined_export.to(device)

        print("[VERIFY] All ONNX/JIT exports verified ✓")

    except ImportError:
        print("[VERIFY] onnxruntime not installed, skipping ONNX verification. "
              "Install with: pip install onnxruntime")
    except Exception as e:
        print(f"[VERIFY] Verification failed: {e}")

    # Get PIE-aware inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting play loop. Press Ctrl+C or close the window to stop.")

    # --- Play loop ---
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard and not args_cli.free_camera:
            camera_follow(env)

        # Real-time delay
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
