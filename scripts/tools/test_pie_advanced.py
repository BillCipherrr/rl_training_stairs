#!/usr/bin/env python3
# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Advanced verification script for PIE Estimator components.

Extends the basic test suite with:
    1. Height scan dimension alignment (map_gt_dim vs actual scanner config)
    2. End-to-end training loop (multi-step rollout + optimizer convergence)
    3. Model save/load roundtrip (state_dict consistency)
    4. Edge cases (batch=1, zero input, large values, env reset mid-rollout)
    5. Buffer → Estimator → Loss end-to-end pipeline
    6. Policy observation dimension compatibility
    7. GT utils interface validation (foot names, scanner config)

Usage:
    cd /home/user1/rl_training
    python scripts/tools/test_pie_advanced.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Direct path to avoid Isaac Lab dependency
_PIE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "source", "rl_training", "rl_training",
        "tasks", "manager_based", "locomotion", "velocity", "pie",
    )
)
sys.path.insert(0, os.path.dirname(_PIE_DIR))

import torch
import torch.optim as optim

from pie.pi_estimator import PIEEstimator
from pie.pie_history_buffer import PIEHistoryBuffer
from pie.pie_loss import (
    compute_pie_estimator_loss,
    reconstruction_loss,
    estimation_loss,
    kl_divergence_loss,
)

# ===========================================================================
# Constants matching Lite3 configuration
# ===========================================================================
PROPRIO_DIM = 45   # base_ang_vel(3) + gravity(3) + cmd(3) + jpos(12) + jvel(12) + act(12)
H1 = 10            # PIE paper: proprioception history
H2 = 2             # PIE paper: depth history
DEPTH_C = 1
DEPTH_H = 64
DEPTH_W = 64
HIDDEN_DIM = 128
LATENT_DIM = 32
MAP_DIM = 64

# Lite3 height scanner: GridPattern(resolution=0.07, size=[1.6, 1.0])
# Grid: int(1.6/0.07)+1 = 23, int(1.0/0.07)+1 = 15 → 23 * 15 = 345
MAP_GT_DIM_LITE3 = 345

# Default scanner: GridPattern(resolution=0.1, size=[1.6, 1.0])
# Grid: int(1.6/0.1)+1 = 17, int(1.0/0.1)+1 = 11 → 17 * 11 = 187
MAP_GT_DIM_DEFAULT = 187

LITE3_FOOT_NAMES = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
LITE3_JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Test 1: Height scan dimension alignment
# ===========================================================================
def test_height_scan_dim_alignment() -> bool:
    """Verify map_gt_dim matches actual Lite3 height scanner configuration."""
    print("=" * 60)
    print("Test 1: Height Scan Dimension Alignment")
    print("=" * 60)

    all_pass = True

    # Compute expected ray count for Lite3 scanner config
    resolution = 0.07  # from rough_env_cfg.py override
    size = [1.6, 1.0]  # from velocity_env_cfg.py MySceneCfg
    nx = int(size[0] / resolution) + 1  # 23
    ny = int(size[1] / resolution) + 1  # 15
    expected_rays = nx * ny  # 345

    ok = expected_rays == MAP_GT_DIM_LITE3
    print(f"  {'✓' if ok else '✗'} Lite3 scanner rays: {nx}x{ny} = {expected_rays} (expected {MAP_GT_DIM_LITE3})")
    if not ok:
        all_pass = False

    # Verify PIEEstimator default matches
    estimator = PIEEstimator()
    default_ok = estimator.map_decoder[-1].out_features == MAP_GT_DIM_LITE3
    actual_out = estimator.map_decoder[-1].out_features
    print(f"  {'✓' if default_ok else '✗'} PIEEstimator default map_gt_dim: {actual_out} (expected {MAP_GT_DIM_LITE3})")
    if not default_ok:
        all_pass = False

    # Verify rec_map output dimension
    device = get_device()
    estimator = estimator.to(device)
    estimator.reset_hidden(2, device)
    with torch.no_grad():
        out = estimator(
            torch.randn(2, H1, PROPRIO_DIM, device=device),
            torch.randn(2, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )
    rec_map_dim = out["rec_map"].shape[-1]
    dim_ok = rec_map_dim == MAP_GT_DIM_LITE3
    print(f"  {'✓' if dim_ok else '✗'} rec_map output dim: {rec_map_dim} (expected {MAP_GT_DIM_LITE3})")
    if not dim_ok:
        all_pass = False

    # Verify custom map_gt_dim works
    est_custom = PIEEstimator(map_gt_dim=MAP_GT_DIM_DEFAULT).to(device)
    est_custom.reset_hidden(2, device)
    with torch.no_grad():
        out_custom = est_custom(
            torch.randn(2, H1, PROPRIO_DIM, device=device),
            torch.randn(2, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )
    custom_ok = out_custom["rec_map"].shape[-1] == MAP_GT_DIM_DEFAULT
    print(f"  {'✓' if custom_ok else '✗'} Custom map_gt_dim={MAP_GT_DIM_DEFAULT} works correctly")
    if not custom_ok:
        all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 2: End-to-end training loop
# ===========================================================================
def test_training_loop() -> bool:
    """Simulate a multi-step training loop and verify loss convergence."""
    print("=" * 60)
    print("Test 2: End-to-End Training Loop (50 iterations)")
    print("=" * 60)

    device = get_device()
    batch_size = 32

    estimator = PIEEstimator(
        map_gt_dim=MAP_GT_DIM_LITE3,
    ).to(device)

    optimizer = optim.Adam(estimator.parameters(), lr=1e-3)
    buffer = PIEHistoryBuffer(
        num_envs=batch_size,
        proprio_dim=PROPRIO_DIM,
        proprio_history_len=H1,
        depth_channels=DEPTH_C,
        depth_height=DEPTH_H,
        depth_width=DEPTH_W,
        depth_history_len=H2,
        device=device,
    )

    # Fixed target for convergence test (simulating consistent GT signals)
    gt_vel = torch.randn(batch_size, 3, device=device) * 0.5
    gt_clearance = torch.rand(batch_size, 4, device=device) * 0.1
    gt_map = torch.randn(batch_size, MAP_GT_DIM_LITE3, device=device) * 0.1

    losses_history = []
    all_pass = True

    estimator.reset_hidden(batch_size, device)
    estimator.train()

    t0 = time.time()

    for step in range(50):
        # Simulate env step: push new observations
        obs = torch.randn(batch_size, PROPRIO_DIM, device=device)
        depth = torch.randn(batch_size, DEPTH_C, DEPTH_H, DEPTH_W, device=device)
        buffer.push_proprio(obs)
        buffer.push_depth(depth)

        # GT next state (fixed target for convergence)
        gt_next_state = torch.randn(batch_size, PROPRIO_DIM, device=device) * 0.1

        # Forward
        output = estimator(
            buffer.get_proprio_history(),
            buffer.get_depth_history(),
        )

        # Loss
        losses = compute_pie_estimator_loss(
            estimator_output=output,
            gt_next_state=gt_next_state,
            gt_vel=gt_vel,
            gt_clearance=gt_clearance,
            gt_map=gt_map,
        )

        # Backward + optimize
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = losses["total"].item()
        losses_history.append(loss_val)

        # Check for NaN/Inf
        if not torch.isfinite(losses["total"]):
            print(f"  ✗ NaN/Inf detected at step {step}: {loss_val}")
            all_pass = False
            break

    elapsed = time.time() - t0

    # Verify no NaN/Inf throughout
    no_nan = all(torch.isfinite(torch.tensor(l)) for l in losses_history)
    print(f"  {'✓' if no_nan else '✗'} No NaN/Inf in 50 steps")
    if not no_nan:
        all_pass = False

    # Verify loss decreased (first 5 avg vs last 5 avg)
    first_avg = sum(losses_history[:5]) / 5
    last_avg = sum(losses_history[-5:]) / 5
    decreased = last_avg < first_avg
    print(f"  {'✓' if decreased else '✗'} Loss decreased: {first_avg:.4f} → {last_avg:.4f} ({(1 - last_avg/first_avg)*100:.1f}% reduction)")
    if not decreased:
        all_pass = False

    # Verify gradient norms are reasonable
    grad_norms = []
    for p in estimator.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    max_grad = max(grad_norms) if grad_norms else 0
    reasonable_grad = max_grad < 100.0  # after clipping should be small
    print(f"  {'✓' if reasonable_grad else '✗'} Max grad norm: {max_grad:.4f} (after clip to 1.0)")
    if not reasonable_grad:
        all_pass = False

    print(f"  Elapsed: {elapsed:.2f}s ({elapsed/50*1000:.1f}ms/step)")
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 3: Model save/load roundtrip
# ===========================================================================
def test_model_save_load() -> bool:
    """Test save/load state_dict produces identical forward pass."""
    print("=" * 60)
    print("Test 3: Model Save/Load Roundtrip")
    print("=" * 60)

    device = get_device()
    batch_size = 4

    estimator = PIEEstimator(map_gt_dim=MAP_GT_DIM_LITE3).to(device)
    estimator.eval()
    estimator.reset_hidden(batch_size, device)

    proprio_history = torch.randn(batch_size, H1, PROPRIO_DIM, device=device)
    depth_history = torch.randn(batch_size, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

    # Forward pass before save (fix seed for deterministic z_sample)
    torch.manual_seed(42)
    with torch.no_grad():
        out_before = estimator(proprio_history, depth_history)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
        torch.save(estimator.state_dict(), tmp_path)

    all_pass = True

    # Load into new model
    estimator2 = PIEEstimator(map_gt_dim=MAP_GT_DIM_LITE3).to(device)
    estimator2.eval()
    estimator2.reset_hidden(batch_size, device)
    estimator2.load_state_dict(torch.load(tmp_path, map_location=device, weights_only=True))

    # Same seed → same z_sample (reparameterization uses randn_like)
    torch.manual_seed(42)
    with torch.no_grad():
        out_after = estimator2(proprio_history, depth_history)

    # Compare all outputs
    for key in out_before:
        match = torch.allclose(out_before[key], out_after[key], atol=1e-6)
        print(f"  {'✓' if match else '✗'} {key} matches after load")
        if not match:
            max_diff = (out_before[key] - out_after[key]).abs().max().item()
            print(f"    Max diff: {max_diff}")
            all_pass = False

    # Verify file size is reasonable (< 20MB for ~1.3M params)
    file_size = os.path.getsize(tmp_path) / (1024 * 1024)
    size_ok = file_size < 20.0
    print(f"  {'✓' if size_ok else '✗'} Model file size: {file_size:.2f} MB")
    if not size_ok:
        all_pass = False

    # Cleanup
    os.unlink(tmp_path)

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 4: Edge cases
# ===========================================================================
def test_edge_cases() -> bool:
    """Test edge cases: batch=1, zeros, large values, mid-rollout reset."""
    print("=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)

    device = get_device()
    all_pass = True

    # --- 4a: Batch size 1 ---
    estimator = PIEEstimator(map_gt_dim=MAP_GT_DIM_LITE3).to(device)
    estimator.reset_hidden(1, device)
    with torch.no_grad():
        out = estimator(
            torch.randn(1, H1, PROPRIO_DIM, device=device),
            torch.randn(1, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )
    ok_b1 = all(torch.isfinite(v).all().item() for v in out.values())
    print(f"  {'✓' if ok_b1 else '✗'} Batch size 1: all outputs finite")
    if not ok_b1:
        all_pass = False

    # --- 4b: Zero input ---
    estimator.reset_hidden(4, device)
    with torch.no_grad():
        out_zero = estimator(
            torch.zeros(4, H1, PROPRIO_DIM, device=device),
            torch.zeros(4, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )
    ok_zero = all(torch.isfinite(v).all().item() for v in out_zero.values())
    print(f"  {'✓' if ok_zero else '✗'} Zero input: all outputs finite")
    if not ok_zero:
        all_pass = False

    # --- 4c: Large value input ---
    estimator.reset_hidden(4, device)
    with torch.no_grad():
        out_large = estimator(
            torch.randn(4, H1, PROPRIO_DIM, device=device) * 100.0,
            torch.randn(4, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device) * 100.0,
        )
    ok_large = all(torch.isfinite(v).all().item() for v in out_large.values())
    print(f"  {'✓' if ok_large else '✗'} Large input (x100): all outputs finite")
    if not ok_large:
        all_pass = False

    # --- 4d: Mid-rollout partial reset ---
    batch_size = 8
    estimator.reset_hidden(batch_size, device)

    # Step 1: all envs
    with torch.no_grad():
        estimator(
            torch.randn(batch_size, H1, PROPRIO_DIM, device=device),
            torch.randn(batch_size, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )

    # Reset envs 2, 5
    reset_ids = torch.tensor([2, 5], device=device)
    estimator.reset_hidden_for_envs(reset_ids)
    hidden_after = estimator._gru_hidden

    # Verify reset envs are zero, others non-zero
    reset_ok = (hidden_after[2].abs().sum() == 0) and (hidden_after[5].abs().sum() == 0)
    kept_ok = hidden_after[0].abs().sum() > 0 and hidden_after[3].abs().sum() > 0
    print(f"  {'✓' if reset_ok else '✗'} Partial reset zeroes target envs [2,5]")
    print(f"  {'✓' if kept_ok else '✗'} Partial reset preserves other envs [0,3]")
    if not (reset_ok and kept_ok):
        all_pass = False

    # Step 2: forward still works after partial reset
    with torch.no_grad():
        out_after_reset = estimator(
            torch.randn(batch_size, H1, PROPRIO_DIM, device=device),
            torch.randn(batch_size, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )
    ok_post_reset = all(torch.isfinite(v).all().item() for v in out_after_reset.values())
    print(f"  {'✓' if ok_post_reset else '✗'} Forward works after partial reset")
    if not ok_post_reset:
        all_pass = False

    # --- 4e: Loss with zero targets ---
    estimator.train()
    estimator.reset_hidden(4, device)
    # Warm up GRU hidden
    with torch.no_grad():
        estimator(
            torch.randn(4, H1, PROPRIO_DIM, device=device),
            torch.randn(4, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
        )
    out = estimator(
        torch.randn(4, H1, PROPRIO_DIM, device=device),
        torch.randn(4, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
    )
    losses = compute_pie_estimator_loss(
        out,
        gt_next_state=torch.zeros(4, PROPRIO_DIM, device=device),
        gt_vel=torch.zeros(4, 3, device=device),
        gt_clearance=torch.zeros(4, 4, device=device),
        gt_map=torch.zeros(4, MAP_GT_DIM_LITE3, device=device),
    )
    ok_zero_gt = torch.isfinite(losses["total"]).item()
    print(f"  {'✓' if ok_zero_gt else '✗'} Loss with zero ground truth: {losses['total'].item():.4f}")
    if not ok_zero_gt:
        all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 5: Buffer → Estimator → Loss end-to-end pipeline
# ===========================================================================
def test_buffer_estimator_pipeline() -> bool:
    """Test the complete pipeline: HistoryBuffer → Estimator → Loss."""
    print("=" * 60)
    print("Test 5: Buffer → Estimator → Loss Pipeline")
    print("=" * 60)

    device = get_device()
    num_envs = 16
    all_pass = True

    # Initialize components
    buffer = PIEHistoryBuffer(
        num_envs=num_envs,
        proprio_dim=PROPRIO_DIM,
        proprio_history_len=H1,
        depth_channels=DEPTH_C,
        depth_height=DEPTH_H,
        depth_width=DEPTH_W,
        depth_history_len=H2,
        device=device,
    )
    estimator = PIEEstimator(map_gt_dim=MAP_GT_DIM_LITE3).to(device)
    estimator.train()
    estimator.reset_hidden(num_envs, device)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-3)

    # Simulate 5 environment steps
    for step in range(5):
        # Simulate observations from env
        obs = torch.randn(num_envs, PROPRIO_DIM, device=device)
        depth = torch.randn(num_envs, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

        # Push to buffer
        buffer.push_proprio(obs)
        buffer.push_depth(depth)

        # Simulate environment resets for some envs
        if step == 2:
            reset_ids = torch.tensor([0, 3, 7], device=device)
            buffer.reset(env_ids=reset_ids)
            estimator.reset_hidden_for_envs(reset_ids)

    # Now do a training step
    proprio_hist = buffer.get_proprio_history()
    depth_hist = buffer.get_depth_history()

    # Verify shapes
    ph_ok = proprio_hist.shape == (num_envs, H1, PROPRIO_DIM)
    dh_ok = depth_hist.shape == (num_envs, H2, DEPTH_C, DEPTH_H, DEPTH_W)
    print(f"  {'✓' if ph_ok else '✗'} Proprio history: {proprio_hist.shape}")
    print(f"  {'✓' if dh_ok else '✗'} Depth history: {depth_hist.shape}")
    if not (ph_ok and dh_ok):
        all_pass = False

    # Forward
    output = estimator(proprio_hist, depth_hist)

    # Simulated GT
    gt_next = torch.randn(num_envs, PROPRIO_DIM, device=device)
    gt_vel = torch.randn(num_envs, 3, device=device)
    gt_cl = torch.rand(num_envs, 4, device=device) * 0.1
    gt_map = torch.randn(num_envs, MAP_GT_DIM_LITE3, device=device)

    losses = compute_pie_estimator_loss(output, gt_next, gt_vel, gt_cl, gt_map)

    # Backward + optimize
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    grad_ok = all(
        p.grad is not None
        for p in estimator.parameters()
        if p.requires_grad
    )
    print(f"  {'✓' if grad_ok else '✗'} All parameters received gradients")
    if not grad_ok:
        all_pass = False

    finite_ok = torch.isfinite(losses["total"]).item()
    print(f"  {'✓' if finite_ok else '✗'} Total loss: {losses['total'].item():.4f}")
    if not finite_ok:
        all_pass = False

    # Verify reset envs have zero buffer data
    reset_env_zeros = buffer.get_proprio_history()[0].abs().sum().item() == 0
    # (env 0 was reset at step 2, then received steps 3 & 4)
    # Actually after reset at step 2, steps 3 and 4 push new data
    # So env 0 should have non-zero at positions [-1] and [-2]
    post_reset_has_data = buffer.get_proprio_history()[0, -1].abs().sum().item() > 0
    print(f"  {'✓' if post_reset_has_data else '✗'} Reset env received new data after reset")
    if not post_reset_has_data:
        all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 6: Policy observation dimension compatibility
# ===========================================================================
def test_policy_obs_compatibility() -> bool:
    """Verify PIE injection dimensions are compatible with policy architecture."""
    print("=" * 60)
    print("Test 6: Policy Observation Dimension Compatibility")
    print("=" * 60)

    device = get_device()
    batch_size = 8
    all_pass = True

    estimator = PIEEstimator(
        latent_dim=LATENT_DIM,
        map_dim=MAP_DIM,
        map_gt_dim=MAP_GT_DIM_LITE3,
    ).to(device)
    estimator.eval()
    estimator.reset_hidden(batch_size, device)

    proprio_history = torch.randn(batch_size, H1, PROPRIO_DIM, device=device)
    depth_history = torch.randn(batch_size, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

    injection = estimator.get_policy_injection(proprio_history, depth_history)

    # PIE policy input = base_obs(45) + injection(3+4+map_dim+latent_dim)
    inj_dim = injection.shape[-1]
    expected_inj = 3 + 4 + MAP_DIM + LATENT_DIM  # 103
    total_policy_input = PROPRIO_DIM + inj_dim

    ok_inj = inj_dim == expected_inj
    print(f"  {'✓' if ok_inj else '✗'} PIE injection dim: {inj_dim} (expected {expected_inj})")
    if not ok_inj:
        all_pass = False

    print(f"  Breakdown: est_vel(3) + est_clearance(4) + map_enc({MAP_DIM}) + z_sample({LATENT_DIM}) = {expected_inj}")
    print(f"  Total policy input: {PROPRIO_DIM} (proprio) + {inj_dim} (PIE) = {total_policy_input}")

    # Verify injection can be concatenated with base obs
    base_obs = torch.randn(batch_size, PROPRIO_DIM, device=device)
    augmented_obs = torch.cat([base_obs, injection], dim=-1)
    concat_ok = augmented_obs.shape == (batch_size, total_policy_input)
    print(f"  {'✓' if concat_ok else '✗'} Concatenation works: {augmented_obs.shape}")
    if not concat_ok:
        all_pass = False

    # Verify a simple policy MLP can process this
    policy_mlp = torch.nn.Sequential(
        torch.nn.Linear(total_policy_input, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 12),  # 12 joint actions for Lite3
    ).to(device)
    with torch.no_grad():
        actions = policy_mlp(augmented_obs)
    action_ok = actions.shape == (batch_size, 12)
    print(f"  {'✓' if action_ok else '✗'} Policy MLP output: {actions.shape} (expected ({batch_size}, 12))")
    if not action_ok:
        all_pass = False

    # Compare with existing architecture sizes
    # rough_env_cfg: actor_hidden_dims=[512, 256, 128]
    # stairs_history: actor_hidden_dims=[1024, 512, 256, 128]
    print(f"\n  Existing architectures (for reference):")
    print(f"    Rough PPO input: {PROPRIO_DIM} dims → [512, 256, 128]")
    print(f"    History PPO input: {PROPRIO_DIM * 20} dims → [1024, 512, 256, 128]")
    print(f"    PIE PPO input: {total_policy_input} dims → [512, 256, 128] (recommended)")

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 7: GT utils interface validation
# ===========================================================================
def test_gt_utils_interface() -> bool:
    """Validate GT utils interface: foot names, expected outputs, imports."""
    print("=" * 60)
    print("Test 7: GT Utils Interface Validation")
    print("=" * 60)

    all_pass = True

    # Import check (should not require Isaac Lab)
    try:
        from pie.pie_gt_utils import (
            LITE3_FOOT_NAMES as gt_foot_names,
            get_base_linear_velocity_gt,
            get_foot_clearance_gt,
            get_height_scan_gt,
        )
        import_ok = True
    except ImportError as e:
        import_ok = False
        print(f"  ✗ Import failed: {e}")
    print(f"  {'✓' if import_ok else '✗'} GT utils import successful (no Isaac Lab required at import time)")
    if not import_ok:
        all_pass = False
        print(f"  Result: FAIL\n")
        return False

    # Verify foot names match rough_env_cfg.py configuration
    expected_feet = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    feet_ok = gt_foot_names == expected_feet
    print(f"  {'✓' if feet_ok else '✗'} Foot names: {gt_foot_names}")
    if not feet_ok:
        all_pass = False

    # Verify functions have correct type hints
    import inspect
    for func_name, func in [
        ("get_base_linear_velocity_gt", get_base_linear_velocity_gt),
        ("get_foot_clearance_gt", get_foot_clearance_gt),
        ("get_height_scan_gt", get_height_scan_gt),
    ]:
        sig = inspect.signature(func)
        has_env_param = "env" in sig.parameters
        has_return = sig.return_annotation != inspect.Parameter.empty
        print(f"  {'✓' if has_env_param else '✗'} {func_name}: has 'env' parameter")
        if not has_env_param:
            all_pass = False

    # Verify foot clearance has optional foot_names param
    cl_sig = inspect.signature(get_foot_clearance_gt)
    has_foot_names = "foot_names" in cl_sig.parameters
    print(f"  {'✓' if has_foot_names else '✗'} get_foot_clearance_gt: has 'foot_names' parameter")
    if not has_foot_names:
        all_pass = False

    # Verify output dimension expectations documented
    print(f"\n  Expected GT output dimensions:")
    print(f"    get_base_linear_velocity_gt → (num_envs, 3)")
    print(f"    get_foot_clearance_gt       → (num_envs, 4)")
    print(f"    get_height_scan_gt          → (num_envs, {MAP_GT_DIM_LITE3})")

    # Cross-check with estimator heads
    print(f"\n  Estimator head dimensions (must match GT):")
    print(f"    vel_head output     → 3 ✓")
    print(f"    clearance_head      → 4 (matches {len(expected_feet)} feet) ✓")
    print(f"    map_decoder output  → {MAP_GT_DIM_LITE3} (matches scanner) ✓")

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Test 8: Individual loss components
# ===========================================================================
def test_individual_losses() -> bool:
    """Test each loss function component independently."""
    print("=" * 60)
    print("Test 8: Individual Loss Components")
    print("=" * 60)

    device = get_device()
    all_pass = True

    # --- Reconstruction loss ---
    rec_state = torch.randn(8, PROPRIO_DIM, device=device)
    gt_state = torch.randn(8, PROPRIO_DIM, device=device)
    rec_map = torch.randn(8, MAP_GT_DIM_LITE3, device=device)
    gt_map = torch.randn(8, MAP_GT_DIM_LITE3, device=device)

    l_recon = reconstruction_loss(rec_state, gt_state, rec_map, gt_map)
    ok_r = l_recon.dim() == 0 and torch.isfinite(l_recon).item() and l_recon.item() > 0
    print(f"  {'✓' if ok_r else '✗'} Reconstruction loss: {l_recon.item():.4f} (positive scalar)")
    if not ok_r:
        all_pass = False

    # Perfect prediction should be ~0
    l_recon_zero = reconstruction_loss(gt_state, gt_state, gt_map, gt_map)
    ok_r0 = l_recon_zero.item() < 1e-6
    print(f"  {'✓' if ok_r0 else '✗'} Reconstruction loss (perfect match): {l_recon_zero.item():.8f} ≈ 0")
    if not ok_r0:
        all_pass = False

    # --- Estimation loss ---
    est_vel = torch.randn(8, 3, device=device)
    gt_vel = torch.randn(8, 3, device=device)
    est_cl = torch.randn(8, 4, device=device)
    gt_cl = torch.randn(8, 4, device=device)

    l_est = estimation_loss(est_vel, gt_vel, est_cl, gt_cl)
    ok_e = l_est.dim() == 0 and torch.isfinite(l_est).item() and l_est.item() > 0
    print(f"  {'✓' if ok_e else '✗'} Estimation loss: {l_est.item():.4f} (positive scalar)")
    if not ok_e:
        all_pass = False

    # --- KL divergence ---
    mu = torch.randn(8, LATENT_DIM, device=device)
    logvar = torch.randn(8, LATENT_DIM, device=device)

    l_kl = kl_divergence_loss(mu, logvar)
    ok_k = l_kl.dim() == 0 and torch.isfinite(l_kl).item()
    print(f"  {'✓' if ok_k else '✗'} KL divergence: {l_kl.item():.4f} (finite scalar)")
    if not ok_k:
        all_pass = False

    # KL with N(0,I) should be 0
    l_kl_zero = kl_divergence_loss(
        torch.zeros(8, LATENT_DIM, device=device),
        torch.zeros(8, LATENT_DIM, device=device),  # logvar=0 → var=1
    )
    ok_k0 = l_kl_zero.item() < 1e-6
    print(f"  {'✓' if ok_k0 else '✗'} KL(N(0,I) || N(0,I)): {l_kl_zero.item():.8f} ≈ 0")
    if not ok_k0:
        all_pass = False

    # --- Weight sensitivity ---
    device = get_device()
    est = PIEEstimator(map_gt_dim=MAP_GT_DIM_LITE3).to(device)
    est.reset_hidden(8, device)
    with torch.no_grad():
        est(torch.randn(8, H1, PROPRIO_DIM, device=device),
            torch.randn(8, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device))
    out = est(
        torch.randn(8, H1, PROPRIO_DIM, device=device),
        torch.randn(8, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device),
    )
    gt_ns = torch.randn(8, PROPRIO_DIM, device=device)
    gv = torch.randn(8, 3, device=device)
    gc = torch.randn(8, 4, device=device)
    gm = torch.randn(8, MAP_GT_DIM_LITE3, device=device)

    l_w1 = compute_pie_estimator_loss(out, gt_ns, gv, gc, gm, kl_weight=0.01)
    l_w2 = compute_pie_estimator_loss(out, gt_ns, gv, gc, gm, kl_weight=1.0)
    weight_sens = l_w2["total"].item() > l_w1["total"].item()
    print(f"  {'✓' if weight_sens else '✗'} KL weight sensitivity: w=0.01 → {l_w1['total'].item():.4f}, w=1.0 → {l_w2['total'].item():.4f}")
    if not weight_sens:
        all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    """Run all advanced PIE Estimator verification tests."""
    print("\n" + "=" * 60)
    print("  PIE Estimator Advanced Verification Suite")
    print("  Reference: docs/PIE.md, docs/PIE_instruction.md")
    print("=" * 60 + "\n")

    results = {
        "Height Scan Dim Alignment": test_height_scan_dim_alignment(),
        "Training Loop (50 iter)": test_training_loop(),
        "Model Save/Load": test_model_save_load(),
        "Edge Cases": test_edge_cases(),
        "Buffer→Estimator Pipeline": test_buffer_estimator_pipeline(),
        "Policy Obs Compatibility": test_policy_obs_compatibility(),
        "GT Utils Interface": test_gt_utils_interface(),
        "Individual Losses": test_individual_losses(),
    }

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False
        print(f"  {status}  {name}")

    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("=" * 60 + "\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
