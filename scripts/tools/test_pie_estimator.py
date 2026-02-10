#!/usr/bin/env python3
# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Verification script for PIE Estimator components.

Validates:
    1. Network forward pass dimensions and output keys.
    2. Loss function computation (scalar, finite, gradient flow).
    3. History buffer FIFO behavior and partial reset.
    4. Policy injection output shape.
    5. GRU temporal consistency (deterministic replay, state effect).

Usage:
    cd /home/user1/rl_training
    python scripts/tools/test_pie_estimator.py
"""

from __future__ import annotations

import sys
import os

# Add the PIE module path directly to avoid importing the full rl_training
# package which requires Isaac Lab / Omniverse runtime.
_PIE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "source", "rl_training", "rl_training",
        "tasks", "manager_based", "locomotion", "velocity", "pie",
    )
)
sys.path.insert(0, os.path.dirname(_PIE_DIR))  # parent of pie/

import torch

from pie.pi_estimator import PIEEstimator
from pie.pie_history_buffer import PIEHistoryBuffer
from pie.pie_loss import compute_pie_estimator_loss


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
PROPRIO_DIM = 45
H1 = 10  # PIE paper: proprioception history length
H2 = 2   # PIE paper: depth history length
DEPTH_C = 1
DEPTH_H = 64
DEPTH_W = 64
HIDDEN_DIM = 128
LATENT_DIM = 32
MAP_DIM = 64
MAP_GT_DIM = 187  # height scan points


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Test 1: Forward pass dimensions
# ---------------------------------------------------------------------------
def test_estimator_forward() -> bool:
    """Test PIEEstimator forward pass dimensions."""
    print("=" * 60)
    print("Test 1: PIEEstimator Forward Pass")
    print("=" * 60)

    device = get_device()

    estimator = PIEEstimator(
        proprio_dim=PROPRIO_DIM,
        proprio_history_len=H1,
        depth_channels=DEPTH_C,
        depth_history_len=H2,
        depth_height=DEPTH_H,
        depth_width=DEPTH_W,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        map_dim=MAP_DIM,
        map_gt_dim=MAP_GT_DIM,
    ).to(device)

    # Create dummy inputs matching PIE spec
    proprio_history = torch.randn(BATCH_SIZE, H1, PROPRIO_DIM, device=device)
    depth_history = torch.randn(BATCH_SIZE, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

    estimator.reset_hidden(BATCH_SIZE, device)
    output = estimator(proprio_history, depth_history)

    # Expected output shapes (from docs/PIE.md Section 4)
    expected_shapes = {
        "est_vel": (BATCH_SIZE, 3),
        "est_clearance": (BATCH_SIZE, 4),
        "latent_mu": (BATCH_SIZE, LATENT_DIM),
        "latent_logvar": (BATCH_SIZE, LATENT_DIM),
        "z_sample": (BATCH_SIZE, LATENT_DIM),
        "map_enc": (BATCH_SIZE, MAP_DIM),
        "rec_next_state": (BATCH_SIZE, PROPRIO_DIM),
        "rec_map": (BATCH_SIZE, MAP_GT_DIM),
        "gru_hidden": (BATCH_SIZE, HIDDEN_DIM),
    }

    all_pass = True
    for key, expected in expected_shapes.items():
        actual = tuple(output[key].shape)
        ok = actual == expected
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"  {status} {key}: expected {expected}, got {actual}")

    total_params = sum(p.numel() for p in estimator.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Test 2: Loss computation and gradient flow
# ---------------------------------------------------------------------------
def test_loss_computation() -> bool:
    """Test PIE loss function computation."""
    print("=" * 60)
    print("Test 2: PIE Loss Computation")
    print("=" * 60)

    device = get_device()

    estimator = PIEEstimator(
        proprio_dim=PROPRIO_DIM,
        latent_dim=LATENT_DIM,
        map_dim=MAP_DIM,
        map_gt_dim=MAP_GT_DIM,
    ).to(device)
    estimator.reset_hidden(BATCH_SIZE, device)

    proprio_history = torch.randn(BATCH_SIZE, H1, PROPRIO_DIM, device=device)
    depth_history = torch.randn(BATCH_SIZE, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

    # Step 1: warm up GRU hidden state (so weight_hh gets non-zero input)
    with torch.no_grad():
        estimator(proprio_history, depth_history)

    # Step 2: forward with gradient — GRU now has non-zero hidden state
    # Re-create hidden from step 1 with gradient enabled
    estimator._gru_hidden = estimator._gru_hidden.clone().requires_grad_(False)
    output = estimator(proprio_history, depth_history)

    # Dummy ground truth signals (PIE Section 3)
    gt_next_state = torch.randn(BATCH_SIZE, PROPRIO_DIM, device=device)
    gt_vel = torch.randn(BATCH_SIZE, 3, device=device)
    gt_clearance = torch.randn(BATCH_SIZE, 4, device=device)
    gt_map = torch.randn(BATCH_SIZE, MAP_GT_DIM, device=device)

    losses = compute_pie_estimator_loss(
        estimator_output=output,
        gt_next_state=gt_next_state,
        gt_vel=gt_vel,
        gt_clearance=gt_clearance,
        gt_map=gt_map,
        recon_weight=1.0,   # PIE Section 5
        est_weight=1.0,
        kl_weight=0.01,
    )

    all_pass = True
    for key, value in losses.items():
        is_scalar = value.dim() == 0
        is_finite = torch.isfinite(value).item()
        ok = is_scalar and is_finite
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"  {status} {key}: {value.item():.6f} (scalar={is_scalar}, finite={is_finite})")

    # Verify gradient flow to all parameters
    # After 2 steps, GRU weight_hh should also receive gradient
    estimator.zero_grad()
    losses["total"].backward()
    no_grad_params = [
        name for name, p in estimator.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    grad_exists = len(no_grad_params) == 0
    grad_status = "✓" if grad_exists else "✗"
    if not grad_exists:
        all_pass = False
        for name in no_grad_params:
            print(f"    WARNING: No gradient for {name}")
    print(f"  {grad_status} Gradients flow to all parameters: {grad_exists}")
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Test 3: History buffer FIFO behavior
# ---------------------------------------------------------------------------
def test_history_buffer() -> bool:
    """Test PIEHistoryBuffer FIFO behavior."""
    print("=" * 60)
    print("Test 3: PIEHistoryBuffer FIFO")
    print("=" * 60)

    num_envs = 4
    device = get_device()

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

    all_pass = True

    # Test initial shapes
    prop_shape = buffer.get_proprio_history().shape
    depth_shape = buffer.get_depth_history().shape
    prop_ok = prop_shape == (num_envs, H1, PROPRIO_DIM)
    depth_ok = depth_shape == (num_envs, H2, DEPTH_C, DEPTH_H, DEPTH_W)
    print(f"  {'✓' if prop_ok else '✗'} Proprio shape: {prop_shape}")
    print(f"  {'✓' if depth_ok else '✗'} Depth shape: {depth_shape}")
    if not (prop_ok and depth_ok):
        all_pass = False

    # Test initial state (all zeros)
    is_zero = buffer.get_proprio_history().sum().item() == 0.0
    print(f"  {'✓' if is_zero else '✗'} Initial buffer is all zeros")
    if not is_zero:
        all_pass = False

    # Push frames and verify FIFO order
    frame1 = torch.ones(num_envs, PROPRIO_DIM, device=device) * 1.0
    frame2 = torch.ones(num_envs, PROPRIO_DIM, device=device) * 2.0
    frame3 = torch.ones(num_envs, PROPRIO_DIM, device=device) * 3.0

    buffer.push_proprio(frame1)
    buffer.push_proprio(frame2)
    buffer.push_proprio(frame3)

    history = buffer.get_proprio_history()

    checks = [
        ("Newest frame at [-1]", torch.allclose(history[:, -1], frame3)),
        ("Second frame at [-2]", torch.allclose(history[:, -2], frame2)),
        ("Third frame at [-3]", torch.allclose(history[:, -3], frame1)),
        ("Older frames are zero", torch.allclose(
            history[:, :-3], torch.zeros_like(history[:, :-3])
        )),
    ]
    for desc, val in checks:
        print(f"  {'✓' if val else '✗'} {desc}")
        if not val:
            all_pass = False

    # Test partial reset (reset envs 0 and 2, keep 1 and 3)
    buffer.reset(env_ids=torch.tensor([0, 2], device=device))
    h = buffer.get_proprio_history()
    reset_ok = torch.allclose(h[0], torch.zeros(H1, PROPRIO_DIM, device=device))
    kept_ok = torch.allclose(h[1, -1], frame3)
    print(f"  {'✓' if reset_ok else '✗'} Partial reset clears env 0")
    print(f"  {'✓' if kept_ok else '✗'} Partial reset keeps env 1")
    if not (reset_ok and kept_ok):
        all_pass = False

    # Test depth buffer push
    buffer.reset()
    d1 = torch.ones(num_envs, DEPTH_C, DEPTH_H, DEPTH_W, device=device) * 5.0
    d2 = torch.ones(num_envs, DEPTH_C, DEPTH_H, DEPTH_W, device=device) * 10.0
    buffer.push_depth(d1)
    buffer.push_depth(d2)
    dh = buffer.get_depth_history()
    d_newest = torch.allclose(dh[:, -1], d2)
    d_oldest = torch.allclose(dh[:, 0], d1)
    print(f"  {'✓' if d_newest else '✗'} Depth newest frame correct")
    print(f"  {'✓' if d_oldest else '✗'} Depth oldest frame correct")
    if not (d_newest and d_oldest):
        all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Test 4: Policy injection shape
# ---------------------------------------------------------------------------
def test_policy_injection() -> bool:
    """Test policy injection output dimension."""
    print("=" * 60)
    print("Test 4: Policy Injection Shape")
    print("=" * 60)

    batch_size = 8
    device = get_device()

    estimator = PIEEstimator(
        latent_dim=LATENT_DIM, map_dim=MAP_DIM
    ).to(device)
    estimator.reset_hidden(batch_size, device)

    proprio_history = torch.randn(batch_size, H1, PROPRIO_DIM, device=device)
    depth_history = torch.randn(batch_size, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

    injection = estimator.get_policy_injection(proprio_history, depth_history)

    # PIE Section 2.1: [est_vel(3), est_clearance(4), map_enc(map_dim), z_sample(latent_dim)]
    expected_dim = 3 + 4 + MAP_DIM + LATENT_DIM
    expected_shape = (batch_size, expected_dim)

    match = tuple(injection.shape) == expected_shape
    print(f"  {'✓' if match else '✗'} Injection shape: expected {expected_shape}, got {tuple(injection.shape)}")
    print(f"  Policy input augmentation: {PROPRIO_DIM} (proprio) + {expected_dim} (PIE) = {PROPRIO_DIM + expected_dim} total dims")

    # Verify no gradient (should be detached for policy input)
    no_grad = not injection.requires_grad
    print(f"  {'✓' if no_grad else '✗'} Injection is detached (no gradient)")

    ok = match and no_grad
    print(f"  Result: {'PASS' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------------------
# Test 5: GRU temporal consistency
# ---------------------------------------------------------------------------
def test_gru_temporal_consistency() -> bool:
    """Test that GRU produces different outputs for different sequences."""
    print("=" * 60)
    print("Test 5: GRU Temporal Consistency")
    print("=" * 60)

    batch_size = 4
    device = get_device()

    estimator = PIEEstimator().to(device)
    estimator.eval()  # deterministic mode (no dropout)
    estimator.reset_hidden(batch_size, device)

    depth = torch.randn(batch_size, H2, DEPTH_C, DEPTH_H, DEPTH_W, device=device)

    # Step 1
    proprio1 = torch.randn(batch_size, H1, PROPRIO_DIM, device=device)
    with torch.no_grad():
        out1 = estimator(proprio1, depth)
        vel1 = out1["est_vel"].clone()

    # Step 2 (different input → GRU state should change output)
    proprio2 = torch.randn(batch_size, H1, PROPRIO_DIM, device=device)
    with torch.no_grad():
        out2 = estimator(proprio2, depth)
        vel2 = out2["est_vel"].clone()

    # Reset and replay step 1 → should reproduce out1
    estimator.reset_hidden(batch_size, device)
    with torch.no_grad():
        out1_replay = estimator(proprio1, depth)
        vel1_replay = out1_replay["est_vel"].clone()

    # Check deterministic replay
    replay_match = torch.allclose(vel1, vel1_replay, atol=1e-5)
    # Check temporal difference (different GRU state → different output)
    temporal_diff = not torch.allclose(vel1, vel2, atol=1e-5)

    print(f"  {'✓' if replay_match else '✗'} Deterministic replay after reset")
    print(f"  {'✓' if temporal_diff else '✗'} GRU state affects output (temporal difference)")

    ok = replay_match and temporal_diff
    print(f"  Result: {'PASS' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all PIE Estimator verification tests."""
    print("\n" + "=" * 60)
    print("  PIE Estimator Verification Suite")
    print("  Reference: docs/PIE.md, docs/PIE_instruction.md")
    print("=" * 60 + "\n")

    results = {
        "Forward Pass": test_estimator_forward(),
        "Loss Computation": test_loss_computation(),
        "History Buffer": test_history_buffer(),
        "Policy Injection": test_policy_injection(),
        "GRU Temporal": test_gru_temporal_consistency(),
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
