# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Ground truth extraction utilities for PIE Estimator training.

Provides helper functions to extract privileged information required
by the PIE loss function (docs/PIE.md Section 3).

These functions require access to the Isaac Lab simulation environment
and are only called during **training** (not deployment).

Ground Truth Signals:
    - Base linear velocity (v_t): from robot.data.root_lin_vel_b
    - Foot clearance (h^f_t): z_foot - z_terrain, via ray-cast
    - Height scan (m_t): from height_scanner sensor

TODO Items (docs/PIE_instruction.md Section 3):
    - get_foot_clearance_gt: Verify terrain.sample_height() API availability.
      If unavailable, implement RayCaster-based per-foot terrain height query.
    - get_height_scan_gt: Verify height_scanner data attribute naming.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Lite3 foot body names (from rough_env_cfg.py)
LITE3_FOOT_NAMES = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]


def get_base_linear_velocity_gt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get ground truth base linear velocity in the base frame.

    PIE Section 3 — required for Estimation Loss MSE(v_hat, v).

    Args:
        env: The Isaac Lab RL environment.

    Returns:
        Tensor of shape ``(num_envs, 3)`` — base linear velocity [vx, vy, vz].
    """
    robot = env.scene["robot"]
    return robot.data.root_lin_vel_b  # (num_envs, 3)


def get_foot_clearance_gt(
    env: ManagerBasedRLEnv,
    foot_names: list[str] | None = None,
) -> torch.Tensor:
    """Get ground truth foot clearance for all four feet.

    Foot clearance is defined as:
        h^f_i = z_foot_i - z_terrain_at_foot_i
    (PIE Section 3.2)

    The function attempts to query the terrain height at each foot's (x, y)
    position. If the terrain object does not support ``sample_height()``,
    it falls back to assuming flat ground at z=0.

    TODO: This implementation needs verification (docs/PIE_instruction.md):
        1. Verify ``terrain.sample_height()`` is available in your Isaac Lab version.
        2. If not available, implement per-foot RayCaster queries instead.
        3. Validate that the terrain height aligns with actual mesh geometry.

    Args:
        env: The Isaac Lab RL environment.
        foot_names: List of foot body names. Defaults to Lite3 feet.

    Returns:
        Tensor of shape ``(num_envs, 4)`` — clearance for [FL, FR, HL, HR].
    """
    if foot_names is None:
        foot_names = LITE3_FOOT_NAMES

    robot = env.scene["robot"]

    # Resolve foot body indices
    foot_indices = []
    for name in foot_names:
        idx = robot.find_bodies(name)
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        foot_indices.append(idx)

    # foot_pos_w shape: (num_envs, num_bodies, 3)
    foot_pos_w = robot.data.body_pos_w[:, foot_indices, :]  # (num_envs, 4, 3)
    foot_z = foot_pos_w[:, :, 2]  # (num_envs, 4)

    # Query terrain height at foot (x, y) positions
    terrain = env.scene.terrain
    if hasattr(terrain, "sample_height"):
        foot_xy = foot_pos_w[:, :, :2]  # (num_envs, 4, 2)
        batch_size, num_feet = foot_xy.shape[:2]
        foot_xy_flat = foot_xy.reshape(-1, 2)  # (num_envs*4, 2)
        terrain_z_flat = terrain.sample_height(foot_xy_flat)  # (num_envs*4,)
        terrain_z = terrain_z_flat.reshape(batch_size, num_feet)  # (num_envs, 4)
    else:
        # Fallback: assume flat ground at z=0
        # TODO: Implement RayCaster-based terrain height query for accurate
        # per-foot clearance on non-flat terrain.
        terrain_z = torch.zeros_like(foot_z)
        warnings.warn(
            "Terrain height query not available. "
            "Foot clearance GT will assume flat ground (z=0). "
            "Please enable height_scanner or implement ray-cast query. "
            "See docs/PIE.md Section 3.2 for details.",
            stacklevel=2,
        )

    clearance = foot_z - terrain_z  # (num_envs, 4)
    return clearance


def get_height_scan_gt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get ground truth terrain height scan from the height scanner sensor.

    PIE Section 3.1 — required for Reconstruction Loss MSE(m_hat, m).

    The height scanner must be enabled in the scene configuration but
    **excluded** from the policy observation group. It is only used by:
    - Critic input (privileged information)
    - Estimator loss computation (reconstruction target)

    Implementation note:
        The scanner is configured in ``MySceneCfg.height_scanner`` with prim
        path offset. In ``rough_env_cfg.py``, policy observation disables it
        via ``self.observations.policy.height_scan = None``.

    Args:
        env: The Isaac Lab RL environment.

    Returns:
        Tensor of shape ``(num_envs, num_scan_points)`` — height scan data.

    Raises:
        RuntimeError: If ``height_scanner`` is not enabled in the scene.
    """
    # Try accessing the height scanner sensor
    try:
        scanner = env.scene["height_scanner"]
    except KeyError:
        raise RuntimeError(
            "Height scanner is not enabled in the scene. "
            "Enable it in StairsEnvCfg (see docs/PIE.md Section 3.1): "
            "self.scene.height_scanner must be configured, but excluded "
            "from observations.policy (set to None in policy group)."
        )

    # The ray caster returns hit positions; we extract z-heights relative
    # to the sensor origin to get a height map.
    # ray_hits_w: (num_envs, num_rays, 3) — world-frame hit positions
    if hasattr(scanner.data, "ray_hits_w"):
        # Relative height: sensor_z - hit_z (distance below sensor)
        # The scanner offset is (0, 0, 20.0) so hits represent terrain surface.
        hit_z = scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays)
        sensor_z = scanner.data.pos_w[:, 2:3]  # (num_envs, 1)
        height_scan = sensor_z - hit_z  # positive = terrain below sensor
        return height_scan
    else:
        raise RuntimeError(
            "Height scanner data does not have 'ray_hits_w' attribute. "
            "Verify the RayCaster sensor configuration. "
            "See velocity_env_cfg.py MySceneCfg.height_scanner."
        )
