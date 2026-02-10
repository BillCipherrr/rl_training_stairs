# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def foot_clearance_gt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Ground truth foot clearance for PIE estimator training.

    Computes per-foot clearance: h^f_i = z_foot_i - z_terrain_at_foot_i.
    Uses the height_scanner_base ray caster to query terrain height at each
    foot's (x, y) position via the terrain mesh.

    Returns:
        Tensor of shape ``(num_envs, 4)`` — clearance for [FL, FR, HL, HR].
    """
    from isaaclab.sensors import RayCaster

    asset = env.scene[asset_cfg.name]

    # body_ids is resolved automatically by the framework from body_names
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, 4, 3)
    foot_z = foot_pos_w[:, :, 2]  # (num_envs, 4)

    # Use height_scanner_base sensor to get terrain height under the robot
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits_z = sensor.data.ray_hits_w[..., 2]  # (num_envs, num_rays)

    # Use mean terrain height as an approximation for terrain under each foot
    if torch.isnan(ray_hits_z).any() or torch.isinf(ray_hits_z).any():
        terrain_z = torch.zeros_like(foot_z)
    else:
        terrain_z_mean = torch.mean(ray_hits_z, dim=1, keepdim=True)  # (num_envs, 1)
        terrain_z = terrain_z_mean.expand_as(foot_z)  # (num_envs, 4)

    clearance = foot_z - terrain_z  # (num_envs, 4)
    return clearance


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor
