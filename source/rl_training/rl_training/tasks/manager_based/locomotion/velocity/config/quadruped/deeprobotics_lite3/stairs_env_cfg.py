# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3StairsEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Environment configuration for Lite3 stair-climbing on randomized stairs."""

    def __post_init__(self):
        super().__post_init__()

        # limit environment spawn to stair tiles and disable curriculum to keep consistent difficulty
        self.scene.terrain.max_init_terrain_level = None
        terrain_generator = deepcopy(self.scene.terrain.terrain_generator)
        terrain_generator.curriculum = False
        terrain_generator.num_rows = 6
        terrain_generator.num_cols = 12

        # keep only stair-type sub-terrains and widen their randomization ranges
        stairs_keys = ("pyramid_stairs", "pyramid_stairs_inv")
        new_sub_terrains: dict[str, object] = {}
        for key in stairs_keys:
            if key not in terrain_generator.sub_terrains:
                continue
            stair_cfg = deepcopy(terrain_generator.sub_terrains[key])
            stair_cfg.proportion = 0.5
            stair_cfg.step_height_range = (0.08, 0.28)
            stair_cfg.step_width = 0.28
            stair_cfg.platform_width = 2.5
            new_sub_terrains[key] = stair_cfg
        if new_sub_terrains:
            terrain_generator.sub_terrains = new_sub_terrains
        self.scene.terrain.terrain_generator = terrain_generator

        # drop any zero-weight rewards inherited from the rough config
        self.disable_zero_weight_rewards()