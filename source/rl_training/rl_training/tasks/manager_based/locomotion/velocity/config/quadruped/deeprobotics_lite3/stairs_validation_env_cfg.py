# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Validation environment configuration for DeepRobotics Lite3 stair climbing.

This module provides deterministic evaluation environments for verifying
whether a trained policy can successfully climb 20 cm stairs.

Key differences from the training environments:
- 100% pyramid stairs at fixed 0.20 m step height (no curriculum, no mixed terrain)
- Fixed forward velocity command (no random commands)
- No observation noise, no push events
- Extended episode length (20 s) to allow full stair traversal
- All curriculum disabled

Registered tasks:
- ``Stairs-Deeprobotics-Lite3-Validation-v0``      (standard policy)
- ``Stairs-Deeprobotics-Lite3-PIE-Validation-v0``  (PIE policy)

Usage::

    # Standard policy validation
    python scripts/reinforcement_learning/rsl_rl/play_validation.py \\
        --task Stairs-Deeprobotics-Lite3-Validation-v0 \\
        --num_envs 50 --num_eval_episodes 200

    # PIE policy validation (use play_pie.py runner)
    python scripts/reinforcement_learning/rsl_rl/play_pie.py \\
        --task Stairs-Deeprobotics-Lite3-PIE-Validation-v0 \\
        --num_envs 50
"""

from copy import deepcopy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .stairs_env_cfg import DeeproboticsLite3StairsEnvCfg
from .stairs_pie_env_cfg import DeeproboticsLite3StairsPIEEnvCfg


# ---------------------------------------------------------------------------
# Terrain: 100% inverted-pyramid stairs, fixed 0.20 m step height
#
# MeshInvertedPyramidStairsTerrainCfg creates a "bowl" shape:
#   - centre platform is the LOWEST point
#   - edges/outer ring are the HIGHEST point
#   - robots spawn on the outer ring (bottom of the staircase) and must
#     climb INWARD/UPWARD toward the centre platform
#
# DO NOT use MeshPyramidStairsTerrainCfg here — that terrain spawns robots
# at the TOP (centre platform) and the robot descends, giving 0 height gain.
#
# step_width=0.28 matches training config to avoid distribution shift.
# ---------------------------------------------------------------------------
VALIDATION_20CM_STAIRS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "stairs_20cm_climb": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.20, 0.20),  # Fixed at target difficulty
            step_width=0.28,                 # Must match training config
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class DeeproboticsLite3StairsValidationEnvCfg(DeeproboticsLite3StairsEnvCfg):
    """Validation environment for standard (non-PIE) stair-climbing policies.

    All randomisation sources are removed so that success/failure reflects
    only the policy capability, not environmental variance.

    Success criterion (measured by play_validation.py):
        max height gain during episode > 0.20 m
        (i.e. the robot climbed at least one full 20 cm step)
    """

    def __post_init__(self):
        super().__post_init__()

        # ------ Terrain ------
        self.scene.terrain.terrain_generator = deepcopy(VALIDATION_20CM_STAIRS_CFG)
        # Spawn robots uniformly across all terrain patches (ignore curriculum levels)
        self.scene.terrain.max_init_terrain_level = None

        # ------ Commands: fixed 0.5 m/s forward ------
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ------ Episode length ------
        # 20 s gives ~18 s of net walking time (substract reset frames)
        # At 0.5 m/s the robot covers ~9 m, enough to traverse the full pyramid
        self.episode_length_s = 20.0

        # ------ Curriculum: all off ------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        # ------ Events: no external perturbations ------
        self.events.push_robot = None
        self.events.randomize_apply_external_force_torque = None

        # ------ Observations: clean (no noise) ------
        self.observations.policy.enable_corruption = False

        # Must call here: DeeproboticsLite3StairsEnvCfg guards disable_zero_weight_rewards()
        # behind `if self.__class__.__name__ == "DeeproboticsLite3StairsEnvCfg"`, so
        # subclasses never inherit the call and must invoke it themselves.
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsLite3StairsPIEValidationEnvCfg(DeeproboticsLite3StairsPIEEnvCfg):
    """Validation environment for PIE stair-climbing policies.

    Identical evaluation conditions as the standard validation env, but
    preserves all PIE observation groups (pie_proprio, pie_gt_vel, etc.)
    required by PIEOnPolicyRunner.

    Use with play_pie.py::

        python scripts/reinforcement_learning/rsl_rl/play_pie.py \\
            --task Stairs-Deeprobotics-Lite3-PIE-Validation-v0 \\
            --num_envs 50
    """

    def __post_init__(self):
        super().__post_init__()

        # ------ Terrain ------
        self.scene.terrain.terrain_generator = deepcopy(VALIDATION_20CM_STAIRS_CFG)
        self.scene.terrain.max_init_terrain_level = None

        # ------ Commands: fixed 0.5 m/s forward ------
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ------ Episode length ------
        self.episode_length_s = 20.0

        # ------ Curriculum: all off ------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        # ------ Events: no external perturbations ------
        self.events.push_robot = None
        self.events.randomize_apply_external_force_torque = None

        # ------ Observations: clean (no noise) ------
        # Only the policy group has corruption; PIE groups are already clean
        self.observations.policy.enable_corruption = False

        # NOTE: do NOT call disable_zero_weight_rewards() here.
        # Same reason as DeeproboticsLite3StairsValidationEnvCfg above.
