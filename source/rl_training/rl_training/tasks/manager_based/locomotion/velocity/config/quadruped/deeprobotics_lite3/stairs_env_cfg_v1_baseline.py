# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Stair climbing environment configuration for DeepRobotics Lite3.

This module provides environment configurations designed for training a
quadruped robot to climb 20cm stairs using curriculum learning.

Key design decisions:
- Uses STRAIGHT staircases (not pyramid) for realistic stair climbing
- 10-level curriculum: Level 0 (flat) -> Level 9 (20cm step height)
- 20% flat terrain preserved at ALL difficulty levels
- Both ascending and descending stairs
- Each staircase has 1.5m flat areas before and after
- Forward-only velocity commands (no backward)
- GRU recurrent policy for temporal reasoning

Environment variants:
- DeeproboticsLite3StairsEnvCfg: Base stairs environment (MLP policy)
- DeeproboticsLite3StairsGRUEnvCfg: Stairs environment for GRU policy
"""

import math
from copy import deepcopy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp.terrains import (
    TrapezoidStairsTerrainCfg,
    InvertedTrapezoidStairsTerrainCfg,
)

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


# ---------------------------------------------------------------------------
# Terrain configuration: trapezoid stairs with curriculum
# ---------------------------------------------------------------------------
# Cell layout (8m x 8m):
#   Trapezoid:         flat(1m) → UP → plateau(1m) → DOWN → flat(1m)
#   Inverted Trapezoid: flat(1m) → DOWN → valley(1m) → UP → flat(1m)
#
# Both ends are at the same height → no cliff at cell boundaries.
# Each side has (8-1-1-1)/2 = 2.5m of stairs → ~8 steps per side.
# At Level 9 (20cm step): peak height = 8 × 0.20 = 1.6m.

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,       # 10 difficulty levels (Level 0 ~ Level 9)
    num_cols=20,        # 20 terrain columns per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,    # Enable terrain curriculum (progressive difficulty)
    sub_terrains={
        # 35% trapezoid stairs (UP then DOWN)
        # Level 0: step_height=0.0 (flat), Level 9: step_height=0.20m
        "trapezoid": TrapezoidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.0, 0.20),
            step_width=0.28,
            flat_mid_length=1.0,
        ),
        # 35% inverted trapezoid stairs (DOWN then UP)
        "inverted_trapezoid": InvertedTrapezoidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.0, 0.20),
            step_width=0.28,
            flat_mid_length=1.0,
        ),
        # 20% flat terrain (preserved at ALL difficulty levels)
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        # 10% random rough terrain (for robustness)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


@configclass
class DeeproboticsLite3StairsEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Lite3 stair climbing environment with straight staircases.

    Curriculum: Level 0 (flat ground) -> Level 9 (20cm step height)
    Terrain mix: 30% ascending + 30% descending + 20% flat + 10% slopes + 10% rough
    """

    def __post_init__(self):
        import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

        # Call parent initialization first
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # Use straight staircase terrain with curriculum
        self.scene.terrain.terrain_generator = deepcopy(STAIRS_TERRAINS_CFG)
        # Start new robots at lower difficulty levels
        self.scene.terrain.max_init_terrain_level = 3

        # ------------------------------Commands------------------------------
        # Forward-only, conservative velocity for stair climbing
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # ------------------------------Rewards------------------------------
        # === Velocity tracking (reduced, prioritize stability over speed) ===
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # === Feet behavior (aggressive tuning for stairs) ===
        # Encourage longer air time for clear stepping over stairs
        self.rewards.feet_air_time.weight = 6.0
        self.rewards.feet_air_time.params["threshold"] = 0.35

        # Higher foot clearance target to clear 20cm steps
        self.rewards.feet_height.weight = -1.0
        self.rewards.feet_height.params["target_height"] = 0.10

        # Strong stumble penalty (hitting step edges)
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_stumble,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )

        # Stronger slide penalty (prevent slipping on step edges)
        self.rewards.feet_slide.weight = -0.2

        # === Body stability (relaxed for stair climbing) ===
        # Allow more tilting when climbing/descending
        self.rewards.flat_orientation_l2.weight = -1.5

        # Relax base height penalty (height varies on stairs)
        self.rewards.base_height_l2.weight = -5.0

        # Reduce vertical velocity penalty (climbing has vertical motion)
        self.rewards.lin_vel_z_l2.weight = -1.0

        # === Action smoothness (slightly increased for stability) ===
        self.rewards.action_rate_l2.weight = -0.05

        # === Forward progress monitoring (weight=0, pure logging) ===
        self.rewards.forward_progress = RewTerm(
            func=mdp.forward_progress,
            weight=0.0,
            params={"command_name": "base_velocity"},
        )

        # ------------------------------Events------------------------------
        # Restrict yaw range for stair training (roughly face forward)
        self.events.randomize_reset_base.params["pose_range"]["yaw"] = (-0.5, 0.5)

        # ------------------------------Curriculum------------------------------
        # Terrain levels: advance difficulty based on walking distance
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # Command levels: gradually increase velocity range
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.3, 1.0),
            },
        )

        # Stair climbing metric: monitor average terrain level in TensorBoard
        self.curriculum.stair_metric = CurrTerm(func=mdp.stair_climbing_metric)

        # Disable zero-weight rewards (except forward_progress which is for logging)
        if self.__class__.__name__ == "DeeproboticsLite3StairsEnvCfg":
            self.disable_zero_weight_rewards()
            # Re-enable forward_progress for monitoring (it was disabled above due to weight=0)
            self.rewards.forward_progress = RewTerm(
                func=mdp.forward_progress,
                weight=0.0,
                params={"command_name": "base_velocity"},
            )


@configclass
class DeeproboticsLite3StairsGRUEnvCfg(DeeproboticsLite3StairsEnvCfg):
    """Lite3 stair climbing environment configured for GRU recurrent policy.

    Uses the same terrain, rewards, and curriculum as DeeproboticsLite3StairsEnvCfg.
    The GRU hidden state captures temporal dependencies without explicit history
    observation buffer, so we use the standard PolicyCfg (not PolicyHistoryCfg).
    """

    def __post_init__(self):
        import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

        super().__post_init__()

        # GRU uses standard observations (no history buffer needed)
        # The parent already sets this up correctly via RoughEnvCfg

        # Disable zero-weight rewards and re-enable forward_progress
        self.disable_zero_weight_rewards()
        self.rewards.forward_progress = RewTerm(
            func=mdp.forward_progress,
            weight=0.0,
            params={"command_name": "base_velocity"},
        )


@configclass
class DeeproboticsLite3StairsHistoryEnvCfg(DeeproboticsLite3StairsEnvCfg):
    """Lite3 stairs environment with observation history.

    Uses 20-timestep observation history for temporal context.
    Input dimension: 45 obs x 20 timesteps = 900 dimensions.
    """

    def __post_init__(self):
        import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
        from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
        from isaaclab.managers import SceneEntityCfg

        # Call parent initialization (sets up terrain, rewards, curriculum)
        super().__post_init__()

        # Switch to history observation group
        self.observations.policy = ObservationsCfg.PolicyHistoryCfg()

        # Re-apply Lite3-specific observation settings
        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None   # type: ignore

        # Re-apply Lite3-specific scales
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Re-apply Lite3-specific joint names
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names, preserve_order=True
        )

        # Disable zero-weight rewards and re-enable forward_progress
        self.disable_zero_weight_rewards()
        self.rewards.forward_progress = RewTerm(
            func=mdp.forward_progress,
            weight=0.0,
            params={"command_name": "base_velocity"},
        )
