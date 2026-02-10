# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Stairs environment configuration for DeepRobotics Lite3.

This module provides environment configurations specifically designed for
stair climbing training. It includes:
- DeeproboticsLite3StairsEnvCfg: Base stairs environment with mixed terrain
- DeeproboticsLite3StairsHistoryEnvCfg: Stairs environment with observation history

Design decisions:
- 50% stairs (25% up + 25% down) + 50% other terrains (slopes, rough)
- Step height curriculum: 0.05m → 0.20m
- Height scan enabled for terrain perception
- Reward tuning focused on stable stair climbing
"""

from copy import deepcopy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


# Custom terrain configuration for stairs training
STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Enable curriculum for terrain difficulty progression
    sub_terrains={
        # 25% ascending stairs
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.20),  # Curriculum: easy to hard
            step_width=0.28,  # Standard step width
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 25% descending stairs
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.20),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 20% slopes (helpful for generalization)
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        # 30% rough terrain (for robustness)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


@configclass
class DeeproboticsLite3StairsEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Lite3 stairs climbing environment configuration.
    
    This environment is designed for training stair climbing behavior with:
    - Mixed terrain: 50% stairs (up/down) + 50% other (slopes, rough)
    - Step height range: 0.05m to 0.20m with curriculum
    - Height scan enabled for terrain perception
    - Reward tuning for stable climbing behavior
    
    Key differences from RoughEnvCfg:
    - Terrain generator uses STAIRS_TERRAINS_CFG
    - terrain_levels curriculum enabled
    - command_levels curriculum enabled  
    - Adjusted rewards for stair climbing
    """

    def __post_init__(self):
        # Call parent initialization first
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # Use custom stairs terrain configuration
        self.scene.terrain.terrain_generator = deepcopy(STAIRS_TERRAINS_CFG)

        # ------------------------------Observations------------------------------
        # Policy does not use height_scan (kept disabled from parent)
        # Height scan is only used by critic if needed

        # ------------------------------Rewards------------------------------
        # Adjust rewards for stair climbing behavior
        
        # Feet behavior - encourage clear leg lifting for stairs
        self.rewards.feet_air_time.weight = 5.0  # Maintain for clear stepping
        self.rewards.feet_air_time.params["threshold"] = 0.4  # Slightly lower threshold
        
        # Increase feet height reward to encourage lifting over steps
        self.rewards.feet_height.weight = -0.5  # Increased from -0.2
        self.rewards.feet_height.params["target_height"] = 0.08  # Higher target for stairs
        
        # Enable feet stumble penalty (hitting step edges)
        self.rewards.feet_stumble = self._create_feet_stumble_reward()
        
        # Reduce flat orientation penalty (allow tilting on stairs)
        self.rewards.flat_orientation_l2.weight = -0.5  # Reduced from -2.0 for stair climbing
        
        # Slightly reduce velocity tracking (prioritize stability)
        self.rewards.track_lin_vel_xy_exp.weight = 2.5  # Reduced from 3.0
        
        # Adjust base height for stair climbing (dynamic target via height_scanner_base)
        self.rewards.base_height_l2.weight = -2.0  # Reduced from -8.0 for stair climbing
        
        # Increase slide penalty (prevent slipping on edges)
        self.rewards.feet_slide.weight = -0.1  # Increased from -0.05

        # ------------------------------Curriculum------------------------------
        # Enable terrain levels curriculum
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
        
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        
        # Enable command levels curriculum with gradual increase
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.3, 1.0),  # Start at 30% of max velocity
            },
        )

        # ------------------------------Commands------------------------------
        # Moderate velocity commands for stairs (safety first)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  # Reduced from ±1.5
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)  # Reduced from ±0.8
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)  # Reduced from ±0.8

        # Disable zero-weight rewards
        if self.__class__.__name__ == "DeeproboticsLite3StairsEnvCfg":
            self.disable_zero_weight_rewards()

    def _create_feet_stumble_reward(self):
        """Create feet stumble penalty reward term."""
        from isaaclab.managers import RewardTermCfg as RewTerm
        import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
        
        return RewTerm(
            func=mdp.feet_stumble,
            weight=-1.0,  # Penalty for hitting obstacles
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            },
        )


@configclass  
class DeeproboticsLite3StairsHistoryEnvCfg(DeeproboticsLite3StairsEnvCfg):
    """Lite3 stairs environment with observation history.
    
    This configuration extends DeeproboticsLite3StairsEnvCfg to use a 20-timestep
    observation history, providing temporal context for the policy network.
    This is particularly useful for stairs where the robot needs to anticipate
    upcoming terrain changes.
    
    For fine-tuning from pretrained model:
        logs/rsl_rl/deeprobotics_lite3_rough_history/2026-02-04_19-09-09/model_9999.pt
    """

    def __post_init__(self):
        # Import here to avoid circular imports
        from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
        from isaaclab.managers import SceneEntityCfg

        # Call parent initialization (sets up terrain, rewards, curriculum)
        super().__post_init__()

        # Switch to history observation group
        self.observations.policy = ObservationsCfg.PolicyHistoryCfg()

        # Re-apply Lite3-specific observation settings
        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None   # type: ignore  # Policy does not use height_scan

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

        # Disable zero-weight rewards
        self.disable_zero_weight_rewards()
