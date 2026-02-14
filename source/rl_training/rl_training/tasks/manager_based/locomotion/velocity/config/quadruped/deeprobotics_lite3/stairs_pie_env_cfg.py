# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PIE-augmented stairs environment configuration for DeepRobotics Lite3.

This module provides environment configurations for PIE (Parkour with
Implicit-Explicit Learning) training. It extends the stairs climbing
environment with additional observation groups required by the PIE Estimator.

Observation Groups:
    - ``policy``: Standard proprioception (45 dims) — base_ang_vel, gravity,
      cmd, joint_pos, joint_vel, actions. Augmented at runtime by PIEOnPolicyRunner
      with PIE injection (+103 dims → total 148 dims).
    - ``critic``: Privileged observations including height scan (for value function).
    - ``pie_proprio``: Raw proprioception for PIE history buffer (same terms as policy,
      but separate group for clean extraction).
    - ``pie_gt_vel``: Ground truth base linear velocity (3 dims).
    - ``pie_gt_height_scan``: Ground truth height scan (345 dims) for map reconstruction loss.

Design:
    The PIE Estimator receives its proprio/depth input through dedicated
    observation groups, NOT through the policy group. The PIEOnPolicyRunner
    extracts these groups, manages the history buffer, runs the estimator,
    and concatenates the injection into the policy observation before
    passing it to the PPO actor.

References:
    - docs/PIE.md Section 2-4
    - docs/PIE_instruction.md
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .stairs_env_cfg import DeeproboticsLite3StairsEnvCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class PIEObservationsCfg:
    """Observation configuration for PIE training.

    Contains the standard policy/critic groups plus PIE-specific
    ground truth groups used by the PIE Estimator loss.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (45 dims, will be augmented by PIE injection at runtime).

        This is the base proprioception that the PPO actor sees.
        The PIEOnPolicyRunner concatenates estimator outputs to this,
        producing a total of 148 dims (45 + 103) as per PIE Section 2.1.
        """

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations (privileged, includes height scan).

        The critic sees ground truth velocity and terrain information
        to learn an accurate value function.
        """

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PIEProprioCfg(ObsGroup):
        """PIE proprioception group (45 dims, no corruption).

        Dedicated group for the PIE history buffer input.
        Uses same terms as PolicyCfg but without noise corruption
        (the estimator should see clean proprio for better GT matching).
        """

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PIEGtVelCfg(ObsGroup):
        """Ground truth base velocity (3 dims) for PIE estimation loss."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PIEGtHeightScanCfg(ObsGroup):
        """Ground truth height scan (345 dims) for PIE map reconstruction loss."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PIEGtClearanceCfg(ObsGroup):
        """Ground truth foot clearance (4 dims) for PIE estimation loss.

        Per-foot clearance: h^f_i = z_foot_i - z_terrain_at_foot_i.
        Order: [FL, FR, HL, HR].
        """

        foot_clearance = ObsTerm(
            func=mdp.foot_clearance_gt,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]),
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            },
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation group instances
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    pie_proprio: PIEProprioCfg = PIEProprioCfg()
    pie_gt_vel: PIEGtVelCfg = PIEGtVelCfg()
    pie_gt_height_scan: PIEGtHeightScanCfg = PIEGtHeightScanCfg()
    pie_gt_clearance: PIEGtClearanceCfg = PIEGtClearanceCfg()


@configclass
class DeeproboticsLite3StairsPIEEnvCfg(DeeproboticsLite3StairsEnvCfg):
    """Lite3 stairs climbing environment with PIE Estimator support.

    This environment extends the stairs environment to provide the additional
    observation groups needed by the PIE framework:

    - ``pie_proprio``: Clean proprioception for the PIE history buffer (45 dims)
    - ``pie_gt_vel``: Ground truth base velocity for estimation loss (3 dims)
    - ``pie_gt_height_scan``: Ground truth height scan for reconstruction loss (345 dims)

    The policy observation (45 dims) is augmented at runtime by PIEOnPolicyRunner
    with PIE injection features (+103 dims), giving the actor a total input of 148 dims.

    Training should use ``PIEOnPolicyRunner`` instead of the standard ``OnPolicyRunner``.

    Usage::

        python scripts/reinforcement_learning/rsl_rl/train_pie.py \\
            --task Stairs-Deeprobotics-Lite3-PIE-v0 \\
            --num_envs 4096
    """

    def __post_init__(self):
        # Call parent initialization first (stairs terrain, rewards, curriculum)
        super().__post_init__()

        # Replace observation config with PIE-specific one
        self.observations = PIEObservationsCfg()

        # Re-apply Lite3-specific joint names to policy/critic/pie_proprio
        from isaaclab.managers import SceneEntityCfg

        lite3_joint_cfg = SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True)

        # Policy
        self.observations.policy.joint_pos.params["asset_cfg"] = lite3_joint_cfg
        self.observations.policy.joint_vel.params["asset_cfg"] = lite3_joint_cfg

        # Critic
        self.observations.critic.joint_pos.params["asset_cfg"] = lite3_joint_cfg
        self.observations.critic.joint_vel.params["asset_cfg"] = lite3_joint_cfg

        # PIE Proprio
        self.observations.pie_proprio.joint_pos.params["asset_cfg"] = lite3_joint_cfg
        self.observations.pie_proprio.joint_vel.params["asset_cfg"] = lite3_joint_cfg

        # ------------------------------Rewards (PIE-specific tuning)------------------------------
        # Increase stand_still penalty to prevent swaying/drifting when velocity command is near zero
        # This is critical for stable standing behavior with PIE estimator
        self.rewards.stand_still.weight = -2.0  # Increased from -0.5 to strongly penalize joint deviation
        self.rewards.stand_still.params["command_threshold"] = 0.15  # Slightly larger threshold for smoother transition
        
        # Increase action rate penalty to reduce high-frequency oscillations
        self.rewards.action_rate_l2.weight = -0.05  # Increased from -0.03 to suppress jitter
        
        # Increase angular velocity penalty to reduce body roll/pitch oscillations when standing
        self.rewards.ang_vel_xy_l2.weight = -0.2  # Increased from -0.05 to reduce swaying

        # Increase velocity tracking reward to prevent "stand still to survive" exploit
        self.rewards.track_lin_vel_xy_exp.weight = 3.5  # Increased from 2.5 to incentivize movement

        # Disable zero-weight rewards
        self.disable_zero_weight_rewards()
