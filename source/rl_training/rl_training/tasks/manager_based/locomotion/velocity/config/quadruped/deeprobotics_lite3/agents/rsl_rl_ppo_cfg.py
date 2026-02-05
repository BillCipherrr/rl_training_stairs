# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsLite3RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_rough"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsLite3RoughHistoryPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    """PPO configuration for Lite3 with observation history.
    
    Uses larger network architecture to handle the 900-dimensional input
    (45 obs dims × 20 timesteps).
    """
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_rough_history"
    empirical_normalization = False
    clip_actions = 100
    # Larger network to handle 900-dim input (45 × 20 history)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[1024, 512, 256, 128],   # Increased first layer width
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )


@configclass
class DeeproboticsLite3FlatPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "deeprobotics_lite3_flat"


@configclass
class DeeproboticsLite3StairsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for Lite3 stairs climbing training.
    
    Key changes from rough config:
    - num_steps_per_env: 32 (increased for complex terrain)
    - max_iterations: 20000 (longer training for curriculum)
    - Larger network for handling height scan input
    """
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_stairs"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsLite3StairsHistoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for Lite3 stairs climbing with observation history.
    
    Uses larger network architecture to handle the expanded input dimension
    (observation dims × 20 timesteps + height scan).
    
    For fine-tuning from pretrained model, use:
        --load_run 2026-02-04_19-09-09 --checkpoint model_9999.pt
    
    Note: When fine-tuning, consider using lower learning rate (1e-4).
    """
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_stairs_history"
    empirical_normalization = False
    clip_actions = 100
    # Larger network to handle history + height scan input
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # Slightly lower for fine-tuning stability
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # Can reduce to 1e-4 for fine-tuning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
