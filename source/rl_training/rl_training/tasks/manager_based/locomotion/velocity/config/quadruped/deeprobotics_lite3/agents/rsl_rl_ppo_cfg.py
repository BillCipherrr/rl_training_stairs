# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


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
class DeeproboticsLite3RoughGRUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO with GRU recurrent policy for Lite3 rough terrain.

    Architecture: obs(45) → GRU(hidden=256) → h_t → MLP[256, 128] → actions(12)
    No history manager needed; GRU hidden state captures temporal dependencies.
    velocity_commands stays in the per-step input (correct for GRU).
    """
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_rough_gru"
    empirical_normalization = False
    clip_actions = 100
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
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
class DeeproboticsLite3FlatPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "deeprobotics_lite3_flat"


@configclass
class DeeproboticsLite3StairsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for Lite3 stairs climbing training (MLP policy).

    Key changes from rough config:
    - num_steps_per_env: 32 (longer rollouts for stair curriculum)
    - max_iterations: 30000 (10-level curriculum needs more iterations)
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
class DeeproboticsLite3StairsGRUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO with GRU recurrent policy for Lite3 stair climbing.

    Architecture: obs(45+345) -> GRU(hidden=256) -> h_t -> MLP[256, 128] -> actions(12)
    obs dim = 45 proprioception + 345 height_scan (23×15 grid, res=0.07m)

    Key changes from v1:
    - num_steps_per_env: 32→48 (longer rollouts cover full stair crossing sequence)
    - gamma: 0.99→0.995 (effective horizon 100→200 steps for long-range planning)
    - entropy_coef: 0.01→0.02 (more exploration; avoids early local optima)
    """
    num_steps_per_env = 48
    max_iterations = 30000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_stairs_gru"
    empirical_normalization = False
    clip_actions = 100
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsLite3StairsHistoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for Lite3 stairs climbing with observation history.

    Uses larger network architecture to handle the expanded input dimension
    (45 obs dims x 20 timesteps = 900 dimensions).
    """
    num_steps_per_env = 32
    max_iterations = 30000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_stairs_history"
    empirical_normalization = False
    clip_actions = 100
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
