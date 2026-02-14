# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PIE-augmented OnPolicyRunner for rsl_rl training.

This module extends the standard ``OnPolicyRunner`` to integrate the
PIE (Parkour with Implicit-Explicit) Estimator into the PPO training loop.

Architecture overview (see docs/PIE.md):

    ┌────────────────────────────────────────────────────────────────────┐
    │  PIEOnPolicyRunner                                                │
    │                                                                    │
    │  ┌─────────────┐  ┌────────────────┐  ┌─────────────────────────┐│
    │  │ PIEEstimator │  │PIEHistoryBuffer│  │ Standard PPO (rsl_rl)   ││
    │  │              │  │  proprio (H1)  │  │  ActorCritic + Storage  ││
    │  │  Encoders    │  │  depth   (H2)  │  │  PPO update             ││
    │  │  Fusion+GRU  │  │                │  │                         ││
    │  │  Heads       │  │                │  │                         ││
    │  └──────┬───────┘  └───────┬────────┘  └────────────┬────────────┘│
    │         │                  │                        │             │
    │         └──────────────────┼─ PIE injection ────────┘             │
    │                            │                                      │
    └────────────────────────────────────────────────────────────────────┘

Training flow per step:
    1. Env provides obs TensorDict with "policy", "critic", "pie_proprio", 
       "pie_height_scan" groups
    2. Push "pie_proprio" into history buffer
    3. (Depth: synthetic or placeholder for now — see TODO)
    4. PIE Estimator forward → injection features
    5. Concatenate injection into "policy" obs → augmented policy obs
    6. PPO act/evaluate on augmented obs
    7. After rollout: compute PIE loss using GT from "pie_gt" obs group
    8. PIE loss backward + optimizer step (separate from PPO optimizer)
"""

from __future__ import annotations

import os
import time
import statistics
import torch
import torch.optim as optim
from collections import deque
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv

from .pi_estimator import PIEEstimator
from .pie_history_buffer import PIEHistoryBuffer
from .pie_loss import compute_pie_estimator_loss


class PIEOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with PIE Estimator integration.

    Extends the standard ``OnPolicyRunner`` by:
    - Creating and managing a ``PIEEstimator`` network
    - Maintaining ``PIEHistoryBuffer`` for proprio/depth history
    - Computing PIE loss (reconstruction + estimation + KL) alongside PPO
    - Injecting estimator outputs into the policy observation

    The PIE estimator has its own optimizer (Adam) and loss, trained
    simultaneously with the PPO policy but without interfering with
    the PPO gradient flow (no policy gradient through the estimator).

    Args:
        env: The vectorized environment.
        train_cfg: Training configuration dictionary (includes ``pie_cfg``).
        log_dir: Directory for logging.
        device: Torch device.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        # Extract PIE-specific configuration before parent init
        self.pie_cfg = train_cfg.pop("pie_cfg", {})

        # Pre-compute PIE injection dimension for obs augmentation
        pie_map_dim = self.pie_cfg.get("map_dim", 64)
        pie_latent_dim = self.pie_cfg.get("latent_dim", 32)
        self._pie_injection_dim = 3 + 4 + pie_map_dim + pie_latent_dim  # vel + clearance + map_enc + z_sample

        # Call parent constructor (this creates the PPO algorithm, storage, etc.)
        # NOTE: We override _construct_algorithm() to augment the initial obs
        #       with PIE injection features BEFORE ActorCritic construction.
        super().__init__(env, train_cfg, log_dir, device)

        # --- PIE Estimator setup ---
        self._setup_pie_estimator()

    def _setup_pie_estimator(self):
        """Initialize PIE Estimator, history buffer, and optimizer."""
        cfg = self.pie_cfg

        # Dimensions
        self.pie_proprio_dim = cfg.get("proprio_dim", 45)
        self.pie_proprio_history_len = cfg.get("proprio_history_len", 10)
        self.pie_depth_channels = cfg.get("depth_channels", 1)
        self.pie_depth_history_len = cfg.get("depth_history_len", 2)
        self.pie_depth_height = cfg.get("depth_height", 64)
        self.pie_depth_width = cfg.get("depth_width", 64)
        self.pie_hidden_dim = cfg.get("hidden_dim", 128)
        self.pie_latent_dim = cfg.get("latent_dim", 32)
        self.pie_map_dim = cfg.get("map_dim", 64)
        self.pie_map_gt_dim = cfg.get("map_gt_dim", 345)

        # Loss weights
        self.pie_recon_weight = cfg.get("recon_weight", 1.0)
        self.pie_est_weight = cfg.get("est_weight", 1.0)
        self.pie_kl_weight = cfg.get("kl_weight", 0.01)

        # Learning rate
        pie_lr = cfg.get("learning_rate", 1e-3)

        # Whether to use depth (disabled by default since we don't have depth camera yet)
        self.pie_use_depth = cfg.get("use_depth", False)

        # Observation group names in the TensorDict
        self.pie_proprio_obs_key = cfg.get("proprio_obs_key", "pie_proprio")
        self.pie_gt_vel_key = cfg.get("gt_vel_key", "pie_gt_vel")
        self.pie_gt_clearance_key = cfg.get("gt_clearance_key", "pie_gt_clearance")
        self.pie_gt_height_scan_key = cfg.get("gt_height_scan_key", "pie_gt_height_scan")
        self.pie_gt_next_state_key = cfg.get("gt_next_state_key", "pie_gt_next_state")

        # Create PIE Estimator
        self.pie_estimator = PIEEstimator(
            proprio_dim=self.pie_proprio_dim,
            proprio_history_len=self.pie_proprio_history_len,
            depth_channels=self.pie_depth_channels,
            depth_history_len=self.pie_depth_history_len,
            depth_height=self.pie_depth_height,
            depth_width=self.pie_depth_width,
            hidden_dim=self.pie_hidden_dim,
            latent_dim=self.pie_latent_dim,
            map_dim=self.pie_map_dim,
            map_gt_dim=self.pie_map_gt_dim,
        ).to(self.device)

        # Create history buffer
        self.pie_history_buffer = PIEHistoryBuffer(
            num_envs=self.env.num_envs,
            proprio_dim=self.pie_proprio_dim,
            proprio_history_len=self.pie_proprio_history_len,
            depth_channels=self.pie_depth_channels,
            depth_height=self.pie_depth_height,
            depth_width=self.pie_depth_width,
            depth_history_len=self.pie_depth_history_len,
            device=torch.device(self.device),
        )

        # Create PIE optimizer (separate from PPO)
        self.pie_optimizer = optim.Adam(
            self.pie_estimator.parameters(), lr=pie_lr
        )

        # Reset hidden state
        self.pie_estimator.reset_hidden(self.env.num_envs, torch.device(self.device))

        # Storage for PIE loss computation during rollout
        self._pie_rollout_data = []

        # Compute injection dim for logging
        self.pie_injection_dim = 3 + 4 + self.pie_map_dim + self.pie_latent_dim
        print(f"[PIE] Estimator created: injection_dim={self.pie_injection_dim}, "
              f"params={sum(p.numel() for p in self.pie_estimator.parameters()):,}")

    def _construct_algorithm(self, obs) -> PPO:
        """Override parent to augment initial obs with PIE injection dimension.

        The rsl_rl ActorCritic reads observation dimensions from the initial obs.
        We need to expand the "policy" observation with PIE injection features
        so the actor MLP is constructed with the correct input dimension (148).

        Args:
            obs: TensorDict from the environment.

        Returns:
            Constructed PPO algorithm with correctly-sized ActorCritic.
        """
        # Augment the "policy" observation with a dummy PIE injection
        # so ActorCritic.actor gets the correct input dimension (45 + 103 = 148)
        if "policy" in obs.keys():
            batch_size = obs["policy"].shape[0]
            dummy_injection = torch.zeros(
                batch_size, self._pie_injection_dim,
                device=obs["policy"].device,
            )
            # Create augmented obs TensorDict
            augmented_obs = obs.clone()
            augmented_obs["policy"] = torch.cat([obs["policy"], dummy_injection], dim=-1)
            print(f"[PIE] Augmented initial policy obs: {obs['policy'].shape[-1]} → "
                  f"{augmented_obs['policy'].shape[-1]} dims")
            return super()._construct_algorithm(augmented_obs)
        else:
            return super()._construct_algorithm(obs)

    def _get_pie_proprio(self, obs) -> torch.Tensor:
        """Extract PIE proprioception from observation TensorDict.

        If the dedicated PIE proprio key exists, use it.
        Otherwise, fall back to extracting from the "policy" group.

        Args:
            obs: TensorDict from the environment.

        Returns:
            Tensor of shape ``(num_envs, proprio_dim)``.
        """
        if self.pie_proprio_obs_key in obs.keys():
            return obs[self.pie_proprio_obs_key]

        # Fallback: use the policy observation (first proprio_dim components)
        policy_obs = obs["policy"]
        return policy_obs[:, :self.pie_proprio_dim]

    def _get_pie_depth(self, obs=None) -> torch.Tensor:
        """Get depth input for PIE estimator.

        Synthesizes a pseudo-depth image from the height_scan observation group.
        The height scan (345 dims from GridPattern 23x15) is reshaped into a 2D
        grid and bilinearly interpolated to the target depth image resolution
        (default 64x64). This provides terrain geometry information to the
        DepthEncoder CNN.

        If height_scan data is not available, returns zeros as fallback.

        Args:
            obs: TensorDict from the environment (used to extract height_scan).

        Returns:
            Tensor of shape ``(num_envs, depth_channels, depth_height, depth_width)``.
        """
        import torch.nn.functional as F

        # Try to get height scan from the observation TensorDict
        height_scan = None
        if obs is not None and self.pie_gt_height_scan_key in obs.keys():
            height_scan = obs[self.pie_gt_height_scan_key]
        elif obs is not None and "critic" in obs.keys():
            # Critic obs has height_scan as the last 345 dims
            critic_obs = obs["critic"]
            if critic_obs.shape[-1] > self.pie_map_gt_dim:
                height_scan = critic_obs[:, -self.pie_map_gt_dim:]

        if height_scan is None:
            return torch.zeros(
                self.env.num_envs,
                self.pie_depth_channels,
                self.pie_depth_height,
                self.pie_depth_width,
                device=self.device,
            )

        # Reshape height scan into 2D grid
        # GridPattern(resolution=0.07, size=[1.6, 1.0]) → 23 x 15 = 345 rays
        num_envs = height_scan.shape[0]
        scan_points = height_scan.shape[-1]

        # Infer grid dimensions (23x15 for Lite3 default)
        # height_scan pattern: size[0]/resolution x size[1]/resolution + 1
        grid_h, grid_w = 23, 15
        if scan_points != grid_h * grid_w:
            # Try to find factors close to sqrt
            import math
            sqrt_n = int(math.sqrt(scan_points))
            for h in range(sqrt_n + 5, sqrt_n - 5, -1):
                if scan_points % h == 0:
                    grid_h, grid_w = h, scan_points // h
                    break

        # Reshape to (N, 1, grid_h, grid_w)
        depth_grid = height_scan.reshape(num_envs, 1, grid_h, grid_w)

        # Bilinear interpolation to target depth image size (64x64)
        depth_image = F.interpolate(
            depth_grid,
            size=(self.pie_depth_height, self.pie_depth_width),
            mode="bilinear",
            align_corners=False,
        )

        # depth_image: (num_envs, 1, depth_height, depth_width)
        return depth_image

    def _augment_obs_with_pie(self, obs, pie_injection: torch.Tensor):
        """Augment the policy observation with PIE injection features.

        Concatenates the PIE estimator output to the existing "policy"
        observation group. This creates the full PIE policy input:
            o_policy = [o_t, v_hat_t, h_hat^f_t, z^m_t, z_t]

        Args:
            obs: TensorDict from the environment.
            pie_injection: ``(num_envs, injection_dim)`` from PIE estimator.

        Returns:
            Modified TensorDict with augmented "policy" observation.
        """
        # Store original policy obs before augmentation
        original_policy = obs["policy"]
        # Safety: replace any NaN/Inf in PIE injection before augmentation
        pie_injection = torch.nan_to_num(pie_injection, nan=0.0, posinf=100.0, neginf=-100.0)
        # Concatenate PIE injection
        augmented_policy = torch.cat([original_policy, pie_injection], dim=-1)
        # Update the TensorDict
        obs["policy"] = augmented_policy
        return obs

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """Main training loop with PIE integration.

        This overrides the parent ``learn()`` method to add:
        1. PIE history buffer management
        2. PIE estimator forward pass for policy injection
        3. PIE loss computation and optimization
        4. PIE-specific logging

        Args:
            num_learning_iterations: Number of PPO iterations to train.
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # PIE loss tracking
        pie_loss_buffer = deque(maxlen=100)

        # Ensure multi-GPU params are synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initial PIE history push
        pie_proprio = self._get_pie_proprio(obs)
        self.pie_history_buffer.push_proprio(pie_proprio)

        # Augment initial obs with PIE injection
        pie_depth = self._get_pie_depth(obs)
        self.pie_history_buffer.push_depth(pie_depth)
        with torch.no_grad():
            pie_injection = self.pie_estimator.get_policy_injection(
                self.pie_history_buffer.get_proprio_history(),
                self.pie_history_buffer.get_depth_history(),
            )
        obs = self._augment_obs_with_pie(obs, pie_injection)

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # Clear PIE rollout data
            self._pie_rollout_data.clear()

            # --- Rollout phase ---
            # NOTE: We do NOT wrap the entire loop in torch.inference_mode() because
            # PIE GT data extraction needs to produce non-inference tensors for the
            # subsequent PIE backward pass. Instead, we use inference_mode() only for
            # PPO-specific operations (act, process_env_step) and torch.no_grad() for
            # PIE operations. This ensures stored tensors can be replayed with gradients.
            for step_idx in range(self.num_steps_per_env):
                # Sample actions from PPO (on augmented obs)
                with torch.inference_mode():
                    actions = self.alg.act(obs)

                # Step the environment
                with torch.inference_mode():
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                # --- PIE: Extract GT and push history ---
                # Use torch.no_grad() (NOT inference_mode) so tensors are normal
                with torch.no_grad():
                    pie_proprio = self._get_pie_proprio(obs)

                    # Store GT data BEFORE pushing new obs to history
                    # So the replay history matches estimator input at step t,
                    # and gt_next_state = o_{t+1} (the current obs after env.step)
                    pie_gt_data = self._extract_pie_gt(obs)
                    # Override gt_next_state with actual o_{t+1}
                    pie_gt_data["gt_next_state"] = pie_proprio.clone().detach()
                    # Record which envs are done so replay can mask their GT
                    # (done envs have GT from the NEW episode, not matching history)
                    pie_gt_data["done_mask"] = dones.clone().detach()
                    self._pie_rollout_data.append(pie_gt_data)

                    # NOW push current obs into history buffer
                    self.pie_history_buffer.push_proprio(pie_proprio)

                    pie_depth = self._get_pie_depth(obs)
                    self.pie_history_buffer.push_depth(pie_depth)

                    # Reset PIE hidden state and history for done envs
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_ids) > 0:
                        self.pie_estimator.reset_hidden_for_envs(done_ids)
                        self.pie_history_buffer.reset(done_ids)

                    # PIE: compute injection for next step's obs
                    pie_injection = self.pie_estimator.get_policy_injection(
                        self.pie_history_buffer.get_proprio_history(),
                        self.pie_history_buffer.get_depth_history(),
                    )

                    # Augment obs with PIE features
                    obs = self._augment_obs_with_pie(obs, pie_injection)

                # Process env step (PPO transition recording)
                with torch.inference_mode():
                    self.alg.process_env_step(obs, rewards, dones, extras)

                # Book keeping
                if self.log_dir is not None:
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    elif "log" in extras:
                        ep_infos.append(extras["log"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Compute returns for PPO
            with torch.inference_mode():
                self.alg.compute_returns(obs)

            # --- PPO update ---
            loss_dict = self.alg.update()

            # --- PIE Estimator update ---
            pie_loss_info = self._update_pie_estimator()
            pie_loss_buffer.append(pie_loss_info["total"])

            # Merge PIE loss into loss dict for logging
            for key, value in pie_loss_info.items():
                loss_dict[f"pie_{key}"] = value

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log info
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            # Save code state
            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save final model
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _extract_pie_gt(self, obs) -> dict:
        """Extract PIE ground truth data from the observation TensorDict.

        Looks for dedicated GT observation groups. Falls back to
        extracting from standard groups if dedicated ones are not present.

        Args:
            obs: TensorDict from the environment.

        Returns:
            Dictionary with GT tensors for PIE loss computation.
        """
        gt_data = {}

        # Ground truth velocity
        if self.pie_gt_vel_key in obs.keys():
            gt_data["gt_vel"] = obs[self.pie_gt_vel_key].clone().detach()
        else:
            # Try from critic obs (base_lin_vel is first 3 dims of critic)
            if "critic" in obs.keys():
                gt_data["gt_vel"] = obs["critic"][:, :3].clone().detach()
            else:
                gt_data["gt_vel"] = torch.zeros(self.env.num_envs, 3, device=self.device)

        # Ground truth foot clearance
        if self.pie_gt_clearance_key in obs.keys():
            gt_data["gt_clearance"] = obs[self.pie_gt_clearance_key].clone().detach()
        else:
            gt_data["gt_clearance"] = torch.zeros(self.env.num_envs, 4, device=self.device)

        # Ground truth height scan
        if self.pie_gt_height_scan_key in obs.keys():
            gt_data["gt_height_scan"] = obs[self.pie_gt_height_scan_key].clone().detach()
        else:
            # Try from critic height_scan (last map_gt_dim dims of critic obs)
            if "critic" in obs.keys() and obs["critic"].shape[-1] > self.pie_map_gt_dim:
                gt_data["gt_height_scan"] = obs["critic"][:, -self.pie_map_gt_dim:].clone().detach()
            else:
                gt_data["gt_height_scan"] = torch.zeros(
                    self.env.num_envs, self.pie_map_gt_dim, device=self.device
                )

        # Ground truth next state — placeholder; overwritten by rollout loop
        # with the actual o_{t+1} proprio from the next env step.
        gt_data["gt_next_state"] = self._get_pie_proprio(obs).clone().detach()

        # Store proprio history for replay (must detach from inference mode)
        gt_data["proprio_history"] = self.pie_history_buffer.get_proprio_history().clone().detach()
        gt_data["depth_history"] = self.pie_history_buffer.get_depth_history().clone().detach()

        # Store GRU hidden state snapshot so replay matches rollout context.
        # At this point in the rollout, _gru_hidden reflects the state BEFORE
        # get_policy_injection() is called (i.e., the hidden from the previous
        # step's forward pass). This is exactly the hidden that should be used
        # when replaying this step's forward pass for PIE loss computation.
        if self.pie_estimator._gru_hidden is not None:
            gt_data["gru_hidden"] = self.pie_estimator._gru_hidden.clone().detach()
        else:
            gt_data["gru_hidden"] = torch.zeros(
                self.env.num_envs, self.pie_hidden_dim, device=self.device
            )

        return gt_data

    def _update_pie_estimator(self) -> dict:
        """Update the PIE Estimator using data collected during the rollout.

        Replays the stored rollout data through the estimator (with gradient)
        and computes the PIE loss.

        Uses **per-step gradient accumulation** to keep memory usage constant
        regardless of ``num_steps_per_env``.  Each step's loss is scaled by
        ``1/num_steps`` and immediately back-propagated so that the computation
        graph is freed before the next step.

        Key fixes vs original implementation:
        - Restores GRU hidden state snapshot from rollout instead of resetting
          to zero, ensuring replay context matches rollout context.
        - Masks out done environments whose GT crosses episode boundaries.

        Returns:
            Dictionary with PIE loss values.
        """
        if len(self._pie_rollout_data) == 0:
            return {"total": 0.0, "reconstruction": 0.0, "estimation": 0.0, "kl": 0.0}

        num_steps = len(self._pie_rollout_data)
        total_loss_val = 0.0
        total_recon = 0.0
        total_est = 0.0
        total_kl = 0.0

        # Zero gradients once; each step will accumulate into the same grad buffers
        self.pie_optimizer.zero_grad()

        for step_data in self._pie_rollout_data:
            # Restore GRU hidden state from rollout snapshot instead of letting
            # it evolve from zero. This ensures the fusion module sees the same
            # temporal context during replay as it did during rollout.
            self.pie_estimator._gru_hidden = step_data["gru_hidden"]

            # Forward pass (with gradient through estimator parameters)
            estimator_output = self.pie_estimator(
                step_data["proprio_history"],
                step_data["depth_history"],
            )

            # Mask out done environments: their GT data crosses episode
            # boundaries (history from old episode, GT from new episode).
            done_mask = step_data.get("done_mask", None)
            if done_mask is not None and done_mask.any():
                alive_mask = (~done_mask.bool()).float()  # 1 for alive, 0 for done
                num_alive = alive_mask.sum().clamp(min=1.0)
                # Scale factor to maintain correct gradient magnitude
                scale = float(self.env.num_envs) / float(num_alive.item())
            else:
                alive_mask = None
                scale = 1.0

            # Compute PIE loss (masked if needed)
            loss_dict = compute_pie_estimator_loss(
                estimator_output=estimator_output,
                gt_next_state=step_data["gt_next_state"],
                gt_vel=step_data["gt_vel"],
                gt_clearance=step_data["gt_clearance"],
                gt_map=step_data["gt_height_scan"],
                recon_weight=self.pie_recon_weight,
                est_weight=self.pie_est_weight,
                kl_weight=self.pie_kl_weight,
                alive_mask=alive_mask,
            )

            # Scale and backward immediately (gradient accumulation)
            step_loss = loss_dict["total"] / num_steps
            step_loss.backward()

            # Record scalar metrics (detached)
            total_loss_val += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_est += loss_dict["estimation"].item()
            total_kl += loss_dict["kl"].item()

        # Single optimizer step with accumulated gradients
        # Check for NaN gradients before stepping — skip update if found
        has_nan_grad = False
        for p in self.pie_estimator.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        if has_nan_grad:
            print("[PIE] WARNING: NaN gradient detected in PIE estimator, skipping optimizer step.")
            self.pie_optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(self.pie_estimator.parameters(), 1.0)
            self.pie_optimizer.step()

        return {
            "total": total_loss_val / num_steps,
            "reconstruction": total_recon / num_steps,
            "estimation": total_est / num_steps,
            "kl": total_kl / num_steps,
        }

    def train_mode(self):
        """Switch to training mode."""
        super().train_mode()
        self.pie_estimator.train()

    def eval_mode(self):
        """Switch to evaluation mode."""
        super().eval_mode()
        self.pie_estimator.eval()

    def save(self, path: str, infos=None):
        """Save model checkpoint including PIE Estimator.

        Args:
            path: Path to save the checkpoint.
            infos: Additional info to save.
        """
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            # PIE-specific
            "pie_estimator_state_dict": self.pie_estimator.state_dict(),
            "pie_optimizer_state_dict": self.pie_optimizer.state_dict(),
        }
        # Save RND if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload to external logging
        if hasattr(self, "logger_type") and self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        """Load model checkpoint including PIE Estimator.

        Args:
            path: Path to load the checkpoint.
            load_optimizer: Whether to load optimizer states.
            map_location: Device mapping for loading.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        # Load PPO policy
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        # Load PIE Estimator
        if "pie_estimator_state_dict" in loaded_dict:
            self.pie_estimator.load_state_dict(loaded_dict["pie_estimator_state_dict"])
            print("[PIE] Loaded PIE Estimator from checkpoint.")
        else:
            print("[PIE] No PIE Estimator found in checkpoint; using fresh initialization.")

        # Load RND if present
        if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        # Load optimizers
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "pie_optimizer_state_dict" in loaded_dict:
                self.pie_optimizer.load_state_dict(loaded_dict["pie_optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        # Load iteration counter
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """Get inference policy with PIE injection.

        Returns a closure that:
        1. Runs PIE estimator to get injection features
        2. Augments the observation
        3. Runs the policy actor

        Args:
            device: Target device.

        Returns:
            Callable inference function.
        """
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
            self.pie_estimator.to(device)

        def inference_fn(obs):
            # Extract proprio and push to history
            pie_proprio = self._get_pie_proprio(obs)
            self.pie_history_buffer.push_proprio(pie_proprio)
            pie_depth = self._get_pie_depth(obs)
            self.pie_history_buffer.push_depth(pie_depth)

            # Get PIE injection
            pie_injection = self.pie_estimator.get_policy_injection(
                self.pie_history_buffer.get_proprio_history(),
                self.pie_history_buffer.get_depth_history(),
            )

            # Augment obs and run actor inference
            obs = self._augment_obs_with_pie(obs, pie_injection)
            return self.alg.policy.act_inference(obs)

        return inference_fn
