# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Depth image Domain Randomization for sim-to-real transfer.

Applies stochastic augmentations to simulated depth images during training
to bridge the gap between perfect RayCaster height scans and noisy RealSense
depth frames observed on the real Lite3 robot.

Augmentation pipeline (applied in order):
    1. Additive Gaussian noise  — simulates sensor quantization noise
    2. Random rectangular dropout — simulates specular/transparent surface holes
    3. Random frame latency      — simulates camera processing delay
    4. Spatial shift (origin jitter) — simulates mounting vibration / FOV mismatch

All augmentations are on-GPU and batched for efficiency.

Usage::

    aug = DepthAugmentation(cfg, num_envs, device)
    depth_image = aug(depth_image)           # training
    depth_image = aug(depth_image, eval=True) # no augmentation

References:
    - Intel RealSense D435i/D455 noise characteristics
    - "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
      (Peng et al., 2018)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class DepthAugmentation:
    """Stochastic depth image augmentation for sim-to-real transfer.

    All operations are in-place or return new tensors on the same device.

    Args:
        cfg: Dictionary with augmentation hyperparameters.
        num_envs: Number of parallel environments.
        device: Torch device.
    """

    def __init__(self, cfg: Dict, num_envs: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.device = device

        # --- Noise ---
        self.noise_std = cfg.get("noise_std", 0.02)  # metres, RealSense D435i typical noise

        # --- Rectangular dropout (simulate specular/transparent holes) ---
        self.dropout_prob = cfg.get("dropout_prob", 0.3)  # probability of applying dropout per env
        self.dropout_num_rects_range = cfg.get("dropout_num_rects_range", (1, 5))  # min/max rectangles
        self.dropout_size_range = cfg.get("dropout_size_range", (0.05, 0.3))  # fraction of image dim

        # --- Frame latency (simulate camera processing delay) ---
        self.latency_prob = cfg.get("latency_prob", 0.2)  # probability of replacing current with stale
        self.latency_max_frames = cfg.get("latency_max_frames", 3)  # max stale frames
        self._stale_buffer: torch.Tensor | None = None  # ring buffer for stale frames
        self._stale_idx = 0

        # --- Spatial shift (simulate mounting vibration / FOV mismatch) ---
        self.shift_prob = cfg.get("shift_prob", 0.3)  # probability of shifting
        self.shift_max_pixels = cfg.get("shift_max_pixels", 4)  # max pixels to shift

        # --- Global depth scale perturbation ---
        self.scale_range = cfg.get("scale_range", (0.95, 1.05))  # multiplicative scale range

    def __call__(
        self,
        depth_image: torch.Tensor,
        is_eval: bool = False,
    ) -> torch.Tensor:
        """Apply augmentation pipeline to a depth image batch.

        Args:
            depth_image: ``(num_envs, C, H, W)`` depth image from ``_get_pie_depth()``.
            is_eval: If True, skip all augmentations (for inference / play).

        Returns:
            Augmented depth image, same shape.
        """
        if is_eval:
            return depth_image

        depth_image = self._apply_noise(depth_image)
        depth_image = self._apply_scale(depth_image)
        depth_image = self._apply_rect_dropout(depth_image)
        depth_image = self._apply_spatial_shift(depth_image)
        depth_image = self._apply_latency(depth_image)

        return depth_image

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------

    def _apply_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian noise simulating RealSense depth quantization.

        RealSense D435i has ~2% depth noise at 1m distance.
        """
        if self.noise_std <= 0:
            return img
        noise = torch.randn_like(img) * self.noise_std
        return img + noise

    def _apply_scale(self, img: torch.Tensor) -> torch.Tensor:
        """Random global depth scale to simulate calibration error."""
        lo, hi = self.scale_range
        if lo >= hi:
            return img
        # Per-environment random scale
        scale = torch.empty(self.num_envs, 1, 1, 1, device=self.device).uniform_(lo, hi)
        return img * scale

    def _apply_rect_dropout(self, img: torch.Tensor) -> torch.Tensor:
        """Random rectangular zero-out patches (simulate depth holes).

        RealSense produces zero-depth on reflective, transparent, or
        out-of-range surfaces. We simulate this by setting random
        rectangular regions to zero.
        """
        if self.dropout_prob <= 0:
            return img

        N, C, H, W = img.shape
        # Decide which envs get dropout
        mask = torch.rand(N, device=self.device) < self.dropout_prob
        if not mask.any():
            return img

        dropout_envs = mask.nonzero(as_tuple=False).flatten()
        num_dropout = dropout_envs.shape[0]

        min_rects, max_rects = self.dropout_num_rects_range
        num_rects = torch.randint(min_rects, max_rects + 1, (1,)).item()

        min_frac, max_frac = self.dropout_size_range

        for _ in range(num_rects):
            # Random rectangle sizes (fraction of image)
            rh = torch.empty(num_dropout, device=self.device).uniform_(min_frac, max_frac)
            rw = torch.empty(num_dropout, device=self.device).uniform_(min_frac, max_frac)
            rect_h = (rh * H).long().clamp(min=1)
            rect_w = (rw * W).long().clamp(min=1)

            # Random top-left corners
            top = (torch.rand(num_dropout, device=self.device) * (H - rect_h.float())).long().clamp(min=0)
            left = (torch.rand(num_dropout, device=self.device) * (W - rect_w.float())).long().clamp(min=0)

            # Apply dropout per env — vectorised via max rect size
            max_h = rect_h.max().item()
            max_w = rect_w.max().item()

            for i in range(num_dropout):
                env_id = dropout_envs[i]
                t, l, h, w = top[i].item(), left[i].item(), rect_h[i].item(), rect_w[i].item()
                img[env_id, :, t:t + h, l:l + w] = 0.0

        return img

    def _apply_spatial_shift(self, img: torch.Tensor) -> torch.Tensor:
        """Random pixel-level translation (simulate mounting vibration).

        Shifts the depth image by a random offset and fills the border
        with zero (simulating unknown depth at edges).
        """
        if self.shift_prob <= 0 or self.shift_max_pixels <= 0:
            return img

        N, C, H, W = img.shape
        mask = torch.rand(N, device=self.device) < self.shift_prob
        if not mask.any():
            return img

        shift_envs = mask.nonzero(as_tuple=False).flatten()
        max_px = self.shift_max_pixels

        # Random shifts for selected envs
        dy = torch.randint(-max_px, max_px + 1, (shift_envs.shape[0],), device=self.device)
        dx = torch.randint(-max_px, max_px + 1, (shift_envs.shape[0],), device=self.device)

        for i in range(shift_envs.shape[0]):
            env_id = shift_envs[i]
            shifted = torch.zeros_like(img[env_id])
            sy, sx = dy[i].item(), dx[i].item()

            # Compute source and destination slices
            src_y_start = max(0, -sy)
            src_y_end = min(H, H - sy)
            src_x_start = max(0, -sx)
            src_x_end = min(W, W - sx)
            dst_y_start = max(0, sy)
            dst_y_end = min(H, H + sy)
            dst_x_start = max(0, sx)
            dst_x_end = min(W, W + sx)

            shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                img[env_id, :, src_y_start:src_y_end, src_x_start:src_x_end]
            img[env_id] = shifted

        return img

    def _apply_latency(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate camera frame latency by randomly using a stale frame.

        RealSense can have 1-3 frame processing delay. We maintain a
        small ring buffer and randomly replace current frames with older ones.
        """
        if self.latency_prob <= 0 or self.latency_max_frames <= 0:
            return img

        N, C, H, W = img.shape

        # Initialize stale buffer on first call
        if self._stale_buffer is None or self._stale_buffer.shape[0] != N:
            self._stale_buffer = torch.zeros(
                self.latency_max_frames, N, C, H, W, device=self.device
            )
            self._stale_idx = 0

        # Store current frame into ring buffer
        self._stale_buffer[self._stale_idx % self.latency_max_frames] = img.clone()
        self._stale_idx += 1

        # Decide which envs get a stale frame
        mask = torch.rand(N, device=self.device) < self.latency_prob
        if not mask.any():
            return img

        stale_envs = mask.nonzero(as_tuple=False).flatten()

        # Pick a random stale frame index for each env
        available = min(self._stale_idx, self.latency_max_frames)
        if available <= 1:
            return img  # not enough history yet

        frame_idx = torch.randint(0, available - 1, (stale_envs.shape[0],), device=self.device)
        for i in range(stale_envs.shape[0]):
            env_id = stale_envs[i]
            img[env_id] = self._stale_buffer[frame_idx[i].item(), env_id]

        return img

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset stale buffer for specific environments.

        Call this when environments reset to avoid leaking frames across episodes.

        Args:
            env_ids: Indices of environments that reset. If None, resets all.
        """
        if self._stale_buffer is None:
            return
        if env_ids is None:
            self._stale_buffer.zero_()
            self._stale_idx = 0
        else:
            self._stale_buffer[:, env_ids] = 0.0
