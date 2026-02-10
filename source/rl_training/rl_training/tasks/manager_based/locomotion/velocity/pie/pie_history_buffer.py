# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Independent FIFO history buffers for PIE Estimator.

PIE requires **separate** history queues for proprioception (H1=10)
and depth images (H2=2), rather than the single 20-frame stack used
in ``DeeproboticsLite3StairsHistoryEnvCfg``.

Key differences from existing history approach:
- ``PolicyHistoryCfg``: single queue, 20 frames, flattened to (N, 900)
- ``PIEHistoryBuffer``: two independent FIFO queues with different lengths

References:
    - docs/PIE.md Section 2.2, Section 3.3
    - docs/PIE_instruction.md
"""

from __future__ import annotations

from typing import Optional

import torch


class PIEHistoryBuffer:
    """FIFO history buffer for PIE Estimator inputs.

    Maintains two independent circular buffers:

    - **Proprioception**: ``(num_envs, H1, proprio_dim)`` — H1=10 frames
    - **Depth images**: ``(num_envs, H2, C, H, W)`` — H2=2 frames

    Usage::

        buffer = PIEHistoryBuffer(num_envs=4096, device=torch.device("cuda:0"))

        # Each environment step:
        buffer.push_proprio(current_obs)    # (num_envs, 45)
        buffer.push_depth(current_depth)    # (num_envs, 1, 64, 64)

        # Feed to estimator:
        proprio_hist = buffer.get_proprio_history()  # (num_envs, 10, 45)
        depth_hist = buffer.get_depth_history()      # (num_envs, 2, 1, 64, 64)

        # On environment reset:
        buffer.reset(env_ids=reset_ids)

    Args:
        num_envs: Number of parallel environments.
        proprio_dim: Proprioception vector dimension (default 45).
        proprio_history_len: H1 frames (PIE paper: 10).
        depth_channels: Depth image channels (default 1).
        depth_height: Depth image height in pixels.
        depth_width: Depth image width in pixels.
        depth_history_len: H2 frames (PIE paper: 2).
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        proprio_dim: int = 45,
        proprio_history_len: int = 10,
        depth_channels: int = 1,
        depth_height: int = 64,
        depth_width: int = 64,
        depth_history_len: int = 2,
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.proprio_history_len = proprio_history_len
        self.depth_channels = depth_channels
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_history_len = depth_history_len
        self.device = device

        # Initialize buffers with zeros (oldest at index 0, newest at index -1)
        self.proprio_buffer = torch.zeros(
            num_envs, proprio_history_len, proprio_dim, device=device
        )
        self.depth_buffer = torch.zeros(
            num_envs, depth_history_len, depth_channels, depth_height, depth_width,
            device=device,
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset history buffers for specified environments.

        Zeroes out all history for the given environment indices.

        Args:
            env_ids: 1-D tensor of environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self.proprio_buffer.zero_()
            self.depth_buffer.zero_()
        else:
            self.proprio_buffer[env_ids] = 0.0
            self.depth_buffer[env_ids] = 0.0

    def push_proprio(self, observation: torch.Tensor) -> None:
        """Push a new proprioception frame into the FIFO queue.

        Oldest frame (index 0) is discarded; new frame is appended at the end.

        Args:
            observation: ``(num_envs, proprio_dim)`` current proprioception.
        """
        # Shift left (discard oldest at index 0)
        self.proprio_buffer[:, :-1] = self.proprio_buffer[:, 1:].clone()
        # Insert newest at the end
        self.proprio_buffer[:, -1] = observation

    def push_depth(self, depth_image: torch.Tensor) -> None:
        """Push a new depth frame into the FIFO queue.

        Oldest frame (index 0) is discarded; new frame is appended at the end.

        Args:
            depth_image: ``(num_envs, C, H, W)`` current depth image.
        """
        self.depth_buffer[:, :-1] = self.depth_buffer[:, 1:].clone()
        self.depth_buffer[:, -1] = depth_image

    def get_proprio_history(self) -> torch.Tensor:
        """Get the full proprioception history.

        Returns:
            Tensor of shape ``(num_envs, H1, proprio_dim)``.
        """
        return self.proprio_buffer

    def get_depth_history(self) -> torch.Tensor:
        """Get the full depth image history.

        Returns:
            Tensor of shape ``(num_envs, H2, C, H, W)``.
        """
        return self.depth_buffer
