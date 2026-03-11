# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PIE (Parkour with Implicit-Explicit Learning) framework modules.

This package contains:
- PIEEstimator: The dual-level implicit-explicit estimator network.
- PIE loss functions for training the estimator.
- History buffer management for proprioception and depth.
- Ground truth extraction utilities.

Reference: docs/PIE.md
"""

from .pi_estimator import PIEEstimator
from .pie_history_buffer import PIEHistoryBuffer
from .pie_loss import compute_pie_estimator_loss
from .depth_augmentation import DepthAugmentation
from .pie_gt_utils import (
    get_base_linear_velocity_gt,
    get_foot_clearance_gt,
    get_height_scan_gt,
)
from .pie_runner import PIEOnPolicyRunner

__all__ = [
    "PIEEstimator",
    "PIEHistoryBuffer",
    "compute_pie_estimator_loss",
    "DepthAugmentation",
    "get_base_linear_velocity_gt",
    "get_foot_clearance_gt",
    "get_height_scan_gt",
    "PIEOnPolicyRunner",
]
