# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PIE Estimator loss functions.

Implements the combined loss from PIE paper (docs/PIE.md Section 5):

    L_total = L_PPO + L_Estimator
    L_Estimator = w_recon * L_reconstruction + w_est * L_estimation + w_kl * L_KL

Where:
    1. L_reconstruction = MSE(o_hat_{t+1}, o_{t+1}) + MSE(m_hat_t, m_t)
    2. L_estimation = MSE(v_hat, v) + MSE(h_hat^f, h^f)
    3. L_KL = D_KL(q(z|history) || N(0, I))

References:
    - docs/PIE.md Section 5
    - docs/PIE_instruction.md
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def reconstruction_loss(
    rec_next_state: torch.Tensor,
    gt_next_state: torch.Tensor,
    rec_map: torch.Tensor,
    gt_map: torch.Tensor,
) -> torch.Tensor:
    """Reconstruction loss: MSE(o_hat_{t+1}, o_{t+1}) + MSE(m_hat_t, m_t).

    PIE Section 5, Loss (1).

    Args:
        rec_next_state: ``(batch, 45)`` reconstructed next proprioception.
        gt_next_state: ``(batch, 45)`` ground truth next proprioception.
        rec_map: ``(batch, map_gt_dim)`` reconstructed terrain map.
        gt_map: ``(batch, map_gt_dim)`` ground truth terrain height scan.

    Returns:
        Scalar reconstruction loss.
    """
    state_loss = F.mse_loss(rec_next_state, gt_next_state)
    map_loss = F.mse_loss(rec_map, gt_map)
    return state_loss + map_loss


def estimation_loss(
    est_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    est_clearance: torch.Tensor,
    gt_clearance: torch.Tensor,
) -> torch.Tensor:
    """Estimation loss: MSE(v_hat, v) + MSE(h_hat^f, h^f).

    PIE Section 5, Loss (2).

    Args:
        est_vel: ``(batch, 3)`` estimated base linear velocity.
        gt_vel: ``(batch, 3)`` ground truth base linear velocity.
        est_clearance: ``(batch, 4)`` estimated foot clearance.
        gt_clearance: ``(batch, 4)`` ground truth foot clearance.

    Returns:
        Scalar estimation loss.
    """
    vel_loss = F.mse_loss(est_vel, gt_vel)
    clearance_loss = F.mse_loss(est_clearance, gt_clearance)
    return vel_loss + clearance_loss


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """KL divergence: D_KL(q(z|history) || N(0, I)).

    PIE Section 5, Loss (3).

    Closed-form KL divergence for diagonal Gaussian:
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: ``(batch, latent_dim)`` mean of the approximate posterior.
        logvar: ``(batch, latent_dim)`` log-variance of the approximate posterior.

    Returns:
        Scalar KL divergence loss (mean over batch and dimensions).
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()


def compute_pie_estimator_loss(
    estimator_output: Dict[str, torch.Tensor],
    gt_next_state: torch.Tensor,
    gt_vel: torch.Tensor,
    gt_clearance: torch.Tensor,
    gt_map: torch.Tensor,
    recon_weight: float = 1.0,
    est_weight: float = 1.0,
    kl_weight: float = 0.01,
    alive_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute the full PIE Estimator loss.

    Aggregates the three loss components with configurable weights:

        L_Estimator = recon_weight * L_recon + est_weight * L_est + kl_weight * L_KL

    Args:
        estimator_output: Output dict from :meth:`PIEEstimator.forward`.
        gt_next_state: ``(batch, 45)`` ground truth next proprioception state.
        gt_vel: ``(batch, 3)`` ground truth base linear velocity.
        gt_clearance: ``(batch, 4)`` ground truth foot clearance.
        gt_map: ``(batch, map_gt_dim)`` ground truth terrain height scan.
        recon_weight: Weight for reconstruction loss.
        est_weight: Weight for estimation loss.
        kl_weight: Weight for KL divergence loss.
        alive_mask: Optional ``(batch,)`` float tensor with 1.0 for alive envs
            and 0.0 for done envs whose GT crosses episode boundaries.
            When provided, per-env losses are masked before averaging.

    Returns:
        Dictionary with keys:

        - ``total``: Scalar weighted total loss.
        - ``reconstruction``: Scalar reconstruction loss.
        - ``estimation``: Scalar estimation loss.
        - ``kl``: Scalar KL divergence loss.
    """
    if alive_mask is not None:
        # Per-env masked loss computation
        # Reconstruction: per-env MSE then mask
        state_loss_per_env = ((estimator_output["rec_next_state"] - gt_next_state) ** 2).mean(dim=-1)
        map_loss_per_env = ((estimator_output["rec_map"] - gt_map) ** 2).mean(dim=-1)
        l_recon = ((state_loss_per_env + map_loss_per_env) * alive_mask).sum() / alive_mask.sum().clamp(min=1.0)

        # Estimation: per-env MSE then mask
        vel_loss_per_env = ((estimator_output["est_vel"] - gt_vel) ** 2).mean(dim=-1)
        clear_loss_per_env = ((estimator_output["est_clearance"] - gt_clearance) ** 2).mean(dim=-1)
        l_est = ((vel_loss_per_env + clear_loss_per_env) * alive_mask).sum() / alive_mask.sum().clamp(min=1.0)

        # KL: per-env then mask
        mu = estimator_output["latent_mu"]
        logvar = estimator_output["latent_logvar"]
        kl_per_env = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        l_kl = (kl_per_env * alive_mask).sum() / alive_mask.sum().clamp(min=1.0)
    else:
        l_recon = reconstruction_loss(
            estimator_output["rec_next_state"],
            gt_next_state,
            estimator_output["rec_map"],
            gt_map,
        )
        l_est = estimation_loss(
            estimator_output["est_vel"],
            gt_vel,
            estimator_output["est_clearance"],
            gt_clearance,
        )
        l_kl = kl_divergence_loss(
            estimator_output["latent_mu"],
            estimator_output["latent_logvar"],
        )

    total = recon_weight * l_recon + est_weight * l_est + kl_weight * l_kl

    return {
        "total": total,
        "reconstruction": l_recon,
        "estimation": l_est,
        "kl": l_kl,
    }
