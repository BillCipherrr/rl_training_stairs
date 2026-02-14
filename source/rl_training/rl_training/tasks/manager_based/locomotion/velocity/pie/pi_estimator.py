# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PIE Estimator Network for Dual-level Implicit-Explicit Learning.

This module implements the core PIE Estimator, which processes proprioception
history (H1=10 frames) and depth image history (H2=2 frames) to produce:
- Explicit estimates: base velocity (3D), foot clearance (4D)
- Implicit latent representations: state VAE latent (z_t), terrain map encoding (z^m_t)
- Reconstructions: next state (o_{t+1}), terrain map (m_t)

The architecture consists of:
1. ProprioEncoder: MLP encoding flattened proprioception history
2. DepthEncoder: CNN encoding stacked depth frames
3. FusionTransformer: Transformer + GRU for cross-modal fusion with temporal memory
4. Output Heads: Separate linear heads for each prediction target

References:
    - PIE: Parkour with Implicit-Explicit Learning Framework
    - Spec: docs/PIE.md Section 4
    - Instructions: docs/PIE_instruction.md Section 1
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):
    """MLP encoder for proprioception history.

    Encodes a flattened proprioception history into a fixed-size feature vector.

    Args:
        input_dim: Single-frame proprioception dimension (default 45).
        history_length: Number of history frames H1 (default 10, PIE paper).
        hidden_dim: Hidden layer width.
        output_dim: Output feature dimension.
    """

    def __init__(
        self,
        input_dim: int = 45,
        history_length: int = 10,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.history_length = history_length
        flat_dim = input_dim * history_length  # 45 * 10 = 450

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ELU(),
        )

    def forward(self, proprio_history: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            proprio_history: Shape ``(batch, H1, 45)`` — proprioception history.

        Returns:
            Feature tensor of shape ``(batch, output_dim)``.
        """
        batch_size = proprio_history.shape[0]
        x = proprio_history.reshape(batch_size, -1)  # (batch, H1 * 45)
        return self.mlp(x)


class DepthEncoder(nn.Module):
    """CNN encoder for depth image history.

    Processes stacked depth frames through convolutional layers followed by
    a fully connected projection.

    Args:
        in_channels: Number of input channels per frame (default 1 for depth).
        history_length: Number of depth frames H2 (default 2, PIE paper).
        output_dim: Output feature dimension.
        image_height: Depth image height in pixels.
        image_width: Depth image width in pixels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        history_length: int = 2,
        output_dim: int = 128,
        image_height: int = 64,
        image_width: int = 64,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        total_channels = in_channels * history_length  # stack along channel dim

        self.cnn = nn.Sequential(
            nn.Conv2d(total_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )

        # Compute flattened CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, total_channels, image_height, image_width)
            cnn_out = self.cnn(dummy)
            cnn_flat_dim = cnn_out.reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, output_dim),
            nn.ELU(),
        )

    def forward(self, depth_history: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth_history: Shape ``(batch, H2, C, H, W)`` — depth image history.

        Returns:
            Feature tensor of shape ``(batch, output_dim)``.
        """
        batch_size = depth_history.shape[0]
        # Stack history frames along channel dimension: (batch, H2*C, H, W)
        x = depth_history.reshape(
            batch_size,
            -1,
            depth_history.shape[-2],
            depth_history.shape[-1],
        )
        x = self.cnn(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class FusionTransformer(nn.Module):
    """Transformer-based multi-modal feature fusion with GRU temporal memory.

    Fuses proprioception and depth features using self-attention on a 2-token
    sequence ``[proprio_feat, depth_feat]``, followed by mean pooling and a
    GRU cell for temporal memory.

    Architecture:
        Transformer Encoder (self-attention) → Mean Pool → GRU Cell

    Args:
        feature_dim: Dimension of each input feature token.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        gru_hidden_dim: GRU hidden state dimension.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        gru_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.gru = nn.GRUCell(feature_dim, gru_hidden_dim)
        self.gru_hidden_dim = gru_hidden_dim

    def forward(
        self,
        proprio_feat: torch.Tensor,
        depth_feat: torch.Tensor,
        gru_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            proprio_feat: ``(batch, feature_dim)`` from ProprioEncoder.
            depth_feat: ``(batch, feature_dim)`` from DepthEncoder.
            gru_hidden: ``(batch, gru_hidden_dim)`` or None for zero-init.

        Returns:
            Tuple of:
                - fused: ``(batch, gru_hidden_dim)`` — fused temporal feature.
                - new_hidden: ``(batch, gru_hidden_dim)`` — updated GRU hidden.
        """
        batch_size = proprio_feat.shape[0]
        # Create 2-token sequence: [proprio, depth]
        tokens = torch.stack([proprio_feat, depth_feat], dim=1)  # (batch, 2, feat)
        attended = self.transformer(tokens)  # (batch, 2, feat)

        # Pool attended features (mean over tokens)
        pooled = attended.mean(dim=1)  # (batch, feat)

        if gru_hidden is None:
            gru_hidden = torch.zeros(
                batch_size, self.gru_hidden_dim, device=pooled.device
            )

        new_hidden = self.gru(pooled, gru_hidden)  # (batch, gru_hidden_dim)
        return new_hidden, new_hidden


class PIEEstimator(nn.Module):
    """PIE Dual-level Implicit-Explicit Estimator.

    Combines proprioception and depth history to produce:

    - **Explicit estimates**: velocity ``(3)``, foot clearance ``(4)``
    - **Implicit estimates**: VAE latent ``z_t``, terrain map encoding ``z^m_t``
    - **Reconstructions**: next state ``o_{t+1}``, terrain map ``m_t``

    The estimator is trained alongside PPO but with its own loss
    (see :func:`pie_loss.compute_pie_estimator_loss`).

    Architecture follows PIE paper (docs/PIE.md Section 4):

    .. code-block:: text

        proprio_history ──► ProprioEncoder ──┐
                                             ├──► FusionTransformer ──► Heads
        depth_history ────► DepthEncoder ────┘

    Args:
        proprio_dim: Single-frame proprioception dimension.
        proprio_history_len: H1 frames for proprioception (PIE: 10).
        depth_channels: Channels per depth frame.
        depth_history_len: H2 frames for depth (PIE: 2).
        depth_height: Depth image height in pixels.
        depth_width: Depth image width in pixels.
        hidden_dim: Shared hidden dimension for encoders and fusion.
        latent_dim: VAE latent dimension.
        map_dim: Terrain map encoding dimension.
        map_gt_dim: Ground truth terrain map size (height scan points).
            Lite3 with resolution=0.07, size=[1.6, 1.0]: 23 x 15 = 345 rays.
    """

    def __init__(
        self,
        proprio_dim: int = 45,
        proprio_history_len: int = 10,  # PIE paper H1
        depth_channels: int = 1,
        depth_history_len: int = 2,  # PIE paper H2
        depth_height: int = 64,
        depth_width: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        map_dim: int = 64,
        map_gt_dim: int = 345,  # Lite3 height scan: GridPattern(res=0.07, size=[1.6,1.0]) = 23x15
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.proprio_history_len = proprio_history_len
        self.latent_dim = latent_dim
        self.map_dim = map_dim

        # --- Encoders (PIE Section 4: Encoders) ---
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim,
            history_length=proprio_history_len,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
        )
        self.depth_encoder = DepthEncoder(
            in_channels=depth_channels,
            history_length=depth_history_len,
            output_dim=hidden_dim,
            image_height=depth_height,
            image_width=depth_width,
        )

        # --- Fusion (PIE Section 4: Transformer + GRU) ---
        self.fusion = FusionTransformer(
            feature_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            gru_hidden_dim=hidden_dim,
        )

        # --- Explicit Heads (PIE Section 4: Explicit Estimation) ---
        # Velocity estimation head (3 dim)
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )
        # Foot clearance estimation head (4 dim, one per foot)
        # PIE Section 3.2: h^f_i = z_foot_i - z_terrain_at_foot_i
        self.clearance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 4),
        )

        # --- Implicit Heads (PIE Section 4: VAE) ---
        # State latent z_t
        self.latent_mu_head = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Terrain map latent z^m_t
        self.map_enc_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, map_dim),
        )

        # --- Reconstruction Decoders (PIE Section 5: Loss) ---
        # Next state reconstruction: o_{t+1}
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, proprio_dim),
        )
        # Terrain map reconstruction: m_t
        self.map_decoder = nn.Sequential(
            nn.Linear(map_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, map_gt_dim),
        )

        # GRU hidden state (managed internally during rollout)
        self._gru_hidden: Optional[torch.Tensor] = None

    def reset_hidden(self, batch_size: int, device: torch.device) -> None:
        """Reset GRU hidden state for all environments.

        Call this at the start of a rollout or when all environments reset.

        Args:
            batch_size: Number of environments.
            device: Target device.
        """
        self._gru_hidden = torch.zeros(
            batch_size, self.fusion.gru_hidden_dim, device=device
        )

    def reset_hidden_for_envs(self, env_ids: torch.Tensor) -> None:
        """Reset GRU hidden state for specific environments that were reset.

        Args:
            env_ids: 1-D tensor of environment indices to reset.
        """
        if self._gru_hidden is not None:
            self._gru_hidden[env_ids] = 0.0

    def detach_hidden(self) -> None:
        """Detach GRU hidden state from the computation graph.

        Call this between replay steps during PIE update to prevent
        back-propagation-through-time (BPTT) across steps, which would
        otherwise cause the computation graph to grow linearly with the
        number of rollout steps and lead to CUDA OOM.
        """
        if self._gru_hidden is not None:
            self._gru_hidden = self._gru_hidden.detach()

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """VAE reparameterization trick: z = mu + std * eps.

        Args:
            mu: Mean of the latent distribution, shape ``(batch, latent_dim)``.
            logvar: Log-variance, shape ``(batch, latent_dim)``.

        Returns:
            Sampled latent vector, shape ``(batch, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
        gru_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the PIE Estimator.

        Args:
            proprio_history: Shape ``(batch, H1, 45)`` — proprioception history.
            depth_history: Shape ``(batch, H2, C, H, W)`` — depth image history.
            gru_hidden: Optional externally managed GRU state.

        Returns:
            Dictionary with keys:

            - ``est_vel``: ``(batch, 3)`` estimated base velocity.
            - ``est_clearance``: ``(batch, 4)`` estimated foot clearance.
            - ``latent_mu``: ``(batch, latent_dim)`` VAE mean.
            - ``latent_logvar``: ``(batch, latent_dim)`` VAE log-variance.
            - ``z_sample``: ``(batch, latent_dim)`` sampled latent.
            - ``map_enc``: ``(batch, map_dim)`` terrain map encoding.
            - ``rec_next_state``: ``(batch, proprio_dim)`` reconstructed next state.
            - ``rec_map``: ``(batch, map_gt_dim)`` reconstructed terrain map.
            - ``gru_hidden``: ``(batch, hidden_dim)`` updated GRU state.
        """
        # Use internal hidden state if not provided
        hidden = gru_hidden if gru_hidden is not None else self._gru_hidden

        # Encode modalities
        proprio_feat = self.proprio_encoder(proprio_history)  # (batch, hidden)
        depth_feat = self.depth_encoder(depth_history)  # (batch, hidden)

        # Fuse with temporal memory
        fused, new_hidden = self.fusion(proprio_feat, depth_feat, hidden)

        # NOTE: Do NOT update self._gru_hidden here. The hidden state is
        # managed externally by the runner:
        # - During rollout: get_policy_injection() updates _gru_hidden after forward.
        # - During replay: _update_pie_estimator() restores _gru_hidden from
        #   saved snapshots before each step's forward call.
        # Updating here would cause unwanted side effects during replay.

        # --- Explicit heads ---
        est_vel = self.vel_head(fused)  # (batch, 3)
        est_clearance = self.clearance_head(fused)  # (batch, 4)

        # --- Implicit heads (VAE) ---
        latent_mu = self.latent_mu_head(fused)  # (batch, latent_dim)
        latent_logvar = self.latent_logvar_head(fused)  # (batch, latent_dim)
        # Clamp logvar to prevent numerical instability in exp()
        latent_logvar = torch.clamp(latent_logvar, min=-20.0, max=2.0)
        z_sample = self.reparameterize(latent_mu, latent_logvar)

        # Terrain map encoding
        map_enc = self.map_enc_head(fused)  # (batch, map_dim)

        # --- Reconstructions ---
        # Next state: condition on fused + sampled latent
        dec_input = torch.cat([fused, z_sample], dim=-1)  # (batch, hidden+latent)
        rec_next_state = self.state_decoder(dec_input)  # (batch, proprio_dim)

        # Terrain map reconstruction
        rec_map = self.map_decoder(map_enc)  # (batch, map_gt_dim)

        return {
            "est_vel": est_vel,
            "est_clearance": est_clearance,
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
            "z_sample": z_sample,
            "map_enc": map_enc,
            "rec_next_state": rec_next_state,
            "rec_map": rec_map,
            "gru_hidden": new_hidden,
        }

    def get_policy_injection(
        self,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
    ) -> torch.Tensor:
        """Get concatenated features to inject into the PPO policy.

        Returns the explicit + implicit estimates concatenated for policy input:
        ``[est_vel(3), est_clearance(4), map_enc(map_dim), latent_mu(latent_dim)]``

        Uses ``latent_mu`` instead of ``z_sample`` for injection to avoid
        stochastic noise from VAE reparameterization corrupting the policy input.
        The VAE sampling is only used during training (PIE loss computation).

        This corresponds to PIE Section 2.1 Policy input:
            o_policy = [o_t, v_hat_t, h_hat^f_t, z^m_t, z_t]

        Args:
            proprio_history: Shape ``(batch, H1, 45)``.
            depth_history: Shape ``(batch, H2, C, H, W)``.

        Returns:
            Injection tensor of shape ``(batch, 3 + 4 + map_dim + latent_dim)``.
        """
        with torch.no_grad():
            out = self.forward(proprio_history, depth_history)
            # Update internal GRU hidden state for the next rollout step.
            # Detach to prevent graph accumulation across rollout steps.
            self._gru_hidden = out["gru_hidden"].detach()
        # Use latent_mu (deterministic) instead of z_sample (stochastic) for
        # stable policy injection. z_sample has std~1.0 randomness that would
        # inject noise into the policy input every step.
        injection = torch.cat(
            [out["est_vel"], out["est_clearance"], out["map_enc"], out["latent_mu"]],
            dim=-1,
        )
        # Guard against NaN/Inf values that can corrupt the PPO actor
        injection = torch.clamp(injection, min=-100.0, max=100.0)
        injection = torch.nan_to_num(injection, nan=0.0, posinf=100.0, neginf=-100.0)
        return injection
