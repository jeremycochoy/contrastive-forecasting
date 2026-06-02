"""
Joint-channel backbone for the contrastive-correlation experiment.

Unlike `ConfigurableModel` (which runs the transformer per-channel via
[B*C, T, H] reshaping), this model concatenates the C channel features and
projects them down before the transformer, so the transformer sees a
**single sequence per sample** with channels living in the feature dim.

Forward shape pipeline:
    x: [B, T_raw, C]
        → patch:               [B, T, C, W]   (W = patch size)
        → encoder (per-chan):  [B, T, C, H]
        → flatten + project:   [B, T, H_joint]
        → transformer:         [B, T, H_joint]   (causal)
    Returns f, o each shaped [B, T, H_joint], and the loss code that expects
    [B, T, C, H] is handed [B, T, 1, H_joint] (a degenerate C=1) so its
    cross-channel negatives vanish — leaving only the cross-batch term.

A small recovery head (`GRUCorrelationHead`) consumes the [B, T, H_joint]
latent and returns `K(K-1)/2` pairwise correlations per sample. The head is
the ARMA-winning recipe (GRU h=128, 2 layers, bidirectional, mean-pool over
time) adapted to a single 6-output linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.encoders import create_encoder
from src.blocks import TransformerBlock


class JointChannelModel(nn.Module):
    def __init__(
        self,
        C: int = 4,
        H: int = 1024,
        W: int = 32,
        encoder_type: str = "gru",
        intermediate_dim: int | None = None,
        num_layers: int = 12,
        nhead: int = 8,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
        depthwise_conv: int = 3,
    ):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

        # Per-channel patch encoder: [B, T, C, W] → [B, T, C, H].
        self.encoder = create_encoder(encoder_type, W, H, intermediate_dim)
        # Joint projection: collapse channels into the feature dim.
        self.joint_proj = nn.Linear(C * H, H)
        # Causal transformer over a single sequence per sample.
        # input_to_latent=None because we already produced [B, T, H].
        self.transformer = TransformerBlock(
            dimension_e=H,
            nhead=nhead,
            num_layers=num_layers,
            feedforward_mult=ffn_mult,
            dropout=dropout,
            input_to_latent=None,
            depthwise_conv=depthwise_conv,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T_raw, C]
        B, T_raw, C = x.shape
        assert C == self.C, f"expected {self.C} channels, got {C}"
        assert T_raw % self.W == 0, f"T_raw {T_raw} not divisible by W {self.W}"
        T = T_raw // self.W

        # Patch and encode per channel.
        x = x.view(B, T, self.W, C).permute(0, 1, 3, 2)  # [B, T, C, W]
        x = self.encoder(x)                              # [B, T, C, H]
        x = x.flatten(2)                                 # [B, T, C*H]
        x = self.joint_proj(x)                           # [B, T, H]

        # TransformerBlock.forward expects [B, T, C', X]; pass C'=1 so the
        # internal reshape becomes [B, T, H] (a no-op in batch terms).
        x = x.unsqueeze(2)                               # [B, T, 1, H]
        f, o = self.transformer(x)                       # [B*1, T, H]
        # Return shape compatible with the existing loss: [B, T, 1, H].
        f = f.view(B, T, 1, self.H)
        o = o.view(B, T, 1, self.H)
        return f, o


class GRUCorrelationHead(nn.Module):
    """ARMA-winning recipe (GRU h=128, 2 layers, bidirectional, mean-pool)
    adapted to output K(K-1)/2 pairwise correlations from a [B, T, H] joint
    latent."""

    def __init__(
        self,
        H: int = 1024,
        K: int = 4,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
        init_bias: float = 0.45,
    ):
        super().__init__()
        self.K = K
        self.num_pairs = K * (K - 1) // 2

        self.input_proj = nn.Sequential(
            nn.Linear(H, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        gru_out = hidden_dim * 2
        self.output_layers = nn.Sequential(
            nn.Linear(gru_out, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, self.num_pairs)
        with torch.no_grad():
            self.out.bias.fill_(float(init_bias))
            self.out.weight.mul_(0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, H] → [B, num_pairs] in [0, 1]."""
        x = self.input_proj(h)               # [B, T, hidden]
        gru_out, _ = self.gru(x)             # [B, T, hidden*2]
        pooled = gru_out.mean(dim=1)         # [B, hidden*2]
        feat = self.output_layers(pooled)    # [B, hidden]
        return self.out(feat).clamp(0.0, 1.0)
