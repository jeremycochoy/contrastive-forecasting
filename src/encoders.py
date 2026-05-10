"""
Encoder variants for architecture search.
Each encoder maps [B, T, C, W] -> [B, T, C, H].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DecoderOnlyTransformerLayer


class MLPEncoder(nn.Module):
    """Original Simple_encoder: Linear->ReLU->Linear + skip + LayerNorm."""
    def __init__(self, W, H, intermediate_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(W, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, H)
        self.linear_skipping = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x1 = self.linear2(x1)
        x2 = self.linear_skipping(x)
        return self.layer_norm(x1 + x2)


class MLPWideEncoder(nn.Module):
    """Wider intermediate dim (256 instead of 64)."""
    def __init__(self, W, H, intermediate_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(W, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, H)
        self.linear_skipping = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        x1 = F.silu(self.linear1(x))
        x1 = self.linear2(x1)
        x2 = self.linear_skipping(x)
        return self.layer_norm(x1 + x2)


class ResidualSiLUEncoder(nn.Module):
    """
    TimeFM-inspired residual block: project to H, then residual MLP with SiLU.
    Linear(W->H) -> SiLU -> Linear(H->H) + skip(W->H) -> LayerNorm
    """
    def __init__(self, W, H, intermediate_dim=None):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = H
        self.proj = nn.Linear(W, H)
        self.mlp = nn.Sequential(
            nn.Linear(H, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, H),
        )
        self.skip = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        h = self.proj(x)
        h = h + self.mlp(h)  # residual within H-space
        s = self.skip(x)
        return self.layer_norm(h + s)


class GRUEncoder(nn.Module):
    """
    GRU processes each patch as a sequence of W scalar time steps.
    Captures temporal ordering within the patch.
    """
    def __init__(self, W, H, intermediate_dim=128, num_gru_layers=2):
        super().__init__()
        self.W = W
        self.gru = nn.GRU(
            input_size=1, hidden_size=intermediate_dim,
            num_layers=num_gru_layers, batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(intermediate_dim * 2, H)  # bidirectional
        self.skip = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        # x: [B, T, C, W]
        shape = x.shape[:-1]  # [B, T, C]
        flat = x.reshape(-1, self.W, 1)  # [B*T*C, W, 1]
        _, hidden = self.gru(flat)  # hidden: [num_layers*2, B*T*C, intermediate_dim]
        # Take last layer's forward and backward hidden states
        h = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [B*T*C, intermediate_dim*2]
        h = self.proj(h)  # [B*T*C, H]
        h = h.reshape(*shape, -1)  # [B, T, C, H]
        s = self.skip(x)
        return self.layer_norm(h + s)


class ConvEncoder(nn.Module):
    """
    1D CNN processes each patch. Captures local patterns within patch.
    """
    def __init__(self, W, H, intermediate_dim=128):
        super().__init__()
        self.W = W
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, intermediate_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(intermediate_dim, H)
        self.skip = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        shape = x.shape[:-1]  # [B, T, C]
        flat = x.reshape(-1, 1, self.W)  # [B*T*C, 1, W]
        h = F.silu(self.conv1(flat))     # [B*T*C, 64, W]
        h = F.silu(self.conv2(h))        # [B*T*C, 128, W]
        h = self.pool(h).squeeze(-1)     # [B*T*C, 128]
        h = self.proj(h)                 # [B*T*C, H]
        h = h.reshape(*shape, -1)        # [B, T, C, H]
        s = self.skip(x)
        return self.layer_norm(h + s)


class TransformerEncoder(nn.Module):
    """N-layer decoder-only transformer encoder, processes ONE PATCH AT A TIME.

    Drop-in for GRU+skip: the W' (= W + patch_stats + freq_emb + seasonality_emb)
    points within a patch are the SEQUENCE the transformer attends over —
    NOT the T patch axis. Each (B, T, C) triple is encoded independently
    in parallel, exactly like the GRU.

    Pipeline per forward:
      [B, T, C, W']  → reshape to [B*T*C, W', 1]    # each scalar = one token
                     → Linear(1 → H)                # upscale scalar → latent
                     → N decoder-only causal layers # attention OVER W', not T
                     → take last token  [B*T*C, H]  # patch summary
                     → reshape to [B, T, C, H]

    "Highway" at init: with norm-first residual chain, attention/FFN
    contributions are near-zero at init, so the encoder approximates a
    plain Linear(1, H) lookup of the last scalar in the patch — same shape
    of inductive bias as the GRU's `linear_skipping` skip path.
    """

    def __init__(self, W, H, num_layers=4, nhead=6, ffn_mult=4,
                 dropout=0.0, depthwise_conv=3, norm_type='layernorm',
                 activation='gelu'):
        super().__init__()
        # W is recorded only for diagnostics; the linear is per-scalar
        # (1 -> H), so the layer doesn't depend on patch width.
        self.W = W
        # Per-scalar upscale: each of the W' positions in a patch is treated
        # as a 1-D token and projected to the H-dim transformer latent.
        self.linear_up = nn.Linear(1, H)
        dim_feedforward = int(ffn_mult * H)
        self.layers = nn.ModuleList([
            DecoderOnlyTransformerLayer(
                d_model=H,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                batch_first=True,
                norm_first=True,
                norm_type=norm_type,
                bias=False,
                dropout=dropout,
                depthwise_conv=depthwise_conv,
            ) for _ in range(num_layers)
        ])
        self.causal_mask = None

    def _get_causal_mask(self, L, device):
        if (self.causal_mask is None
                or self.causal_mask.size(0) != L
                or self.causal_mask.device != device):
            mask = (torch.triu(torch.ones(L, L, device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')) \
                               .masked_fill(mask == 1, float(0.0))
            self.causal_mask = mask
        return self.causal_mask

    def forward(self, x):
        # x: [B, T, C, W']  — W' = W + patch_stats + freq_emb + seasonality_emb
        B, T, C, Wp = x.shape
        # Each of the W' scalars per patch becomes its own token. Patches
        # are encoded independently in parallel along the (B, T, C) axes.
        h = x.reshape(B * T * C, Wp, 1)              # [N, W', 1]
        h = self.linear_up(h)                        # [N, W', H]
        H = h.shape[-1]

        mask = self._get_causal_mask(Wp, h.device)
        for layer in self.layers:
            h = layer(h, tgt_mask=mask, tgt_is_causal=True)

        # Patch summary = last token (causal attention; position W'-1 has
        # access to all positions 0..W'-1).
        h = h[:, -1, :]                              # [N, H]
        h = h.reshape(B, T, C, H)                    # [B, T, C, H]
        return h


def create_encoder(encoder_type, W, H, intermediate_dim=None,
                   transformer_num_layers=4, transformer_nhead=6,
                   transformer_ffn_mult=4, transformer_dropout=0.0,
                   transformer_depthwise_conv=3):
    """Factory function for encoder creation."""
    if encoder_type == 'mlp':
        return MLPEncoder(W, H, intermediate_dim=intermediate_dim or 64)
    elif encoder_type == 'mlp_wide':
        return MLPWideEncoder(W, H, intermediate_dim=intermediate_dim or 256)
    elif encoder_type == 'residual_silu':
        return ResidualSiLUEncoder(W, H, intermediate_dim=intermediate_dim)
    elif encoder_type == 'gru':
        return GRUEncoder(W, H, intermediate_dim=intermediate_dim or 128)
    elif encoder_type == 'conv':
        return ConvEncoder(W, H, intermediate_dim=intermediate_dim or 128)
    elif encoder_type == 'transformer':
        return TransformerEncoder(
            W, H,
            num_layers=transformer_num_layers,
            nhead=transformer_nhead,
            ffn_mult=transformer_ffn_mult,
            dropout=transformer_dropout,
            depthwise_conv=transformer_depthwise_conv,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
