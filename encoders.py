"""
Encoder variants for architecture search.
Each encoder maps [B, T, C, W] -> [B, T, C, H].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def create_encoder(encoder_type, W, H, intermediate_dim=None):
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
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
