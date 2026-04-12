"""
Recovery head models for ARMA parameter recovery.

Contains multiple architectures (MLP, GRU, ResidualMLP, Attention, etc.)
and a factory function to create them by name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Model Architectures
# =============================================================================

class ParameterRecoveryHead(nn.Module):
    """Original MLP head from the notebook. Works per-timestep on H dimension."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(H, hidden_dim),
            nn.CELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(0.1),
        )
        self.ar_head = nn.Linear(hidden_dim, num_arma_params)
        self.ma_head = nn.Linear(hidden_dim, num_arma_params)

    def forward(self, x):
        """x: [B*C, T, H] -> ar_params, ma_params: [B*C, T, num_arma_params]"""
        shared = self.shared_layers(x)
        return torch.tanh(self.ar_head(shared)), torch.tanh(self.ma_head(shared))


class GRURecoveryHead(nn.Module):
    """GRU-based model that processes the temporal dimension to aggregate information
    before predicting parameters. The idea: ARMA parameters are constant across time,
    so a GRU can accumulate evidence over patches and produce a better estimate."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4, num_gru_layers=2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        dir_mult = 2 if bidirectional else 1

        # Project down from H to hidden_dim first (saves GRU parameters)
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
            dropout=0.1 if num_gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * dir_mult

        self.output_layers = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.ar_head = nn.Linear(hidden_dim, num_arma_params)
        self.ma_head = nn.Linear(hidden_dim, num_arma_params)

    def forward(self, x):
        """x: [B*C, T, H] -> ar_params, ma_params: [B*C, T, num_arma_params]"""
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        features = self.output_layers(gru_out)
        return torch.tanh(self.ar_head(features)), torch.tanh(self.ma_head(features))


class ResidualMLPRecoveryHead(nn.Module):
    """Deeper MLP with residual connections. More expressive than the original
    3-layer MLP while still being stable to train."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4, num_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(H, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(0.05),
            ))
        self.block_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_blocks)])

        self.ar_head = nn.Linear(hidden_dim, num_arma_params)
        self.ma_head = nn.Linear(hidden_dim, num_arma_params)

    def forward(self, x):
        """x: [B*C, T, H] -> ar_params, ma_params: [B*C, T, num_arma_params]"""
        x = self.input_proj(x)
        for block, norm in zip(self.blocks, self.block_norms):
            x = norm(x + block(x))
        return torch.tanh(self.ar_head(x)), torch.tanh(self.ma_head(x))


class AttentionRecoveryHead(nn.Module):
    """Uses self-attention over the time dimension to aggregate temporal information
    before predicting parameters. The parameters are constant, so attention can learn
    which time steps are most informative."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4, nhead=4, num_attn_layers=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(H, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.ar_head = nn.Linear(hidden_dim, num_arma_params)
        self.ma_head = nn.Linear(hidden_dim, num_arma_params)

    def forward(self, x):
        """x: [B*C, T, H] -> ar_params, ma_params: [B*C, T, num_arma_params]"""
        x = self.input_proj(x)
        x = self.transformer(x)
        features = self.output_layers(x)
        return torch.tanh(self.ar_head(features)), torch.tanh(self.ma_head(features))


class GRUPoolRecoveryHead(nn.Module):
    """GRU that processes the full sequence and outputs a single prediction per channel.
    Uses the final hidden state (or mean pool of all states) for prediction.
    This is arguably the right approach: since ARMA params are constant over time,
    we should aggregate first, then predict."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4, num_gru_layers=2, pool='mean'):
        super().__init__()
        self.pool = pool

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
            dropout=0.1 if num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.ar_head = nn.Linear(hidden_dim, num_arma_params)
        self.ma_head = nn.Linear(hidden_dim, num_arma_params)

    def forward(self, x):
        """x: [B*C, T, H] -> ar_params, ma_params: [B*C, num_arma_params]
        Note: returns [B*C, 1, num_arma_params] to match interface (broadcast over T)."""
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)  # [B*C, T, hidden*2]

        if self.pool == 'mean':
            pooled = gru_out.mean(dim=1)  # [B*C, hidden*2]
        elif self.pool == 'last':
            pooled = gru_out[:, -1, :]
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        features = self.output_layers(pooled)  # [B*C, hidden]
        ar = torch.tanh(self.ar_head(features))  # [B*C, num_arma_params]
        ma = torch.tanh(self.ma_head(features))
        # Unsqueeze to [B*C, 1, num_arma_params] for compatibility
        return ar.unsqueeze(1), ma.unsqueeze(1)


class DeepGRURecoveryHead(nn.Module):
    """GRU with deeper non-linear processing. Uses SiLU (Swish) activations
    which have been shown to work better than GELU/ReLU for regression tasks.
    Has separate per-coefficient output heads to allow specialization."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4, num_gru_layers=3):
        super().__init__()
        # Two-stage projection to give the network more capacity to extract features
        self.input_proj = nn.Sequential(
            nn.Linear(H, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

        gru_out = hidden_dim * 2  # bidirectional

        # Deep non-linear output with residual
        self.mid_proj = nn.Sequential(
            nn.Linear(gru_out, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Per-coefficient heads for AR and MA
        self.ar_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_arma_params)
        ])
        self.ma_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_arma_params)
        ])
        self.num_arma_params = num_arma_params

    def forward(self, x):
        """x: [B*C, T, H] -> ar_params, ma_params: [B*C, T, num_arma_params]"""
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        x = self.mid_proj(gru_out)
        x = self.norm1(x + self.res_block1(x))
        x = self.norm2(x + self.res_block2(x))

        ar = torch.cat([head(x) for head in self.ar_heads], dim=-1)
        ma = torch.cat([head(x) for head in self.ma_heads], dim=-1)
        return torch.tanh(ar), torch.tanh(ma)


class DeepGRUPoolRecoveryHead(nn.Module):
    """Like DeepGRU but with global pooling - predicts one set of parameters per channel.
    Uses attention-weighted pooling over time dimension."""

    def __init__(self, H=1024, hidden_dim=256, num_arma_params=4, num_gru_layers=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(H, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

        gru_out = hidden_dim * 2

        # Attention-weighted pooling
        self.attn_score = nn.Sequential(
            nn.Linear(gru_out, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Deep output
        self.output = nn.Sequential(
            nn.Linear(gru_out, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.ar_head = nn.Linear(hidden_dim, num_arma_params)
        self.ma_head = nn.Linear(hidden_dim, num_arma_params)

    def forward(self, x):
        """x: [B*C, T, H] -> ar, ma: [B*C, 1, num_arma_params]"""
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)  # [B*C, T, hidden*2]

        # Attention-weighted pooling
        scores = self.attn_score(gru_out)  # [B*C, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (gru_out * weights).sum(dim=1)  # [B*C, hidden*2]

        features = self.output(pooled)
        ar = torch.tanh(self.ar_head(features))
        ma = torch.tanh(self.ma_head(features))
        return ar.unsqueeze(1), ma.unsqueeze(1)


# =============================================================================
# Factory and loss
# =============================================================================

def create_recovery_head(model_type, H=1024, hidden_dim=256, num_arma_params=4, num_gru_layers=None):
    """Factory function to create recovery heads."""
    gru_kwargs = {}
    if num_gru_layers is not None:
        gru_kwargs['num_gru_layers'] = num_gru_layers
    if model_type == 'mlp':
        return ParameterRecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params)
    elif model_type == 'gru':
        return GRURecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params, **gru_kwargs)
    elif model_type == 'resmlp':
        return ResidualMLPRecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params)
    elif model_type == 'attention':
        return AttentionRecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params)
    elif model_type == 'grupool':
        return GRUPoolRecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params, **gru_kwargs)
    elif model_type == 'deepgru':
        return DeepGRURecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params, **gru_kwargs)
    elif model_type == 'deepgrupool':
        return DeepGRUPoolRecoveryHead(H=H, hidden_dim=hidden_dim, num_arma_params=num_arma_params, **gru_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: mlp, gru, resmlp, attention, grupool, deepgru, deepgrupool")


def parameter_loss(pred_ar, pred_ma, true_ar, true_ma):
    """Compute loss between predicted and true parameters."""
    # pred shapes: [B*C, T, num_arma_params] or [B*C, 1, num_arma_params]
    pred_ar_avg = pred_ar.mean(dim=1)
    pred_ma_avg = pred_ma.mean(dim=1)

    ar_loss = F.mse_loss(pred_ar_avg, true_ar)
    ma_loss = F.mse_loss(pred_ma_avg, true_ma)

    return ar_loss + ma_loss, ar_loss, ma_loss
