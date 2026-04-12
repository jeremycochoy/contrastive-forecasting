"""
Model definitions for contrastive forecasting.

Contains the ConfigurableModel (configurable encoder + transformer backbone)
and helper functions for batch generation and metric computation.
"""

import torch
import torch.nn.functional as F

from .arma import generate_arma_batch
from .encoders import create_encoder
from .blocks import TransformerBlock, Simple_channel_mixing_module
from .norm import RevEWMNorm


class ConfigurableModel(torch.nn.Module):
    """SimpleModel with configurable encoder and transformer."""
    def __init__(self, C, H, W, encoder_type='mlp', intermediate_dim=None,
                 num_layers=12, nhead=4, ffn_mult=2, dropout=0.1,
                 activation='gelu', depthwise_conv=3,
                 rev_norm_span=None):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

        # Reversible EWM normalization (optional)
        if rev_norm_span is not None:
            self.rev_norm = RevEWMNorm(
                num_features=C, span=rev_norm_span, patch_size=W)
        else:
            self.rev_norm = None

        self.encoder = create_encoder(encoder_type, W, H, intermediate_dim)

        self.transformer = TransformerBlock(
            dimension_e=H,
            nhead=nhead,
            num_layers=num_layers,
            feedforward_mult=ffn_mult,
            dropout=dropout,
            input_to_latent=self.encoder,
            depthwise_conv=depthwise_conv,
        )
        # Override activation if requested
        if activation != 'gelu':
            act_fn = torch.nn.functional.silu if activation == 'silu' else torch.nn.functional.gelu
            for layer in self.transformer.layers:
                layer.activation = act_fn

        self.channel_mixing_module = Simple_channel_mixing_module(H=H, C=C)

    def forward(self, x):
        B, T_raw, C = x.shape
        W = self.W
        H = self.H
        assert T_raw % W == 0
        T = T_raw // W

        # Apply reversible normalization before patching
        if self.rev_norm is not None:
            x = self.rev_norm(x, mode='norm')

        x = x.view(B, T, W, C).permute(0, 1, 3, 2)  # [B, T, C, W]
        x, x_original = self.transformer(x)
        x = x.reshape(B, C, T, H).permute(0, 2, 1, 3).reshape(B, T, C * H)
        x_original = x_original.reshape(B, C, T, H).permute(0, 2, 1, 3)
        x = self.channel_mixing_module(x)
        x = x.reshape(B, T, C, H)
        return x, x_original


def generate_random_batch(batch_size=16, T_raw=4096, C=4, seed=None, dimension=4):
    """Generate a random ARMA batch (discarding parameters)."""
    X, _ = generate_arma_batch(batch_size=batch_size, T_raw=T_raw, C=C, seed=seed, dimension=dimension)
    return X


def compute_metrics(f_lat, o_lat, cld):
    """Compute contrastive metrics: FF, FP, TP, and cross-batch similarity."""
    fn = F.normalize(f_lat, p=2, dim=-1)
    on = F.normalize(o_lat, p=2, dim=-1)
    hyh = fn[:, :-cld, :, :]
    hyn = on[:, cld:, :, :]
    hxn = on[:, :-cld, :, :]

    ff = (hyh * hyn).sum(-1).mean().item()
    fp = (hyh * hxn).sum(-1).mean().item()
    tp = (hyn * hxn).sum(-1).mean().item()

    B, T, C, H = hyh.shape
    hyh_exp = hyh.unsqueeze(0)
    hyn_exp = hyn.unsqueeze(1)
    sims_cross_batch = (hyh_exp * hyn_exp).sum(-1)
    mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
    mask_batch = mask_batch.view(B, B, 1, 1)
    sims_masked = sims_cross_batch.masked_fill(~mask_batch, 0)
    cross_batch = sims_masked.mean().item()

    return ff, fp, tp, cross_batch


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
