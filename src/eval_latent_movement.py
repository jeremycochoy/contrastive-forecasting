"""Post-hoc latent-movement diagnostic (#379).

Loads a backbone checkpoint into a :class:`~src.models.ConfigurableModel`,
runs a fixed held-out batch through it, and returns the encoder-output
latent ``h_t`` and the patch-embedding latent ``e_t`` in ``[B, T, C, H]``
layout — the same layout the trainer's loss operates on.

Latent movement between two checkpoints of the same arm is then

    movement_h = mean over (b, t, c) of 1 - cos(h_prev, h_next)
    movement_e = mean over (b, t, c) of 1 - cos(e_prev, e_next)

``compute_latents`` runs under ``torch.no_grad()`` and requires the model
to be in ``eval()``; ``load_backbone`` does both.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import ConfigurableModel


def load_backbone(ckpt_path: str, model_kwargs: dict,
                  device: str | torch.device) -> ConfigurableModel:
    """Build a ConfigurableModel from ``model_kwargs``, load the state dict, eval mode."""
    model = ConfigurableModel(**model_kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # ncpc arms (cpc-off) skip building cpc_w1/cpc_w2 heads → tolerate
    # missing cpc_* keys since compute_latents doesn't touch them.
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def compute_latents(model: ConfigurableModel, x: torch.Tensor,
                    freq_ids: torch.Tensor | None = None,
                    seasonality_ids: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(h_t, e_t)`` from one forward pass.

    ``x`` is a raw ``[B, T_raw, C]`` batch; the reversible normaliser +
    patch splitter run inside so downstream axes match the training path.
    Both returned tensors have shape ``[B, T_patches, C, H]``.
    """
    B, T_raw, C = x.shape
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    xr = model.prepare_encoder_input(
        x, freq_ids=freq_ids, seasonality_ids=seasonality_ids)
    _, o_flat, e_lat = model.transformer(xr, return_embed=True)
    T = T_raw // model.W
    H = model.H
    h_t = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    return h_t.float(), e_lat.float()


def mean_one_minus_cos(a: torch.Tensor, b: torch.Tensor,
                       dim: int = -1) -> float:
    """Mean over all axes except ``dim`` of ``1 - cos(a, b)`` on ``dim``.

    Uses ``F.cosine_similarity`` (default eps=1e-8) so zero-norm rows
    return a defined value rather than NaN.
    """
    cos = F.cosine_similarity(a.float(), b.float(), dim=dim)
    return (1.0 - cos).mean().item()


def small_backbone_kwargs(C: int = 1, freq_emb_dim: int = 3,
                          seasonality_emb_dim: int = 3) -> dict:
    """Model kwargs matching the #379 small-backbone recipe.

    Every arm in ``experiments/2026-07-21_split_pred_rep_small/`` uses
    the exact same architecture (only loss flags differ), so a single
    factory suffices for reloading their checkpoints.
    """
    return dict(
        C=C, H=64, W=16,
        encoder_type='gru',
        num_layers=3, nhead=8,
        num_encoder_layers=3,
        ffn_mult=4.0, activation='gelu',
        depthwise_conv=3, deprecated_depthwise_conv=0, dropout=0.1,
        rev_norm_kind='ewma', rev_norm_span=128,
        freq_emb_dim=freq_emb_dim, seasonality_emb_dim=seasonality_emb_dim,
        encoder_dropkey=0.70,
        encoder_dropkey_share_heads=True,
        encoder_dropkey_share_layers=True,
        residual_dtype='fp32', attn_dtype='fp16',
        ffn_dtype='fp16', conv_dtype='fp16', patch_emb_dtype='fp32',
        qk_norm=True, attn_out_norm=True, log_attn_amplitude=True,
        ema_embedding=True, ema_encoder=True,
        cpc_infonce=True,
    )
