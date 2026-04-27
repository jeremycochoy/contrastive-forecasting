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
from .norm import RevEWMNorm, RevIN, compute_patch_stats, PATCH_STATS_DIM
from .freq_embedding import FrequencyEmbedding


class ConfigurableModel(torch.nn.Module):
    """SimpleModel with configurable encoder and transformer.

    Parameters
    ----------
    freq_emb_dim : int
        If > 0, add a learned frequency embedding of this dimension. The
        embedding is concatenated to each patch along the time/feature axis,
        widening the encoder's input from W to W+freq_emb_dim. Default 0
        (disabled — keeps checkpoint compatibility with pre-freq-emb runs).
    num_freqs : int
        Number of freq classes (only used when freq_emb_dim > 0).
    patch_stats_kind : str
        ``'none'`` (default) — backwards compatible, no extra features.
        ``'diff'`` — append two per-patch scale-free diff stats from the
        reversible normaliser (dmean in std units, dlogstd) to each patch
        along the feature axis. Encoder input widens from W to W+2.
        ``'raw'`` — append centred mean and centred log_std as an ablation.
        Only meaningful with ``rev_norm_kind='ewma'``; with RevIN the
        per-step stats are constant and the diff/centred values are all 0.
    """
    def __init__(self, C, H, W, encoder_type='mlp', intermediate_dim=None,
                 num_layers=12, nhead=4, ffn_mult=2, dropout=0.1,
                 activation='gelu', depthwise_conv=3,
                 rev_norm_span=None, norm_type='layernorm',
                 freq_emb_dim=0, num_freqs=10,
                 rev_norm_kind='ewma',
                 patch_stats_kind='none'):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.freq_emb_dim = freq_emb_dim

        # Reversible normalization (optional). Two kinds:
        #   ewma  → RevEWMNorm(span=rev_norm_span)  (default, dynamic)
        #   revin → RevIN()                         (single per-instance z-score)
        # rev_norm_span is only used by ewma; revin ignores it.
        if rev_norm_kind == 'ewma' and rev_norm_span is not None:
            self.rev_norm = RevEWMNorm(
                num_features=C, span=rev_norm_span, patch_size=W)
        elif rev_norm_kind == 'revin':
            self.rev_norm = RevIN(num_features=C)
        elif rev_norm_kind in (None, 'none') or rev_norm_span is None:
            self.rev_norm = None
        else:
            raise ValueError(
                f"Unknown rev_norm_kind={rev_norm_kind!r} (expected 'ewma', 'revin', or 'none')")
        self.rev_norm_kind = rev_norm_kind

        if patch_stats_kind not in {'none', 'diff', 'raw'}:
            raise ValueError(f"Unknown patch_stats_kind={patch_stats_kind!r}")
        if patch_stats_kind != 'none' and self.rev_norm is None:
            raise ValueError(
                "patch_stats_kind requires a reversible normaliser; got rev_norm=None")
        if patch_stats_kind != 'none' and rev_norm_kind == 'revin':
            # RevIN's mean/std are time-invariant: per-patch diffs are
            # identically zero. Refuse rather than silently waste capacity.
            raise ValueError(
                "patch_stats_kind incompatible with rev_norm_kind='revin' "
                "(per-patch stats are constant under RevIN, so the feature "
                "carries zero information).")
        self.patch_stats_kind = patch_stats_kind
        patch_stats_dim = PATCH_STATS_DIM if patch_stats_kind != 'none' else 0

        # Frequency embedding (optional)
        if freq_emb_dim > 0:
            self.freq_embedding = FrequencyEmbedding(
                emb_dim=freq_emb_dim, num_freqs=num_freqs)
        else:
            self.freq_embedding = None

        # Encoder input width = W (patch values) + patch_stats_dim + freq_emb_dim.
        encoder_input = W + patch_stats_dim + freq_emb_dim
        self.encoder = create_encoder(
            encoder_type, encoder_input, H, intermediate_dim)

        self.transformer = TransformerBlock(
            dimension_e=H,
            nhead=nhead,
            num_layers=num_layers,
            feedforward_mult=ffn_mult,
            dropout=dropout,
            input_to_latent=self.encoder,
            depthwise_conv=depthwise_conv,
            norm_type=norm_type,
        )
        # Override activation if requested
        if activation != 'gelu':
            act_fn = torch.nn.functional.silu if activation == 'silu' else torch.nn.functional.gelu
            for layer in self.transformer.layers:
                layer.activation = act_fn

        self.channel_mixing_module = Simple_channel_mixing_module(H=H, C=C)

    def _apply_freq_embedding(self, x_patch, freq_ids=None, freq_embs=None):
        """Widen the per-patch time axis with a broadcast freq embedding.

        x_patch: [B, T, C, W]
        freq_ids: LongTensor of shape [B], if provided does a lookup.
        freq_embs: FloatTensor [B, freq_emb_dim], if provided skips lookup
                   (used by mixup to pass an interpolated embedding).

        Returns: [B, T, C, W + freq_emb_dim]
        """
        if self.freq_embedding is None:
            return x_patch
        if freq_embs is None:
            if freq_ids is None:
                raise ValueError(
                    "freq_embedding is configured but neither freq_ids nor "
                    "freq_embs was passed to forward()")
            freq_embs = self.freq_embedding(freq_ids)      # [B, E]
        # Broadcast [B, E] → [B, T, C, E]
        B, T, C, _ = x_patch.shape
        E = self.freq_emb_dim
        emb_b = freq_embs.view(B, 1, 1, E).expand(B, T, C, E)
        return torch.cat([x_patch, emb_b], dim=-1)         # [B, T, C, W+E]

    def prepare_encoder_input(self, x_norm, freq_ids=None, freq_embs=None):
        """Build the per-patch encoder input from an *already-normalised* series.

        Output shape: ``[B, T_patches, C, W + (2 if patch_stats else 0) + freq_emb_dim]``.

        The reversible-norm step has to be done by the caller — this helper
        assumes ``x_norm = rev_norm(x, 'norm')``. We split it out so that
        downstream code (``extract_*_latents`` in :mod:`src.forecasting_head`)
        can reuse the exact same patching pipeline as the train-time forward.
        """
        B, T_raw, C = x_norm.shape
        W = self.W
        assert T_raw % W == 0, f"T_raw={T_raw} must be a multiple of W={W}"
        T = T_raw // W

        x = x_norm.view(B, T, W, C).permute(0, 1, 3, 2)  # [B, T, C, W]

        # Append per-patch summary stats to the patch feature axis BEFORE
        # the freq embedding, so the order is [W | stats | freq]. This
        # matches the user-facing description "stats concatenated at the
        # end of the patch" while keeping the freq embedding as a
        # series-level (not patch-level) tail.
        if self.patch_stats_kind != 'none':
            # rev_norm is RevEWMNorm here (RevIN+stats is rejected in __init__);
            # mean and stdev are [B, T_raw, C].
            stats = compute_patch_stats(
                self.rev_norm.mean, self.rev_norm.stdev,
                W, kind=self.patch_stats_kind,
            )                                            # [B, T, C, 2]
            x = torch.cat([x, stats], dim=-1)            # [B, T, C, W+2]

        x = self._apply_freq_embedding(x, freq_ids=freq_ids, freq_embs=freq_embs)
        return x

    def forward(self, x, freq_ids=None, freq_embs=None):
        B, T_raw, C = x.shape
        W = self.W
        H = self.H
        assert T_raw % W == 0
        T = T_raw // W

        # Apply reversible normalization before patching
        if self.rev_norm is not None:
            x = self.rev_norm(x, mode='norm')

        x = self.prepare_encoder_input(x, freq_ids=freq_ids, freq_embs=freq_embs)
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
