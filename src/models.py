"""
Model definitions for contrastive forecasting.

Contains the ConfigurableModel (configurable encoder + transformer backbone)
and helper functions for batch generation and metric computation.
"""

import math
import torch
import torch.nn.functional as F

from .arma import generate_arma_batch
from .encoders import create_encoder
from .blocks import TransformerBlock, Simple_channel_mixing_module, AttentionChannelMixing
from .norm import RevEWMNorm, RevIN, compute_patch_stats, PATCH_STATS_DIM
from .freq_embedding import (
    FrequencyEmbedding, SeasonalityEmbedding,
    NUM_FREQS, NUM_SEASONALITIES,
)


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
    seasonality_emb_dim : int
        If > 0, add a parallel learned seasonality embedding (period in
        samples bucket). Concatenated to each patch alongside the freq
        embedding, widening the encoder's input by an additional
        ``seasonality_emb_dim``. Default 0 (disabled).
    num_seasonalities : int
        Number of seasonality classes (only used when ``seasonality_emb_dim > 0``).
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
                 activation='gelu', depthwise_conv: int = 3,
                 deprecated_depthwise_conv: int = 0,
                 rev_norm_span=None, norm_type='layernorm',
                 freq_emb_dim=0, num_freqs=NUM_FREQS,
                 seasonality_emb_dim=0, num_seasonalities=NUM_SEASONALITIES,
                 rev_norm_kind='ewma',
                 patch_stats_kind='none',
                 learnable_tau=False, tau_init=0.07,
                 enc_transformer_num_layers=4,
                 enc_transformer_nhead=6,
                 enc_transformer_ffn_mult=4,
                 enc_transformer_dropout=0.0,
                 enc_transformer_depthwise_conv=3,
                 enc_transformer_chunk_size=8192,
                 enc_transformer_use_grad_checkpoint=True,
                 num_encoder_layers=0,
                 encoder_dropkey: float = 0.0,
                 encoder_dropkey_share_heads: bool = False,
                 encoder_dropkey_share_layers: bool = False,
                 residual_dtype: str = "fp32",
                 attn_dtype: str = "fp32",
                 ffn_dtype: str = "fp32",
                 conv_dtype: str | None = None,
                 patch_emb_dtype: str = "fp32",
                 forecaster_d_model: int | None = None,
                 forecaster_n_heads: int | None = None,
                 forecaster_kind: str = "transformer",
                 cpc_k_steps: int = 12,
                 qk_norm: bool = False,
                 log_attn_amplitude: bool = False,
                 channel_mixing_kind='simple', channel_mixing_n_heads=8):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.freq_emb_dim = freq_emb_dim
        self.seasonality_emb_dim = seasonality_emb_dim

        # Optional CLIP-style learnable temperature for the contrastive loss.
        # Stored as `log_inv_tau` so τ = exp(-log_inv_tau). Trainer is
        # responsible for clamping log_inv_tau to [0, log(100)] after each
        # optimizer.step (→ τ ∈ [0.01, 1.0]). When learnable_tau=False the
        # parameter is not registered; loss falls back to the dict-driven
        # scalar.
        self.learnable_tau = bool(learnable_tau)
        if self.learnable_tau:
            self.log_inv_tau = torch.nn.Parameter(
                torch.tensor(math.log(1.0 / float(tau_init)),
                             dtype=torch.float32))

        # Reversible normalization (optional). Two kinds:
        #   ewma  → RevEWMNorm(span=rev_norm_span)  (default, dynamic)
        #   revin → RevIN()                         (single per-instance z-score)
        # rev_norm_span is only used by ewma; revin ignores it.
        if rev_norm_kind == 'ewma' and rev_norm_span is not None:
            self.rev_norm = RevEWMNorm(
                num_features=C, span=rev_norm_span, patch_size=W,
                patch_emb_dtype=patch_emb_dtype)
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

        # Seasonality embedding (optional, parallel to freq)
        if seasonality_emb_dim > 0:
            self.seasonality_embedding = SeasonalityEmbedding(
                emb_dim=seasonality_emb_dim,
                num_seasonalities=num_seasonalities)
        else:
            self.seasonality_embedding = None

        # Encoder input width: W (patch values) + patch_stats + freq + seasonality.
        encoder_input = W + patch_stats_dim + freq_emb_dim + seasonality_emb_dim
        self.encoder = create_encoder(
            encoder_type, encoder_input, H, intermediate_dim,
            transformer_num_layers=enc_transformer_num_layers,
            transformer_nhead=enc_transformer_nhead,
            transformer_ffn_mult=enc_transformer_ffn_mult,
            transformer_dropout=enc_transformer_dropout,
            transformer_depthwise_conv=enc_transformer_depthwise_conv,
            transformer_chunk_size=enc_transformer_chunk_size,
            transformer_use_grad_checkpoint=enc_transformer_use_grad_checkpoint,
            patch_emb_dtype=patch_emb_dtype)

        # Forecaster bottleneck (#286 follow-up, v13). When
        # `forecaster_d_model` is None, the forecaster runs at the encoder's
        # `H`/`nhead` (legacy behaviour). When set, the forecaster's
        # `self.layers` are constructed at the smaller dim with
        # `forecaster_n_heads`, wrapped by Linear down/up projections so the
        # JEPA contrastive loss still operates in the encoder's d_model space.
        # Validation: `forecaster_d_model % forecaster_n_heads == 0`.
        eff_fcst_d_model = H if forecaster_d_model is None else int(forecaster_d_model)
        eff_fcst_n_heads = nhead if forecaster_n_heads is None else int(forecaster_n_heads)
        if eff_fcst_d_model % eff_fcst_n_heads != 0:
            raise ValueError(
                f"forecaster_d_model={eff_fcst_d_model} must be divisible by "
                f"forecaster_n_heads={eff_fcst_n_heads}")
        self.forecaster_d_model = eff_fcst_d_model
        self.forecaster_n_heads = eff_fcst_n_heads
        # Forecaster variant (#316). "transformer" (default) is the legacy
        # causal forecaster; "linear_cpc" replaces it with K linear heads
        # predicting h_{t+1..t+K}. Stored so forward_step / downstream code
        # can branch on the multi-step output contract.
        self.forecaster_kind = forecaster_kind
        self.cpc_k_steps = int(cpc_k_steps)
        self.transformer = TransformerBlock(
            dimension_e=H,
            nhead=nhead,
            num_layers=num_layers,
            feedforward_mult=ffn_mult,
            dropout=dropout,
            input_to_latent=self.encoder,
            depthwise_conv=depthwise_conv,
            deprecated_depthwise_conv=deprecated_depthwise_conv,
            norm_type=norm_type,
            num_encoder_layers=num_encoder_layers,
            encoder_dropkey=encoder_dropkey,
            encoder_dropkey_share_heads=encoder_dropkey_share_heads,
            encoder_dropkey_share_layers=encoder_dropkey_share_layers,
            residual_dtype=residual_dtype,
            attn_dtype=attn_dtype,
            ffn_dtype=ffn_dtype,
            conv_dtype=conv_dtype,
            forecaster_d_model=eff_fcst_d_model,
            forecaster_nhead=eff_fcst_n_heads,
            forecaster_kind=forecaster_kind,
            cpc_k_steps=cpc_k_steps,
            qk_norm=qk_norm,
            log_attn_amplitude=log_attn_amplitude,
        )
        # Override activation if requested
        if activation != 'gelu':
            act_fn = torch.nn.functional.silu if activation == 'silu' else torch.nn.functional.gelu
            for layer in self.transformer.layers:
                layer.activation = act_fn
            for layer in self.transformer.encoder_layers:
                layer.activation = act_fn

        if channel_mixing_kind == 'simple':
            self.channel_mixing_module = Simple_channel_mixing_module(H=H, C=C)
        elif channel_mixing_kind == 'attention':
            self.channel_mixing_module = AttentionChannelMixing(
                H=H, C=C, n_heads=channel_mixing_n_heads)
        else:
            raise ValueError(
                f"Unknown channel_mixing_kind={channel_mixing_kind!r} "
                "(expected 'simple' or 'attention')")
        self.channel_mixing_kind = channel_mixing_kind

    def tau(self):
        """Current contrastive temperature.

        Returns the learnable τ as a 0-d tensor (gradient-tracking) when
        ``learnable_tau=True``; otherwise returns ``None`` and the loss
        falls back to the spec dict's `contrastive_divergence_temperature`.

        τ = exp(-log_inv_tau). With log_inv_tau clamped to [0, log(100)],
        τ is bounded to [0.01, 1.0].
        """
        if not getattr(self, 'learnable_tau', False):
            return None
        return torch.exp(-self.log_inv_tau)

    def clamp_log_inv_tau(self):
        """Clamp log_inv_tau in-place to [0, log(100)] → τ ∈ [0.01, 1.0].

        Caller MUST hold a no_grad context (this is for use right after
        optimizer.step(), to keep τ in a sane range)."""
        if getattr(self, 'learnable_tau', False):
            self.log_inv_tau.data.clamp_(0.0, math.log(100.0))

    def _apply_freq_embedding(self, x_patch, freq_ids=None, freq_embs=None):
        """Widen the per-patch time axis with a broadcast freq embedding.

        x_patch: [B, T, C, *]
        freq_ids: LongTensor of shape [B], if provided does a lookup.
        freq_embs: FloatTensor [B, freq_emb_dim], if provided skips lookup
                   (used by mixup to pass an interpolated embedding).

        Returns: [B, T, C, * + freq_emb_dim]
        """
        if self.freq_embedding is None:
            return x_patch
        if freq_embs is None:
            if freq_ids is None:
                raise ValueError(
                    "freq_embedding is configured but neither freq_ids nor "
                    "freq_embs was passed to forward()")
            freq_embs = self.freq_embedding(freq_ids)      # [B, E]
        B, T, C, _ = x_patch.shape
        E = self.freq_emb_dim
        emb_b = freq_embs.view(B, 1, 1, E).expand(B, T, C, E)
        return torch.cat([x_patch, emb_b], dim=-1)

    def _apply_seasonality_embedding(self, x_patch, seasonality_ids=None,
                                      seasonality_embs=None):
        """Widen the per-patch time axis with a broadcast seasonality embedding.

        Mirrors :meth:`_apply_freq_embedding`; concatenated AFTER freq so
        the patch tail is ``[W | stats | freq | seasonality]``.
        """
        if self.seasonality_embedding is None:
            return x_patch
        if seasonality_embs is None:
            if seasonality_ids is None:
                raise ValueError(
                    "seasonality_embedding is configured but neither "
                    "seasonality_ids nor seasonality_embs was passed to forward()")
            seasonality_embs = self.seasonality_embedding(seasonality_ids)
        B, T, C, _ = x_patch.shape
        E = self.seasonality_emb_dim
        emb_b = seasonality_embs.view(B, 1, 1, E).expand(B, T, C, E)
        return torch.cat([x_patch, emb_b], dim=-1)

    def prepare_encoder_input(self, x_norm, freq_ids=None, freq_embs=None,
                               seasonality_ids=None, seasonality_embs=None):
        """Build the per-patch encoder input from an *already-normalised* series.

        Output shape: ``[B, T_patches, C, W + patch_stats + freq_emb + seasonality_emb]``.

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

        # Order: [W | stats | freq | seasonality]. Stats first so they sit
        # next to the raw patch values; the two label embeddings are
        # series-level so they tail.
        if self.patch_stats_kind != 'none':
            stats = compute_patch_stats(
                self.rev_norm.mean, self.rev_norm.stdev,
                W, kind=self.patch_stats_kind,
            )                                            # [B, T, C, 2]
            x = torch.cat([x, stats], dim=-1)

        x = self._apply_freq_embedding(
            x, freq_ids=freq_ids, freq_embs=freq_embs)
        x = self._apply_seasonality_embedding(
            x, seasonality_ids=seasonality_ids, seasonality_embs=seasonality_embs)
        return x

    def forward(self, x, freq_ids=None, freq_embs=None,
                 seasonality_ids=None, seasonality_embs=None):
        B, T_raw, C = x.shape
        W = self.W
        H = self.H
        assert T_raw % W == 0
        T = T_raw // W

        # Apply reversible normalization before patching
        if self.rev_norm is not None:
            x = self.rev_norm(x, mode='norm')

        x = self.prepare_encoder_input(
            x, freq_ids=freq_ids, freq_embs=freq_embs,
            seasonality_ids=seasonality_ids, seasonality_embs=seasonality_embs)
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
    # Cross-batch cosine sims, contracting H *inside* the reduction via
    # einsum so the [B, B, T, C, H] broadcast intermediate is never
    # materialised (~26 GB fp32 at B=256, T-cld=255, H=384 — the OOM that
    # this diagnostic used to trigger every training step). einsum routes
    # to a batched matmul; peak is the [B, B, T, C] output (~67 MB).
    # sims[i,j,t,c] = Σ_h hyn[i,t,c,h]·hyh[j,t,c,h]. This is *directly*
    # the previous (hyh.unsqueeze(0) * hyn.unsqueeze(1)).sum(-1): there
    # hyn.unsqueeze(1)=[B,1,…] put hyn on dim0 and hyh.unsqueeze(0)=[1,B,…]
    # was the broadcast axis, so old[i,j]=Σ_h hyn[i]·hyh[j] — element-for-
    # element equal to this einsum, no transpose. (The downstream
    # symmetric eye-mask + full mean would mask any i/j swap anyway.)
    sims_cross_batch = torch.einsum('itch,jtch->ijtc', hyn, hyh)
    mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
    mask_batch = mask_batch.view(B, B, 1, 1)
    sims_masked = sims_cross_batch.masked_fill(~mask_batch, 0)
    cross_batch = sims_masked.mean().item()

    return ff, fp, tp, cross_batch


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
