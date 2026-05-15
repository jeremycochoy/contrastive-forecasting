import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional
from torch import Tensor


# --- Attention-amplitude diagnostic (opt-in, zero-overhead when off) ---------
# A process-global singleton the transformer layers append to during forward
# *only* when `ATTN_AMP_DIAG.active` is True (set by the training loop every
# N steps via `set_active()`), and only on layers whose
# `log_attn_amplitude=True`. The training loop calls `take_rows()` after the
# forward to drain the buffer into a sidecar CSV.
#
# When `active` is False the per-layer hook is a single boolean check and an
# immediate return — no extra compute, no allocations, no graph nodes, and the
# training math is byte-identical to the pre-diagnostic code path. Everything
# recorded is under `torch.no_grad()` + `.detach()`, so gradients / the loss
# are never affected when on.
class _AttnAmpDiag:
    __slots__ = ("active", "_rows")

    def __init__(self):
        self.active = False
        self._rows = []

    def set_active(self, flag: bool):
        self.active = bool(flag)

    def record(self, layer_idx, block, qk_logit_maxabs,
               sa_in_maxabs, sa_out_maxabs, resid_maxabs):
        self._rows.append((layer_idx, block, qk_logit_maxabs,
                           sa_in_maxabs, sa_out_maxabs, resid_maxabs))

    def take_rows(self):
        """Return and clear the buffered (per-layer) rows."""
        rows = self._rows
        self._rows = []
        return rows


# Process-global instance. Imported by the training loop.
ATTN_AMP_DIAG = _AttnAmpDiag()


class CausalConv(nn.Module):
    # nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=depthwise_conv, padding=1,
    #                                         groups=d_model, bias=bias)
    def __init__(self, c_in, c_out, kernel_size, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, **kwargs)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Pad left side with kernel_size-1 zeros
        padding = self.kernel_size - 1
        x = F.pad(x, (padding, 0))  # pad only left side
        return self.conv(x)

class DecoderOnlyTransformerLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.functional.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = True,
                 bias: bool = True, device=None, dtype=None, depthwise_conv=3,
                 norm_type: str = 'layernorm',
                 log_attn_amplitude: bool = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # Attention-amplitude diagnostic (opt-in). When True AND the global
        # ATTN_AMP_DIAG.active flag is set, _sa_block records per-layer
        # max-abs of the pre-softmax QK^T logits / SA in / SA out / residual.
        # Default False → strict no-op (see _AttnAmpDiag docstring). The
        # layer_idx / block_tag below are set by TransformerBlock so the
        # CSV rows are attributable to a specific (block, layer).
        self.log_attn_amplitude = bool(log_attn_amplitude)
        self.attn_amp_layer_idx = -1
        self.attn_amp_block_tag = "?"
        # Partial diagnostic row stashed by _sa_block, finalized with the
        # residual-stream max-abs by forward(). None when inactive.
        self._attn_amp_pending = None
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        if norm_type == 'rmsnorm':
            self.norm1 = nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Depth-wise convolution over 3 timesteps
        if depthwise_conv > 0:
            # transform
            # nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=depthwise_conv, padding=1,
            #                                         groups=d_model, bias=bias)
            # into causal conv:
            self.depthwise_conv = CausalConv(c_in=d_model, c_out=d_model, kernel_size=depthwise_conv, groups=d_model, bias=bias)
        else:
            self.depthwise_conv = None


        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super().__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False) -> Tensor:
        x = tgt

        if self.depthwise_conv is not None:
            # Reshape and apply depth-wise convolution
            # Input tensor has shape (batch_size, seq_len, d_model),
            # Conv1d expects (batch_size, d_model, seq_len), so we need to transpose
            x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
            x = self.depthwise_conv(x)
            x = x.transpose(1, 2)  # Back to (batch_size, seq_len, d_model)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # Diagnostic-only (strict no-op unless flag+active): record
            # residual-stream max-abs at this layer post-SA-add.
            self._finalize_attn_amplitude(x)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # Diagnostic-only (strict no-op unless flag+active): record
            # residual-stream max-abs at this layer post-SA-add+norm.
            self._finalize_attn_amplitude(x)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        sa_in = x
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        self._maybe_log_attn_amplitude(sa_in, x)
        return self.dropout1(x)

    def _maybe_log_attn_amplitude(self, sa_in: Tensor, sa_out: Tensor) -> None:
        """Diagnostic-only: record max-abs of the pre-softmax QK^T logits
        and the SA-block in/out tensors for this layer.

        STRICT NO-OP unless (self.log_attn_amplitude AND
        ATTN_AMP_DIAG.active). When it runs it is fully under
        torch.no_grad() + .detach(), recomputing q/k from the module's
        in_proj weights (Option A) so the real attention forward above is
        byte-identical whether the flag is on or off. The QK^T recompute
        is wasteful but only fires every N steps when the flag is on.

        The residual-stream max-abs is filled in later by forward() (it
        owns the residual tensor after the SA add).
        """
        if not (self.log_attn_amplitude and ATTN_AMP_DIAG.active):
            return
        with torch.no_grad():
            mha = self.self_attn
            # Self-attention with shared embed dim → single packed
            # in_proj_weight of shape (3*d_model, d_model). bias=False in
            # all configs here, but handle in_proj_bias defensively.
            W = mha.in_proj_weight
            d = W.shape[1]
            Wq = W[:d, :]
            Wk = W[d:2 * d, :]
            b = mha.in_proj_bias
            if b is not None:
                bq = b[:d]
                bk = b[d:2 * d]
            else:
                bq = bk = None
            xin = sa_in.detach()
            # Compute in fp32 so the diagnostic itself never overflows
            # while measuring whether the real (fp16/bf16) path would.
            xin32 = xin.float()
            Wq32 = Wq.float()
            Wk32 = Wk.float()
            q = torch.nn.functional.linear(
                xin32, Wq32, bq.float() if bq is not None else None)
            k = torch.nn.functional.linear(
                xin32, Wk32, bk.float() if bk is not None else None)
            # (B, T, d) → (B, nhead, T, hd); scaled dot product matches
            # nn.MultiheadAttention's internal pre-softmax scores.
            B, T, _ = q.shape
            hd = d // self.nhead
            qh = q.view(B, T, self.nhead, hd).transpose(1, 2)
            kh = k.view(B, T, self.nhead, hd).transpose(1, 2)
            logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(hd)
            qk_logit_maxabs = float(logits.abs().max().item())
            sa_in_maxabs = float(xin.abs().max().item())
            sa_out_maxabs = float(sa_out.detach().abs().max().item())
            # Stash partial; forward() finalizes with resid_maxabs
            # (it owns the residual tensor after the SA add).
            self._attn_amp_pending = (
                qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs)

    def _finalize_attn_amplitude(self, resid: Tensor) -> None:
        """Complete the pending diagnostic row with the residual-stream
        max-abs and push it to ATTN_AMP_DIAG. STRICT NO-OP when there's no
        pending row (flag off / not a logged step)."""
        if self._attn_amp_pending is None:
            return
        qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs = self._attn_amp_pending
        self._attn_amp_pending = None
        with torch.no_grad():
            resid_maxabs = float(resid.detach().abs().max().item())
        ATTN_AMP_DIAG.record(
            self.attn_amp_layer_idx, self.attn_amp_block_tag,
            qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs, resid_maxabs)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return nn.functional.relu
        elif activation == "gelu":
            return nn.functional.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer layers for causal language modeling
    with optional input reshaping
    """
    support_streaming = False

    def __init__(self, dimension_e, nhead=8, num_layers=6, feedforward_mult=None,
                 activation=None, input_to_latent=None, dropout=0, depthwise_conv=3,
                 norm_first=True, norm_type='layernorm', num_encoder_layers=0,
                 encoder_dropkey: float = 0.0,
                 encoder_dropkey_share_heads: bool = False,
                 encoder_dropkey_share_layers: bool = False,
                 forecaster_d_model: int | None = None,
                 forecaster_nhead: int | None = None,
                 log_attn_amplitude: bool = False):
        super().__init__()

        if feedforward_mult is None:
            feedforward_mult = 3
        dim_feedforward = int(feedforward_mult * dimension_e)

        self.input_to_latent = input_to_latent
        # DropKey on the ENCODER stack only (not the forecaster). At each
        # training step, every encoder layer redraws a fresh causal mask
        # where strictly-below-diagonal entries are set to −∞ with probability
        # p. Diagonal stays 0 (self-attention preserved), above-diagonal stays
        # −∞ (causal). Eval / p=0 falls back to the cached pure causal mask
        # so checkpoints stay bit-identical to pre-flag behaviour.
        # Mask is per-(B, head) independent by default — sharing across
        # the batch correlated noise across all 256 rows in lockstep and
        # triggered NaN at step 11700 of attempt-1 (shared (T,T) mask).
        # encoder_dropkey_share_heads=True ties heads within each batch
        # row: same mask for all heads of a given (B, layer) — drops
        # variance by ~num_heads× and forces heads to disagree on which
        # positions they attend to (rather than cooperating to count).
        # Used after attempt-2 (per-(B,head) at p=0.7) diverged at step
        # ~14900 with sustained loss climb 2 → 5 → 7 with no recovery.
        # encoder_dropkey_share_layers=True draws the mask ONCE per
        # forward pass and reuses it for ALL encoder layers. Combined
        # with share_heads, only the (batch_row, step) axes carry
        # randomness — a given token is either fully visible or fully
        # blocked across the whole encoder stack. This pushes the
        # effective per-token block-rate from p^L (independent layers)
        # up to p, much more hostile to a position counter that
        # otherwise survives via the union of "visible-at-some-layer".
        self.encoder_dropkey = float(encoder_dropkey)
        self.encoder_dropkey_share_heads = bool(encoder_dropkey_share_heads)
        self.encoder_dropkey_share_layers = bool(encoder_dropkey_share_layers)
        self.nhead = nhead

        # Pre-forecaster causal encoder stack. When num_encoder_layers=0 the
        # ModuleList is empty and forward() degenerates to the prior
        # patch-encoder-only contrastive target — checkpoint-compatible with
        # all pre-encoder-stage runs.
        self.encoder_layers = nn.ModuleList([
            DecoderOnlyTransformerLayer(
                d_model=dimension_e,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation or 'gelu',
                batch_first=True,
                norm_first=norm_first,
                norm_type=norm_type,
                bias=False,
                dropout=dropout,
                depthwise_conv=depthwise_conv,
                log_attn_amplitude=log_attn_amplitude,
            ) for _ in range(num_encoder_layers)
        ])
        # Tag encoder layers so the amplitude diagnostic CSV rows are
        # attributable to (block='enc', layer_idx).
        for i, lyr in enumerate(self.encoder_layers):
            lyr.attn_amp_layer_idx = i
            lyr.attn_amp_block_tag = "enc"

        # Forecaster bottleneck (#286 follow-up, v13). When
        # `forecaster_d_model` is None it inherits `dimension_e` (legacy),
        # the projections are nn.Identity, and FFN width still uses
        # `feedforward_mult * dimension_e` — a no-op for all prior runs.
        # When set to a smaller value, the forecaster's `self.layers` are
        # built at `forecaster_d_model`, with `forecaster_nhead` heads,
        # and `dim_feedforward = feedforward_mult * forecaster_d_model`
        # (so FFN expansion stays proportional). `down_proj` shrinks the
        # encoder-output stream into the bottleneck, `up_proj` widens it
        # back to `dimension_e` for downstream channel-mixing + loss.
        eff_fcst_d_model = dimension_e if forecaster_d_model is None else int(forecaster_d_model)
        eff_fcst_nhead = nhead if forecaster_nhead is None else int(forecaster_nhead)
        if eff_fcst_d_model % eff_fcst_nhead != 0:
            raise ValueError(
                f"forecaster_d_model={eff_fcst_d_model} must be divisible by "
                f"forecaster_nhead={eff_fcst_nhead}")
        self.forecaster_d_model = eff_fcst_d_model
        self.forecaster_nhead = eff_fcst_nhead
        fcst_dim_feedforward = int(feedforward_mult * eff_fcst_d_model)
        if eff_fcst_d_model != dimension_e:
            self.fcst_down_proj = nn.Linear(dimension_e, eff_fcst_d_model, bias=False)
            self.fcst_up_proj = nn.Linear(eff_fcst_d_model, dimension_e, bias=False)
        else:
            self.fcst_down_proj = nn.Identity()
            self.fcst_up_proj = nn.Identity()

        self.layers = nn.ModuleList([
            DecoderOnlyTransformerLayer(
                d_model=eff_fcst_d_model,
                nhead=eff_fcst_nhead,
                dim_feedforward=fcst_dim_feedforward,
                activation=activation or 'gelu',
                batch_first=True,
                norm_first=norm_first,
                norm_type=norm_type,
                bias=False,
                dropout=dropout,
                depthwise_conv=depthwise_conv,
                log_attn_amplitude=log_attn_amplitude,
            ) for _ in range(num_layers)
        ])
        # Tag forecaster layers so the amplitude diagnostic CSV rows are
        # attributable to (block='fcst', layer_idx).
        for i, lyr in enumerate(self.layers):
            lyr.attn_amp_layer_idx = i
            lyr.attn_amp_block_tag = "fcst"

        self.causal_mask = None

    def forward(self, x):
        # Apply input_to_latent if provided
        if self.input_to_latent is not None:
            x = self.input_to_latent(x)

        B,T,C,H = x.size()
        x = x.permute(0,2,1,3)
        x = x.reshape(B*C, T, H)

        # x shape after potential reshaping: (batch_size, sequence_length, dimension_e)
        if self.causal_mask is None or self.causal_mask.size(0) != x.size(1):
            self.causal_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Encoder layers run BEFORE x_original is captured: the contrastive
        # loss normalises x_original on the unit sphere, so encoder vs
        # forecaster are forced apart by the asymmetric position of the L2.
        # When encoder_dropkey > 0 and we are training, each encoder layer
        # gets a *fresh* random causal mask (diagonal preserved, above-diag
        # still −∞, below-diag −∞ with prob p) — per-layer, per-step random
        # is the regularization. Eval / p=0 falls back to is_causal=True.
        use_dropkey = self.training and self.encoder_dropkey > 0.0
        shared_mask = None
        if use_dropkey and self.encoder_dropkey_share_layers:
            shared_mask = self._dropkey_causal_mask(
                x.size(1), x.size(0), self.nhead,
                x.device, self.causal_mask.dtype,
                self.encoder_dropkey,
                self.encoder_dropkey_share_heads)
        for layer in self.encoder_layers:
            if use_dropkey:
                dk_mask = shared_mask if shared_mask is not None else \
                    self._dropkey_causal_mask(
                        x.size(1), x.size(0), self.nhead,
                        x.device, self.causal_mask.dtype,
                        self.encoder_dropkey,
                        self.encoder_dropkey_share_heads)
                x = layer(x, tgt_mask=dk_mask, tgt_is_causal=False)
            else:
                x = layer(x, tgt_mask=self.causal_mask, tgt_is_causal=True)

        x_original = x.clone()

        # Forecaster bottleneck (#286 follow-up, v13). When configured
        # smaller than `dimension_e`, `fcst_down_proj` is a Linear that
        # shrinks per-token; otherwise it's nn.Identity (no-op). The
        # projection is per-token, so it commutes with the causal mask.
        x = self.fcst_down_proj(x)

        # Forecaster (decoder) layers — always pure causal.
        for layer in self.layers:
            x = layer(x, tgt_mask=self.causal_mask, tgt_is_causal=True)

        # Project the forecaster output back up to dimension_e so the
        # downstream contrastive loss / channel-mixing operates in the
        # same H-dim space as `x_original` (the encoder-side latent).
        x = self.fcst_up_proj(x)

        return x, x_original

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def _dropkey_causal_mask(T, B, num_heads, device, dtype, p, share_heads=False):
        """Causal mask with random −∞ on strictly-below-diagonal entries.

        Above-diagonal: −∞ (causal).
        Diagonal: 0 (self always allowed).
        Strictly below-diagonal: 0 w.p. (1-p), −∞ w.p. p.

        Returns shape (B*num_heads, T, T) — the form `nn.MultiheadAttention`
        accepts as a per-(batch, head) attn_mask.

        share_heads=False (default): independent per (batch_row, head) draw.
            Within-batch averaging absorbs the noise. Fixed attempt-1's
            shared-(T,T) NaN at step 11700, but at p=0.7 attempt-2 still
            diverged at step ~14900 (sustained 2 → 7 climb).

        share_heads=True: independent per batch row but ALL heads of a
            given row see the same mask. Drops variance by ~num_heads×
            and forces heads to disagree on which positions they attend
            to (rather than cooperating to count). Used after attempt-2
            divergence; this is the user-pre-authorized fallback.
        """
        causal = TransformerBlock._generate_square_subsequent_mask(T).to(
            device=device, dtype=dtype)
        if p <= 0.0:
            return causal.unsqueeze(0).expand(B * num_heads, -1, -1)
        below = torch.tril(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1)
        neg_inf = torch.tensor(float('-inf'), device=device, dtype=dtype)
        if share_heads:
            # One mask per batch row, replicated across heads.
            drop = torch.rand(B, T, T, device=device) < p
            drop_below = drop & below.unsqueeze(0)
            mask_b = torch.where(
                drop_below, neg_inf, causal.unsqueeze(0).expand(B, -1, -1))
            # (B, T, T) → (B, num_heads, T, T) → (B*num_heads, T, T)
            mask = mask_b.unsqueeze(1).expand(-1, num_heads, -1, -1)
            mask = mask.reshape(B * num_heads, T, T)
            return mask
        # Independent per (batch_row, head).
        N = B * num_heads
        drop = torch.rand(N, T, T, device=device) < p
        drop_below = drop & below.unsqueeze(0)
        mask = torch.where(
            drop_below, neg_inf, causal.unsqueeze(0).expand(N, -1, -1))
        return mask
        
class Simple_channel_mixing_module(nn.Module):
    def __init__(self, H, C):
        super().__init__()
        self.H = H
        self.C = C

        # Learnable H×H matrices
        self.R = nn.Parameter(torch.randn(H, H)) #channel with itself
        self.Q = nn.Parameter(torch.randn(H, H)) #channel with other channels

        # to build the matrix with R on the diagonal and Q elsewhere
        I = torch.eye(C)
        mask = torch.ones(C, C) - I

        # Register them as buffers so they move with .to(device) but aren't parameters
        self.register_buffer('I', I)
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x:    [B, T, C*H]
        # I:       [C, C]
        # mask:    [C, C]
        # R, Q:    [H, H]
        # kron(I, R) → [H*C, H*C], puts R on diagonal blocks
        # kron(mask, Q) → [H*C, H*C], fills off‑diag with Q
        M = torch.kron(self.I, self.R) + torch.kron(self.mask, self.Q)
        x_hat = x.matmul(M.T)

        return x_hat