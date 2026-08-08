import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
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
               sa_in_maxabs, sa_out_maxabs, resid_post_sa_maxabs,
               resid_post_ffn_maxabs):
        self._rows.append((layer_idx, block, qk_logit_maxabs,
                           sa_in_maxabs, sa_out_maxabs,
                           resid_post_sa_maxabs, resid_post_ffn_maxabs))

    def take_rows(self):
        """Return and clear the buffered (per-layer) rows."""
        rows = self._rows
        self._rows = []
        return rows


# Process-global instance. Imported by the training loop.
ATTN_AMP_DIAG = _AttnAmpDiag()


# --- Centralized precision API -----------------------------------------------
# Four independent dtype knobs for the transformer body:
#   * residual_dtype : outer-most precision (residual stream + LayerNorm).
#                      fp32 = safe default; bf16 unstable in our
#                      high-aligned-cos-sim regime.
#   * attn_dtype     : dtype for SA-block matmuls (Q/K/V proj, scores,
#                      softmax, output proj). Independent of residual_dtype.
#   * ffn_dtype      : dtype for FFN block (x4 expansion + projection).
#   * conv_dtype     : dtype for the depthwise causal conv. Independent of
#                      residual_dtype. Default None = inherit residual_dtype
#                      (preserves the legacy behaviour, where the conv ran
#                      under the residual-stream autocast, byte-identically
#                      for every existing run/checkpoint — including the
#                      historical residual=fp16/bf16 arms).
# When an inner block dtype (attn/ffn/conv) differs from residual_dtype, its
# output is cast back to residual_dtype before the residual ADD so the
# residual stream stays in residual_dtype. fp32 maps to an
# `autocast(enabled=False)` no-op so weight casting is fully disabled — not
# just an autocast at fp32 dtype (which is different from "disabled" because
# torch's autocast cache still kicks in).
_DTYPE_MAP = {"fp32": torch.float32,
              "fp16": torch.float16,
              "bf16": torch.bfloat16}


def _autocast_ctx(dtype_str):
    """Return an autocast context for `dtype_str`.

    fp32 → a *disabled* autocast (no-op for casts and weight conversion).
    fp16 / bf16 → enabled autocast at that dtype.
    """
    if dtype_str == "fp32":
        return torch.amp.autocast('cuda', enabled=False)
    return torch.amp.autocast('cuda', dtype=_DTYPE_MAP[dtype_str], enabled=True)


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
                 bias: bool = True, device=None, dtype=None, depthwise_conv: int = 3,
                 deprecated_depthwise_conv: int = 0,
                 norm_type: str = 'layernorm',
                 residual_dtype: str = "fp32",
                 attn_dtype: str = "fp32",
                 ffn_dtype: str = "fp32",
                 conv_dtype: Optional[str] = None,
                 qk_norm: bool = False,
                 attn_out_norm: bool = False,
                 log_attn_amplitude: bool = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # Four independent precision knobs (see module-level docstring).
        # The residual stream + LayerNorm run in `residual_dtype`; the SA
        # matmuls run in `attn_dtype`; the FFN matmuls run in `ffn_dtype`;
        # the depthwise conv runs in `conv_dtype` (None → inherit
        # residual_dtype, i.e. the legacy behaviour). When an inner block
        # dtype differs from residual_dtype, its output is cast back to
        # residual_dtype before the residual ADD so the residual stream
        # stays uniform.
        assert residual_dtype in _DTYPE_MAP, \
            f"residual_dtype must be one of {list(_DTYPE_MAP)}, got {residual_dtype!r}"
        assert attn_dtype in _DTYPE_MAP, \
            f"attn_dtype must be one of {list(_DTYPE_MAP)}, got {attn_dtype!r}"
        assert ffn_dtype in _DTYPE_MAP, \
            f"ffn_dtype must be one of {list(_DTYPE_MAP)}, got {ffn_dtype!r}"
        assert conv_dtype is None or conv_dtype in _DTYPE_MAP, \
            f"conv_dtype must be None or one of {list(_DTYPE_MAP)}, got {conv_dtype!r}"
        self.residual_dtype = residual_dtype
        self.attn_dtype = attn_dtype
        self.ffn_dtype = ffn_dtype
        # None → inherit residual_dtype: byte-identical to the pre-conv_dtype
        # code path (conv ran under the residual-stream autocast) for every
        # existing run/checkpoint, including residual=fp16/bf16 arms.
        self.conv_dtype = conv_dtype if conv_dtype is not None else residual_dtype
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
        # residual-stream max-abs by _forward_impl. None when inactive.
        self._attn_amp_pending = None
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               bias=bias, **factory_kwargs)
        # QK-norm (PaLM / Gemma / ViT-22B): RMSNorm on Q and K per head before
        # the dot-product, to bound the pre-softmax logits independently of how
        # large the q/k projection weights grow (the batch-1024 divergence mode,
        # #322). OFF (default) → byte-identical to the nn.MultiheadAttention path
        # above. ON → _sa_block runs an SDPA forward that REUSES this MHA's
        # in_proj/out_proj weights (so ON = OFF + q/k RMSNorm only) plus the two
        # small RMSNorm γ vectors below; same fused kernel, ~no perf cost.
        self.qk_norm = bool(qk_norm)
        # Sandwich norm on the ATTENTION OUTPUT only (Gemma2-style post-sublayer
        # RMSNorm): bounds sa_out — #322's residual-runaway driver — before it
        # enters the residual stream. Attention only (the FFN output does not grow,
        # measured); add to the FFN later only if it does.
        self.attn_out_norm = bool(attn_out_norm)
        self._attn_dropout_p = float(dropout)
        head_dim = d_model // nhead
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, **factory_kwargs)
            self.k_norm = nn.RMSNorm(head_dim, **factory_kwargs)
        if self.attn_out_norm:
            self.attn_out_rms = nn.RMSNorm(d_model, **factory_kwargs)
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

        # Depth-wise causal convolution placement is mutually exclusive between
        # the NEW (Conformer-style, conv feeds SA-branch input only) and the
        # LEGACY (in-place on residual, all-prior-runs behaviour). Use the
        # legacy mode only when resuming an old checkpoint trained with that
        # graph. Both modes use a single `self.depthwise_conv` module; the
        # placement is dispatched in forward() via `depthwise_conv_placement`.
        if depthwise_conv > 0 and deprecated_depthwise_conv > 0:
            raise ValueError(
                "depthwise_conv and deprecated_depthwise_conv are mutually "
                "exclusive (got depthwise_conv={}, deprecated_depthwise_conv={})"
                .format(depthwise_conv, deprecated_depthwise_conv))
        if depthwise_conv > 0:
            self.depthwise_conv = CausalConv(
                c_in=d_model, c_out=d_model, kernel_size=depthwise_conv,
                groups=d_model, bias=bias)
            self.depthwise_conv_placement = "sa_input"
        elif deprecated_depthwise_conv > 0:
            self.depthwise_conv = CausalConv(
                c_in=d_model, c_out=d_model, kernel_size=deprecated_depthwise_conv,
                groups=d_model, bias=bias)
            self.depthwise_conv_placement = "deprecated"
        else:
            self.depthwise_conv = None
            self.depthwise_conv_placement = "off"


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
        # Outer autocast = residual_dtype. fp32 is a *disabled* autocast
        # (no-op); fp16/bf16 enable mixed precision for the residual stream
        # + LayerNorm. The inner SA/FFN/conv blocks open their own autocast
        # contexts at attn_dtype/ffn_dtype/conv_dtype if those differ from
        # residual_dtype, and cast their outputs back to residual_dtype
        # before the residual ADD so the residual stream stays uniform.
        with _autocast_ctx(self.residual_dtype):
            return self._forward_impl(tgt, tgt_mask, tgt_key_padding_mask, tgt_is_causal)

    def _forward_impl(self, tgt: Tensor, tgt_mask: Optional[Tensor],
                      tgt_key_padding_mask: Optional[Tensor],
                      tgt_is_causal: bool) -> Tensor:
        x = tgt

        # LEGACY placement: conv mutates the residual stream in-place.
        # Used ONLY when resuming an old checkpoint trained with this graph.
        # Runs in conv_dtype (default None → residual_dtype, byte-identical
        # to the pre-conv_dtype path), then cast back to residual_dtype.
        if self.depthwise_conv is not None and self.depthwise_conv_placement == "deprecated":
            x = self._conv_block(x)

        if self.norm_first:
            if self.depthwise_conv is not None and self.depthwise_conv_placement == "sa_input":
                # NEW placement (Conformer-style): conv feeds the SA-branch
                # pre-norm input only. Residual stream (x_res) stays clean.
                # Conv runs in conv_dtype (see _conv_block).
                x_res = x
                y = self._conv_block(x)
                sa_out = self._sa_block(self.norm1(y), tgt_mask,
                                        tgt_key_padding_mask, tgt_is_causal)
                x = x_res + sa_out
            else:
                x = x + self._sa_block(self.norm1(x), tgt_mask,
                                       tgt_key_padding_mask, tgt_is_causal)
            # Diagnostic-only (strict no-op unless flag+active): stash
            # residual max-abs post-SA-add, then record full row (incl.
            # end-of-layer residual) after the FFN add.
            self._stash_resid_post_sa(x)
            x = x + self._ff_block(self.norm2(x))
            self._record_attn_amplitude(x)
        else:
            # norm-last path — preserve legacy semantics; current runs use
            # norm_first=True, so this branch is for completeness.
            if self.depthwise_conv is not None and self.depthwise_conv_placement == "sa_input":
                y = self._conv_block(x)
                x = self.norm1(x + self._sa_block(y, tgt_mask,
                                                  tgt_key_padding_mask, tgt_is_causal))
            else:
                x = self.norm1(x + self._sa_block(x, tgt_mask,
                                                  tgt_key_padding_mask, tgt_is_causal))
            # Diagnostic-only (strict no-op unless flag+active): stash
            # residual max-abs post-SA-add+norm, then record full row
            # (incl. end-of-layer residual) after the FFN add.
            self._stash_resid_post_sa(x)
            x = self.norm2(x + self._ff_block(x))
            self._record_attn_amplitude(x)

        return x

    # depthwise causal conv block
    def _conv_block(self, x: Tensor) -> Tensor:
        """Depthwise causal conv (channels-first transpose dance), run in
        `conv_dtype`.

        Mirrors `_sa_block` / `_ff_block`: when conv_dtype != residual_dtype
        the conv runs under its own autocast and the output is cast back to
        residual_dtype before it re-enters the residual stream. When equal
        (the default, conv_dtype inherits residual_dtype) it is a
        byte-identical no-op vs. the pre-conv_dtype code path.
        """
        if self.conv_dtype != self.residual_dtype:
            with _autocast_ctx(self.conv_dtype):
                y = x.transpose(1, 2)  # (B, d_model, T)
                y = self.depthwise_conv(y)
                y = y.transpose(1, 2)  # back to (B, T, d_model)
            # Cast back to residual_dtype before it re-enters the stream.
            return y.to(_DTYPE_MAP[self.residual_dtype])
        else:
            y = x.transpose(1, 2)  # (B, d_model, T)
            y = self.depthwise_conv(y)
            y = y.transpose(1, 2)  # back to (B, T, d_model)
            return y

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        if self.qk_norm:
            out = self._attn_qkv(x, attn_mask, key_padding_mask, is_causal)
        elif self.attn_dtype != self.residual_dtype:
            with _autocast_ctx(self.attn_dtype):
                out = self.self_attn(x, x, x,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     is_causal=is_causal,
                                     need_weights=False)[0]
            # Cast back to residual_dtype before residual ADD.
            out = out.to(_DTYPE_MAP[self.residual_dtype])
        else:
            out = self.self_attn(x, x, x,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask,
                                 is_causal=is_causal,
                                 need_weights=False)[0]
        # Log the RAW attention output (so sa_out shows the value-path growth),
        # then optionally NormFormer-normalise it before it enters the residual.
        self._maybe_log_attn_amplitude(x, out)
        if self.attn_out_norm:
            out = self.attn_out_rms(out)
        return self.dropout1(out)

    def _attn_qkv(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        """SDPA self-attention reusing self.self_attn's in_proj/out_proj weights,
        with RMSNorm on Q,K (QK-norm). Same fused F.scaled_dot_product_attention
        kernel nn.MultiheadAttention dispatches to (no perf regression).
        DropKey/causal additive mask (B*nhead,T,T) → reshaped to (B,nhead,T,T).
        key_padding_mask unused here (fixed-length) — asserted None. Returns the RAW
        attention output in residual_dtype; the caller does the amplitude logging,
        optional attention-output-norm, and dropout."""
        assert key_padding_mask is None, "qk-norm path: key_padding_mask unsupported"
        mha = self.self_attn
        W = mha.in_proj_weight
        bsum = mha.in_proj_bias
        d = W.shape[1]
        B, T, _ = x.shape
        hd = d // self.nhead
        with _autocast_ctx(self.attn_dtype):
            qkv = torch.nn.functional.linear(x, W, bsum)          # [B, T, 3d]
            q, k, v = qkv.split(d, dim=-1)
            q = q.view(B, T, self.nhead, hd).transpose(1, 2)      # [B, nhead, T, hd]
            k = k.view(B, T, self.nhead, hd).transpose(1, 2)
            v = v.view(B, T, self.nhead, hd).transpose(1, 2)
            # RMSNorm over head_dim (autocasts to fp32) then back to compute dtype
            # so SDPA sees q,k,v in one dtype.
            vdt = v.dtype
            if self.qk_norm:
                q = self.q_norm(q).to(vdt)
                k = self.k_norm(k).to(vdt)
            am = attn_mask
            if am is not None:
                if am.dim() == 3:                                  # (B*nhead,T,T) → (B,nhead,T,T)
                    am = am.reshape(B, self.nhead, T, T)
                am = am.to(vdt)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=am,
                dropout_p=(self._attn_dropout_p if self.training else 0.0),
                is_causal=(is_causal and am is None))
            out = out.transpose(1, 2).reshape(B, T, d)             # [B, T, d]
            out = torch.nn.functional.linear(out, mha.out_proj.weight, mha.out_proj.bias)
        return out.to(_DTYPE_MAP[self.residual_dtype])

    def _maybe_log_attn_amplitude(self, sa_in: Tensor, sa_out: Tensor) -> None:
        """Diagnostic-only: record max-abs of the pre-softmax QK^T logits
        and the SA-block in/out tensors for this layer.

        STRICT NO-OP unless (self.log_attn_amplitude AND
        ATTN_AMP_DIAG.active). When it runs it is fully under
        torch.no_grad() + .detach(), recomputing q/k from the module's
        in_proj weights (Option A) so the real attention forward above is
        byte-identical whether the flag is on or off. The QK^T recompute
        is wasteful but only fires every N steps when the flag is on.

        The residual-stream max-abs is filled in later by
        TransformerBlock (it owns the residual tensor); here we stash a
        partial row keyed by (block, layer_idx) via ATTN_AMP_DIAG and the
        caller patches resid_maxabs.
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
            # QK-norm path: report the POST-norm logits (what the kernel sees).
            if self.qk_norm:
                qh = self.q_norm(qh)
                kh = self.k_norm(kh)
            logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(hd)
            qk_logit_maxabs = float(logits.abs().max().item())
            sa_in_maxabs = float(xin.abs().max().item())
            sa_out_maxabs = float(sa_out.detach().abs().max().item())
            # Stash partial; _forward_impl finalizes with resid_maxabs
            # (it owns the residual tensor after the SA add).
            self._attn_amp_pending = (
                qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs)

    def _stash_resid_post_sa(self, resid: Tensor) -> None:
        """Stash the residual-stream max-abs immediately after the SA add.
        STRICT NO-OP when there's no pending row (flag off / not a logged
        step). The full row (incl. post-FFN residual) is pushed by
        _record_attn_amplitude after the FFN add."""
        if self._attn_amp_pending is None:
            return
        with torch.no_grad():
            resid_post_sa = float(resid.detach().abs().max().item())
        # extend the pending tuple: (qk, sa_in, sa_out) -> (..., resid_post_sa)
        self._attn_amp_pending = self._attn_amp_pending + (resid_post_sa,)

    def _record_attn_amplitude(self, resid: Tensor) -> None:
        """Complete + push the diagnostic row with BOTH the post-SA-add and
        the post-FFN-add (end-of-layer) residual max-abs. STRICT NO-OP when
        there's no pending row."""
        if self._attn_amp_pending is None:
            return
        qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs, resid_post_sa = \
            self._attn_amp_pending
        self._attn_amp_pending = None
        with torch.no_grad():
            resid_post_ffn = float(resid.detach().abs().max().item())
        ATTN_AMP_DIAG.record(
            self.attn_amp_layer_idx, self.attn_amp_block_tag,
            qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs,
            resid_post_sa, resid_post_ffn)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        if self.ffn_dtype != self.residual_dtype:
            with _autocast_ctx(self.ffn_dtype):
                y = self.linear2(self.dropout(self.activation(self.linear1(x))))
            # Cast back to residual_dtype before residual ADD.
            y = y.to(_DTYPE_MAP[self.residual_dtype])
            return self.dropout2(y)
        else:
            y = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(y)

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
                 activation=None, input_to_latent=None, dropout=0,
                 depthwise_conv: int = 3,
                 deprecated_depthwise_conv: int = 0,
                 norm_first=True, norm_type='layernorm', num_encoder_layers=0,
                 encoder_dropkey: float = 0.0,
                 encoder_dropkey_share_heads: bool = False,
                 encoder_dropkey_share_layers: bool = False,
                 residual_dtype: str = "fp32",
                 attn_dtype: str = "fp32",
                 ffn_dtype: str = "fp32",
                 conv_dtype: str | None = None,
                 forecaster_d_model: int | None = None,
                 forecaster_nhead: int | None = None,
                 forecaster_kind: str = "transformer",
                 cpc_k_steps: int = 12,
                 qk_norm: bool = False,
                 attn_out_norm: bool = False,
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
                deprecated_depthwise_conv=deprecated_depthwise_conv,
                residual_dtype=residual_dtype,
                attn_dtype=attn_dtype,
                ffn_dtype=ffn_dtype,
                conv_dtype=conv_dtype,
                qk_norm=qk_norm,
                attn_out_norm=attn_out_norm,
                log_attn_amplitude=log_attn_amplitude,
            ) for _ in range(num_encoder_layers)
        ])
        self._qk_norm = bool(qk_norm)
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
        self.forecaster_kind = forecaster_kind
        self.cpc_k_steps = int(cpc_k_steps)

        if forecaster_kind == "linear_cpc":
            # CPC multi-step with LINEAR heads (#316, study arm #2/#3). K linear
            # maps W_k: H → H on the encoder output h_t, head k → h_{t+k}. This
            # is the linear-forecaster family (vs the transformer-head 'cpc');
            # it confounds forecaster-type with k vs β, so it is a control
            # family for the trend-consistency check, not the headline.
            self.fcst_down_proj = nn.Identity()
            self.fcst_up_proj = nn.Identity()
            self.layers = nn.ModuleList()
            self.cpc_heads = nn.ModuleList([
                nn.Linear(dimension_e, dimension_e, bias=False)
                for _ in range(self.cpc_k_steps)
            ])
        elif forecaster_kind == "cpc":
            # CPC multi-step (#316). K independent forecaster heads, each
            # architecturally IDENTICAL to β's forecaster (Linear down →
            # `num_layers` causal transformer layer(s) → Linear up). Head k
            # forecasts the encoder latent k steps ahead (h_{t+k}); for K = 1
            # the single head is byte-identical to β's forecaster. ONLY the
            # number of forecast steps differs from β — the per-head
            # architecture (a transformer 1L with the d=128 bottleneck) is
            # unchanged, so there is no linear-head confound. The next-step
            # head (k=1) is the single forecaster latent the downstream
            # head-trainer / GIFT-Eval consumes (forward default).
            fcst_dim_feedforward = int(feedforward_mult * eff_fcst_d_model)
            self.fcst_down_proj = nn.Identity()   # unused in cpc mode
            self.fcst_up_proj = nn.Identity()     # (per-head projections below)
            self.layers = nn.ModuleList()         # unused

            def _mk_proj(a, b):
                return nn.Linear(a, b, bias=False) if a != b else nn.Identity()
            self.cpc_down = nn.ModuleList(
                [_mk_proj(dimension_e, eff_fcst_d_model) for _ in range(self.cpc_k_steps)])
            self.cpc_up = nn.ModuleList(
                [_mk_proj(eff_fcst_d_model, dimension_e) for _ in range(self.cpc_k_steps)])
            self.cpc_layers = nn.ModuleList([
                nn.ModuleList([
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
                        deprecated_depthwise_conv=deprecated_depthwise_conv,
                        residual_dtype=residual_dtype,
                        attn_dtype=attn_dtype,
                        ffn_dtype=ffn_dtype,
                        conv_dtype=conv_dtype,
                        log_attn_amplitude=False,
                    ) for _ in range(num_layers)
                ]) for _ in range(self.cpc_k_steps)
            ])
        else:
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
                    deprecated_depthwise_conv=deprecated_depthwise_conv,
                    residual_dtype=residual_dtype,
                    attn_dtype=attn_dtype,
                    ffn_dtype=ffn_dtype,
                    conv_dtype=conv_dtype,
                    qk_norm=qk_norm,
                    attn_out_norm=attn_out_norm,
                    log_attn_amplitude=log_attn_amplitude,
                ) for _ in range(num_layers)
            ])
            # Tag forecaster layers so the amplitude diagnostic CSV rows are
            # attributable to (block='fcst', layer_idx).
            for i, lyr in enumerate(self.layers):
                lyr.attn_amp_layer_idx = i
                lyr.attn_amp_block_tag = "fcst"

        self.causal_mask = None

    def causal_mask_for(self, x):
        """The cached [T, T] causal mask for `x`'s sequence length.

        Rebuilt only when the length changes, so a caller that grows the
        sequence (the eval rollout) pays for one mask per new length.
        """
        if self.causal_mask is None or self.causal_mask.size(0) != x.size(1):
            self.causal_mask = self._generate_square_subsequent_mask(
                x.size(1)).to(x.device)
        return self.causal_mask

    def forecaster_forward(self, x):
        """Run the forecaster stack on a [B*C, T, H] latent sequence.

        `fcst_down_proj` → causal decoder layers (the last in fp32) →
        `fcst_up_proj`, i.e. the forecaster operator F on its own. Three
        callers compose the SAME entry point, so they cannot drift apart:
        :meth:`forward` applies it to the encoder output, the eval rollout
        (`src.forecasting_head.rollout_latent`) applies it to its own output
        one token at a time, and the training rollout depth (#373) re-enters
        it over the whole sequence. Returns [B*C, T, H] in `dimension_e`.
        """
        causal_mask = self.causal_mask_for(x)
        # Forecaster bottleneck (#286 follow-up, v13). When configured
        # smaller than `dimension_e`, `fcst_down_proj` is a Linear that
        # shrinks per-token; otherwise it's nn.Identity (no-op). The
        # projection is per-token, so it commutes with the causal mask.
        x = self.fcst_down_proj(x)

        # Forecaster (decoder) layers — always pure causal. Same hybrid as
        # the encoder stack: all-but-last under outer autocast, last layer in
        # fp32 so the forecaster latent feeding the loss is full precision.
        n_fcst = len(self.layers)
        bb_ckpt = self.training and os.environ.get("BACKBONE_CKPT", "0") == "1"
        # Optional gradient-checkpointing of the (non-last) forecaster layers,
        # mirroring the encoder-layer checkpointing — env-gated and training-only,
        # so it is BYTE-IDENTICAL (exact recompute) and a no-op for every existing
        # run/checkpoint. Lets the full-width-forecaster arm fit a single 24 GB card.
        fcst_ckpt = (os.environ.get("FCST_GRAD_CKPT", "0") == "1"
                     and self.training and x.requires_grad)
        for i, layer in enumerate(self.layers):
            if i == n_fcst - 1:
                with torch.amp.autocast('cuda', enabled=False):
                    x = x.float()
                    x = layer(x, tgt_mask=causal_mask, tgt_is_causal=True)
            elif fcst_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp, lyr=layer: lyr(inp, tgt_mask=causal_mask,
                                               tgt_is_causal=True),
                    x, use_reentrant=False)
            else:
                x = self._run_layer(layer, x, causal_mask, True, bb_ckpt)
        if n_fcst > 0:
            x = x.float()

        # Project the forecaster output back up to dimension_e so the
        # downstream contrastive loss / channel-mixing operates in the
        # same H-dim space as `x_original` (the encoder-side latent).
        return self.fcst_up_proj(x)

    def _run_layer(self, layer, x, mask, is_causal, ckpt):
        """Run one decoder layer, optionally gradient-checkpointed (#327).

        Checkpointing is BYTE-IDENTICAL in the forward — it only trades stored
        activations for recompute in the backward. The attention mask is built
        outside and captured here, so it is the same tensor on forward and
        recompute (no RNG dependence); any layer-internal dropout is matched by
        checkpoint's preserve_rng_state. Env-gated (BACKBONE_CKPT=1, training
        only) so the default path is unchanged; lets global batch 2048 fit the
        backbone-transformer forward on one 24 GB card (the GRU-encoder path's
        main transformer is otherwise not checkpointed)."""
        if ckpt:
            return torch.utils.checkpoint.checkpoint(
                lambda t: layer(t, tgt_mask=mask, tgt_is_causal=is_causal),
                x, use_reentrant=False)
        return layer(x, tgt_mask=mask, tgt_is_causal=is_causal)

    def forward(self, x, return_multi=False, return_embed=False):
        # Apply input_to_latent if provided. Capture the patch-embedding
        # output e_t when requested (#355 SIGReg attaches a regulariser to
        # this pre-encoder latent); the captured tensor is in [B,T,C,H]
        # layout, before the [B*C,T,H] reshape used inside this block.
        if self.input_to_latent is not None:
            x = self.input_to_latent(x)
        embed = x if return_embed else None

        B,T,C,H = x.size()
        x = x.permute(0,2,1,3)
        x = x.reshape(B*C, T, H)

        # x shape after potential reshaping: (batch_size, sequence_length, dimension_e)
        self.causal_mask_for(x)

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
        # Run all encoder layers EXCEPT the last under the outer autocast
        # context (typically bf16 for speed). The LAST encoder layer runs in
        # fp32 so the latent at the encoder boundary (x_original) is full
        # precision — required by the contrastive loss's small-τ logsumexp.
        n_enc = len(self.encoder_layers)
        # #327: optionally gradient-checkpoint the non-last (non-fp32) layers so
        # a global batch of 2048 fits the backbone forward on one 24 GB card.
        bb_ckpt = self.training and os.environ.get("BACKBONE_CKPT", "0") == "1"
        for i, layer in enumerate(self.encoder_layers):
            if i == n_enc - 1:
                # fp32 region for the last encoder layer
                with torch.amp.autocast('cuda', enabled=False):
                    x = x.float()
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
            else:
                if use_dropkey:
                    dk_mask = shared_mask if shared_mask is not None else \
                        self._dropkey_causal_mask(
                            x.size(1), x.size(0), self.nhead,
                            x.device, self.causal_mask.dtype,
                            self.encoder_dropkey,
                            self.encoder_dropkey_share_heads)
                    x = self._run_layer(layer, x, dk_mask, False, bb_ckpt)
                else:
                    x = self._run_layer(layer, x, self.causal_mask, True, bb_ckpt)

        # Keep x in fp32 across the encoder/forecaster boundary so x_original
        # remains fp32 even after the outer autocast would re-cast the
        # tensor. (When n_enc == 0, x stays in whatever dtype the caller fed.)
        if n_enc > 0:
            x = x.float()
        x_original = x.clone()

        # CPC multi-step (#316). K transformer-1L forecaster heads (each = β's
        # forecaster: down → causal transformer layer(s) → up) map the causal
        # encoder output h_t to a prediction of the latent k steps ahead.
        # Computed in fp32 (x is already fp32 at the encoder boundary). The full
        # [B*C, T, K, H] stack feeds the multi-step InfoNCE loss
        # (return_multi=True); the default path returns only the next-step (k=1)
        # head so all existing callers — ConfigurableModel.forward,
        # extract_forecaster_latents — see the usual [B*C, T, H] forecaster
        # latent (the analogue of β's forecaster output).
        if self.forecaster_kind == "linear_cpc":
            with torch.amp.autocast('cuda', enabled=False):
                h = x.float()
                f_stack = torch.stack(
                    [head(h) for head in self.cpc_heads], dim=2)  # [B*C, T, K, H]
            f_out = f_stack if return_multi else f_stack[:, :, 0, :]
            if return_embed:
                return f_out, x_original, embed
            return f_out, x_original

        if self.forecaster_kind == "cpc":
            with torch.amp.autocast('cuda', enabled=False):
                h = x.float()
                preds = []
                for kdown, klayers, kup in zip(self.cpc_down, self.cpc_layers, self.cpc_up):
                    xk = kdown(h)
                    for layer in klayers:
                        xk = layer(xk, tgt_mask=self.causal_mask, tgt_is_causal=True)
                    preds.append(kup(xk))
                f_stack = torch.stack(preds, dim=2)               # [B*C, T, K, H]
            f_out = f_stack if return_multi else f_stack[:, :, 0, :]
            if return_embed:
                return f_out, x_original, embed
            return f_out, x_original

        x = self.forecaster_forward(x)

        if return_embed:
            return x, x_original, embed
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
        
class AttentionChannelMixing(nn.Module):
    """Per-time-step softmax attention across channels.

    Q, K, V are 1x1 Conv1d projections on H (sample-independent linear maps),
    but attention scores Q.K^T are computed from the per-sample features, so
    the effective channel mix is sample-dependent. Different samples mix
    channels differently based on their content — exactly what
    Simple_channel_mixing_module cannot do.

    Ported from rnd_jeremy/pytrade/blocks/latent_blocks.py:ChannelMixingBlock
    (mode='attention' branch), simplified for our shape.

    Input  : [B, T, C*H] (matches Simple_channel_mixing_module API)
    Output : [B, T, C*H]
    """

    def __init__(self, H: int, C: int, n_heads: int = 8,
                 residual_weight: float = 1.0):
        super().__init__()
        assert H % n_heads == 0, f"H={H} must be divisible by n_heads={n_heads}"
        self.H = H
        self.C = C
        self.n_heads = n_heads
        self.head_dim = H // n_heads
        self.residual_weight = residual_weight
        # 1x1 Conv1d over (H, T) — equivalent to Linear on H, broadcast over T
        self.q = nn.Conv1d(H, H, kernel_size=1, bias=True)
        self.k = nn.Conv1d(H, H, kernel_size=1, bias=True)
        self.v = nn.Conv1d(H, H, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(H)
        # Init v small so initial output is approximately a residual identity
        with torch.no_grad():
            self.v.weight.mul_(0.01)
            self.v.bias.zero_()

    def forward(self, x_btch_flat: Tensor) -> Tensor:
        # [B, T, C*H]
        B, T, CH = x_btch_flat.shape
        H, C = self.H, self.C
        assert CH == C * H, f"got C*H={CH}, expected {C * H}"

        # [B, T, C, H] -> [B, C, H, T]
        x = x_btch_flat.view(B, T, C, H).permute(0, 2, 3, 1).contiguous()

        x_conv = x.reshape(B * C, H, T)
        q = self.q(x_conv).view(B, C, self.n_heads, self.head_dim, T)
        k = self.k(x_conv).view(B, C, self.n_heads, self.head_dim, T)
        v = self.v(x_conv).view(B, C, self.n_heads, self.head_dim, T)

        # Reshape for attention (across channels at each time)
        q = q.permute(0, 2, 4, 1, 3)  # [B, n_heads, T, C, head_dim]
        k = k.permute(0, 2, 4, 3, 1)  # [B, n_heads, T, head_dim, C]
        v = v.permute(0, 2, 4, 1, 3)  # [B, n_heads, T, C, head_dim]

        scores = (q @ k) / (self.head_dim ** 0.5)         # [B, n_heads, T, C, C]
        scores = torch.softmax(scores, dim=-1)
        y = scores @ v                                    # [B, n_heads, T, C, head_dim]

        # back to [B, C, H, T]
        y = y.permute(0, 3, 1, 4, 2).contiguous().view(B, C, H, T)

        y = x * self.residual_weight + y

        # LayerNorm over H — last dim
        y = y.permute(0, 1, 3, 2).contiguous()             # [B, C, T, H]
        y = self.norm(y)
        # back to [B, T, C, H] -> flatten to [B, T, C*H]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C * H)
        return y


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