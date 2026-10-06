"""
Encoder variants for architecture search.
Each encoder maps [B, T, C, W] -> [B, T, C, H].
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blocks import DecoderOnlyTransformerLayer, _autocast_ctx


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
    def __init__(self, W, H, intermediate_dim=128, num_gru_layers=2,
                 patch_emb_dtype: str = "fp32", input_bound: float = 0.0):
        super().__init__()
        self.W = W
        # #421: with input_bound c > 0 the GRU reads c * tanh(x / c), and the
        # skip layer keeps the raw x. On patches with values above about 20,
        # the size-128 GRU of the #421z run multiplied the gradient by 1e7 to
        # 1e13 in its backward pass. On inputs within 12 its gain stayed
        # below 0.5. The buffer records c, so a loader rebuilds it.
        if input_bound:
            self.register_buffer("input_bound",
                                 torch.tensor(float(input_bound)))
        else:
            self.input_bound = None
        # Dtype for the GRU compute path. fp32 = disabled autocast (no-op).
        # The GRU's W-step recurrence is sensitive to bf16 truncation. fp32
        # is the safe default. fp16 trades precision for ~25-30% speedup.
        self.patch_emb_dtype = patch_emb_dtype
        self.gru = nn.GRU(
            input_size=1, hidden_size=intermediate_dim,
            num_layers=num_gru_layers, batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(intermediate_dim * 2, H)  # bidirectional
        self.skip = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)
        # --patch-rms-weight (#421): a list the forward appends the vector
        # before the LayerNorm to. None: nothing is kept.
        self.pre_norm_sink = None

    def forward(self, x):
        with _autocast_ctx(self.patch_emb_dtype):
            # x: [B, T, C, W]
            shape = x.shape[:-1]  # [B, T, C]
            flat = x.reshape(-1, self.W, 1)  # [B*T*C, W, 1]
            if self.input_bound is not None:
                flat = self.input_bound * torch.tanh(flat / self.input_bound)
            # The GRU over B*T*C independent sequences is the patch-encoder's memory
            # wall at large batch. Optionally chunk it over the sequence dim and/or
            # gradient-checkpoint it (recompute in backward). Both are BYTE-IDENTICAL
            # in the forward output — each sequence is processed independently, so
            # chunking dim 0 and concatenating the last hidden states is exact, and
            # checkpointing only trades stored activations for recompute. #322 uses
            # this to fit global batch 1024 on a single 24 GB card (GRU bwd-activation
            # storage at 65 k sequences is what OOMs otherwise). Env-gated + training-
            # only, so eval and any other caller keep the original behaviour.
            ckpt = os.environ.get("PATCH_ENC_CKPT", "0") == "1"
            nchunk = max(1, int(os.environ.get("PATCH_ENC_CHUNK", "1")))

            def _last_hidden(fl):
                _, hidden = self.gru(fl)  # [num_layers*2, N, intermediate_dim]
                return torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [N, dim*2]

            if ckpt and self.training:
                if nchunk > 1:
                    h = torch.cat(
                        [torch.utils.checkpoint.checkpoint(
                            _last_hidden, fl, use_reentrant=False)
                         for fl in torch.chunk(flat, nchunk, dim=0)], dim=0)
                else:
                    h = torch.utils.checkpoint.checkpoint(
                        _last_hidden, flat, use_reentrant=False)
            else:
                h = _last_hidden(flat)
            h = self.proj(h)  # [B*T*C, H]
            h = h.reshape(*shape, -1)  # [B, T, C, H]
            s = self.skip(x)
            pre = h + s
            if self.pre_norm_sink is not None:
                self.pre_norm_sink.append(pre)
            return self.layer_norm(pre)


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
                 dropout=0.0, depthwise_conv: int = 3,
                 deprecated_depthwise_conv: int = 0,
                 norm_type='layernorm',
                 activation='gelu', use_grad_checkpoint=True,
                 chunk_size=8192):
        super().__init__()
        # W is recorded only for diagnostics. The linear is per-scalar
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
                deprecated_depthwise_conv=deprecated_depthwise_conv,
            ) for _ in range(num_layers)
        ])
        self.causal_mask = None
        # B*T*C = 65k independent length-22 sequences with FFN-mult=4 blows
        # past 24 GB at full N: QKV projection alone allocates ~6.6 GB and
        # the FFN intermediate is ~8.6 GB per layer. We split N into chunks
        # processed sequentially, and gradient-checkpoint each layer inside
        # the chunk so only the layer inputs are kept for backward. Chunks
        # are independent so this is exact (no approximation).
        # chunk_size=0 / None disables chunking (process all N at once).
        self.use_grad_checkpoint = bool(use_grad_checkpoint)
        self.chunk_size = int(chunk_size) if chunk_size else 0

    def _get_causal_mask(self, L, device):
        if (self.causal_mask is None
                or self.causal_mask.size(0) != L
                or self.causal_mask.device != device):
            mask = (torch.triu(torch.ones(L, L, device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')) \
                               .masked_fill(mask == 1, float(0.0))
            self.causal_mask = mask
        return self.causal_mask

    def _run_layer(self, layer, h, mask):
        return layer(h, tgt_mask=mask, tgt_is_causal=True)

    def _encode_chunk(self, h, mask, ckpt):
        """Run all encoder layers on one chunk and pool to last token."""
        for layer in self.layers:
            if ckpt:
                h = torch.utils.checkpoint.checkpoint(
                    self._run_layer, layer, h, mask, use_reentrant=False)
            else:
                h = self._run_layer(layer, h, mask)
        return h[:, -1, :]

    def forward(self, x):
        # x: [B, T, C, W']  — W' = W + patch_stats + freq_emb + seasonality_emb
        B, T, C, Wp = x.shape
        # Each of the W' scalars per patch becomes its own token. Patches
        # are encoded independently in parallel along the (B, T, C) axes.
        h = x.reshape(B * T * C, Wp, 1)              # [N, W', 1]
        h = self.linear_up(h)                        # [N, W', H]
        H = h.shape[-1]
        N = h.shape[0]

        mask = self._get_causal_mask(Wp, h.device)
        ckpt = self.use_grad_checkpoint and self.training and h.requires_grad
        cs = self.chunk_size if self.chunk_size > 0 else N
        if cs >= N:
            out = self._encode_chunk(h, mask, ckpt)             # [N, H]
        else:
            outs = []
            for s in range(0, N, cs):
                outs.append(self._encode_chunk(h[s:s + cs], mask, ckpt))
            out = torch.cat(outs, dim=0)                        # [N, H]

        return out.reshape(B, T, C, H)                          # [B, T, C, H]


class PatchEncoderBank(nn.Module):
    """One patch encoder per patch size (#417).

    A patch of P values reaches the encoder as P + ``tail`` features: the
    values, then the patch statistics and the label embeddings. The width
    of the input therefore names its patch size, and the bank hands the
    patch to that size's encoder. So every caller that runs
    ``transformer(prepare_encoder_input(x, patch_size=P))`` reads size P
    with no other change.
    """

    def __init__(self, encoders: dict, tail: int):
        super().__init__()
        self.tail = int(tail)
        self.encoders = nn.ModuleDict(
            {str(size): encoders[size] for size in sorted(encoders)})

    def forward(self, x):
        size = str(x.shape[-1] - self.tail)
        if size not in self.encoders:
            raise ValueError(f"no patch encoder for patch size {size}; the "
                             f"bank holds {list(self.encoders)}")
        return self.encoders[size](x)


def create_encoder(encoder_type, W, H, intermediate_dim=None,
                   transformer_num_layers=4, transformer_nhead=6,
                   transformer_ffn_mult=4, transformer_dropout=0.0,
                   transformer_depthwise_conv=3,
                   transformer_chunk_size=8192,
                   transformer_use_grad_checkpoint=True,
                   patch_emb_dtype: str = "fp32",
                   gru_input_bound: float = 0.0):
    """Factory function for encoder creation.

    ``patch_emb_dtype`` is wired into encoders whose forward compute is
    precision-sensitive (currently GRU). Other encoders accept-and-ignore
    the kwarg — they can opt in later by wrapping their forward in
    ``_autocast_ctx(self.patch_emb_dtype)``.

    ``gru_input_bound`` (#421) bounds the values the GRU reads (see
    :class:`GRUEncoder`). Only the GRU encoder takes it.
    """
    if gru_input_bound and encoder_type != 'gru':
        raise ValueError("gru_input_bound applies to the GRU encoder only, "
                         f"not to encoder_type={encoder_type!r}")
    if encoder_type == 'mlp':
        return MLPEncoder(W, H, intermediate_dim=intermediate_dim or 64)
    elif encoder_type == 'mlp_wide':
        return MLPWideEncoder(W, H, intermediate_dim=intermediate_dim or 256)
    elif encoder_type == 'residual_silu':
        return ResidualSiLUEncoder(W, H, intermediate_dim=intermediate_dim)
    elif encoder_type == 'gru':
        return GRUEncoder(W, H, intermediate_dim=intermediate_dim or 128,
                          patch_emb_dtype=patch_emb_dtype,
                          input_bound=gru_input_bound)
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
            chunk_size=transformer_chunk_size,
            use_grad_checkpoint=transformer_use_grad_checkpoint,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
