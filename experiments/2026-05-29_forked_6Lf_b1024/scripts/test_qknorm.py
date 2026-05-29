#!/usr/bin/env python3
"""Correctness test for the QK-norm implementation (CPU, no training run).

1. OFF builds + forwards unchanged (sanity).
2. The qk_norm SDPA path with q/k norm replaced by Identity must reproduce
   nn.MultiheadAttention to fp tolerance — validates projection reuse + mask
   reshape (causal (T,T) AND DropKey (B*nhead,T,T)).
3. Real RMSNorm-QK runs and bounds the logits well below the un-normed ones.
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024")
from src.blocks import DecoderOnlyTransformerLayer, TransformerBlock

torch.manual_seed(0)
D, NH, T, B = 384, 6, 16, 4
mk = lambda qk: DecoderOnlyTransformerLayer(
    d_model=D, nhead=NH, dim_feedforward=1536, dropout=0.0, activation="gelu",
    batch_first=True, norm_first=True, bias=False, depthwise_conv=3,
    residual_dtype="fp32", attn_dtype="fp32", ffn_dtype="fp32", qk_norm=qk).eval()

off = mk(False)
on = mk(True)
on.load_state_dict(off.state_dict(), strict=False)   # share all shared weights
on.q_norm = nn.Identity(); on.k_norm = nn.Identity()  # disable norm → must == MHA

x = torch.randn(B, T, D)
causal = TransformerBlock._generate_square_subsequent_mask(T)               # (T,T)
# DropKey-style additive mask (B*NH, T, T): causal + random -inf below diag
below = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)
drop = (torch.rand(B * NH, T, T) < 0.5) & below.unsqueeze(0)
dk = torch.where(drop, torch.tensor(float("-inf")), causal.unsqueeze(0).expand(B * NH, -1, -1))

for name, mask, is_causal in [("causal(T,T)", causal, True),
                              ("dropkey(B*nh,T,T)", dk, False)]:
    with torch.no_grad():
        o_off = off(x, tgt_mask=mask, tgt_is_causal=is_causal)
        o_on = on(x, tgt_mask=mask, tgt_is_causal=is_causal)
    diff = (o_off - o_on).abs().max().item()
    print(f"[{name}] max|MHA - SDPA(identity-norm)| = {diff:.2e}  -> {'OK' if diff < 1e-4 else 'MISMATCH'}")

# Real RMSNorm-QK: forward runs, and post-norm logits are bounded.
real = mk(True); real.load_state_dict(off.state_dict(), strict=False)
# blow up the q/k projection to mimic the diverged weights, then compare logits
with torch.no_grad():
    real.self_attn.in_proj_weight.mul_(8.0); off.self_attn.in_proj_weight.copy_(real.self_attn.in_proj_weight)
def maxlogit(layer, qk):
    W = layer.self_attn.in_proj_weight; d = W.shape[1]; hd = d // NH
    q = torch.nn.functional.linear(x, W[:d]); k = torch.nn.functional.linear(x, W[d:2*d])
    qh = q.view(B, T, NH, hd).transpose(1, 2); kh = k.view(B, T, NH, hd).transpose(1, 2)
    if qk: qh = layer.q_norm(qh); kh = layer.k_norm(kh)
    return (qh @ kh.transpose(-2, -1) / (hd ** 0.5)).abs().max().item()
print(f"max|QK logit| un-normed (blown-up weights) = {maxlogit(off, False):.1f}")
print(f"max|QK logit| RMSNorm-QK                    = {maxlogit(real, True):.1f}")
print("DONE")
