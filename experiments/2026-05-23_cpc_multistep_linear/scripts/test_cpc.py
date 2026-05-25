#!/usr/bin/env python3
"""CPU sanity tests for the CPC multi-step linear forecaster (#316).

Run BEFORE any GPU launch to de-risk the multi-hour training run. Checks:
  1. cpc model builds; transformer.forward(return_multi) shapes.
  2. forward_step-style reshape to [B,T,C,K,H]; k=1 default path [B,T,C,H].
  3. cpc_multistep_loss is finite, ≥0 with pos-in-denominator, and gradients
     reach BOTH the cpc_heads and the encoder.
  4. A few optimiser steps on a fixed batch drive the loss down (the heads +
     encoder can actually fit the objective).
  5. state_dict has cpc_heads.* and NO transformer.layers.* / fcst_*proj;
     round-trips through a freshly-built cpc model (load strict).
  6. extract_forecaster_latents returns [B*C,T,H] (downstream contract).
  7. The legacy transformer forecaster path is unchanged (regression).
"""
import sys, os
import torch
from types import SimpleNamespace

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)

from src.models import ConfigurableModel
from src.loss import contrastive_latent_loss, cpc_multistep_loss
from src.forecasting_head import extract_forecaster_latents, extract_encoder_latents

torch.manual_seed(0)

B, C, H, W, Traw = 4, 1, 32, 16, 64
T = Traw // W
K = 3

def make_model(kind):
    return ConfigurableModel(
        C=C, H=H, W=W, encoder_type="gru", num_layers=1, nhead=4,
        ffn_mult=4.0, depthwise_conv=3, dropout=0.0,
        rev_norm_kind="ewma", rev_norm_span=16,
        num_encoder_layers=2, freq_emb_dim=3, seasonality_emb_dim=3,
        forecaster_kind=kind, cpc_k_steps=K,
    ).eval()

def fwd_multi(model, x, freq_ids, seas_ids):
    """Mirror train.forward_step for the CPC path."""
    xn = model.rev_norm(x, mode="norm")
    xr = model.prepare_encoder_input(xn, freq_ids=freq_ids, seasonality_ids=seas_ids)
    f_flat, o_flat = model.transformer(xr, return_multi=True)
    f_lat = f_flat.reshape(B, C, T, K, H).permute(0, 2, 1, 3, 4)
    o_lat = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    return f_lat, o_lat

x = torch.randn(B, Traw, C)
freq_ids = torch.zeros(B, dtype=torch.long)
seas_ids = torch.zeros(B, dtype=torch.long)

# --- 1 & 2: shapes -----------------------------------------------------------
m = make_model("cpc")
xn = m.rev_norm(x, mode="norm")
xr = m.prepare_encoder_input(xn, freq_ids=freq_ids, seasonality_ids=seas_ids)
f_multi, o_flat = m.transformer(xr, return_multi=True)
f_one, _ = m.transformer(xr, return_multi=False)
assert f_multi.shape == (B * C, T, K, H), f_multi.shape
assert f_one.shape == (B * C, T, H), f_one.shape
assert torch.allclose(f_multi[:, :, 0, :], f_one), "k=1 slice must equal default output"
f_lat, o_lat = fwd_multi(m, x, freq_ids, seas_ids)
assert f_lat.shape == (B, T, C, K, H) and o_lat.shape == (B, T, C, H)
print("[1,2] shapes OK  f_multi", tuple(f_multi.shape), " f_lat", tuple(f_lat.shape))

# --- 3: loss finite, >=0, gradients to heads AND encoder ---------------------
m.train()
f_lat, o_lat = fwd_multi(m, x, freq_ids, seas_ids)
loss = cpc_multistep_loss(f_lat.float(), o_lat.float(), tau=0.1,
                          include_positive_in_denominator=True)
assert torch.isfinite(loss) and loss.item() >= 0, loss.item()
loss.backward()
g_head = next(p for p in m.transformer.cpc_layers[0].parameters() if p.grad is not None).grad
g_enc = next(p.grad for p in m.transformer.encoder_layers.parameters()
             if p.grad is not None)
assert g_head is not None and g_head.abs().sum() > 0, "no grad to cpc head"
assert g_enc is not None and g_enc.abs().sum() > 0, "no grad to encoder"
# Dispatch through the public entrypoint too (loss_shape switch).
spec = SimpleNamespace(train_configuration={
    "loss_shape": "cpc_multistep", "contrastive_divergence_temperature": 0.1,
    "include_positive_in_denominator": True})
loss2 = contrastive_latent_loss((f_lat.float(), o_lat.float()), False, spec)
assert torch.isfinite(loss2) and loss2.item() >= 0
print(f"[3] loss={loss.item():.4f} (>=0), grads reach heads+encoder; "
      f"dispatch loss={loss2.item():.4f} OK")

# --- 4: optimiser drives loss down -------------------------------------------
m2 = make_model("cpc").train()
opt = torch.optim.AdamW(m2.parameters(), lr=1e-2)
xf = torch.randn(B, Traw, C)
losses = []
for _ in range(40):
    opt.zero_grad()
    fl, ol = fwd_multi(m2, xf, freq_ids, seas_ids)
    l = cpc_multistep_loss(fl.float(), ol.float(), tau=0.1,
                           include_positive_in_denominator=True)
    l.backward(); opt.step(); losses.append(l.item())
assert losses[-1] < losses[0] - 0.05, f"loss did not drop: {losses[0]:.3f}->{losses[-1]:.3f}"
print(f"[4] overfit loss {losses[0]:.3f} -> {losses[-1]:.3f} OK")

# --- 5: state_dict keys + round-trip -----------------------------------------
sd = m.state_dict()
has_heads = [k for k in sd if k.startswith("transformer.cpc_layers.")]
has_fcst_layers = [k for k in sd if k.startswith("transformer.layers.")]
has_proj = [k for k in sd if "fcst_down_proj.weight" in k or "fcst_up_proj.weight" in k]
assert has_heads, "no cpc_layers in state_dict"
assert not has_fcst_layers, f"unexpected transformer.layers.*: {has_fcst_layers}"
assert not has_proj, f"unexpected fcst projections: {has_proj}"
detK = max(int(k.split(".")[2]) for k in has_heads) + 1
assert detK == K, detK
m_fresh = make_model("cpc")
m_fresh.load_state_dict(sd)  # strict
print(f"[5] state_dict: {len(has_heads)} cpc_head tensors, no transformer "
      f"forecaster keys, auto-detect K={detK}, strict load OK")

# --- 6: downstream extraction contract ---------------------------------------
ff, xnorm = extract_forecaster_latents(m, x, freq_ids=freq_ids, seasonality_ids=seas_ids)
ee, _ = extract_encoder_latents(m, x, freq_ids=freq_ids, seasonality_ids=seas_ids)
assert ff.shape == (B * C, T, H), ff.shape
assert ee.shape == (B * C, T, H), ee.shape
print(f"[6] extract_forecaster_latents -> {tuple(ff.shape)}, "
      f"extract_encoder_latents -> {tuple(ee.shape)} OK")

# --- 7: legacy transformer path regression -----------------------------------
mt = make_model("transformer").eval()
xrt = mt.prepare_encoder_input(mt.rev_norm(x, mode="norm"),
                               freq_ids=freq_ids, seasonality_ids=seas_ids)
ft, ot = mt.transformer(xrt)  # default call, no return_multi
assert ft.shape == (B * C, T, H) and ot.shape == (B * C, T, H)
sdt = mt.state_dict()
assert any(k.startswith("transformer.layers.") for k in sdt)
assert not any(k.startswith("transformer.cpc_layers.") for k in sdt)
print(f"[7] transformer path unchanged -> {tuple(ft.shape)}, has layers.*, no cpc_heads")

# --- 8: at k=1, cpc_multistep == β's cosine_similarity_batch_full_hh_negs (C=1) ---
# This is the key correctness property (#316 review): the multi-step loss keeps
# β's negatives and only changes the positive, so the single-step (k=1) case
# must reproduce β's loss exactly at C=1.
torch.manual_seed(3)
Bc, Cc, Hc, Tc = 6, 1, 16, 9
f1 = torch.randn(Bc, Tc, Cc, Hc)
hc = torch.randn(Bc, Tc, Cc, Hc)
tau_c = 0.1
cpc_k1 = cpc_multistep_loss(f1.unsqueeze(3), hc, tau_c,
                            include_positive_in_denominator=True)
beta_spec = SimpleNamespace(train_configuration={
    "loss_shape": "cosine_similarity_batch_full_hh_negs",
    "contrastive_divergence_temperature": tau_c,
    "include_positive_in_denominator": True})
beta_loss = contrastive_latent_loss((f1, hc), False, beta_spec)
assert torch.allclose(cpc_k1, beta_loss, atol=1e-5), \
    f"k=1 CPC ({cpc_k1.item():.6f}) != β full_hh_negs ({beta_loss.item():.6f})"
print(f"[8] k=1 cpc_multistep={cpc_k1.item():.6f} == β full_hh_negs="
      f"{beta_loss.item():.6f} (C=1) OK — only the positive differs for k>1")

print("\nALL CPC SANITY TESTS PASSED")
