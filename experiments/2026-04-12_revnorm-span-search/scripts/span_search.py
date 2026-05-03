#!/usr/bin/env python3
"""RevEWMNorm span search on ARIMA(1, p, q) data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python span_search.py none
    CUDA_VISIBLE_DEVICES=1 python span_search.py 128
"""
import sys
import time
import torch
import torch.optim as optim
from types import SimpleNamespace
from src.models import ConfigurableModel, compute_metrics, count_parameters
from src.arma import generate_arima_batch
from src.loss import contrastive_latent_loss

# Parse span from CLI
span_arg = sys.argv[1] if len(sys.argv) > 1 else "none"
span = None if span_arg == "none" else int(span_arg)

device = torch.device("cuda")
W, H, L, C = 16, 512, 6, 4
T_RAW, BS, LR = 4096, 8, 1e-4
TOTAL_STEPS, VAL_EVERY = 3000, 500
DIMENSION = 8  # ARIMA(1, 8, 8)
LOSS_SPEC = SimpleNamespace(train_configuration={
    "contrastive_divergence_temperature": 0.07, "contrastive_latent_noise": None,
    "loss_shape": "cosine_similarity_batch_no_time_neg", "contrastive_latent_delay": 0,
})

span_label = f"span={span}" if span is not None else "no_norm"
print(f"Tiny W={W}: H={H}, L={L}, ARIMA(1,8,8), {span_label}, bs={BS}", flush=True)

model = ConfigurableModel(
    C=C, H=H, W=W, encoder_type="gru", num_layers=L,
    nhead=8, ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=span,
).to(device)
print(f"Parameters: {count_parameters(model):,}", flush=True)

optimizer = optim.AdamW(model.parameters(), lr=LR)

# Fixed ARIMA validation batch
x_val, _ = generate_arima_batch(batch_size=BS, T_raw=T_RAW, C=C, seed=0, dimension=DIMENSION)
x_val = x_val.to(device)

best_gap = -float("inf")
t0 = time.time()

for step in range(1, TOTAL_STEPS + 1):
    model.train()
    optimizer.zero_grad()

    x, _ = generate_arima_batch(batch_size=BS, T_raw=T_RAW, C=C, dimension=DIMENSION)
    x = x.to(device)

    # Apply RevEWMNorm if present, then transformer (skip channel mixing)
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    B, Tr, Cc = x.shape
    T = Tr // W
    xr = x.view(B, T, W, Cc).permute(0, 1, 3, 2)
    f_flat, o_flat = model.transformer(xr)
    f_lat = f_flat.reshape(B, Cc, T, H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(B, Cc, T, H).permute(0, 2, 1, 3)

    loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=LOSS_SPEC)
    loss.backward()
    optimizer.step()

    if step % VAL_EVERY == 0 or step == TOTAL_STEPS:
        model.eval()
        with torch.no_grad():
            xv = x_val.clone()
            if model.rev_norm is not None:
                xv = model.rev_norm(xv, mode='norm')
            Bv, Trv, Cv = xv.shape
            Tv = Trv // W
            xvr = xv.view(Bv, Tv, W, Cv).permute(0, 1, 3, 2)
            fv, ov = model.transformer(xvr)
            fv = fv.reshape(Bv, Cv, Tv, H).permute(0, 2, 1, 3)
            ov = ov.reshape(Bv, Cv, Tv, H).permute(0, 2, 1, 3)
            val_ff, val_fp, _, _ = compute_metrics(fv, ov, 1)
        gap = val_ff - val_fp
        if gap > best_gap:
            best_gap = gap
        sps = step / (time.time() - t0)
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{step:>5d}] loss={loss.item():.4f} FF={val_ff:.4f} FP={val_fp:.4f} "
              f"gap={gap:.4f} best={best_gap:.4f} {sps:.1f}sps vram={vram:.1f}GB", flush=True)

elapsed = time.time() - t0
print(f"\n{span_label} DONE in {elapsed/60:.1f}min. Best gap: {best_gap:.4f}", flush=True)
