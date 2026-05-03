#!/usr/bin/env python3
"""Window size experiment: W=16 on Tiny backbone, bs=28 (max VRAM)."""
import time, torch, torch.optim as optim
from types import SimpleNamespace
from src.models import ConfigurableModel, compute_metrics, count_parameters
from src.arma import generate_arma_batch
from src.loss import contrastive_latent_loss

device = torch.device("cuda")
W, H, L, C = 16, 512, 6, 4
T_RAW, BS, LR = 4096, 28, 1e-4
TOTAL_STEPS, VAL_EVERY = 15000, 1000
WALL_LIMIT = 29.2 * 60  # same wall time as W=32 10k run
LOSS_SPEC = SimpleNamespace(train_configuration={
    "contrastive_divergence_temperature": 0.07, "contrastive_latent_noise": None,
    "loss_shape": "cosine_similarity_batch_no_time_neg", "contrastive_latent_delay": 0,
})

print(f"Tiny W={W}: H={H}, L={L}, patches={T_RAW//W}, bs={BS}", flush=True)
model = ConfigurableModel(C=C, H=H, W=W, encoder_type="gru", num_layers=L,
    nhead=8, ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1).to(device)
print(f"Parameters: {count_parameters(model):,}", flush=True)
optimizer = optim.AdamW(model.parameters(), lr=LR)
x_val, _ = generate_arma_batch(batch_size=BS, T_raw=T_RAW, C=C, seed=0, dimension=4)
x_val = x_val.to(device)
best_gap = -float("inf")
t0 = time.time()
for step in range(1, TOTAL_STEPS + 1):
    if time.time() - t0 > WALL_LIMIT:
        print(f"\nWall time limit reached at step {step-1}.", flush=True)
        break
    model.train(); optimizer.zero_grad()
    x, _ = generate_arma_batch(batch_size=BS, T_raw=T_RAW, C=C, dimension=4)
    x = x.to(device)
    B, Tr, Cc = x.shape; T = Tr // W
    xr = x.view(B, T, W, Cc).permute(0, 1, 3, 2)
    f_flat, o_flat = model.transformer(xr)
    f_lat = f_flat.reshape(B, Cc, T, H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(B, Cc, T, H).permute(0, 2, 1, 3)
    loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=LOSS_SPEC)
    loss.backward(); optimizer.step()
    if step % VAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            Bv, Trv, Cv = x_val.shape; Tv = Trv // W
            xv = x_val.view(Bv, Tv, W, Cv).permute(0, 1, 3, 2)
            fv, ov = model.transformer(xv)
            fv = fv.reshape(Bv, Cv, Tv, H).permute(0, 2, 1, 3)
            ov = ov.reshape(Bv, Cv, Tv, H).permute(0, 2, 1, 3)
            val_ff, val_fp, _, _ = compute_metrics(fv, ov, 1)
        gap = val_ff - val_fp
        if gap > best_gap: best_gap = gap
        elapsed = time.time() - t0
        sps = step / elapsed
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{step:>6d}] loss={loss.item():.4f} FF={val_ff:.4f} FP={val_fp:.4f} gap={gap:.4f} best={best_gap:.4f} {sps:.1f}sps vram={vram:.1f}GB", flush=True)
elapsed = time.time() - t0
print(f"\nW=16 bs=28 DONE in {elapsed/60:.1f}min. Best gap: {best_gap:.4f}", flush=True)
