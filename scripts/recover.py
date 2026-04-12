#!/usr/bin/env python3
"""
ARMA parameter recovery training script.

Trains a lightweight GRU head on top of a frozen contrastive backbone to
recover the AR/MA coefficients of the generating ARMA process. This tests
whether the backbone's latent representations encode process structure.

Usage:
    # Train recovery head (requires a trained backbone checkpoint)
    python scripts/recover.py --device cuda --model-path model_best.pth

    # Longer training
    python scripts/recover.py --device cuda --model-path model_best.pth --epochs 30000
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.models import ConfigurableModel
from src.recovery import create_recovery_head, parameter_loss
from src.arma import generate_arma_batch

# -- Backbone architecture (must match the saved checkpoint) -----------------
BACKBONE_CONFIG = dict(
    C=4, H=1024, W=32,
    encoder_type="gru", num_layers=12, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
)

# -- Recovery head config (best from experiments: DeepGRU) -------------------
NUM_ARMA_PARAMS = 4
RECOVERY_CONFIG = dict(model_type="deepgru", H=1024, hidden_dim=128, num_gru_layers=2,
                       num_arma_params=NUM_ARMA_PARAMS)


def parse_args():
    p = argparse.ArgumentParser(description="Train ARMA parameter recovery head")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-path", required=True, help="Path to trained backbone checkpoint")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-path", default="recovery_head.pth")
    p.add_argument("--eval-samples", type=int, default=200, help="Samples for final evaluation")
    return p.parse_args()


def extract_latent(model, x):
    """Extract per-channel latent features from the frozen backbone."""
    with torch.no_grad():
        _, h = model(x)                          # [B, T, C, H]
        B, T, C, H = h.shape
        return h.permute(0, 2, 1, 3).reshape(B * C, T, H)


def make_batch(batch_size, device, seed=None):
    """Generate ARMA data and extract true AR/MA coefficients."""
    x, params = generate_arma_batch(batch_size=batch_size, T_raw=4096,
                                    C=BACKBONE_CONFIG["C"], seed=seed, dimension=4)
    ar_list, ma_list = [], []
    for ar_poly, ma_poly in params:
        ar = np.pad(-ar_poly[1:], (0, max(0, NUM_ARMA_PARAMS - len(ar_poly) + 1)))[:NUM_ARMA_PARAMS]
        ma = np.pad( ma_poly[1:], (0, max(0, NUM_ARMA_PARAMS - len(ma_poly) + 1)))[:NUM_ARMA_PARAMS]
        ar_list.append(ar)
        ma_list.append(ma)
    true_ar = torch.tensor(np.array(ar_list), dtype=torch.float32).to(device)
    true_ma = torch.tensor(np.array(ma_list), dtype=torch.float32).to(device)
    return x.to(device), true_ar, true_ma


def evaluate(head, backbone, device, n_samples):
    """Run evaluation: compute MSE, improvement ratio, and sign agreement."""
    head.eval()
    errors, baselines = [], []
    pred_all, true_all = [], []

    with torch.no_grad():
        for i in range(n_samples):
            x, ar_true, ma_true = make_batch(1, device, seed=i + 1000)
            h = extract_latent(backbone, x)
            pred_ar, pred_ma = head(h)
            _, ar_e, ma_e = parameter_loss(pred_ar, pred_ma, ar_true, ma_true)
            errors.append(ar_e.item() + ma_e.item())
            baselines.append(
                F.mse_loss(torch.zeros_like(ar_true), ar_true).item()
                + F.mse_loss(torch.zeros_like(ma_true), ma_true).item()
            )
            # Collect for sign agreement
            pred_all.append(torch.cat([pred_ar.mean(1), pred_ma.mean(1)], dim=-1).cpu())
            true_all.append(torch.cat([ar_true, ma_true], dim=-1).cpu())

    pred_t = torch.cat(pred_all).numpy()
    true_t = torch.cat(true_all).numpy()
    nonzero = true_t != 0
    sign_agree = float(np.mean(np.sign(pred_t[nonzero]) == np.sign(true_t[nonzero])))
    mean_err = float(np.mean(errors))
    mean_base = float(np.mean(baselines))
    return mean_err, mean_base, mean_base / mean_err, sign_agree


def main():
    args = parse_args()
    device = torch.device(args.device)

    # -- Load and freeze backbone --------------------------------------------
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    backbone.load_state_dict(torch.load(args.model_path, map_location=device))
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"Backbone loaded from {args.model_path}")

    # -- Create recovery head ------------------------------------------------
    head = create_recovery_head(**RECOVERY_CONFIG).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Recovery head: {RECOVERY_CONFIG['model_type']}, {n_params:,} params")

    optimizer = optim.Adam(head.parameters(), lr=args.lr)

    # -- Fixed validation set ------------------------------------------------
    x_val, ar_val, ma_val = make_batch(args.batch_size, device, seed=0)
    h_val = extract_latent(backbone, x_val)

    # -- Training loop -------------------------------------------------------
    best_val, best_ep = float("inf"), 0
    print(f"Training for {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        x, ar, ma = make_batch(args.batch_size, device, seed=epoch)
        h = extract_latent(backbone, x)
        pred_ar, pred_ma = head(h)
        loss, _, _ = parameter_loss(pred_ar, pred_ma, ar, ma)
        loss.backward()
        optimizer.step()

        # Validation
        head.eval()
        with torch.no_grad():
            pav, pmv = head(h_val)
            vl, _, _ = parameter_loss(pav, pmv, ar_val, ma_val)
        vl_item = vl.item()

        if vl_item < best_val:
            best_val, best_ep = vl_item, epoch
            torch.save(head.state_dict(), args.save_path.replace(".pth", "_best.pth"))

        if epoch % 100 == 0:
            print(f"[{epoch:6d}] train={loss.item():.6f}  val={vl_item:.6f}  best={best_val:.6f}@{best_ep}")

    torch.save(head.state_dict(), args.save_path)

    # -- Final evaluation with best checkpoint -------------------------------
    head.load_state_dict(torch.load(args.save_path.replace(".pth", "_best.pth"), map_location=device))
    err, base, ratio, sign = evaluate(head, backbone, device, args.eval_samples)

    print(f"\n{'='*50}")
    print(f"Best val loss:      {best_val:.6f} at epoch {best_ep}")
    print(f"Mean error:         {err:.6f}")
    print(f"Baseline error:     {base:.6f}")
    print(f"Improvement ratio:  {ratio:.2f}x")
    print(f"Sign agreement:     {sign:.1%}")
    print(f"Saved to:           {args.save_path}")


if __name__ == "__main__":
    main()
