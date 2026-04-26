#!/usr/bin/env python3
"""Plot 12 synthetic-sample forecasts from our best 30k-class config:
fe+mu backbone + R1 quantile head.

Layout: 4×3 grid (12 panels). Per panel:
- ground truth (black)
- seasonal-naive with the known synth period P (green dashed)
- qhead median (q=0.5)         (purple)
- qhead [0.1, 0.9] band         (purple semi-transparent fill)

Synthetic series are drawn fresh from src.synthetic_periodic with a
deterministic seed and *one extra horizon's worth* of timesteps appended
so we can split into context+truth.
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from src.models import ConfigurableModel
from src.forecasting_head import (
    QuantileForecastingHead, FORECAST_LEN, forecast_with_strategy,
)
from src.synthetic_periodic import (
    generate_periodic_batch, primitive_name,
)


BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
HEAD_CONFIG = dict(H=512, hidden_dim=128, num_gru_layers=2,
                    forecast_len=FORECAST_LEN, dropout=0.1)
T_RAW = 1024
HORIZON = 16
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_pair(backbone_path, head_path, device):
    sd = torch.load(backbone_path, map_location=device, weights_only=True)
    cfg = dict(BACKBONE_CONFIG)
    w = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = (w.shape[1] if w is not None else 0)
    bb = ConfigurableModel(**cfg).to(device).eval()
    bb.load_state_dict(sd)
    head_cfg = dict(HEAD_CONFIG); head_cfg["forecast_len"] = HORIZON
    head = QuantileForecastingHead(**head_cfg).to(device).eval()
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    return bb, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="experiments/freq-embedding/plots/synth_qhead_grid.png")
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.backbone} + {args.head} on {device}")
    bb, head = load_pair(args.backbone, args.head, device)

    # Generate N synthetic series of length T_RAW + HORIZON so we have both
    # context (first T_RAW) and ground-truth (last HORIZON).
    T_full = T_RAW + HORIZON
    X, meta = generate_periodic_batch(
        batch_size=args.n, T_raw=T_full, C=1, seed=args.seed, return_meta=True,
    )
    x = X.squeeze(-1).numpy()                # [N, T_full]

    rows, cols = 4, 3
    assert args.n == rows * cols, f"need {rows*cols} samples"

    fig, axes = plt.subplots(rows, cols, figsize=(18, 14), dpi=90)
    for i in range(args.n):
        ax = axes[i // cols, i % cols]
        full = x[i]
        context = full[:T_RAW]
        truth = full[T_RAW:T_RAW + HORIZON]
        P = max(2, int(round(meta["spp"][i])))

        # Seasonal-naive with known period
        last_period = context[-P:]
        sn = np.array([last_period[h % P] for h in range(HORIZON)], dtype=np.float32)

        # Quantile forecast
        ctx_t = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            y = forecast_with_strategy(
                "B4", bb, head, ctx_t, horizon=HORIZON, device=device)
        # y shape: (Q, horizon, C) — channel 0
        q_pred = np.asarray(y[:, :, 0], dtype=np.float32)
        median = q_pred[QUANTILE_LEVELS.index(0.5)]
        lo = q_pred[QUANTILE_LEVELS.index(0.1)]
        hi = q_pred[QUANTILE_LEVELS.index(0.9)]

        # Plot last few periods of context + horizon
        c_show = min(int(3 * P) if P < 100 else 200, len(context))
        xs_ctx = np.arange(-c_show, 0)
        xs_fut = np.arange(0, HORIZON)

        ax.plot(xs_ctx, context[-c_show:], color="gray", lw=0.9, label="context")
        ax.plot(xs_fut, truth, color="black", lw=1.8, label="truth")
        ax.plot(xs_fut, sn, color="tab:green", lw=1.0, ls="--", label="SN (P known)")
        ax.fill_between(xs_fut, lo, hi, color="tab:purple", alpha=0.20,
                        label="qhead [0.1, 0.9]")
        ax.plot(xs_fut, median, color="tab:purple", lw=1.4, label="qhead median")
        ax.axvline(0, color="k", ls="--", alpha=0.3, lw=0.5)
        ax.grid(True, alpha=0.3)

        scale = float(np.abs(np.diff(context)).mean()) + 1e-12
        m_sn = float(np.abs(sn - truth).mean() / scale)
        m_qh = float(np.abs(median - truth).mean() / scale)
        title = (f"#{i:02d} {primitive_name(meta['primitive'][i])}  P={P}  "
                 f"{'env' if meta['use_env'][i] else 'no-env'}   "
                 f"MASE: SN={m_sn:.2f}  qmed={m_qh:.2f}")
        ax.set_title(title, fontsize=8)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=5)

    fig.suptitle(
        "fe+mu backbone + R1 quantile head — 12 random synthetic samples\n"
        "(seed={}; truth black, SN green, qhead median purple, [0.1, 0.9] band)".format(args.seed),
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
