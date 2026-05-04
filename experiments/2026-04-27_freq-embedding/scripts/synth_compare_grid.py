#!/usr/bin/env python3
"""Multi-arm synth compare grid on held-out synthetic data.

3×4 (or any --rows × --cols) panel grid of DIFFERENT synth samples.
Each panel overlays ALL arm forecasts so they can be directly compared.

Each panel shows:
  - context tail (gray, last ~3 periods)
  - ground truth horizon (black, thick)
  - seasonal-naive with known period (green dashed)
  - one colored line per arm (qhead median; bands omitted for clarity)

Usage:
    python experiments/2026-04-27_freq-embedding/scripts/synth_compare_grid.py \\
        --arms 'label_a:bb_a.pth,head_a.pth label_b:bb_b.pth,head_b.pth ...' \\
        --out experiments/2026-04-27_freq-embedding/plots/synth_compare_grid.png \\
        --rows 3 --cols 4 --device cpu
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.models import ConfigurableModel
from src.forecasting_head import (
    QuantileForecastingHead, FORECAST_LEN, forecast_with_strategy,
)
from src.synthetic_periodic import generate_periodic_batch, primitive_name


BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_kind="ewma", rev_norm_span=32,
)
HEAD_CONFIG = dict(H=512, hidden_dim=128, num_gru_layers=2,
                   forecast_len=FORECAST_LEN, dropout=0.1)
T_RAW = 1024
HORIZON = 16
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_pair(backbone_path, head_path, device, rev_norm_kind="ewma",
              rev_norm_span=32):
    from src.norm import PATCH_STATS_DIM
    sd = torch.load(backbone_path, map_location=device, weights_only=True)
    cfg = dict(BACKBONE_CONFIG)
    cfg["rev_norm_kind"] = rev_norm_kind
    if rev_norm_kind == "ewma":
        cfg["rev_norm_span"] = rev_norm_span
    elif "rev_norm_span" in cfg:
        del cfg["rev_norm_span"]
    w = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = (w.shape[1] if w is not None else 0)
    ref = sd.get("encoder.skip.weight")
    if ref is None:
        ref = sd.get("encoder.linear1.weight")
    if ref is not None:
        extra = ref.shape[1] - cfg["W"] - cfg["freq_emb_dim"]
        if extra == 0:
            cfg["patch_stats_kind"] = "none"
        elif extra == PATCH_STATS_DIM:
            cfg["patch_stats_kind"] = "diff"
    bb = ConfigurableModel(**cfg).to(device).eval()
    bb.load_state_dict(sd)
    head_cfg = dict(HEAD_CONFIG); head_cfg["forecast_len"] = HORIZON
    head = QuantileForecastingHead(**head_cfg).to(device).eval()
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    return bb, head, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="Space-separated 'label:backbone.pth,head.pth'."
                         " Optionally append ',revin' or ',ewma:SPAN' to override")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    arms = []
    for entry in args.arms.split():
        if ":" not in entry:
            raise SystemExit(f"--arms entry '{entry}' must be 'label:bb,head'")
        label, rest = entry.split(":", 1)
        parts = rest.split(",")
        bb_path, head_path = parts[0], parts[1]
        rev_norm_kind = "ewma"
        rev_norm_span = 32
        if len(parts) > 2:
            spec = parts[2]
            if spec == "revin":
                rev_norm_kind = "revin"
            elif spec.startswith("ewma:"):
                rev_norm_kind = "ewma"
                rev_norm_span = int(spec.split(":", 1)[1])
        arms.append((label, bb_path, head_path, rev_norm_kind, rev_norm_span))

    n_panels = args.rows * args.cols
    print(f"  {len(arms)} arms × {n_panels} samples = "
          f"{len(arms) * n_panels} forecasts")

    # Generate n_panels random synth samples (deterministic via --seed).
    X, meta = generate_periodic_batch(
        batch_size=n_panels, T_raw=T_RAW + HORIZON, C=1,
        seed=args.seed, return_meta=True,
    )
    spp_arr = np.asarray(meta["spp"], dtype=float)
    prim_arr = np.asarray(meta["primitive"])

    # Per-arm forecasts, computed once per (arm, panel).
    arm_forecasts = []  # list of (label, color, [median per panel])
    arm_colors = ["tab:red", "tab:orange", "tab:blue", "tab:purple",
                  "tab:brown", "tab:cyan", "tab:olive", "tab:pink"]
    for ai, (label, bb_path, head_path, rk, rs) in enumerate(arms):
        print(f"  [{ai+1}/{len(arms)}] {label}: forecasting {n_panels} samples…")
        bb, head, _ = load_pair(bb_path, head_path, device,
                                rev_norm_kind=rk, rev_norm_span=rs)
        medians = []
        for p in range(n_panels):
            ctx = X[p, :T_RAW, :].to(device)
            ctx_t = ctx.unsqueeze(0)
            with torch.no_grad():
                y = forecast_with_strategy(
                    "B4", bb, head, ctx_t, horizon=HORIZON, device=device)
            q_pred = np.asarray(y[:, :, 0], dtype=np.float32)
            medians.append(q_pred[QUANTILE_LEVELS.index(0.5)])
        arm_forecasts.append((label, arm_colors[ai % len(arm_colors)],
                              np.stack(medians, axis=0)))

    fig, axes = plt.subplots(args.rows, args.cols,
                             figsize=(5 * args.cols, 3.5 * args.rows),
                             dpi=120, squeeze=False)

    for p in range(n_panels):
        r, c = p // args.cols, p % args.cols
        full = X[p, :, 0].numpy()
        context = full[:T_RAW]
        truth = full[T_RAW:T_RAW + HORIZON]
        P = max(2, int(round(spp_arr[p])))
        last_period = context[-P:]
        sn = np.array([last_period[h % P] for h in range(HORIZON)],
                      dtype=np.float32)

        ax = axes[r, c]
        c_show = min(int(3 * P) if P < 100 else 200, len(context))
        xs_ctx = np.arange(-c_show, 0)
        xs_fut = np.arange(0, HORIZON)

        ax.plot(xs_ctx, context[-c_show:], color="gray", lw=0.7,
                label="context" if p == 0 else None)
        ax.plot(xs_fut, truth, color="black", lw=2.0,
                label="truth" if p == 0 else None)
        ax.plot(xs_fut, sn, color="tab:green", lw=1.2, ls="--",
                label=f"SN (P known)" if p == 0 else None)
        for label, color, medians in arm_forecasts:
            ax.plot(xs_fut, medians[p], color=color, lw=1.1,
                    label=label if p == 0 else None)
        ax.axvline(0, color="k", ls="--", alpha=0.3, lw=0.5)
        ax.grid(True, alpha=0.3)

        scale = float(np.abs(np.diff(context)).mean()) + 1e-12
        m_sn = float(np.abs(sn - truth).mean() / scale)
        title = (f"#{p:02d}  {primitive_name(prim_arr[p])}  P={P}  "
                 f"SN MASE={m_sn:.2f}")
        ax.set_title(title, fontsize=8)
        if p == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        f"Synth-only eval: {len(arms)}-arm forecast comparison on "
        f"{n_panels} random synth samples (seed={args.seed})",
        fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
