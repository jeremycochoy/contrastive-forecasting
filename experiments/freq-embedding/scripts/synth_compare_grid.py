#!/usr/bin/env python3
"""4-arm × 3-sample comparison grid on held-out synthetic data.

Rows = model arms (e.g. fe+mu peak_gap / 30k / 60k / RevIN-synth).
Cols = 3 picked synth samples (short / medium / long period).

Each panel shows:
  - context tail (gray, last ~3 periods)
  - ground truth horizon (black)
  - seasonal-naive with known period (green dashed)
  - quantile head median (purple)
  - quantile head [0.1, 0.9] band (purple shaded)

Usage:
    python experiments/freq-embedding/scripts/synth_compare_grid.py \\
        --arms 'label_a:bb_a.pth,head_a.pth label_b:bb_b.pth,head_b.pth ...' \\
        --out experiments/freq-embedding/plots/synth_compare_grid.png \\
        --short-period 16 --medium-period 64 --long-period 200 \\
        --device cpu
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


def pick_samples_with_periods(target_periods, seed, search_pool=2048):
    """Generate `search_pool` random synth samples and pick the ones whose
    spp is closest to each target. Returns the sample tensors and metadata.
    Uses C=1 to mirror plot_synth_qhead."""
    X, meta = generate_periodic_batch(
        batch_size=search_pool, T_raw=T_RAW + HORIZON, C=1,
        seed=seed, return_meta=True,
    )
    spp = np.asarray(meta["spp"], dtype=float)
    picks = []
    for tgt in target_periods:
        idx = int(np.argmin(np.abs(spp - tgt)))
        picks.append(idx)
    sub_X = torch.stack([X[i] for i in picks], dim=0)  # (K, T_full, 1)
    sub_meta = {
        "spp": spp[picks],
        "primitive": np.asarray(meta["primitive"])[picks],
        "use_env": np.asarray(meta.get("use_env", [False] * search_pool))[picks],
    }
    return sub_X, sub_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="Space-separated 'label:backbone.pth,head.pth'."
                         " Optionally append ',revin' or ',ewma:SPAN' to override")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--short-period", type=float, default=16.0)
    ap.add_argument("--medium-period", type=float, default=64.0)
    ap.add_argument("--long-period", type=float, default=200.0)
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
        # Optional norm spec
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

    # Pick 3 representative samples
    target_periods = [args.short_period, args.medium_period, args.long_period]
    X, meta = pick_samples_with_periods(target_periods, seed=args.seed)
    n_rows, n_cols = len(arms), 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), dpi=120,
                             squeeze=False)

    for r, (label, bb_path, head_path, rk, rs) in enumerate(arms):
        print(f"[{r+1}/{n_rows}] {label}: {bb_path}")
        bb, head, cfg = load_pair(bb_path, head_path, device,
                                  rev_norm_kind=rk, rev_norm_span=rs)
        for c in range(n_cols):
            full = X[c, :, 0].numpy()
            context = full[:T_RAW]
            truth = full[T_RAW:T_RAW + HORIZON]
            P = max(2, int(round(meta["spp"][c])))

            # Seasonal-naive with known period
            last_period = context[-P:]
            sn = np.array([last_period[h % P] for h in range(HORIZON)],
                          dtype=np.float32)

            ctx_t = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                y = forecast_with_strategy(
                    "B4", bb, head, ctx_t, horizon=HORIZON, device=device)
            q_pred = np.asarray(y[:, :, 0], dtype=np.float32)
            median = q_pred[QUANTILE_LEVELS.index(0.5)]
            lo = q_pred[QUANTILE_LEVELS.index(0.1)]
            hi = q_pred[QUANTILE_LEVELS.index(0.9)]

            ax = axes[r, c]
            c_show = min(int(3 * P) if P < 100 else 200, len(context))
            xs_ctx = np.arange(-c_show, 0)
            xs_fut = np.arange(0, HORIZON)
            ax.plot(xs_ctx, context[-c_show:], color="gray", lw=0.9, label="context")
            ax.plot(xs_fut, truth, color="black", lw=1.8, label="truth")
            ax.plot(xs_fut, sn, color="tab:green", lw=1.0, ls="--",
                    label=f"SN (P={P})")
            ax.fill_between(xs_fut, lo, hi, color="tab:purple", alpha=0.18,
                            label="qhead [0.1, 0.9]")
            ax.plot(xs_fut, median, color="tab:purple", lw=1.4,
                    label="qhead median")
            ax.axvline(0, color="k", ls="--", alpha=0.3, lw=0.5)
            ax.grid(True, alpha=0.3)

            scale = float(np.abs(np.diff(context)).mean()) + 1e-12
            m_sn = float(np.abs(sn - truth).mean() / scale)
            m_qh = float(np.abs(median - truth).mean() / scale)
            ax.set_title(
                f"{label} | {primitive_name(meta['primitive'][c])} P={P}  "
                f"MASE: SN={m_sn:.2f} qmed={m_qh:.2f}",
                fontsize=8)
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "Synth-only eval: backbone + qhead forecasts vs ground truth\n"
        f"3 representative samples (short / medium / long period; seed={args.seed})",
        fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
