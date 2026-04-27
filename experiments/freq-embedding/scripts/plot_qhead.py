#!/usr/bin/env python3
"""Focused multi-model prediction plots with quantile uncertainty band.

Per user feedback (5+ curves was too cluttered): show only what's needed
to compare the new architecture against its baseline.

Per-panel:
- ground truth (black)
- seasonal-naive (green dashed)
- fe+mu (30k MSE head)        — orange line          [point baseline]
- qhead median (q=0.5)         — purple line          [new architecture]
- qhead [0.1, 0.9] band        — purple semi-transparent fill

Same 6 periodic focus configs, 10 random test windows each.
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
    ForecastingHead, QuantileForecastingHead, FORECAST_LEN,
    forecast_with_strategy,
)
from gift_eval.data import Dataset as GiftDataset
from gluonts.time_feature import get_seasonality


BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
HEAD_CONFIG = dict(H=512, hidden_dim=128, num_gru_layers=2,
                    forecast_len=FORECAST_LEN, dropout=0.1)
T_RAW = 1024


CONFIG_MAP = {
    "ett1/15T/short":   ("ett1/15T",  "short"),
    "ett2/W/short":     ("ett2/W",    "short"),
    "m4_hourly/H/short":("m4_hourly", "short"),
    "solar/10T/long":   ("solar/10T", "long"),
    "solar/10T/medium": ("solar/10T", "medium"),
    "solar/H/short":    ("solar/H",   "short"),
}


def load_pair(backbone_path, head_path, device, quantile=False):
    sd = torch.load(backbone_path, map_location=device, weights_only=True)
    w = sd.get("freq_embedding.embedding.weight")
    cfg = dict(BACKBONE_CONFIG)
    cfg["freq_emb_dim"] = (w.shape[1] if w is not None else 0)
    bb = ConfigurableModel(**cfg).to(device).eval()
    bb.load_state_dict(sd)
    head_cfg = dict(HEAD_CONFIG); head_cfg["forecast_len"] = 16
    if quantile:
        head = QuantileForecastingHead(**head_cfg).to(device).eval()
    else:
        head = ForecastingHead(**head_cfg).to(device).eval()
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    return bb, head


def _to_uni(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1: return a
    return a[0] if a.shape[0] < a.shape[1] else a[:, 0]


def _ffill(a):
    a = _to_uni(a).copy()
    mask = np.isnan(a)
    if mask.all(): a[:] = 0.0; return a
    first = np.where(~mask)[0][0]
    a[:first] = a[first]
    for i in range(1, len(a)):
        if np.isnan(a[i]): a[i] = a[i-1]
    return a


def _prep_ctx(target, t_raw=T_RAW):
    n = len(target)
    if n >= t_raw:
        ctx = target[-t_raw:]
    else:
        ctx = np.concatenate([np.full(t_raw - n, target[0], dtype=np.float32), target])
    return torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)


def forecast_point(backbone, head, context, horizon, device):
    """MSE head: returns (horizon,) point forecast."""
    x = _prep_ctx(context).to(device)
    with torch.no_grad():
        y = forecast_with_strategy("B4", backbone, head, x, horizon=horizon, device=device)
    return np.asarray(y[:, 0], dtype=np.float32)


def forecast_quantiles(backbone, head, context, horizon, device):
    """Quantile head: returns (Q, horizon) array."""
    x = _prep_ctx(context).to(device)
    with torch.no_grad():
        y = forecast_with_strategy("B4", backbone, head, x, horizon=horizon, device=device)
    # y shape: (Q, horizon, C) — take channel 0
    return np.asarray(y[:, :, 0], dtype=np.float32)


def seasonal_naive(context, horizon, season):
    n = len(context)
    if season is None or season <= 0 or season > n:
        return np.full(horizon, context[-1], dtype=np.float32)
    last = context[-season:]
    return last[np.arange(horizon) % season].astype(np.float32)


def mase(p, t, ctx):
    denom = np.mean(np.abs(np.diff(ctx))) + 1e-12
    return float(np.mean(np.abs(p - t)) / denom)


QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_IDX = QUANTILE_LEVELS.index(0.5)
LO_IDX = QUANTILE_LEVELS.index(0.1)
HI_IDX = QUANTILE_LEVELS.index(0.9)


def plot_config(cfg_name, term, horizon, season, n_samples,
                femu_pair, qhead_pair, device, out_path):
    ds = GiftDataset(name=cfg_name, term=term, to_univariate=False)
    items = list(ds.test_data)
    rng = np.random.default_rng(42)
    if len(items) > n_samples:
        idx = np.sort(rng.choice(len(items), n_samples, replace=False))
        sel = [items[i] for i in idx]
    else:
        sel = items

    fig, axes = plt.subplots(len(sel), 1, figsize=(15, 2.6 * len(sel)), dpi=90)
    if len(sel) == 1:
        axes = [axes]

    for ax, item in zip(axes, sel):
        if isinstance(item, tuple):
            context = _ffill(item[0]["target"])
            truth = _ffill(item[1]["target"])
        else:
            full = _ffill(item["target"])
            context = full[:-horizon]; truth = full[-horizon:]
        H = len(truth)

        sn = seasonal_naive(context, H, season)
        femu = forecast_point(*femu_pair, context, H, device)
        qhead_q = forecast_quantiles(*qhead_pair, context, H, device)
        qmedian = qhead_q[MEDIAN_IDX]
        qlo = qhead_q[LO_IDX]
        qhi = qhead_q[HI_IDX]

        c_show = min(400, len(context))
        xs_ctx = np.arange(-c_show, 0)
        xs_fut = np.arange(0, H)

        ax.plot(xs_ctx, context[-c_show:], color="gray", linewidth=0.8, label="context")
        ax.plot(xs_fut, truth, color="black", linewidth=1.8, label="truth")
        ax.plot(xs_fut, sn, color="tab:green", linewidth=1.0, linestyle="--", label="SN")
        ax.plot(xs_fut, femu, color="tab:orange", linewidth=1.2, label="fe+mu (MSE)")
        # Quantile band — must come BEFORE the median line so the line is on top.
        ax.fill_between(xs_fut, qlo, qhi, color="tab:purple", alpha=0.20,
                        label="qhead [0.1, 0.9]")
        ax.plot(xs_fut, qmedian, color="tab:purple", linewidth=1.4,
                label="qhead median")
        ax.axvline(0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.grid(True, alpha=0.3)

        m_sn = mase(sn, truth, context)
        m_femu = mase(femu, truth, context)
        m_qmed = mase(qmedian, truth, context)
        ax.set_title(
            f"{cfg_name}/{term}   P={season}   H={H}   "
            f"MASE: SN={m_sn:.2f}  fe+mu={m_femu:.2f}  qmed={m_qmed:.2f}",
            fontsize=9,
        )
        ax.legend(loc="upper left", fontsize=7, ncol=6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--femu-backbone", required=True)
    ap.add_argument("--femu-head", required=True)
    ap.add_argument("--qhead-backbone", required=True,
                    help="(typically the same fe+mu backbone)")
    ap.add_argument("--qhead-head", required=True)
    ap.add_argument("--output-dir", default="experiments/freq-embedding/plots/predictions_qhead")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Loading models...")
    femu_pair  = load_pair(args.femu_backbone,  args.femu_head,  device, quantile=False)
    qhead_pair = load_pair(args.qhead_backbone, args.qhead_head, device, quantile=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for cfg in args.configs:
        if cfg not in CONFIG_MAP:
            print(f"  SKIP {cfg}"); continue
        ds_name, term = CONFIG_MAP[cfg]
        print(f"\n=== {cfg} ===")
        ds = GiftDataset(name=ds_name, term=term, to_univariate=False)
        H = ds.prediction_length
        season = get_seasonality(ds.freq)
        if season <= 1:
            if "15T" in ds_name: season = 96
            elif "10T" in ds_name: season = 144
            elif "5T" in ds_name: season = 288
            elif ds_name.endswith("/H"): season = 24
            elif ds_name.endswith("/D"): season = 7
            elif ds_name.endswith("/W"): season = 52
        print(f"  horizon={H} season={season}")
        out_path = os.path.join(args.output_dir, cfg.replace("/", "_") + ".png")
        plot_config(ds_name, term, H, season, args.n_samples,
                    femu_pair, qhead_pair, device, out_path)


if __name__ == "__main__":
    main()
