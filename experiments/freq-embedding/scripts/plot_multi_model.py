#!/usr/bin/env python3
"""Multi-model qualitative prediction plots.

Runs 4 models + seasonal-naive + ground truth on the 6 periodic focus
configs from the periodic-synth-mix experiment, 10 random test windows
per config, one PNG per config.

Arms:
- v2 (500k backbone + original R1 head) — best-aggregate reference
- mix90 (90k synth-mix backbone + R1v3c_mix_90k head)
- fe (freq-embedding backbone, no mixup)
- fe+mu (freq-embedding backbone + mixup)
Plus: seasonal-naive, ground truth.

Needs GIFT-Eval data available locally at $GIFT_EVAL or mounted dir.
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

# Project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from src.models import ConfigurableModel
from src.forecasting_head import ForecastingHead, FORECAST_LEN, forecast_with_strategy
from gift_eval.data import Dataset as GiftDataset
from gluonts.time_feature import get_seasonality


BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
HEAD_CONFIG = dict(
    H=512, hidden_dim=128, num_gru_layers=2,
    forecast_len=FORECAST_LEN, dropout=0.1,
)
T_RAW = 1024


CONFIG_MAP = {
    "ett1/15T/short":      ("ett1/15T",   "short"),
    "ett2/W/short":        ("ett2/W",     "short"),
    "m4_hourly/H/short":   ("m4_hourly",  "short"),
    "solar/10T/long":      ("solar/10T",  "long"),
    "solar/10T/medium":    ("solar/10T",  "medium"),
    "solar/H/short":       ("solar/H",    "short"),
}


def load_model_pair(backbone_path, head_path, device):
    """Load a (backbone, head) pair, auto-detecting freq_emb_dim."""
    sd = torch.load(backbone_path, map_location=device, weights_only=True)
    w = sd.get("freq_embedding.embedding.weight")
    cfg = dict(BACKBONE_CONFIG)
    cfg["freq_emb_dim"] = (w.shape[1] if w is not None else 0)
    bb = ConfigurableModel(**cfg).to(device).eval()
    bb.load_state_dict(sd)

    head_cfg = dict(HEAD_CONFIG); head_cfg["forecast_len"] = 16
    head = ForecastingHead(**head_cfg).to(device).eval()
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    return bb, head


def _to_univariate(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.shape[0] < a.shape[1]:
        return a[0]
    return a[:, 0]


def _ffill(a):
    a = _to_univariate(a).copy()
    mask = np.isnan(a)
    if mask.all():
        a[:] = 0.0; return a
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
    t = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(-1)
    return t


def forecast_model(backbone, head, context, horizon, device):
    x = _prep_ctx(context).to(device)
    with torch.no_grad():
        y = forecast_with_strategy("B4", backbone, head, x, horizon=horizon, device=device)
    return np.asarray(y[:, 0], dtype=np.float32)


def seasonal_naive(context, horizon, season):
    n = len(context)
    if season is None or season <= 0 or season > n:
        return np.full(horizon, context[-1], dtype=np.float32)
    last = context[-season:]
    return last[np.arange(horizon) % season].astype(np.float32)


def mase_per_panel(pred, truth, ctx):
    denom = np.mean(np.abs(np.diff(ctx))) + 1e-12
    return float(np.mean(np.abs(pred - truth)) / denom)


MODEL_STYLE = {
    "truth":           dict(color="black",    lw=1.8, ls="-",  label="truth"),
    "seasonal-naive":  dict(color="tab:green", lw=1.0, ls="--", label="SN"),
    "v2 (500k)":       dict(color="tab:blue",  lw=1.1, ls="-",  label="v2 500k"),
    "mix90 (90k)":     dict(color="tab:orange",lw=1.1, ls="-",  label="mix90"),
    "fe (30k)":        dict(color="tab:red",   lw=1.1, ls="-",  label="fe 30k"),
    "fe+mu (30k)":     dict(color="tab:purple",lw=1.4, ls="-",  label="fe+mu 30k"),
}


def plot_config(cfg_name, term, horizon, season, n_samples, models, device,
                out_path, max_context_show=400):
    ds = GiftDataset(name=cfg_name, term=term, to_univariate=False)
    items = list(ds.test_data)
    rng = np.random.default_rng(42)
    if len(items) > n_samples:
        idx = np.sort(rng.choice(len(items), n_samples, replace=False))
        sel = [items[i] for i in idx]
    else:
        sel = items

    fig, axes = plt.subplots(len(sel), 1, figsize=(16, 2.8 * len(sel)), dpi=90)
    if len(sel) == 1:
        axes = [axes]

    for ax, item in zip(axes, sel):
        if isinstance(item, tuple):
            ctx_raw = item[0]["target"]
            truth_raw = item[1]["target"]
        else:
            full = item["target"]
            ctx_raw = full[:-horizon]
            truth_raw = full[-horizon:]
        context = _ffill(ctx_raw)
        truth = _ffill(truth_raw)
        H = len(truth)

        # Compute forecasts from every arm + SN
        arm_preds = {}
        arm_preds["seasonal-naive"] = seasonal_naive(context, H, season)
        for name, (bb, hd) in models.items():
            arm_preds[name] = forecast_model(bb, hd, context, H, device)

        # Plot
        c_show = min(max_context_show, len(context))
        xs_ctx = np.arange(-c_show, 0)
        xs_fut = np.arange(0, H)
        ax.plot(xs_ctx, context[-c_show:], color="gray", linewidth=0.8, label="context")
        ax.plot(xs_fut, truth, **MODEL_STYLE["truth"])
        ax.plot(xs_fut, arm_preds["seasonal-naive"], **MODEL_STYLE["seasonal-naive"])
        # Model curves in consistent order
        for name in ["v2 (500k)", "mix90 (90k)", "fe (30k)", "fe+mu (30k)"]:
            if name in arm_preds:
                ax.plot(xs_fut, arm_preds[name], **MODEL_STYLE[name])
        ax.axvline(0, color="k", ls="--", alpha=0.3, lw=0.5)
        ax.grid(True, alpha=0.3)
        # Per-panel MASE
        mase_parts = []
        mase_sn = mase_per_panel(arm_preds["seasonal-naive"], truth, context)
        mase_parts.append(f"SN={mase_sn:.2f}")
        for name in ["v2 (500k)", "mix90 (90k)", "fe (30k)", "fe+mu (30k)"]:
            if name in arm_preds:
                m = mase_per_panel(arm_preds[name], truth, context)
                short = name.split()[0]
                mase_parts.append(f"{short}={m:.2f}")
        ax.set_title(
            f"{cfg_name}/{term}   P={season}   H={H}   MASE: {'  '.join(mase_parts)}",
            fontsize=8,
        )
        ax.legend(loc="upper left", fontsize=6, ncol=6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-backbone", required=True)
    ap.add_argument("--v2-head", required=True)
    ap.add_argument("--mix90-backbone", required=True)
    ap.add_argument("--mix90-head", required=True)
    ap.add_argument("--fe-backbone", required=True)
    ap.add_argument("--fe-head", required=True)
    ap.add_argument("--femu-backbone", required=True)
    ap.add_argument("--femu-head", required=True)
    ap.add_argument("--output-dir", default="experiments/freq-embedding/plots/predictions")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading models on {device}...")
    models = {}
    models["v2 (500k)"]   = load_model_pair(args.v2_backbone,   args.v2_head,   device)
    models["mix90 (90k)"] = load_model_pair(args.mix90_backbone, args.mix90_head, device)
    models["fe (30k)"]    = load_model_pair(args.fe_backbone,   args.fe_head,   device)
    models["fe+mu (30k)"] = load_model_pair(args.femu_backbone, args.femu_head, device)

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
        plot_config(ds_name, term, H, season, args.n_samples, models, device, out_path)


if __name__ == "__main__":
    main()
