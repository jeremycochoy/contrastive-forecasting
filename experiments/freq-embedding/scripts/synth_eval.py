#!/usr/bin/env python3
"""Eval a backbone+quantile-head pair on a held-out synthetic test set.

Generates a fresh batch from src.synthetic_periodic.generate_periodic_batch
with a SEED that is never used during training, splits each series into
context (T_RAW=1024) + truth (HORIZON=16), and computes per-sample MASE
and WQL averaged over the batch.

This deliberately stays in the *training distribution* — the question is
whether the backbone has learned to reproduce its own data.

MASE definition (per-sample, per-channel):
    mase = mean(|qmedian - truth|) / scale
    scale = mean(|series[1:] - series[:-1]|)  (1-step seasonal-naive scale)

WQL definition (mean weighted-sum quantile loss):
    qloss(q, p, t) = max(q*(t-p), (q-1)*(t-p))
    wql = sum_q[ qloss[q] ] / sum |t|
    averaged across samples and channels.

Output:
- A row appended to `results/synth_eval/all_results.csv` with arm name,
  steps, n_samples, gm_mase, gm_wql, plus seasonal-naive baseline values
  for the same samples.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.models import ConfigurableModel
from src.forecasting_head import (
    QuantileForecastingHead, FORECAST_LEN, forecast_with_strategy,
)
from src.synthetic_periodic import generate_periodic_batch


BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_kind="ewma",
)
HEAD_CONFIG = dict(H=512, hidden_dim=128, num_gru_layers=2,
                   forecast_len=FORECAST_LEN, dropout=0.1)
T_RAW = 1024
HORIZON = 16
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_pair(backbone_path, head_path, device, rev_norm_span=32):
    from src.norm import PATCH_STATS_DIM
    sd = torch.load(backbone_path, map_location=device, weights_only=True)
    cfg = dict(BACKBONE_CONFIG)
    cfg["rev_norm_span"] = rev_norm_span
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
        else:
            raise ValueError(
                f"Unexpected encoder in_features={ref.shape[1]}; extra "
                f"width={extra} (expected 0 or {PATCH_STATS_DIM}).")
    bb = ConfigurableModel(**cfg).to(device).eval()
    bb.load_state_dict(sd)
    head_cfg = dict(HEAD_CONFIG); head_cfg["forecast_len"] = HORIZON
    head = QuantileForecastingHead(**head_cfg).to(device).eval()
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    return bb, head, cfg


def quantile_loss(q_preds, truth, levels):
    """q_preds: (Q, T, C); truth: (T, C); levels: tuple of len Q.
    Returns sum_q over each (T, C) of pinball loss."""
    Q, T, C = q_preds.shape
    out = np.zeros_like(truth)  # (T, C)
    for qi, q in enumerate(levels):
        diff = truth - q_preds[qi]
        out += np.maximum(q * diff, (q - 1) * diff)
    return out  # (T, C), already summed over Q


def compute_metrics(forecast_q, truth, scale, levels):
    """forecast_q: (Q, T, C); truth: (T, C); scale: per-(C,) scale."""
    median_idx = levels.index(0.5)
    median = forecast_q[median_idx]                                 # (T, C)
    abs_err = np.abs(median - truth)                                # (T, C)
    mase_per_c = abs_err.mean(axis=0) / np.maximum(scale, 1e-8)     # (C,)
    qloss = quantile_loss(forecast_q, truth, levels)                # (T, C)
    wql_per_c = qloss.sum(axis=0) / np.maximum(np.abs(truth).sum(axis=0), 1e-8)
    return mase_per_c, wql_per_c


def seasonal_naive_forecast(context, P, horizon):
    """context: (T, C); P: (C,) array of periods; returns (horizon, C)."""
    T, C = context.shape
    out = np.zeros((horizon, C), dtype=context.dtype)
    for c in range(C):
        Pc = max(2, int(P[c]))
        last_period = context[-Pc:, c]
        for h in range(horizon):
            out[h, c] = last_period[h % Pc]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--arm", required=True, help="Label written to the output CSV row.")
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=99999999,
                    help="Eval-only seed; never used during training.")
    ap.add_argument("--out-csv", default="results/synth_eval/all_results.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rev-norm-span", type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.backbone} + {args.head} on {device}")
    bb, head, cfg = load_pair(args.backbone, args.head, device,
                              rev_norm_span=args.rev_norm_span)
    print(f"  freq_emb_dim={cfg['freq_emb_dim']} "
          f"patch_stats_kind={cfg['patch_stats_kind']} "
          f"rev_norm_span={cfg.get('rev_norm_span', 'N/A')}")

    n_done = 0
    mase_all = []
    wql_all = []
    sn_mase_all = []
    sn_wql_all = []

    # Use C=1 single-channel samples (mirroring plot_synth_qhead.py).
    # generate_periodic_batch returns spp of shape [batch_size * C], so with
    # C=1 spp is one period per sample — clean and unambiguous. The trained
    # backbone handles arbitrary C at inference (channel-mixing is a no-op
    # at C=1).
    rng = np.random.default_rng(args.seed)
    while n_done < args.n_samples:
        bs = min(args.batch_size, args.n_samples - n_done)
        sub_seed = int(rng.integers(0, 2**31 - 1))
        X, meta = generate_periodic_batch(
            batch_size=bs, T_raw=T_RAW + HORIZON, C=1,
            seed=sub_seed, return_meta=True,
        )
        x_full = X.numpy()                              # (bs, T_full, 1)
        ctx = x_full[:, :T_RAW, :]                      # (bs, T_RAW, 1)
        truth = x_full[:, T_RAW:T_RAW + HORIZON, :]    # (bs, H, 1)
        spp = np.asarray(meta["spp"], dtype=float)      # shape (bs,) since C=1

        ctx_t = torch.from_numpy(ctx).float().to(device)
        with torch.no_grad():
            for b in range(bs):
                ctx_b = ctx_t[b:b+1]                    # (1, T_RAW, 1)
                y = forecast_with_strategy(
                    "B4", bb, head, ctx_b, horizon=HORIZON, device=device)
                # y: (Q, H, 1)
                forecast_q = np.asarray(y, dtype=np.float64)
                truth_b = truth[b].astype(np.float64)   # (H, 1)
                scale = np.abs(np.diff(ctx[b], axis=0)).mean(axis=0)  # (1,)
                mase_per_c, wql_per_c = compute_metrics(
                    forecast_q, truth_b, scale, QUANTILE_LEVELS)
                mase_all.append(mase_per_c)
                wql_all.append(wql_per_c)

                # Seasonal-naive baseline: known period — essentially optimal
                # on this clean data. Each sample has one P (since C=1).
                P_per_c = np.array([spp[b]], dtype=float)  # (1,)
                sn = seasonal_naive_forecast(
                    ctx[b].astype(np.float64), P_per_c, HORIZON)  # (H, 1)
                sn_q = np.broadcast_to(sn[None, :, :], (9, HORIZON, 1))
                sn_mase_c, sn_wql_c = compute_metrics(
                    sn_q, truth_b, scale, QUANTILE_LEVELS)
                sn_mase_all.append(sn_mase_c)
                sn_wql_all.append(sn_wql_c)
        n_done += bs
        if n_done % 256 == 0 or n_done == args.n_samples:
            print(f"  [{n_done}/{args.n_samples}] running…", flush=True)

    mase_all = np.concatenate(mase_all)            # (n_samples * C,)
    wql_all = np.concatenate(wql_all)
    sn_mase_all = np.concatenate(sn_mase_all)
    sn_wql_all = np.concatenate(sn_wql_all)

    def gm(x):
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) == 0:
            return float("nan")
        return float(np.exp(np.mean(np.log(x))))

    gm_mase = gm(mase_all)
    gm_wql = gm(wql_all)
    sn_gm_mase = gm(sn_mase_all)
    sn_gm_wql = gm(sn_wql_all)
    rel_mase = gm_mase / sn_gm_mase
    rel_wql = gm_wql / sn_gm_wql
    skill_mase = (1 - rel_mase) * 100
    skill_wql = (1 - rel_wql) * 100

    print()
    print(f"  arm = {args.arm}")
    print(f"  GM-MASE      = {gm_mase:.4f}")
    print(f"  GM-WQL       = {gm_wql:.4f}")
    print(f"  SN GM-MASE   = {sn_gm_mase:.4f}")
    print(f"  SN GM-WQL    = {sn_gm_wql:.4f}")
    print(f"  MASE skill   = {skill_mase:+.1f}%   (vs SN; higher is better)")
    print(f"  WQL skill    = {skill_wql:+.1f}%")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "arm", "n_samples", "n_per_channel", "freq_emb_dim",
                "patch_stats_kind", "rev_norm_span",
                "gm_mase", "gm_wql", "sn_gm_mase", "sn_gm_wql",
                "mase_skill_pct", "wql_skill_pct",
            ])
        w.writerow([
            args.arm, args.n_samples, len(mase_all),
            cfg["freq_emb_dim"], cfg["patch_stats_kind"], cfg.get("rev_norm_span", ""),
            f"{gm_mase:.5f}", f"{gm_wql:.5f}",
            f"{sn_gm_mase:.5f}", f"{sn_gm_wql:.5f}",
            f"{skill_mase:.2f}", f"{skill_wql:.2f}",
        ])
    print(f"  appended row to {args.out_csv}")


if __name__ == "__main__":
    main()
