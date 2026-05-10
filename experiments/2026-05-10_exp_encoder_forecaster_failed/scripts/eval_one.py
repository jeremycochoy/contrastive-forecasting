#!/usr/bin/env python3
"""N=50 multisample held-out eval — encoder+forecaster arm vs τ=0.10 baseline.

Same data, same model construction, same metrics as
`experiments/2026-05-08_exp_tau_sweep/scripts/eval_multisample.py`. Reuses
its helpers; writes two arms to:
  experiments/2026-05-10_exp_encoder_forecaster/results/
      encoder_forecaster_metrics_persample_n50.csv
      encoder_forecaster_metrics_multisample_n50.csv

Run from the worktree (so num_encoder_layers detection + extract_encoder_latents
encoder-pass are picked up):
    cd /home/jupyter/cf-encoder-forecaster
    PYTHONPATH=. python experiments/2026-05-10_exp_encoder_forecaster/scripts/eval_one.py
"""

from __future__ import annotations

import csv
import os
import statistics
import sys

import torch

# Reuse the canonical eval helpers.
_TAU_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "2026-05-08_exp_tau_sweep", "scripts"))
if _TAU_DIR not in sys.path:
    sys.path.insert(0, _TAU_DIR)

from eval_multisample import (  # noqa: E402
    BATCH_SIZE, T_RAW, T, HF_REPO, HF_PATH,
    N_SAMPLES, SKIP_ROWS_LIST, METRIC_KEYS, TAU_COLUMNS,
    build_backbone, eval_one_sample, load_held_out_batch, pick_device,
)


REPO_ROOT = os.environ.get(
    "CFCAST_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
EXP_DIR = os.path.join(REPO_ROOT, "experiments", "2026-05-10_exp_encoder_forecaster")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
OUT_PERSAMPLE = os.path.join(RESULTS_DIR, "encoder_forecaster_metrics_persample_n50.csv")
OUT_MULTISAMPLE = os.path.join(RESULTS_DIR, "encoder_forecaster_metrics_multisample_n50.csv")

# (display_name, tau, declared_encoder_type, relpath_to_FINAL_pth_under_REPO_ROOT)
BACKBONES: list[tuple[str, float, str, str]] = [
    ("enc_fcst_tau_0_10_50k", 0.10, "gru",
     "checkpoints/enc_fcst_tau_0_10_50k_FINAL.pth"),
    ("tau_sweep_0_10_baseline", 0.10, "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_10_FINAL.pth"),
]

PERSAMPLE_COLUMNS = (
    ["name", "tau", "encoder_type", "sample_idx", "skip_rows"] + METRIC_KEYS
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = pick_device()
    print(f"[eval] device={device}", flush=True)

    persample_rows = []
    aggregated_rows = []

    for name, tau, declared_enc, relpath in BACKBONES:
        ckpt_path = os.path.join(REPO_ROOT, relpath)
        if not os.path.isfile(ckpt_path):
            print(f"[eval] SKIP {name}: {ckpt_path} not found", file=sys.stderr)
            continue

        print(f"\n[eval] === {name}  (tau={tau})  {ckpt_path}", flush=True)
        sd = torch.load(ckpt_path, map_location="cpu")
        bb, enc_type = build_backbone(name, declared_enc, sd)
        if bb is None:
            print(f"[eval] SKIP {name}: build_backbone failed", file=sys.stderr)
            continue
        bb = bb.to(device).eval()

        # Per-arm parameter counts (factual, for the report).
        total_p = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        enc_p = sum(p.numel() for layer in bb.transformer.encoder_layers
                    for p in layer.parameters() if p.requires_grad)
        fcst_p = sum(p.numel() for layer in bb.transformer.layers
                     for p in layer.parameters() if p.requires_grad)
        n_enc_layers = len(bb.transformer.encoder_layers)
        n_fcst_layers = len(bb.transformer.layers)
        print(f"  params: total={total_p:,}  "
              f"enc({n_enc_layers}L)={enc_p:,}  fcst({n_fcst_layers}L)={fcst_p:,}",
              flush=True)

        per_metric: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
        for i, skip_rows in enumerate(SKIP_ROWS_LIST):
            x_dev, freq_ids, seas_ids = load_held_out_batch(skip_rows, device)
            row = eval_one_sample(bb, bb.H, x_dev, freq_ids, seas_ids)
            for k in METRIC_KEYS:
                per_metric[k].append(float(row[k]))
            persample_rows.append({
                "name": name, "tau": tau, "encoder_type": enc_type,
                "sample_idx": i, "skip_rows": skip_rows,
                **{k: float(row[k]) for k in METRIC_KEYS},
            })
            if (i + 1) % 10 == 0 or i == len(SKIP_ROWS_LIST) - 1:
                print(f"    [{i+1}/{len(SKIP_ROWS_LIST)}] "
                      f"auc={row['auc']:.4f} top1={row['top1']:.4f} "
                      f"r2_naive={row['r2_naive']:.4f}",
                      flush=True)

        agg = {"name": name, "tau": tau, "encoder_type": enc_type}
        for k in METRIC_KEYS:
            agg[f"{k}_mean"] = statistics.mean(per_metric[k])
            agg[f"{k}_std"] = statistics.stdev(per_metric[k]) \
                if len(per_metric[k]) > 1 else 0.0
        agg["n_samples"] = len(per_metric[METRIC_KEYS[0]])
        aggregated_rows.append(agg)
        print(f"  AGG: auc={agg['auc_mean']:.4f}±{agg['auc_std']:.4f}  "
              f"top1={agg['top1_mean']:.4f}±{agg['top1_std']:.4f}  "
              f"r2_naive={agg['r2_naive_mean']:.4f}±{agg['r2_naive_std']:.4f}",
              flush=True)

        del bb, sd
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not aggregated_rows:
        print("[eval] No backbones evaluated; nothing written.", file=sys.stderr)
        sys.exit(1)

    with open(OUT_PERSAMPLE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERSAMPLE_COLUMNS)
        w.writeheader()
        for r in persample_rows:
            w.writerow(r)
    with open(OUT_MULTISAMPLE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TAU_COLUMNS)
        w.writeheader()
        for r in aggregated_rows:
            w.writerow(r)

    print(f"\n[eval] wrote {OUT_PERSAMPLE}", flush=True)
    print(f"[eval] wrote {OUT_MULTISAMPLE}", flush=True)


if __name__ == "__main__":
    main()
