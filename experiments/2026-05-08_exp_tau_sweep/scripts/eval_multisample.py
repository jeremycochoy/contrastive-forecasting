#!/usr/bin/env python3
"""Multi-sample (N=10) backbone diagnostic for τ-sweep + loss-extension arms.

Adapted from `eval_tau_sweep_metrics_v2.py` — same architecture detection,
same 6 metrics, same B=256 forward pass — but draws N=10 disjoint held-out
batches via 10 different `skip_rows` values, then aggregates mean ± stdev
per backbone.

The 10 skip_rows values are spaced (total_rows // 10) apart, starting at
50M, so each wraps to a different region of the 42.7M-row corpus and the
batches do not overlap (4.27M-row stride ≫ B=256).

Outputs two CSVs (one per experiment), each with mean/std columns:
  experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_multisample.csv
  experiments/2026-05-09_exp_loss_extensions/results/loss_extensions_metrics_multisample.csv

Run:
    PYTHONPATH=. python experiments/2026-05-08_exp_tau_sweep/scripts/eval_multisample.py
"""

from __future__ import annotations

import csv
import os
import statistics
import sys

import torch

import src.dataloader as dataloader
from src.dataloader import create_hf_dataloader
from src.forecasting_head import (
    extract_encoder_latents,
    extract_forecaster_latents,
)
from src.metrics import (
    dim_usage,
    q_naive_latent,
    q_random,
    retrieval_auc_top1,
)
from src.models import ConfigurableModel


# Run from the main repo by default, but allow override (so this script in a
# worktree can still read sync_* checkpoints from the main checkout).
REPO_ROOT = os.environ.get(
    "CFCAST_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
EXP_TAU_DIR = os.path.join(REPO_ROOT, "experiments", "2026-05-08_exp_tau_sweep")
EXP_LOSS_DIR = os.path.join(REPO_ROOT, "experiments", "2026-05-09_exp_loss_extensions")
TAU_OUT_CSV = os.path.join(EXP_TAU_DIR, "results", "tau_sweep_metrics_multisample.csv")
LOSS_OUT_CSV = os.path.join(EXP_LOSS_DIR, "results", "loss_extensions_metrics_multisample.csv")

BATCH_SIZE = 256
T_RAW = 4096
W = 16
T = T_RAW // W   # 256
HF_REPO = "jeremycochoy/gift-pretrain-full-4096"
HF_PATH = "small_v1"
TOTAL_ROWS = 42_740_000

# N well-spread skip_rows values. The dataloader wraps mod total_rows, so
# these land at N distinct row offsets covering the entire corpus.
N_SAMPLES = 50
SKIP_ROWS_LIST = [50_000_000 + i * (TOTAL_ROWS // N_SAMPLES) for i in range(N_SAMPLES)]

BACKBONE_BASE = dict(
    C=1, H=384, W=W, num_layers=6,
    nhead=6, ffn_mult=4.0, activation="gelu",
    depthwise_conv=3, dropout=0.1,
)

# τ-sweep arms: (display_name, tau_for_csv, encoder_type, relpath_FINAL)
TAU_BACKBONES: list[tuple[str, float, str, str]] = [
    ("tau_sweep_0_03",            0.03, "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_03_FINAL.pth"),
    ("tau_sweep_0_05",            0.05, "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_05_FINAL.pth"),
    ("tau_sweep_0_07",            0.07, "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_07_FINAL.pth"),
    ("tau_sweep_0_10",            0.10, "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_10_FINAL.pth"),
    ("tau_sweep_0_20",            0.20, "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_20_FINAL.pth"),
    ("tau_sweep_0_20_v2",         0.20, "gru",
     "sync_tau_sweep_arm5_v2/checkpoints/tau_sweep_0_20_v2_FINAL.pth"),
    ("tau_sweep_learnable_0_10",  0.10, "gru",
     "sync_tau_sweep_learnable/checkpoints/tau_sweep_learnable_0_10_FINAL.pth"),
]

# Loss-extension arms (baseline + 3 extensions).
LOSS_BACKBONES: list[tuple[str, str, str, str]] = [
    ("tau_sweep_0_20",            "baseline_cosine_similarity_batch",          "gru",
     "sync_tau_sweep/checkpoints/tau_sweep_0_20_FINAL.pth"),
    ("exp3_pos_htft_tau_0_20",    "cosine_similarity_batch_add_pos_htft",      "gru",
     "sync_exp3_pos_htft/checkpoints/exp3_pos_htft_tau_0_20_FINAL.pth"),
    ("exp4_only_fnegs_tau_0_20",  "cosine_similarity_batch_add_f_cross_negs",  "gru",
     "sync_exp4_only_fnegs/checkpoints/exp4_only_fnegs_tau_0_20_FINAL.pth"),
    ("exp5_skip_fnegs_tau_0_20",  "cosine_similarity_batch_add_skip_f_negs",   "gru",
     "sync_exp5_skip_fnegs/checkpoints/exp5_skip_fnegs_tau_0_20_FINAL.pth"),
]

METRIC_KEYS = ["r2_random", "r2_naive", "u_temporal", "u_batch", "auc", "top1"]

TAU_COLUMNS = (
    ["name", "tau", "encoder_type"]
    + [f"{m}_mean" for m in METRIC_KEYS]
    + [f"{m}_std" for m in METRIC_KEYS]
    + ["n_samples"]
)
LOSS_COLUMNS = (
    ["name", "loss_shape", "encoder_type"]
    + [f"{m}_mean" for m in METRIC_KEYS]
    + [f"{m}_std" for m in METRIC_KEYS]
    + ["n_samples"]
)


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        free_b, _ = torch.cuda.mem_get_info(i)
        free.append((free_b // (1024 * 1024), i))
    free.sort(reverse=True)
    free_mib, idx = free[0]
    if free_mib < 6000:
        return "cpu"
    return f"cuda:{idx}"


def detect_encoder_type(sd: dict, fallback: str = "gru") -> str:
    keys = list(sd.keys())
    if any(k.startswith("encoder.gru.") for k in keys):
        return "gru"
    if any(k.startswith("encoder.mlp.") for k in keys):
        return "residual_silu"
    if any(k.startswith("encoder.conv1.") for k in keys):
        return "conv"
    if "encoder.linear1.weight" in sd:
        return "mlp"
    return fallback


def load_held_out_batch(skip_rows: int, device: str):
    dataloader.T_RAW = T_RAW
    torch.manual_seed(0)
    loader = create_hf_dataloader(
        repo_id=HF_REPO, batch_size=BATCH_SIZE, C=1,
        path_in_repo=HF_PATH, skip_rows=skip_rows,
    )
    batch = next(iter(loader))
    if isinstance(batch, tuple):
        if len(batch) == 3:
            x, freq_ids, seas_ids = batch
        else:
            x = batch[0]
            freq_ids = None
            seas_ids = None
    else:
        x = batch
        freq_ids = None
        seas_ids = None
    assert x.shape == (BATCH_SIZE, T_RAW, 1), x.shape
    return x.to(device), \
        (freq_ids.to(device) if freq_ids is not None else None), \
        (seas_ids.to(device) if seas_ids is not None else None)


def _forward_in_chunks(bb, x_dev, freq_ids, seas_ids, chunk: int = 32):
    e_parts, f_parts = [], []
    for s in range(0, x_dev.shape[0], chunk):
        x_c = x_dev[s:s + chunk]
        f_c = freq_ids[s:s + chunk] if freq_ids is not None else None
        s_c = seas_ids[s:s + chunk] if seas_ids is not None else None
        e_part, _ = extract_encoder_latents(
            bb, x_c, freq_ids=f_c, seasonality_ids=s_c)
        f_part, _ = extract_forecaster_latents(
            bb, x_c, freq_ids=f_c, seasonality_ids=s_c)
        e_parts.append(e_part)
        f_parts.append(f_part)
    return torch.cat(e_parts, dim=0), torch.cat(f_parts, dim=0)


def build_backbone(name: str, declared_enc: str, sd: dict):
    """Construct ConfigurableModel matching the checkpoint's state-dict.

    Returns (bb, encoder_type) on success, (None, None) on shape mismatch.
    """
    cfg = dict(BACKBONE_BASE)
    detected_enc = detect_encoder_type(sd, fallback=declared_enc)
    if detected_enc != declared_enc:
        print(f"  [eval] WARN {name}: declared encoder_type={declared_enc!r} "
              f"differs from detected {detected_enc!r}; using detected.",
              file=sys.stderr)
    cfg["encoder_type"] = detected_enc
    fw = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = int(fw.shape[1]) if fw is not None else 0
    sw = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = int(sw.shape[1]) if sw is not None else 0
    cfg["learnable_tau"] = "log_inv_tau" in sd
    cfg["rev_norm_kind"] = "ewma"
    cfg["rev_norm_span"] = 128
    ref = sd.get("encoder.skip.weight")
    if ref is None:
        ref = sd.get("encoder.linear1.weight")
    if ref is None:
        cfg["patch_stats_kind"] = "none"
    else:
        extra = (ref.shape[1] - cfg["W"]
                 - cfg["freq_emb_dim"] - cfg["seasonality_emb_dim"])
        if extra == 0:
            cfg["patch_stats_kind"] = "none"
        elif extra == 2:
            cfg["patch_stats_kind"] = "diff"
        else:
            print(f"  [eval] skipped: {name}: unexpected encoder "
                  f"in_features={ref.shape[1]} (extra={extra})",
                  file=sys.stderr)
            return None, None
    try:
        bb = ConfigurableModel(**cfg)
        bb.load_state_dict(sd)
    except (RuntimeError, ValueError) as e:
        print(f"  [eval] skipped: {name}: load_state_dict failed ({e})",
              file=sys.stderr)
        return None, None
    return bb, cfg["encoder_type"]


def eval_one_sample(bb, H, x_dev, freq_ids, seas_ids) -> dict:
    e_bc, f_bc = _forward_in_chunks(bb, x_dev, freq_ids, seas_ids)
    B, C = BATCH_SIZE, 1
    h = e_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    f = f_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    assert h.shape == (B, T, C, H), h.shape

    torch.manual_seed(0)
    q_r = q_random(f[:, :T - 1], h[:, 1:T]).item()
    q_n = q_naive_latent(f[:, :T - 1], h[:, 1:T], h[:, :T - 1]).item()
    u_temp = dim_usage(h, axis=1).item()
    u_batch_v = dim_usage(h, axis=0).item()
    auc, top1 = retrieval_auc_top1(f[:, :T - 1], h[:, :T])
    return dict(
        r2_random=1.0 - q_r,
        r2_naive=1.0 - q_n,
        u_temporal=u_temp,
        u_batch=u_batch_v,
        auc=auc.item(),
        top1=top1.item(),
    )


def aggregate(samples: list[dict]) -> dict:
    """mean / stdev across the per-sample metric dicts."""
    out = {}
    n = len(samples)
    for k in METRIC_KEYS:
        vals = [s[k] for s in samples]
        m = statistics.fmean(vals)
        std = statistics.pstdev(vals) if n > 1 else 0.0
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = std
    out["n_samples"] = n
    return out


def main() -> None:
    device = pick_device()
    print(f"  [eval] device={device}")
    print(f"  [eval] N_SAMPLES={N_SAMPLES}; "
          f"skip_rows wraps to: "
          f"{[s % TOTAL_ROWS for s in SKIP_ROWS_LIST]}")

    # Pre-fetch all 10 batches once and reuse across every backbone.
    batches: list[tuple] = []
    for i, skip in enumerate(SKIP_ROWS_LIST):
        print(f"  [eval] loading batch {i+1}/{N_SAMPLES} "
              f"(skip_rows={skip}, wraps to row "
              f"{skip % TOTAL_ROWS})...", flush=True)
        batches.append(load_held_out_batch(skip, device))

    # Union of all backbones we need to eval, deduplicated by relative path
    # so we only load the τ=0.20 checkpoint once even though it appears in
    # both CSVs.
    all_specs: dict[str, dict] = {}
    for name, tau, enc, rel in TAU_BACKBONES:
        all_specs[rel] = {"name": name, "tau": tau, "enc": enc, "rel": rel}
    for name, ls, enc, rel in LOSS_BACKBONES:
        if rel not in all_specs:
            all_specs[rel] = {"name": name, "loss_shape": ls, "enc": enc, "rel": rel}

    # Per-rel aggregate cache: {rel: {metric_mean, metric_std, encoder_type, n_samples}}
    cache: dict[str, dict] = {}

    for rel, spec in all_specs.items():
        path = os.path.join(REPO_ROOT, rel)
        name = spec["name"]
        if not os.path.exists(path):
            print(f"  [eval] skipped: {name}: missing {path}", file=sys.stderr)
            continue
        sd = torch.load(path, map_location="cpu", weights_only=True)
        bb, enc_type = build_backbone(name, spec["enc"], sd)
        if bb is None:
            continue
        bb.eval().to(device)
        H = BACKBONE_BASE["H"]

        per_sample = []
        for i, (x_dev, freq_ids, seas_ids) in enumerate(batches):
            with torch.no_grad():
                m = eval_one_sample(bb, H, x_dev, freq_ids, seas_ids)
            per_sample.append(m)
            print(f"    [{name}] sample {i+1}/{N_SAMPLES}: "
                  f"r2_r={m['r2_random']:+.4f}  auc={m['auc']:.4f}  "
                  f"top1={m['top1']:.4f}", flush=True)

        agg = aggregate(per_sample)
        agg["encoder_type"] = enc_type
        cache[rel] = agg

        del bb
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"  [eval] {name:>30}  "
              f"r2_r={agg['r2_random_mean']:+.4f}±{agg['r2_random_std']:.4f}  "
              f"auc={agg['auc_mean']:.4f}±{agg['auc_std']:.4f}  "
              f"top1={agg['top1_mean']:.4f}±{agg['top1_std']:.4f}")

    # ---- Write τ-sweep CSV ----
    os.makedirs(os.path.dirname(TAU_OUT_CSV), exist_ok=True)
    with open(TAU_OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TAU_COLUMNS)
        w.writeheader()
        for name, tau, enc, rel in TAU_BACKBONES:
            if rel not in cache:
                continue
            agg = cache[rel]
            row = dict(name=name, tau=tau, encoder_type=agg["encoder_type"],
                       n_samples=agg["n_samples"])
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = agg[f"{k}_mean"]
                row[f"{k}_std"] = agg[f"{k}_std"]
            w.writerow(row)
    print(f"  [eval] wrote {TAU_OUT_CSV}")

    # ---- Write loss-extensions CSV ----
    os.makedirs(os.path.dirname(LOSS_OUT_CSV), exist_ok=True)
    with open(LOSS_OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOSS_COLUMNS)
        w.writeheader()
        for name, ls, enc, rel in LOSS_BACKBONES:
            if rel not in cache:
                continue
            agg = cache[rel]
            row = dict(name=name, loss_shape=ls, encoder_type=agg["encoder_type"],
                       n_samples=agg["n_samples"])
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = agg[f"{k}_mean"]
                row[f"{k}_std"] = agg[f"{k}_std"]
            w.writerow(row)
    print(f"  [eval] wrote {LOSS_OUT_CSV}")


if __name__ == "__main__":
    main()
