#!/usr/bin/env python3
"""Cross-backbone diagnostic eval at the canonical best-checkpoint of each run.

Distinct from ``eval_backbone_metrics.py`` (one trajectory). Here we load the
"best_loss" (or highest periodic) checkpoint of each completed backbone
training run available locally and evaluate the same fixed held-out HF batch
through each.

Reports R² = 1 − Q where Q is the error ratio
``mean_b e(forecast, target) / mean_b e(reference)``. Higher R² is better;
R² = 0 means the forecast is no better than the reference, R² = 1 means a
perfect forecast. We keep the Q metric functions in ``src/metrics.py``
unchanged and convert here.

Run:
    PYTHONPATH=. python experiments/2026-05-05_exp_qhead_improvements/\
scripts/eval_backbones_cross.py
"""

from __future__ import annotations

import csv
import os
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
    retrieval_auc_top1_legacy as retrieval_auc_top1,
)
from src.models import ConfigurableModel


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "experiments", "2026-05-05_exp_qhead_improvements")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "backbone_metrics_cross.csv")

BATCH_SIZE = 256
T_RAW = 4096
W = 16
T = T_RAW // W   # 256
HF_REPO = "jeremycochoy/gift-pretrain-full-4096"
HF_PATH = "small_v1"
SKIP_ROWS = 50_000_000

BACKBONE = dict(
    C=1, H=384, W=W, encoder_type="gru", num_layers=6,
    nhead=6, ffn_mult=4.0, activation="gelu",
    depthwise_conv=3, dropout=0.1,
)

# Hardcoded list of (name, relative-path-from-repo-root). Verified by
# `find sync_realonly_*/ -name "*best_loss.pth" -size +30M` plus the
# highest periodic save for runs without a best_loss.
BACKBONES: list[tuple[str, str]] = [
    ("moirai_hp_early",
     "sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/"
     "tiny_realonly_full4096_moirai_hp_best_loss.pth"),
    ("learnable_tau",
     "sync_realonly_full4096_learnable_tau/learnable/checkpoints/"
     "tiny_realonly_full4096_learnable_tau_best_loss.pth"),
    ("FRESH_50k",
     "sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/"
     "tiny_full4096_moirai_hp_FRESH_best_loss.pth"),
    ("backbone_beta_167k",
     "sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/"
     "moirai_hp_FRESH_RESUME50k/checkpoints/"
     "tiny_full4096_moirai_hp_FRESH_RESUME50k_best_loss.pth"),
    ("moirai_hp_FINAL_run1",
     "sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/"
     "checkpoints/tiny_full4096_moirai_hp_FINAL_190k.pth"),
]

COLUMNS = ["name", "r2_random", "r2_naive",
           "u_temporal", "u_batch", "auc", "top1"]


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


def load_held_out_batch(device: str):
    dataloader.T_RAW = T_RAW
    torch.manual_seed(0)
    loader = create_hf_dataloader(
        repo_id=HF_REPO, batch_size=BATCH_SIZE, C=1,
        path_in_repo=HF_PATH, skip_rows=SKIP_ROWS,
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


def eval_backbone(name: str, path: str, x_dev, freq_ids, seas_ids,
                  device: str) -> dict | None:
    if not os.path.exists(path):
        print(f"  [eval] skipped: {name}: missing {path}", file=sys.stderr)
        return None
    sd = torch.load(path, map_location="cpu", weights_only=True)

    # Auto-detect arch from state-dict (same logic as the trajectory script).
    cfg = dict(BACKBONE)
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
                  f"in_features={ref.shape[1]} (extra={extra})", file=sys.stderr)
            return None

    try:
        bb = ConfigurableModel(**cfg)
        bb.load_state_dict(sd)
    except (RuntimeError, ValueError) as e:
        print(f"  [eval] skipped: {name}: load_state_dict failed ({e})",
              file=sys.stderr)
        return None
    bb.eval().to(device)

    e_bc, f_bc = _forward_in_chunks(bb, x_dev, freq_ids, seas_ids)

    B, C = BATCH_SIZE, 1
    H = cfg["H"]
    h = e_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    f = f_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    assert h.shape == (B, T, C, H), h.shape

    torch.manual_seed(0)
    q_r = q_random(f[:, :T - 1], h[:, 1:T]).item()
    q_n = q_naive_latent(f[:, :T - 1], h[:, 1:T], h[:, :T - 1]).item()
    u_temp = dim_usage(h, axis=1).item()
    u_batch_v = dim_usage(h, axis=0).item()
    auc, top1 = retrieval_auc_top1(f[:, :T - 1], h[:, :T])

    del bb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # R² = 1 − Q is a presentation transform; Q stays Q in src/metrics.py.
    return dict(
        name=name,
        r2_random=1.0 - q_r,
        r2_naive=1.0 - q_n,
        u_temporal=u_temp,
        u_batch=u_batch_v,
        auc=auc.item(),
        top1=top1.item(),
    )


def main() -> None:
    device = pick_device()
    print(f"  [eval] device={device}")
    print(f"  [eval] {len(BACKBONES)} backbones to evaluate")

    x_dev, freq_ids, seas_ids = load_held_out_batch(device)
    print(f"  [eval] held-out batch shape={tuple(x_dev.shape)}")

    rows: list[dict] = []
    for name, rel in BACKBONES:
        path = os.path.join(REPO_ROOT, rel)
        row = eval_backbone(name, path, x_dev, freq_ids, seas_ids, device)
        if row is None:
            continue
        rows.append(row)
        print(f"  [eval] {row['name']:>22}  "
              f"r2_r={row['r2_random']:+.4f}  r2_n={row['r2_naive']:+.4f}  "
              f"ut={row['u_temporal']:.4f}  ub={row['u_batch']:.4f}  "
              f"auc={row['auc']:.4f}  top1={row['top1']:.4f}")

    rows.sort(key=lambda r: r["r2_random"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  [eval] wrote {CSV_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
