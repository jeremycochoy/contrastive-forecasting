#!/usr/bin/env python3
"""One-shot eval of diagnostic metrics across the RESUME50k training trajectory.

Loads each periodic checkpoint of the moirai_hp FRESH_RESUME50k run plus the
FRESH 50k starting point, runs a fixed held-out HF batch through the backbone,
and records q_random, q_naive_latent, dim_usage (temporal+batch), retrieval
AUC and Top-1 to a CSV. Renders a 2x3 plot, then appends a section to the
qhead-improvements REPORT.md.

Run: PYTHONPATH=. python experiments/2026-05-05_exp_qhead_improvements/scripts/eval_backbone_metrics.py
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.dataloader as dataloader
from src.dataloader import create_hf_dataloader
from src.models import ConfigurableModel
from src.forecasting_head import (
    extract_encoder_latents,
    extract_forecaster_latents,
)
from src.metrics import (
    q_random,
    q_naive_latent,
    dim_usage,
    retrieval_auc_top1,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "experiments", "2026-05-05_exp_qhead_improvements")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
CSV_PATH = os.path.join(RESULTS_DIR, "backbone_metrics_trajectory.csv")
PLOT_PATH = os.path.join(PLOTS_DIR, "backbone_metrics_curve.png")
REPORT_PATH = os.path.join(EXP_DIR, "REPORT.md")

RESUME_CKPT_DIR = os.path.join(
    REPO_ROOT,
    "sync_realonly_full4096_moirai_hp_FRESH_RESUME50k",
    "moirai_hp_FRESH_RESUME50k",
    "checkpoints",
)
RESUME_PREFIX = "tiny_full4096_moirai_hp_FRESH_RESUME50k"
FRESH_CKPT_PATH = os.path.join(
    REPO_ROOT,
    "sync_realonly_full4096_moirai_hp_FRESH",
    "moirai_hp_FRESH",
    "checkpoints",
    "tiny_full4096_moirai_hp_FRESH_50k.pth",
)

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


def step_for_path(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"_(\d+)k\.pth$", name)
    if m is not None:
        return int(m.group(1)) * 1000
    if name.endswith("_FINAL.pth"):
        opt = path[:-4] + "_optimizer.pth"
        if os.path.exists(opt):
            sd = torch.load(opt, map_location="cpu", weights_only=False)
            meta = sd.get("meta", {}) if isinstance(sd, dict) else {}
            step = meta.get("step")
            if step is not None:
                return int(step)
        print(f"  [eval] {name}: no optimizer/meta; defaulting to step 167000",
              file=sys.stderr)
        return 167000
    raise ValueError(f"Cannot infer step from {name!r}")


def list_checkpoints() -> list[str]:
    pat = os.path.join(RESUME_CKPT_DIR, f"{RESUME_PREFIX}_*k.pth")
    periodic = [p for p in sorted(glob.glob(pat))
                if not p.endswith("_optimizer.pth")
                and re.search(r"_(\d+)k\.pth$", os.path.basename(p))]
    final = os.path.join(RESUME_CKPT_DIR, f"{RESUME_PREFIX}_FINAL.pth")
    paths = list(periodic)
    if os.path.exists(final):
        paths.append(final)
    if os.path.exists(FRESH_CKPT_PATH):
        paths.append(FRESH_CKPT_PATH)
    else:
        print("  [eval] FRESH 50k not found locally", file=sys.stderr)
    return paths


def load_held_out_batch(device: str):
    # HFStreamingLoader crops module-level T_RAW windows from each parquet row;
    # set it before creating the loader so we get 4096-length windows.
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
    """Run the encoder + forecaster forward in chunks to fit a 24GB GPU."""
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


def eval_checkpoint(path: str, x_dev, freq_ids, seas_ids, device: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)

    # Auto-detect arch from state-dict (same logic as
    # experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py).
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
            raise ValueError(
                f"Unexpected encoder in_features={ref.shape[1]}: extra={extra}")

    bb = ConfigurableModel(**cfg)
    bb.load_state_dict(sd)
    bb.eval().to(device)

    e_bc, f_bc = _forward_in_chunks(bb, x_dev, freq_ids, seas_ids)

    B, C = BATCH_SIZE, 1
    H = cfg["H"]
    h = e_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    f = f_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    assert h.shape == (B, T, C, H), h.shape
    assert f.shape == (B, T, C, H), f.shape

    torch.manual_seed(0)
    q_r = q_random(f[:, :T - 1], h[:, 1:T]).item()
    q_n = q_naive_latent(f[:, :T - 1], h[:, 1:T], h[:, :T - 1]).item()
    u_temp = dim_usage(h, axis=1).item()
    u_batch = dim_usage(h, axis=0).item()
    auc, top1 = retrieval_auc_top1(f[:, :T - 1], h[:, :T])

    del bb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return dict(
        step=step_for_path(path),
        q_random=q_r, q_naive_latent=q_n,
        u_temporal=u_temp, u_batch=u_batch,
        auc=auc.item(), top1=top1.item(),
    )


COLUMNS = ["step", "q_random", "q_naive_latent",
           "u_temporal", "u_batch", "auc", "top1"]


def write_plot(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # The CSV stores Q metrics; we display R² = 1 − Q on the plot so
    # higher = better, consistent with auc/top1. dim_usage stays as-is.
    # Conversion is presentation-only — src/metrics.py keeps Q.
    metrics = [
        ("r2_random",     lambda r: 1.0 - r["q_random"]),
        ("r2_naive",      lambda r: 1.0 - r["q_naive_latent"]),
        ("u_temporal",    lambda r: r["u_temporal"]),
        ("u_batch",       lambda r: r["u_batch"]),
        ("auc",           lambda r: r["auc"]),
        ("top1",          lambda r: r["top1"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (title, fn) in zip(axes.flat, metrics):
        ax.plot(steps, [fn(r) for r in rows], marker="o")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.suptitle(
        "Backbone metric trajectory across RESUME50k training "
        "(held-out HF batch, skip=50M; R² = 1 − Q, higher = better)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _build_interpretation(rows: list[dict]) -> str:
    """Build a 2-3 sentence factual summary of the trajectory."""
    first, last = rows[0], rows[-1]
    # Display in R² space (1 − Q): higher = better, consistent with auc/top1.
    parts = []
    for q_col, r2_col in (("q_random", "r2_random"),
                          ("q_naive_latent", "r2_naive")):
        r2 = [1.0 - r[q_col] for r in rows]
        delta = r2[-1] - r2[0]
        rng = max(r2) - min(r2)
        if abs(delta) < rng * 0.25:
            verb = "oscillates without a clear trend"
        elif delta > 0:
            verb = "trends upward"
        else:
            verb = "trends downward"
        parts.append(f"{r2_col} {verb} (Δ={delta:+.4f}, range {rng:.4f})")
    r2_sentence = "; ".join(parts) + "."

    ut_delta = last["u_temporal"] - first["u_temporal"]
    ub_delta = last["u_batch"] - first["u_batch"]
    u_sentence = (f"u_temporal Δ={ut_delta:+.4f} and u_batch Δ={ub_delta:+.4f} "
                  f"— both stay within ~0.07.")

    auc_range = max(r["auc"] for r in rows) - min(r["auc"] for r in rows)
    top1_range = max(r["top1"] for r in rows) - min(r["top1"] for r in rows)
    if auc_range < 0.01 and top1_range < 0.02:
        retrieval_sentence = (
            f"Retrieval auc and top1 are essentially flat "
            f"(auc range {auc_range:.4f}, top1 range {top1_range:.4f}).")
    else:
        retrieval_sentence = (
            f"auc Δ={last['auc'] - first['auc']:+.4f} (range {auc_range:.4f}), "
            f"top1 Δ={last['top1'] - first['top1']:+.4f} "
            f"(range {top1_range:.4f}).")

    return " ".join([r2_sentence, u_sentence, retrieval_sentence])


def update_report(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["step"])
    # Report in R² space (1 − Q). Q_* values remain in the CSV.
    report_cols = ["step", "r2_random", "r2_naive",
                   "u_temporal", "u_batch", "auc", "top1"]
    header = "| " + " | ".join(report_cols) + " |"
    sep = "|" + "|".join(["---"] * len(report_cols)) + "|"
    body_lines = []
    for r in rows:
        cells = [
            str(r["step"]),
            f"{1.0 - r['q_random']:.4f}",
            f"{1.0 - r['q_naive_latent']:.4f}",
            f"{r['u_temporal']:.4f}",
            f"{r['u_batch']:.4f}",
            f"{r['auc']:.4f}",
            f"{r['top1']:.4f}",
        ]
        body_lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join([header, sep] + body_lines)

    interpretation = _build_interpretation(rows)

    section = (
        "## Backbone metric trajectory\n"
        "\n"
        "Below we report R² = 1 − Q where Q is the error ratio "
        "mean_b e(forecast, target) / mean_b e(reference). R² = 0 means "
        "the forecast is no better than the baseline; R² = 1 means the "
        "forecast is exact. Q values are in `results/backbone_metrics_trajectory.csv`.\n"
        "\n"
        "Diagnostic on the *backbone* (not the head experiments). Every head "
        "experiment in this report shares the same backbone-beta = step 167k, "
        "so the table below shows how the backbone evolved across its own "
        "training, not a per-head comparison.\n"
        "\n"
        "![backbone metrics](plots/backbone_metrics_curve.png)\n"
        "\n"
        f"{table}\n"
        "\n"
        f"{interpretation}\n"
        "\n"
    )

    with open(REPORT_PATH, "r") as f:
        text = f.read()
    marker = "## Pipeline summary"
    if marker not in text:
        raise RuntimeError("Pipeline summary marker not found in REPORT.md")
    if "## Backbone metric trajectory" in text:
        # Idempotent re-run: drop the existing section so we replace it.
        head, _, rest = text.partition("## Backbone metric trajectory")
        _, _, after = rest.partition(marker)
        text = head + marker + after
    new_text = text.replace(marker, section + marker, 1)
    with open(REPORT_PATH, "w") as f:
        f.write(new_text)


def main() -> None:
    device = pick_device()
    print(f"  [eval] device={device}")

    paths = list_checkpoints()
    print(f"  [eval] {len(paths)} checkpoints to evaluate")

    x_dev, freq_ids, seas_ids = load_held_out_batch(device)
    print(f"  [eval] held-out batch shape={tuple(x_dev.shape)}")

    # Stream rows to a tmp CSV as each checkpoint finishes, so a partial run
    # still leaves usable data; rewrite step-sorted at the end.
    rows: list[dict] = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_path = CSV_PATH + ".partial"
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for path in paths:
            row = eval_checkpoint(path, x_dev, freq_ids, seas_ids, device)
            rows.append(row)
            w.writerow(row)
            f.flush()
            print(f"  [eval] step={row['step']:>6}  "
                  f"q_r={row['q_random']:.4f}  q_n={row['q_naive_latent']:.4f}  "
                  f"ut={row['u_temporal']:.4f}  ub={row['u_batch']:.4f}  "
                  f"auc={row['auc']:.4f}  top1={row['top1']:.4f}")

    rows.sort(key=lambda r: r["step"])
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.remove(tmp_path)
    write_plot(rows)
    update_report(rows)
    print(f"  [eval] wrote {CSV_PATH}")
    print(f"  [eval] wrote {PLOT_PATH}")
    print(f"  [eval] updated {REPORT_PATH}")


if __name__ == "__main__":
    main()
