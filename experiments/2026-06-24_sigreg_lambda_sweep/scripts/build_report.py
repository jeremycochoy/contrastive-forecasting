#!/usr/bin/env python3
# #363 SIGReg λ-sweep report builder: plots + gm_table.csv from training CSV
# files and GIFT-Eval summaries. Loads four prior anchors (#344 enc3+CPC,
# #353 EMA-target, #355 SIGReg λ_e=λ_h=0.1, #359 SIGReg λ_e=1.0/λ_h=0.1) and
# whichever sweep arms have a `runs/bb_<…>_<suffix>_losses.csv` present:
#
#   emb100_enc01   λ_e=10.0, λ_h=0.1
#   emb100_enc10   λ_e=10.0, λ_h=1.0
#   emb100_enc100  λ_e=10.0, λ_h=10.0
#   emb10_enc10    λ_e=1.0,  λ_h=1.0   (optional 4th)
#
# Outputs (under <report_dir>):
#   results/gm_table.csv        — anchors + present sweep arms × 4 cells
#   plots/loss_curve.png        — training loss for present arms vs anchors
#   plots/sigreg_e_inspection.png — sigreg_e/h, u_batch_e/temporal_e
#   plots/dim_usage.png         — U (dimension usage) on h_t and e_t, cross-batch and cross-time
#   plots/gm_rel_mase.png       — per-cell GM-Rel MASE bars across arms
#   results/final_trajectories.txt — Tail-50 means per arm
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Prior anchor GM values (transcribed verbatim, all `n=97`). All four anchors
# already coexist in the #359 published table — see
# reports/2026-06-22_lejepa_sigreg_emb10/results/gm_table.csv (the cpc_enc3 /
# ema_enc3 / sigreg01_enc3 / sigreg10_enc3 rows). Re-derived from there to
# avoid drift across separate per-issue tables.
ANCHOR_GM = {
    # #344 enc3+CPC, B=1024
    ("cpc_enc3",      "2L", "best"): 1.1846,
    ("cpc_enc3",      "2L", "last"): 1.1531,
    ("cpc_enc3",      "6L", "best"): 1.1584,
    ("cpc_enc3",      "6L", "last"): 1.1436,
    # #353 EMA-target enc3+CPC, B=1024
    ("ema_enc3",      "2L", "best"): 1.1614,
    ("ema_enc3",      "2L", "last"): 1.1817,
    ("ema_enc3",      "6L", "best"): 1.1576,
    ("ema_enc3",      "6L", "last"): 1.1597,
    # #355 SIGReg λ_e=λ_h=0.1
    ("sigreg01_enc3", "2L", "best"): 1.1610,
    ("sigreg01_enc3", "2L", "last"): 1.1758,
    ("sigreg01_enc3", "6L", "best"): 1.1543,
    ("sigreg01_enc3", "6L", "last"): 1.1556,
    # #359 SIGReg λ_e=1.0, λ_h=0.1
    ("sigreg10_enc3", "2L", "best"): 1.1470,
    ("sigreg10_enc3", "2L", "last"): 1.1681,
    ("sigreg10_enc3", "6L", "best"): 1.1408,
    ("sigreg10_enc3", "6L", "last"): 1.1482,
}

ARM_LABEL = {
    "cpc_enc3":          "enc3+CPC, B=1024",
    "ema_enc3":          "EMA-target enc3+CPC, B=1024",
    "sigreg01_enc3":     "SIGReg + EMA-target, B=512 (λ_e=λ_h=0.1)",
    "sigreg10_enc3":     "SIGReg + EMA-target, B=512 (λ_e=1.0, λ_h=0.1)",
    "emb100_enc01":      "SIGReg + EMA-target, B=512 (λ_e=10.0, λ_h=0.1)",
    "emb100_enc10":      "SIGReg + EMA-target, B=512 (λ_e=10.0, λ_h=1.0)",
    "emb100_enc100":     "SIGReg + EMA-target, B=512 (λ_e=10.0, λ_h=10.0)",
    "emb10_enc10":       "SIGReg + EMA-target, B=512 (λ_e=1.0, λ_h=1.0)",
    "emb1000_enc01":     "SIGReg + EMA-target, B=512 (λ_e=100.0, λ_h=0.1)",
    "emb10000_enc10":    "SIGReg + EMA-target, B=512 (λ_e=1000.0, λ_h=1.0)",
}

ARM_COLOR = {
    "cpc_enc3":      "#888888",
    "ema_enc3":      "#1f77b4",
    "sigreg01_enc3": "#d62728",
    "sigreg10_enc3": "#2ca02c",
    "emb100_enc01":  "#9467bd",
    "emb100_enc10":  "#8c564b",
    "emb100_enc100": "#e377c2",
    "emb10_enc10":   "#17becf",
    "emb1000_enc01": "#bcbd22",
    "emb10000_enc10":"#7f7f7f",
}

SWEEP_ARMS = [
    # (suffix, λ_e, λ_h)
    ("emb100_enc01",  10.0, 0.1),
    ("emb100_enc10",  10.0, 1.0),
    ("emb100_enc100", 10.0, 10.0),
    ("emb10_enc10",    1.0, 1.0),
    ("emb1000_enc01", 100.0, 0.1),
    ("emb10000_enc10",1000.0, 1.0),
]

ANCHOR_ORDER = ["cpc_enc3", "ema_enc3", "sigreg01_enc3", "sigreg10_enc3"]


def parse_gm(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE.*?:\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def parse_n_configs(summary_path: Path, default: int = 97) -> int:
    if not summary_path.exists():
        return default
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE \((\d+) configs\)", txt)
    return int(m.group(1)) if m else default


def base_tag() -> str:
    return "allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc"


def sweep_run_csv(runs: Path, suffix: str) -> Path:
    return runs / f"bb_{base_tag()}_{suffix}_losses.csv"


def sweep_summary(results: Path, suffix: str, head: str, ckpt: str) -> Path:
    last = "_last" if ckpt == "last" else ""
    return results / f"gift_eval_full_{base_tag()}_{suffix}{last}_{head}" / "summary.txt"


def present_arms(runs: Path) -> list[str]:
    return [s for s, _, _ in SWEEP_ARMS if sweep_run_csv(runs, s).exists()]


def write_gm_table(runs: Path, results: Path, out_csv: Path) -> list[dict]:
    rows: list[dict] = []
    for (arm, head, ckpt), gm in ANCHOR_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97))
    for suffix in present_arms(runs):
        for head in ("2L", "6L"):
            for ckpt in ("best", "last"):
                sp = sweep_summary(results, suffix, head, ckpt)
                gm = parse_gm(sp)
                n = parse_n_configs(sp)
                if gm is not None:
                    rows.append(dict(arm=suffix, label=ARM_LABEL[suffix],
                                     head=head, ckpt=ckpt, gm=gm, n=n))
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "label", "head", "ckpt", "gm", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gm": f"{r['gm']:.4f}"})
    return rows


def plot_loss_curves(runs: Path, ema_csv: Path | None, cpc_csv: Path | None,
                     sig01_csv: Path | None, sig10_csv: Path | None, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for arm, p in (("cpc_enc3", cpc_csv), ("ema_enc3", ema_csv),
                   ("sigreg01_enc3", sig01_csv), ("sigreg10_enc3", sig10_csv)):
        if p and Path(p).exists():
            d = pd.read_csv(p)
            ax.plot(d["step"], d["loss"].rolling(50, min_periods=1).mean(),
                    label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.0)
    for suffix in present_arms(runs):
        d = pd.read_csv(sweep_run_csv(runs, suffix))
        ax.plot(d["step"], d["loss"].rolling(50, min_periods=1).mean(),
                label=ARM_LABEL[suffix], color=ARM_COLOR[suffix], lw=1.6)
    ax.set_xlabel("step"); ax.set_ylabel("loss (50-step rolling mean)")
    ax.set_title("Training loss")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_sigreg_inspection(runs: Path, sig01_csv: Path | None, sig10_csv: Path | None, out: Path):
    """L_SIGReg(e_t), L_SIGReg(h_t), u_batch_e, u_temporal_e across sweep arms + anchors."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    K = 384  # d_model (embedding dim), not batch — sets the 1/K dimension-usage floor
    panels = [
        ("sigreg_e",     "L_SIGReg(e_t)",                                          True),
        ("sigreg_h",     "L_SIGReg(h_t)",                                          True),
        ("u_batch_e",    "U_batch (e_t) — cross-batch dimension usage",            False),
        ("u_temporal_e", "U_temporal (e_t) — cross-time dimension usage",          False),
    ]
    sweep_csvs = [(s, sweep_run_csv(runs, s)) for s in present_arms(runs)]
    anchor_csvs = [
        ("sigreg01_enc3", sig01_csv),
        ("sigreg10_enc3", sig10_csv),
    ]
    for ax, (col, title, logy) in zip(axes.ravel(), panels):
        for arm, p in anchor_csvs:
            if p and Path(p).exists():
                d = pd.read_csv(p)
                if col in d.columns:
                    ax.plot(d["step"], d[col].rolling(50, min_periods=1).mean(),
                            label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.0)
        for suffix, p in sweep_csvs:
            d = pd.read_csv(p)
            if col in d.columns:
                ax.plot(d["step"], d[col].rolling(50, min_periods=1).mean(),
                        label=ARM_LABEL[suffix], color=ARM_COLOR[suffix], lw=1.6)
        if col in ("u_batch_e", "u_temporal_e"):
            ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5,
                       label=f"1/K = 1/{K} ≈ {1/K:.4f} (all K dims evenly used)")
            ax.set_ylim(0, 1)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_title(title)
        ax.legend(fontsize=6); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Embedding-side SIGReg trajectory across the λ-sweep arms (#363)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_dim_usage(runs: Path, sig01_csv: Path | None, sig10_csv: Path | None,
                   ema_csv: Path | None, cpc_csv: Path | None, out: Path):
    """Dimension usage U on h_t (solid) and e_t (dashed) for sweep arms + 2
    anchors. U measures how many of the K=384 latent dimensions are actively
    used; 1/K floor = all K dims evenly used, values near 1 = collapsed to a
    single direction."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    K = 384
    for ax, kind in zip(axes, ("batch", "temporal")):
        for arm, p in (("cpc_enc3", cpc_csv), ("ema_enc3", ema_csv),
                       ("sigreg01_enc3", sig01_csv), ("sigreg10_enc3", sig10_csv)):
            if p and Path(p).exists():
                d = pd.read_csv(p)
                if f"u_{kind}" in d.columns:
                    ax.plot(d["step"], d[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                            label=f"{ARM_LABEL[arm]} · h_t",
                            color=ARM_COLOR[arm], lw=0.9)
        for suffix in present_arms(runs):
            d = pd.read_csv(sweep_run_csv(runs, suffix))
            if f"u_{kind}" in d.columns:
                ax.plot(d["step"], d[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                        label=f"{ARM_LABEL[suffix]} · h_t",
                        color=ARM_COLOR[suffix], lw=1.5)
            if f"u_{kind}_e" in d.columns:
                ax.plot(d["step"], d[f"u_{kind}_e"].rolling(50, min_periods=1).mean(),
                        label=f"{ARM_LABEL[suffix]} · e_t",
                        color=ARM_COLOR[suffix], lw=1.5, ls="--")
        ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5,
                   label=f"1/K = 1/{K} ≈ {1/K:.4f} (all K dims evenly used)")
        ax.set_xlabel("step"); ax.set_ylabel("U (dimension usage)")
        ax.set_title(f"U_{kind} ({'cross-batch' if kind=='batch' else 'cross-time'})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle("Dimension usage U — cos²-based; clipped to [1/K, 1] (floor = all K dims evenly used)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_gm_bars(rows: list[dict], runs: Path, out: Path):
    df = pd.DataFrame(rows)
    arms_order = ANCHOR_ORDER + present_arms(runs)
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    x = np.arange(len(cells))
    w = 0.78 / max(1, len(arms_order))
    for i, arm in enumerate(arms_order):
        vals = []
        for head, ckpt in cells:
            r = df[(df["arm"] == arm) & (df["head"] == head) & (df["ckpt"] == ckpt)]
            vals.append(r["gm"].values[0] if len(r) else np.nan)
        offs = (i - (len(arms_order) - 1) / 2) * w
        ax.bar(x + offs, vals, w, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        for xi, vi in zip(x + offs, vals):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.003, f"{vi:.3f}",
                        ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}/{c}" for h, c in cells])
    ax.set_ylabel("GM-Rel MASE (lower = better)")
    ax.set_title("GIFT-Eval full-97 GM-Rel MASE — head-matched")
    ax.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.5)
    vals_all = [r["gm"] for r in rows]
    if vals_all:
        ax.set_ylim(min(vals_all) - 0.02, max(vals_all) + 0.04)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def final_trajectories(csv_path: Path, n_tail: int = 50) -> dict[str, float]:
    s = pd.read_csv(csv_path)
    tail = s.tail(n_tail)
    out: dict[str, float] = {"final_step": int(s["step"].iloc[-1])}
    for col in ("u_batch", "u_batch_e", "u_temporal", "u_temporal_e",
                "sigreg_e", "sigreg_h", "loss"):
        if col in s.columns:
            out[col] = float(tail[col].mean())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--sig01-csv", type=Path)
    p.add_argument("--sig10-csv", type=Path)
    p.add_argument("--ema-csv", type=Path)
    p.add_argument("--cpc-csv", type=Path)
    args = p.parse_args(argv)

    report = args.report_dir
    runs = report / "runs"
    results = report / "results"
    plots = report / "plots"
    runs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    present = present_arms(runs)
    if not present:
        print("WARN: no sweep arm CSVs found; emitting anchors-only table + bar plot.",
              file=sys.stderr)

    rows = write_gm_table(runs, results, results / "gm_table.csv")
    plot_loss_curves(runs, args.ema_csv, args.cpc_csv, args.sig01_csv, args.sig10_csv,
                     plots / "loss_curve.png")
    plot_sigreg_inspection(runs, args.sig01_csv, args.sig10_csv,
                           plots / "sigreg_e_inspection.png")
    plot_dim_usage(runs, args.sig01_csv, args.sig10_csv, args.ema_csv, args.cpc_csv,
                   plots / "dim_usage.png")
    plot_gm_bars(rows, runs, plots / "gm_rel_mase.png")

    traj_lines = []
    for suffix in present:
        t = final_trajectories(sweep_run_csv(runs, suffix))
        traj_lines.append((suffix, t))
    with (results / "final_trajectories.txt").open("w") as fh:
        for suffix, t in traj_lines:
            fh.write(f"== {suffix} ({ARM_LABEL[suffix]}) ==\n")
            for k, v in t.items():
                fh.write(f"{k}\t{v}\n")
            fh.write("\n")
    print(f"present sweep arms: {present}")
    print(f"rows in gm_table: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
