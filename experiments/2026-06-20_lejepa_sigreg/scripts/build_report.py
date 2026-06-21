#!/usr/bin/env python3
# #355 SIGReg report builder: plots + gm_table.csv from training CSV + GIFT-Eval summaries.
import argparse, csv, os, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REF_GM = {
    ("cpc_enc3", "2L", "best"): 1.1846,
    ("cpc_enc3", "2L", "last"): 1.1531,
    ("cpc_enc3", "6L", "best"): 1.1584,
    ("cpc_enc3", "6L", "last"): 1.1436,
}
EMA_GM = {
    ("ema_enc3", "2L", "best"): 1.1614,
    ("ema_enc3", "2L", "last"): 1.1817,
    ("ema_enc3", "6L", "best"): 1.1576,
    ("ema_enc3", "6L", "last"): 1.1597,
}

ARM_LABEL = {
    "cpc_enc3":    "enc3+CPC, B=1024",
    "ema_enc3":    "EMA-target enc3+CPC, B=1024",
    "sigreg_enc3": "SIGReg + EMA-target enc3+CPC, B=512",
}


def parse_gm(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE.*?:\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def parse_n_configs(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE \((\d+) configs\)", txt)
    return int(m.group(1)) if m else 0


def write_gm_table(results_dir: Path, out_csv: Path, tag: str):
    rows: list[dict] = []
    for (arm, head, ckpt), gm in REF_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97))
    for (arm, head, ckpt), gm in EMA_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97))

    sigreg_map = {
        ("2L", "best"): results_dir / f"gift_eval_full_{tag}_2L" / "summary.txt",
        ("2L", "last"): results_dir / f"gift_eval_full_{tag}_last_2L" / "summary.txt",
        ("6L", "best"): results_dir / f"gift_eval_full_{tag}_6L" / "summary.txt",
        ("6L", "last"): results_dir / f"gift_eval_full_{tag}_last_6L" / "summary.txt",
    }
    for (head, ckpt), sp in sigreg_map.items():
        gm = parse_gm(sp)
        n = parse_n_configs(sp)
        if gm is not None:
            rows.append(dict(arm="sigreg_enc3", label=ARM_LABEL["sigreg_enc3"],
                             head=head, ckpt=ckpt, gm=gm, n=n))

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "label", "head", "ckpt", "gm", "n"])
        w.writeheader()
        for r in rows:
            # Force 4 dp for gm — DictWriter would otherwise strip trailing zeros
            # (`1.1610` → `1.161`), breaking column precision consistency.
            w.writerow({**r, "gm": f"{r['gm']:.4f}"})
    return rows


def plot_loss_curves(sigreg_csv: Path, ema_csv: Path | None, cpc_csv: Path | None, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = pd.read_csv(sigreg_csv)
    ax.plot(s["step"], s["loss"].rolling(50, min_periods=1).mean(),
            label=ARM_LABEL["sigreg_enc3"], color="C0", lw=1.5)
    if ema_csv and ema_csv.exists():
        e = pd.read_csv(ema_csv)
        ax.plot(e["step"], e["loss"].rolling(50, min_periods=1).mean(),
                label=ARM_LABEL["ema_enc3"], color="C1", lw=1.5)
    if cpc_csv and cpc_csv.exists():
        c = pd.read_csv(cpc_csv)
        ax.plot(c["step"], c["loss"].rolling(50, min_periods=1).mean(),
                label=ARM_LABEL["cpc_enc3"], color="C2", lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("loss (50-step rolling mean)")
    ax.set_title("Training loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_uniformity(sigreg_csv: Path, ema_csv: Path | None, cpc_csv: Path | None, out: Path):
    s = pd.read_csv(sigreg_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    sig_lab = ARM_LABEL["sigreg_enc3"]
    ema_lab = ARM_LABEL["ema_enc3"]
    cpc_lab = ARM_LABEL["cpc_enc3"]
    for ax, kind in zip(axes, ("batch", "temporal")):
        ax.plot(s["step"], s[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind} (h_t) — {sig_lab}", color="C0", lw=1.5)
        ax.plot(s["step"], s[f"u_{kind}_e"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind}_e (e_t) — {sig_lab}", color="C0", lw=1.5, ls="--")
        if ema_csv and ema_csv.exists():
            e = pd.read_csv(ema_csv)
            if f"u_{kind}" in e.columns:
                ax.plot(e["step"], e[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                        label=f"u_{kind} (h_t) — {ema_lab}", color="C1", lw=1.0)
        if cpc_csv and cpc_csv.exists():
            c = pd.read_csv(cpc_csv)
            if f"u_{kind}" in c.columns:
                ax.plot(c["step"], c[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                        label=f"u_{kind} (h_t) — {cpc_lab}", color="C2", lw=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("effective dimensionality")
        ax.set_title(f"u_{kind} ({'cross-batch' if kind=='batch' else 'cross-time'})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Uniformity (cos²-based dim_usage; clipped to [1/K, 1])")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_gm_bars(rows: list[dict], out: Path):
    df = pd.DataFrame(rows)
    df = df.sort_values(["head", "ckpt", "arm"]).reset_index(drop=True)
    arms_order = ["cpc_enc3", "ema_enc3", "sigreg_enc3"]
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(cells))
    w = 0.27
    colors = {"cpc_enc3": "#888888", "ema_enc3": "#1f77b4", "sigreg_enc3": "#d62728"}
    for i, arm in enumerate(arms_order):
        vals = []
        for head, ckpt in cells:
            r = df[(df["arm"] == arm) & (df["head"] == head) & (df["ckpt"] == ckpt)]
            vals.append(r["gm"].values[0] if len(r) else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=ARM_LABEL[arm], color=colors[arm])
        for xi, vi in zip(x + (i - 1) * w, vals):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.003, f"{vi:.4f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}/{c}" for h, c in cells])
    ax.set_ylabel("GM-Rel MASE (lower = better)")
    ax.set_title("GIFT-Eval full-97 GM-Rel MASE — head-matched")
    ax.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.5)
    ymin = max(0.95, min((r["gm"] for r in rows)) - 0.02)
    ymax = max(r["gm"] for r in rows) + 0.04
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def final_trajectories(sigreg_csv: Path, n_tail: int = 50) -> dict[str, float]:
    s = pd.read_csv(sigreg_csv)
    tail = s.tail(n_tail)
    return {
        "u_batch":      float(tail["u_batch"].mean()),
        "u_batch_e":    float(tail["u_batch_e"].mean()),
        "u_temporal":   float(tail["u_temporal"].mean()),
        "u_temporal_e": float(tail["u_temporal_e"].mean()),
        "sigreg_e":     float(tail["sigreg_e"].mean()),
        "sigreg_h":     float(tail["sigreg_h"].mean()),
        "loss":         float(tail["loss"].mean()),
        "final_step":   int(s["step"].iloc[-1]),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--sigreg-tag", default="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc")
    p.add_argument("--ema-csv", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-19_ema_target_encoder/runs/bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv"))
    p.add_argument("--cpc-csv", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv"))
    args = p.parse_args()

    report = args.report_dir
    runs = report / "runs"
    results = report / "results"
    plots = report / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    sigreg_csv = runs / f"bb_{args.sigreg_tag}_losses.csv"
    if not sigreg_csv.exists():
        sys.exit(f"missing: {sigreg_csv}")

    rows = write_gm_table(results, results / "gm_table.csv", args.sigreg_tag)
    plot_loss_curves(sigreg_csv, args.ema_csv, args.cpc_csv, plots / "loss_curve.png")
    plot_uniformity(sigreg_csv, args.ema_csv, args.cpc_csv, plots / "uniformity.png")
    plot_gm_bars(rows, plots / "gm_rel_mase.png")

    traj = final_trajectories(sigreg_csv)
    with (results / "final_trajectories.txt").open("w") as fh:
        for k, v in traj.items():
            fh.write(f"{k}\t{v}\n")
    print("FINAL trajectories (last 50 rows):")
    for k, v in traj.items():
        print(f"  {k}: {v}")
    print(f"\nrows in gm_table: {len(rows)}")
    for r in rows:
        if r["arm"] == "sigreg_enc3":
            print(f"  {r['head']}/{r['ckpt']}  GM={r['gm']:.4f}  (n={r['n']})")
