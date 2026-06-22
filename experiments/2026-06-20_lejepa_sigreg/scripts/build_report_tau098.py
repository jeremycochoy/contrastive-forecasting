#!/usr/bin/env python3
# #357 SIGReg-tau098 report builder: plots + gm_table.csv from training CSV + GIFT-Eval summaries.
# Mirrors #355's build_report.py with one extra reference column (#355 τ=0.99).
import argparse, csv, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# #344 enc3+CPC baseline (head-matched, B=1024)
REF_GM = {
    ("cpc_enc3", "2L", "best"): 1.1846,
    ("cpc_enc3", "2L", "last"): 1.1531,
    ("cpc_enc3", "6L", "best"): 1.1584,
    ("cpc_enc3", "6L", "last"): 1.1436,
}
# #353 EMA-target enc3+CPC, B=1024, τ=0.99
EMA_GM = {
    ("ema_enc3", "2L", "best"): 1.1614,
    ("ema_enc3", "2L", "last"): 1.1817,
    ("ema_enc3", "6L", "best"): 1.1576,
    ("ema_enc3", "6L", "last"): 1.1597,
}
# #355 SIGReg + EMA-target enc3+CPC, B=512, τ=0.99 (the direct single-axis ref for #357)
SIGREG_TAU099_GM = {
    ("sigreg_enc3_tau099", "2L", "best"): 1.1610,
    ("sigreg_enc3_tau099", "2L", "last"): 1.1758,
    ("sigreg_enc3_tau099", "6L", "best"): 1.1543,
    ("sigreg_enc3_tau099", "6L", "last"): 1.1556,
}

ARM_LABEL = {
    "cpc_enc3":            "enc3+CPC, B=1024",
    "ema_enc3":            "EMA-target enc3+CPC, B=1024, τ=0.99",
    "sigreg_enc3_tau099":  "SIGReg + EMA-target, B=512, τ=0.99",
    "sigreg_enc3_tau098":  "SIGReg + EMA-target, B=512, τ=0.98",
}

ARM_COLOR = {
    "cpc_enc3":           "#888888",
    "ema_enc3":           "#1f77b4",
    "sigreg_enc3_tau099": "#d62728",
    "sigreg_enc3_tau098": "#2ca02c",
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
    for (arm, head, ckpt), gm in SIGREG_TAU099_GM.items():
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
            rows.append(dict(arm="sigreg_enc3_tau098", label=ARM_LABEL["sigreg_enc3_tau098"],
                             head=head, ckpt=ckpt, gm=gm, n=n))

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "label", "head", "ckpt", "gm", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gm": f"{r['gm']:.4f}"})
    return rows


def plot_loss_curves(this_csv: Path, sigreg099_csv: Path | None,
                     ema_csv: Path | None, cpc_csv: Path | None, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = pd.read_csv(this_csv)
    ax.plot(s["step"], s["loss"].rolling(50, min_periods=1).mean(),
            label=ARM_LABEL["sigreg_enc3_tau098"], color=ARM_COLOR["sigreg_enc3_tau098"], lw=1.5)
    if sigreg099_csv and sigreg099_csv.exists():
        r = pd.read_csv(sigreg099_csv)
        ax.plot(r["step"], r["loss"].rolling(50, min_periods=1).mean(),
                label=ARM_LABEL["sigreg_enc3_tau099"], color=ARM_COLOR["sigreg_enc3_tau099"], lw=1.5)
    if ema_csv and ema_csv.exists():
        e = pd.read_csv(ema_csv)
        ax.plot(e["step"], e["loss"].rolling(50, min_periods=1).mean(),
                label=ARM_LABEL["ema_enc3"], color=ARM_COLOR["ema_enc3"], lw=1.5)
    if cpc_csv and cpc_csv.exists():
        c = pd.read_csv(cpc_csv)
        ax.plot(c["step"], c["loss"].rolling(50, min_periods=1).mean(),
                label=ARM_LABEL["cpc_enc3"], color=ARM_COLOR["cpc_enc3"], lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("loss (50-step rolling mean)")
    ax.set_title("Training loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_uniformity(this_csv: Path, sigreg099_csv: Path | None,
                    ema_csv: Path | None, cpc_csv: Path | None, out: Path):
    s = pd.read_csv(this_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    new_lab = ARM_LABEL["sigreg_enc3_tau098"]
    sig_lab = ARM_LABEL["sigreg_enc3_tau099"]
    ema_lab = ARM_LABEL["ema_enc3"]
    cpc_lab = ARM_LABEL["cpc_enc3"]
    for ax, kind in zip(axes, ("batch", "temporal")):
        ax.plot(s["step"], s[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind} (h_t) — {new_lab}", color=ARM_COLOR["sigreg_enc3_tau098"], lw=1.5)
        ax.plot(s["step"], s[f"u_{kind}_e"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind}_e (e_t) — {new_lab}", color=ARM_COLOR["sigreg_enc3_tau098"], lw=1.5, ls="--")
        if sigreg099_csv and sigreg099_csv.exists():
            r = pd.read_csv(sigreg099_csv)
            if f"u_{kind}" in r.columns:
                ax.plot(r["step"], r[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                        label=f"u_{kind} (h_t) — {sig_lab}", color=ARM_COLOR["sigreg_enc3_tau099"], lw=1.0)
            if f"u_{kind}_e" in r.columns:
                ax.plot(r["step"], r[f"u_{kind}_e"].rolling(50, min_periods=1).mean(),
                        label=f"u_{kind}_e (e_t) — {sig_lab}", color=ARM_COLOR["sigreg_enc3_tau099"], lw=1.0, ls="--")
        if ema_csv and ema_csv.exists():
            e = pd.read_csv(ema_csv)
            if f"u_{kind}" in e.columns:
                ax.plot(e["step"], e[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                        label=f"u_{kind} (h_t) — {ema_lab}", color=ARM_COLOR["ema_enc3"], lw=1.0)
        if cpc_csv and cpc_csv.exists():
            c = pd.read_csv(cpc_csv)
            if f"u_{kind}" in c.columns:
                ax.plot(c["step"], c[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                        label=f"u_{kind} (h_t) — {cpc_lab}", color=ARM_COLOR["cpc_enc3"], lw=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("effective dimensionality")
        ax.set_title(f"u_{kind} ({'cross-batch' if kind=='batch' else 'cross-time'})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    fig.suptitle("Uniformity (cos²-based dim_usage; clipped to [1/K, 1])")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_gm_bars(rows: list[dict], out: Path):
    df = pd.DataFrame(rows)
    df = df.sort_values(["head", "ckpt", "arm"]).reset_index(drop=True)
    arms_order = ["cpc_enc3", "ema_enc3", "sigreg_enc3_tau099", "sigreg_enc3_tau098"]
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cells))
    w = 0.2
    for i, arm in enumerate(arms_order):
        vals = []
        for head, ckpt in cells:
            r = df[(df["arm"] == arm) & (df["head"] == head) & (df["ckpt"] == ckpt)]
            vals.append(r["gm"].values[0] if len(r) else np.nan)
        ax.bar(x + (i - 1.5) * w, vals, w, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        for xi, vi in zip(x + (i - 1.5) * w, vals):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.003, f"{vi:.4f}",
                        ha="center", va="bottom", fontsize=6)
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


def plot_sigreg_e_inspection(this_csv: Path, out: Path, lam: float = 0.1):
    s = pd.read_csv(this_csv)
    if "sigreg_e" not in s.columns or "sigreg_h" not in s.columns:
        return None
    rwin = 50
    se = s["sigreg_e"].rolling(rwin, min_periods=1).mean()
    sh = s["sigreg_h"].rolling(rwin, min_periods=1).mean()
    loss = s["loss"].rolling(rwin, min_periods=1).mean()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(s["step"], se, label="L_SIGReg(e_t)", color="C3", lw=1.5)
    ax.plot(s["step"], sh, label="L_SIGReg(h_t)", color="C0", lw=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("SIGReg term (log)")
    ax.set_title("SIGReg term trajectories (50-step rolling mean)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(s["step"], lam * se / loss.clip(lower=1e-6), label=f"λ·L_SIGReg(e_t) / loss (λ={lam})", color="C3", lw=1.5)
    ax.plot(s["step"], lam * sh / loss.clip(lower=1e-6), label=f"λ·L_SIGReg(h_t) / loss", color="C0", lw=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("share of total loss (log)")
    ax.set_xlabel("step")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return float(se.tail(50).mean()), float(sh.tail(50).mean()), float(loss.tail(50).mean())


def final_trajectories(this_csv: Path, n_tail: int = 50) -> dict[str, float]:
    s = pd.read_csv(this_csv)
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


def trajectory_table(this_csv: Path, steps=(250, 500, 1000, 2000, 5000, 7500, 10000, 12500)):
    s = pd.read_csv(this_csv).set_index("step")
    cols = ["sigreg_e", "sigreg_h", "u_batch_e", "u_batch", "u_temporal_e", "u_temporal", "loss"]
    rwin = 50
    out = []
    for st in steps:
        if st not in s.index:
            continue
        i = s.index.get_loc(st)
        lo = max(0, i - rwin + 1)
        win = s.iloc[lo:i+1]
        row = {"step": st}
        for c in cols:
            if c in win.columns:
                row[c] = float(win[c].mean())
        out.append(row)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--sigreg-tag", default="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau098")
    p.add_argument("--sigreg099-csv", type=Path,
        default=Path("/tmp/contrastive-forecasting-357/reports/2026-06-20_lejepa_sigreg/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv"))
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

    this_csv = runs / f"bb_{args.sigreg_tag}_losses.csv"
    if not this_csv.exists():
        sys.exit(f"missing: {this_csv}")

    rows = write_gm_table(results, results / "gm_table.csv", args.sigreg_tag)
    plot_loss_curves(this_csv, args.sigreg099_csv, args.ema_csv, args.cpc_csv, plots / "loss_curve.png")
    plot_uniformity(this_csv, args.sigreg099_csv, args.ema_csv, args.cpc_csv, plots / "uniformity.png")
    plot_gm_bars(rows, plots / "gm_rel_mase.png")
    plot_sigreg_e_inspection(this_csv, plots / "sigreg_e_inspection.png")

    traj = final_trajectories(this_csv)
    with (results / "final_trajectories.txt").open("w") as fh:
        for k, v in traj.items():
            fh.write(f"{k}\t{v}\n")

    table = trajectory_table(this_csv)
    with (results / "trajectory_table.csv").open("w", newline="") as fh:
        cols = ["step", "sigreg_e", "sigreg_h", "u_batch_e", "u_batch", "u_temporal_e", "u_temporal", "loss"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: r.get(c, "") for c in cols})

    print("FINAL trajectories (last 50 rows):")
    for k, v in traj.items():
        print(f"  {k}: {v}")
    print(f"\nrows in gm_table: {len(rows)}")
    for r in rows:
        if r["arm"] == "sigreg_enc3_tau098":
            print(f"  {r['head']}/{r['ckpt']}  GM={r['gm']:.4f}  (n={r['n']})")
