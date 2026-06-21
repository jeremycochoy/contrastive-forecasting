#!/usr/bin/env python3
# #355 SIGReg report builder: plots + gm_table.csv from training CSV + GIFT-Eval summaries.
import argparse, csv, math, os, re, sys
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
# Per-arm GM-MASE = geomean of `eval_metrics/MASE[0.5]` over the 97 configs in
# `all_results.csv`. Independent of seasonal-naive denominators.
REF_GM_MASE = {
    ("cpc_enc3", "2L", "best"): 1.6559,
    ("cpc_enc3", "2L", "last"): 1.6119,
    ("cpc_enc3", "6L", "best"): 1.6193,
    ("cpc_enc3", "6L", "last"): 1.5986,
}
EMA_GM_MASE = {
    ("ema_enc3", "2L", "best"): 1.6235,
    ("ema_enc3", "2L", "last"): 1.6519,
    ("ema_enc3", "6L", "best"): 1.6182,
    ("ema_enc3", "6L", "last"): 1.6211,
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


def gm_mase_from_all_results(all_csv: Path) -> float | None:
    if not all_csv.exists():
        return None
    mase = pd.read_csv(all_csv)["eval_metrics/MASE[0.5]"].astype(float)
    mase = mase[mase > 0]
    if mase.empty:
        return None
    return float(math.exp(mase.apply(math.log).mean()))


def write_gm_table(results_dir: Path, out_csv: Path, tag: str):
    rows: list[dict] = []
    for (arm, head, ckpt), gm in REF_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt,
                         gm=gm, gm_mase=REF_GM_MASE[(arm, head, ckpt)], n=97))
    for (arm, head, ckpt), gm in EMA_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt,
                         gm=gm, gm_mase=EMA_GM_MASE[(arm, head, ckpt)], n=97))

    sigreg_map = {
        ("2L", "best"): results_dir / f"gift_eval_full_{tag}_2L",
        ("2L", "last"): results_dir / f"gift_eval_full_{tag}_last_2L",
        ("6L", "best"): results_dir / f"gift_eval_full_{tag}_6L",
        ("6L", "last"): results_dir / f"gift_eval_full_{tag}_last_6L",
    }
    for (head, ckpt), sub in sigreg_map.items():
        gm = parse_gm(sub / "summary.txt")
        n = parse_n_configs(sub / "summary.txt")
        gm_mase = gm_mase_from_all_results(sub / "all_results.csv")
        if gm is not None:
            rows.append(dict(arm="sigreg_enc3", label=ARM_LABEL["sigreg_enc3"],
                             head=head, ckpt=ckpt, gm=gm, gm_mase=gm_mase, n=n))

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["arm", "label", "head", "ckpt", "gm", "gm_mase", "n"])
        w.writeheader()
        for r in rows:
            # Force 4 dp for both gm columns — DictWriter would otherwise strip
            # trailing zeros (`1.1610` → `1.161`), breaking column precision.
            w.writerow({**r,
                        "gm": f"{r['gm']:.4f}",
                        "gm_mase": f"{r['gm_mase']:.4f}" if r['gm_mase'] is not None else ""})
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


def _dataset_to_domain(all_csv: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    if not all_csv.exists():
        return m
    for r in csv.DictReader(open(all_csv)):
        m[r["dataset"]] = r.get("domain", "Other")
    return m


def _config_relatives(summary_txt: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not summary_txt.exists():
        return out
    for line in open(summary_txt):
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try:
                out[p[0]] = float(p[3])
            except ValueError:
                pass
    return out


def _gm_by_domain(rels: dict[str, float], dmap: dict[str, str]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for cfg, rel in rels.items():
        dom = dmap.get(cfg)
        if rel <= 0 or dom is None:
            continue
        # Strip the freq/horizon suffix: per-dataset domain key is the prefix
        # before the first '/'.
        acc.setdefault(dom, []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def plot_perdomain_radar(
    sig_results: Path, ema_results: Path, cpc_results: Path,
    sig_tag: str, ema_tag: str, cpc_tag: str, out: Path,
):
    """2 panels (2L | 6L), 6 curves each (3 arms × {best, last})."""
    from matplotlib.lines import Line2D
    HEADS = ["2L", "6L"]
    GREY, BLUE, RED = "#888888", "#1f77b4", "#d62728"
    ARMS = [  # (arm-key, root, tag, colour)
        ("cpc_enc3",    cpc_results, cpc_tag, GREY),
        ("ema_enc3",    ema_results, ema_tag, BLUE),
        ("sigreg_enc3", sig_results, sig_tag, RED),
    ]
    CKPTS = [("", "-"), ("_last", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), subplot_kw=dict(polar=True))
    for ax, head in zip(axes, HEADS):
        cells = []
        for arm, root, tag, col in ARMS:
            for suf, ls in CKPTS:
                sub = root / f"gift_eval_full_{tag}{suf}_{head}"
                rel = _config_relatives(sub / "summary.txt")
                dmap = _dataset_to_domain(sub / "all_results.csv")
                gm = _gm_by_domain(rel, dmap)
                if gm:
                    cells.append((gm, col, ls))
        if not cells:
            ax.text(0.5, 0.5, "no eval", transform=ax.transAxes); continue
        domains = sorted(set().union(*(g for g, _, _ in cells)))
        theta = np.linspace(0, 2 * np.pi, len(domains), endpoint=False)
        theta_closed = np.concatenate([theta, theta[:1]])
        vals = [v for g, _, _ in cells for v in g.values()]
        lo, hi = max(0.5, min(vals) * 0.92), max(vals) * 1.06
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(theta); ax.set_xticklabels(domains, fontsize=8)
        ax.set_rscale("log"); ax.set_ylim(lo, hi)
        rticks = [t for t in (0.8, 1.0, 1.2, 1.5, 2.0) if lo < t < hi]
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{t:g}" for t in rticks], fontsize=7, color="0.4")
        ax.set_rlabel_position(90)
        ax.plot(theta_closed, [1.0] * len(theta_closed),
                color="k", ls=(0, (2, 2)), lw=0.8, alpha=0.6, zorder=1)
        for g, col, ls in cells:
            v = np.array([g.get(d, np.nan) for d in domains]
                         + [g.get(domains[0], np.nan)])
            ax.plot(theta_closed, v, color=col, ls=ls, lw=1.6, zorder=3,
                    marker="o", markersize=3)
        ax.set_title(f"{head} q-head", fontsize=11, pad=14)
    arm_handles = [Line2D([0], [0], color=col, lw=1.6, marker="o", markersize=4,
                          label=ARM_LABEL[arm]) for arm, _, _, col in ARMS]
    ckpt_handles = [
        Line2D([0], [0], color="0.2", lw=1.6, ls="-",  label="solid = best-loss"),
        Line2D([0], [0], color="0.2", lw=1.6, ls="--", label="dashed = last"),
    ]
    fig.legend(handles=arm_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.06), ncol=3, fontsize=9, frameon=False)
    fig.legend(handles=ckpt_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=2, fontsize=9, frameon=False)
    fig.suptitle(
        "Per-domain GM relative MASE on GIFT-Eval full-97 "
        "(radial log scale; ring at 1.0 = seasonal-naive; lower = better)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


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
    p.add_argument("--ema-results", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-19_ema_target_encoder/results"))
    p.add_argument("--cpc-results", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/results"))
    p.add_argument("--ema-tag", default="allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc")
    p.add_argument("--cpc-tag", default="allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc")
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
    plot_perdomain_radar(
        sig_results=results, ema_results=args.ema_results, cpc_results=args.cpc_results,
        sig_tag=args.sigreg_tag, ema_tag=args.ema_tag, cpc_tag=args.cpc_tag,
        out=plots / "perdomain_radar.png",
    )

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
