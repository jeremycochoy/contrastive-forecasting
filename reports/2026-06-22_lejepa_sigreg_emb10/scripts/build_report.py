#!/usr/bin/env python3
# #359 SIGReg-emb10 report builder: plots + gm_table.csv from training CSV + GIFT-Eval summaries.
# Compares against four references: enc3+CPC (#344), EMA-target (#353), SIGReg λ_e=λ_h=0.1 (#355),
# and the new SIGReg λ_embedding=1.0 / λ_encoding=0.1 arm (#359).
import argparse, csv, math, os, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference GM values, transcribed from each arm's published gm_table.csv at its own code rev.
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
SIGREG01_GM = {  # the #355 SIGReg arm with shared λ=0.1 (transcribed from #355 gm_table.csv)
    ("sigreg01_enc3", "2L", "best"): 1.1610,
    ("sigreg01_enc3", "2L", "last"): 1.1758,
    ("sigreg01_enc3", "6L", "best"): 1.1543,
    ("sigreg01_enc3", "6L", "last"): 1.1556,
}

ARM_LABEL = {
    "cpc_enc3":      "enc3+CPC, B=1024",
    "ema_enc3":      "EMA-target enc3+CPC, B=1024",
    "sigreg01_enc3": "SIGReg + EMA-target, B=512 (λ_e=λ_h=0.1)",
    "sigreg10_enc3": "SIGReg + EMA-target, B=512 (λ_e=1.0, λ_h=0.1)",
}

ARM_COLOR = {
    "cpc_enc3":      "#888888",
    "ema_enc3":      "#1f77b4",
    "sigreg01_enc3": "#d62728",
    "sigreg10_enc3": "#2ca02c",
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
    for d in (REF_GM, EMA_GM, SIGREG01_GM):
        for (arm, head, ckpt), gm in d.items():
            rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97))

    sigreg10_map = {
        ("2L", "best"): results_dir / f"gift_eval_full_{tag}_2L" / "summary.txt",
        ("2L", "last"): results_dir / f"gift_eval_full_{tag}_last_2L" / "summary.txt",
        ("6L", "best"): results_dir / f"gift_eval_full_{tag}_6L" / "summary.txt",
        ("6L", "last"): results_dir / f"gift_eval_full_{tag}_last_6L" / "summary.txt",
    }
    for (head, ckpt), sp in sigreg10_map.items():
        gm = parse_gm(sp)
        n = parse_n_configs(sp)
        if gm is not None:
            rows.append(dict(arm="sigreg10_enc3", label=ARM_LABEL["sigreg10_enc3"],
                             head=head, ckpt=ckpt, gm=gm, n=n))

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "label", "head", "ckpt", "gm", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gm": f"{r['gm']:.4f}"})
    return rows


def plot_loss_curves(
    sig10_csv: Path, sig01_csv: Path | None, ema_csv: Path | None, cpc_csv: Path | None, out: Path,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = pd.read_csv(sig10_csv)
    ax.plot(s["step"], s["loss"].rolling(50, min_periods=1).mean(),
            label=ARM_LABEL["sigreg10_enc3"], color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
    for arm, csv_path in (("sigreg01_enc3", sig01_csv), ("ema_enc3", ema_csv), ("cpc_enc3", cpc_csv)):
        if csv_path and csv_path.exists():
            d = pd.read_csv(csv_path)
            ax.plot(d["step"], d["loss"].rolling(50, min_periods=1).mean(),
                    label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("loss (50-step rolling mean)")
    ax.set_title("Training loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_sigreg_inspection(sig10_csv: Path, sig01_csv: Path | None, out: Path):
    """Compare sigreg_e / sigreg_h / u_batch_e / u_temporal_e between the two λ_e weights."""
    s10 = pd.read_csv(sig10_csv)
    s01 = pd.read_csv(sig01_csv) if (sig01_csv and sig01_csv.exists()) else None
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("sigreg_e",     "L_SIGReg(e_t)",     True),
        ("sigreg_h",     "L_SIGReg(h_t)",     True),
        ("u_batch_e",    "u_batch (e_t)",     False),
        ("u_temporal_e", "u_temporal (e_t)",  False),
    ]
    K = 384
    for ax, (col, title, logy) in zip(axes.ravel(), panels):
        ax.plot(s10["step"], s10[col].rolling(50, min_periods=1).mean(),
                label="λ_e=1.0 (this arm, #359)", color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
        if s01 is not None and col in s01.columns:
            ax.plot(s01["step"], s01[col].rolling(50, min_periods=1).mean(),
                    label="λ_e=0.1 (#355)", color=ARM_COLOR["sigreg01_enc3"], lw=1.2)
        if col in ("u_batch_e", "u_temporal_e"):
            ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5, label=f"1/K = 1/{K} ≈ {1/K:.4f}")
            ax.set_ylim(0, 1)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Embedding-side SIGReg trajectory: λ_e=1.0 (#359) vs λ_e=0.1 (#355)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_uniformity(
    sig10_csv: Path, sig01_csv: Path | None, ema_csv: Path | None, cpc_csv: Path | None, out: Path,
):
    s10 = pd.read_csv(sig10_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, kind in zip(axes, ("batch", "temporal")):
        ax.plot(s10["step"], s10[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind} (h_t) — {ARM_LABEL['sigreg10_enc3']}",
                color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
        ax.plot(s10["step"], s10[f"u_{kind}_e"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind}_e (e_t) — {ARM_LABEL['sigreg10_enc3']}",
                color=ARM_COLOR["sigreg10_enc3"], lw=1.6, ls="--")
        if sig01_csv and sig01_csv.exists():
            s01 = pd.read_csv(sig01_csv)
            ax.plot(s01["step"], s01[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                    label=f"u_{kind} (h_t) — {ARM_LABEL['sigreg01_enc3']}",
                    color=ARM_COLOR["sigreg01_enc3"], lw=1.0)
            ax.plot(s01["step"], s01[f"u_{kind}_e"].rolling(50, min_periods=1).mean(),
                    label=f"u_{kind}_e (e_t) — {ARM_LABEL['sigreg01_enc3']}",
                    color=ARM_COLOR["sigreg01_enc3"], lw=1.0, ls="--")
        for arm, csv_path in (("ema_enc3", ema_csv), ("cpc_enc3", cpc_csv)):
            if csv_path and csv_path.exists():
                d = pd.read_csv(csv_path)
                if f"u_{kind}" in d.columns:
                    ax.plot(d["step"], d[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                            label=f"u_{kind} (h_t) — {ARM_LABEL[arm]}",
                            color=ARM_COLOR[arm], lw=0.9)
        ax.set_xlabel("step")
        ax.set_ylabel("effective dimensionality")
        ax.set_title(f"u_{kind} ({'cross-batch' if kind=='batch' else 'cross-time'})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle("Uniformity (cos²-based dim_usage; clipped to [1/K, 1])")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_gm_bars(rows: list[dict], out: Path):
    df = pd.DataFrame(rows)
    arms_order = ["cpc_enc3", "ema_enc3", "sigreg01_enc3", "sigreg10_enc3"]
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(cells))
    w = 0.2
    for i, arm in enumerate(arms_order):
        vals = []
        for head, ckpt in cells:
            r = df[(df["arm"] == arm) & (df["head"] == head) & (df["ckpt"] == ckpt)]
            vals.append(r["gm"].values[0] if len(r) else np.nan)
        offs = (i - (len(arms_order) - 1) / 2) * w
        ax.bar(x + offs, vals, w, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        for xi, vi in zip(x + offs, vals):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.003, f"{vi:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}/{c}" for h, c in cells])
    ax.set_ylabel("GM-Rel MASE (lower = better)")
    ax.set_title("GIFT-Eval full-97 GM-Rel MASE — head-matched")
    ax.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.5)
    vals_all = [r["gm"] for r in rows]
    ax.set_ylim(min(vals_all) - 0.02, max(vals_all) + 0.04)
    ax.legend(loc="upper left", fontsize=7)
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
        acc.setdefault(dom, []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def plot_perdomain_radar(
    sig10_results: Path, sig01_results: Path, ema_results: Path, cpc_results: Path,
    sig10_tag: str, sig01_tag: str, ema_tag: str, cpc_tag: str, out: Path,
):
    """2 panels (2L | 6L), 8 curves each (4 arms × {best, last})."""
    HEADS = ["2L", "6L"]
    CURVES = [
        ("cpc_enc3",      cpc_results,   cpc_tag,   "",      ARM_COLOR["cpc_enc3"],      "-",  f"{ARM_LABEL['cpc_enc3']} · best"),
        ("cpc_enc3",      cpc_results,   cpc_tag,   "_last", ARM_COLOR["cpc_enc3"],      "--", f"{ARM_LABEL['cpc_enc3']} · last"),
        ("ema_enc3",      ema_results,   ema_tag,   "",      ARM_COLOR["ema_enc3"],      "-",  f"{ARM_LABEL['ema_enc3']} · best"),
        ("ema_enc3",      ema_results,   ema_tag,   "_last", ARM_COLOR["ema_enc3"],      "--", f"{ARM_LABEL['ema_enc3']} · last"),
        ("sigreg01_enc3", sig01_results, sig01_tag, "",      ARM_COLOR["sigreg01_enc3"], "-",  f"{ARM_LABEL['sigreg01_enc3']} · best"),
        ("sigreg01_enc3", sig01_results, sig01_tag, "_last", ARM_COLOR["sigreg01_enc3"], "--", f"{ARM_LABEL['sigreg01_enc3']} · last"),
        ("sigreg10_enc3", sig10_results, sig10_tag, "",      ARM_COLOR["sigreg10_enc3"], "-",  f"{ARM_LABEL['sigreg10_enc3']} · best"),
        ("sigreg10_enc3", sig10_results, sig10_tag, "_last", ARM_COLOR["sigreg10_enc3"], "--", f"{ARM_LABEL['sigreg10_enc3']} · last"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), subplot_kw=dict(polar=True))
    for ax, head in zip(axes, HEADS):
        cells = []
        for arm, root, tag, suf, col, ls, lab in CURVES:
            sub = root / f"gift_eval_full_{tag}{suf}_{head}"
            rel = _config_relatives(sub / "summary.txt")
            dmap = _dataset_to_domain(sub / "all_results.csv")
            gm = _gm_by_domain(rel, dmap)
            if gm:
                cells.append((lab, gm, col, ls))
        if not cells:
            ax.text(0.5, 0.5, "no eval", transform=ax.transAxes); continue
        domains = sorted(set().union(*(g for _, g, _, _ in cells)))
        theta = np.linspace(0, 2 * np.pi, len(domains), endpoint=False)
        theta_closed = np.concatenate([theta, theta[:1]])
        vals = [v for _, g, _, _ in cells for v in g.values()]
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
        for lab, g, col, ls in cells:
            v = np.array([g.get(d, np.nan) for d in domains]
                         + [g.get(domains[0], np.nan)])
            ax.plot(theta_closed, v, color=col, ls=ls, lw=1.4, zorder=3,
                    marker="o", markersize=3, label=lab)
        ax.set_title(f"{head} q-head", fontsize=11, pad=14)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.06),
                  fontsize=7, frameon=False, ncol=1)
    fig.suptitle(
        "Per-domain GM relative MASE on GIFT-Eval full-97 "
        "(radial log scale; ring at 1.0 = seasonal-naive; lower = better; "
        "solid = best-loss, dashed = last)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def final_trajectories(sigreg_csv: Path, n_tail: int = 50) -> dict[str, float]:
    s = pd.read_csv(sigreg_csv)
    tail = s.tail(n_tail)
    out = {
        "u_batch":      float(tail["u_batch"].mean()),
        "u_batch_e":    float(tail["u_batch_e"].mean()),
        "u_temporal":   float(tail["u_temporal"].mean()),
        "u_temporal_e": float(tail["u_temporal_e"].mean()),
        "sigreg_e":     float(tail["sigreg_e"].mean()),
        "sigreg_h":     float(tail["sigreg_h"].mean()),
        "loss":         float(tail["loss"].mean()),
        "final_step":   int(s["step"].iloc[-1]),
    }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--sig10-tag", default="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_emb10")
    p.add_argument("--sig01-csv", type=Path,
        default=Path("/tmp/contrastive-forecasting-359/reports/2026-06-20_lejepa_sigreg/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv"))
    p.add_argument("--sig01-results", type=Path,
        default=Path("/tmp/contrastive-forecasting-359/reports/2026-06-20_lejepa_sigreg/results"))
    p.add_argument("--sig01-tag", default="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc")
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

    sig10_csv = runs / f"bb_{args.sig10_tag}_losses.csv"
    if not sig10_csv.exists():
        sys.exit(f"missing: {sig10_csv}")

    rows = write_gm_table(results, results / "gm_table.csv", args.sig10_tag)
    plot_loss_curves(sig10_csv, args.sig01_csv, args.ema_csv, args.cpc_csv, plots / "loss_curve.png")
    plot_sigreg_inspection(sig10_csv, args.sig01_csv, plots / "sigreg_e_inspection.png")
    plot_uniformity(sig10_csv, args.sig01_csv, args.ema_csv, args.cpc_csv, plots / "uniformity.png")
    plot_gm_bars(rows, plots / "gm_rel_mase.png")
    plot_perdomain_radar(
        sig10_results=results, sig01_results=args.sig01_results,
        ema_results=args.ema_results, cpc_results=args.cpc_results,
        sig10_tag=args.sig10_tag, sig01_tag=args.sig01_tag,
        ema_tag=args.ema_tag, cpc_tag=args.cpc_tag,
        out=plots / "perdomain_radar.png",
    )

    traj_new = final_trajectories(sig10_csv)
    with (results / "final_trajectories.txt").open("w") as fh:
        for k, v in traj_new.items():
            fh.write(f"{k}\t{v}\n")
    print("FINAL trajectories (last 50 rows) — λ_e=1.0:")
    for k, v in traj_new.items():
        print(f"  {k}: {v}")
    if args.sig01_csv.exists():
        traj_ref = final_trajectories(args.sig01_csv)
        print("\nReference trajectories (last 50 rows) — λ_e=0.1 (#355):")
        for k, v in traj_ref.items():
            print(f"  {k}: {v}")
    print(f"\nrows in gm_table: {len(rows)}")
    for r in rows:
        if r["arm"] == "sigreg10_enc3":
            print(f"  {r['head']}/{r['ckpt']}  GM={r['gm']:.4f}  (n={r['n']})")
