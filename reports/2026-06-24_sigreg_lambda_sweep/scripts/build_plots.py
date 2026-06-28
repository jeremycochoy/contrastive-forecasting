#!/usr/bin/env python3
# #363 SIGReg λ-sweep — plot builder.
#
# Plots produced under reports/2026-06-24_sigreg_lambda_sweep/plots/:
#   gm_rel_mase.png         — head-matched bar chart, CI whiskers (vs #359),
#                             dashed baseline lines for the 4 anchors.
#   lambda_e_ladder.png     — λ_e ladder (0.1 → 100.0) at fixed λ_h=0.1,
#                             one curve per (head, ckpt) cell.
#   best_vs_last.png        — per-arm best-minus-last gap across the 4 cells.
#   loss_curve.png          — total loss across arms + anchors (50-step rolling mean).
#   sigreg_e_inspection.png — L_SIGReg(e_t), L_SIGReg(h_t), u_batch_e, u_temporal_e.
#   dim_usage.png           — U (dimension usage) on h_t/e_t, cross-batch and cross-time.
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_heatmap import heatmap as plot_heatmap_grid

BASE_TAG = "allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc"

ANCHOR_GM = {
    ("cpc_enc3",      "2L", "best"): 1.1846,
    ("cpc_enc3",      "2L", "last"): 1.1531,
    ("cpc_enc3",      "6L", "best"): 1.1584,
    ("cpc_enc3",      "6L", "last"): 1.1436,
    ("ema_enc3",      "2L", "best"): 1.1614,
    ("ema_enc3",      "2L", "last"): 1.1817,
    ("ema_enc3",      "6L", "best"): 1.1576,
    ("ema_enc3",      "6L", "last"): 1.1597,
    ("sigreg01_enc3", "2L", "best"): 1.1610,
    ("sigreg01_enc3", "2L", "last"): 1.1758,
    ("sigreg01_enc3", "6L", "best"): 1.1543,
    ("sigreg01_enc3", "6L", "last"): 1.1556,
    ("sigreg10_enc3", "2L", "best"): 1.1470,
    ("sigreg10_enc3", "2L", "last"): 1.1681,
    ("sigreg10_enc3", "6L", "best"): 1.1408,
    ("sigreg10_enc3", "6L", "last"): 1.1482,
}

ARM_LABEL = {
    "cpc_enc3":          "enc3+CPC, B=1024 (#344)",
    "ema_enc3":          "EMA enc3+CPC, B=1024 (#353)",
    "sigreg01_enc3":     "SIGReg λ_e=0.1, λ_h=0.1 (#355)",
    "sigreg10_enc3":     "SIGReg λ_e=1.0, λ_h=0.1 (#359)",
    "emb100_enc01":      "SIGReg λ_e=10.0, λ_h=0.1 (arm 1)",
    "emb100_enc10":      "SIGReg λ_e=10.0, λ_h=1.0 (arm 2)",
    "emb100_enc100":     "SIGReg λ_e=10.0, λ_h=10.0 (arm 3)",
    "emb10_enc10":       "SIGReg λ_e=1.0, λ_h=1.0 (arm 4)",
    "emb1000_enc01":     "SIGReg λ_e=100.0, λ_h=0.1 (arm 5)",
    "emb10000_enc10":    "SIGReg λ_e=1000.0, λ_h=1.0 (arm 6)",
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
    "emb10000_enc10":"#ff7f0e",
}

SWEEP_ARMS = [
    "emb100_enc01", "emb100_enc10", "emb100_enc100",
    "emb10_enc10", "emb1000_enc01", "emb10000_enc10",
]
ANCHOR_ORDER = ["cpc_enc3", "ema_enc3", "sigreg01_enc3", "sigreg10_enc3"]
CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
# Plot start step — cuts the warm-up regime so its early-step divergence
# does not dominate the y-range. Matches the parent #359 report's log-y
# convention; here the loss curve is log-log so the start step is required
# (log x at step ≤ 0 is undefined).
PLOT_START_STEP = 100


def load_gm_table(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def load_ci(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return df


def plot_gm_bars(gm: pd.DataFrame, ci_vs_359: pd.DataFrame, out: Path):
    """Bars: anchors + 4 sweep arms × 4 cells; whiskers = paired-bootstrap 95% CI vs #359."""
    arms = ANCHOR_ORDER + SWEEP_ARMS
    fig, ax = plt.subplots(figsize=(14, 6.0))
    x = np.arange(len(CELLS))
    w = 0.78 / max(1, len(arms))
    vals_all = gm["gm"].values
    baseline = float(min(vals_all)) - 0.015
    for i, arm in enumerate(arms):
        vals, err_lo, err_hi = [], [], []
        for head, ckpt in CELLS:
            r = gm[(gm["arm"] == arm) & (gm["head"] == head) & (gm["ckpt"] == ckpt)]
            v = float(r.gm.values[0]) if len(r) else np.nan
            vals.append(v)
            if arm in SWEEP_ARMS:
                cr = ci_vs_359[(ci_vs_359["arm"] == arm) & (ci_vs_359["head"] == head) & (ci_vs_359["ckpt"] == ckpt)]
                if len(cr) and not np.isnan(cr.gm_delta_lo.values[0]):
                    lo = float(cr.gm_delta_lo.values[0])
                    hi = float(cr.gm_delta_hi.values[0])
                    point = float(cr.gm_delta_abs.values[0])
                    err_lo.append(point - lo)
                    err_hi.append(hi - point)
                else:
                    err_lo.append(0); err_hi.append(0)
            else:
                err_lo.append(0); err_hi.append(0)
        offs = (i - (len(arms) - 1) / 2) * w
        heights = [v - baseline if not np.isnan(v) else np.nan for v in vals]
        ax.bar(x + offs, heights, w, bottom=baseline,
               label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        if arm in SWEEP_ARMS:
            ax.errorbar(x + offs, vals, yerr=[err_lo, err_hi],
                        fmt="none", ecolor="black", capsize=2.5, lw=0.9)
        for xi, vi in zip(x + offs, vals):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.004, f"{vi:.3f}",
                        ha="center", va="bottom", fontsize=5.6, rotation=90)
    # per-cell baseline lines at each of 4 anchors
    anchor_styles = [("cpc_enc3", ":"), ("ema_enc3", ":"),
                     ("sigreg01_enc3", "--"), ("sigreg10_enc3", "-")]
    for arm, ls in anchor_styles:
        for ci_idx, (head, ckpt) in enumerate(CELLS):
            v = ANCHOR_GM[(arm, head, ckpt)]
            ax.hlines(v, x[ci_idx] - 0.5, x[ci_idx] + 0.5,
                      colors=ARM_COLOR[arm], linestyles=ls, lw=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}/{c}" for h, c in CELLS])
    ax.set_ylabel("GM-Rel MASE (lower = better)")
    ax.set_title("GIFT-Eval full-97 GM-Rel MASE — whiskers = paired-bootstrap 95% CI vs #359 (sweep arms only)")
    ax.set_ylim(baseline, max(vals_all) + 0.020)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_lambda_e_ladder(gm: pd.DataFrame, ci_vs_359: pd.DataFrame, out: Path):
    """Two λ_e ladders, one per λ_h row:
       λ_h=0.1:  0.1 (#355) → 1.0 (#359) → 10.0 (arm 1) → 100.0 (arm 5).
       λ_h=1.0:  1.0 (arm 4) → 10.0 (arm 2) → 1000.0 (arm 6).
    Whiskers: paired-bootstrap 95% CI on the absolute GM scale vs #359."""
    ladders = {
        0.1: [("sigreg01_enc3", 0.1), ("sigreg10_enc3", 1.0),
              ("emb100_enc01",  10.0), ("emb1000_enc01", 100.0)],
        1.0: [("emb10_enc10",    1.0), ("emb100_enc10",  10.0),
              ("emb10000_enc10", 1000.0)],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (lam_h, ladder) in zip(axes, ladders.items()):
        for head, ckpt in CELLS:
            xs, ys, lo, hi = [], [], [], []
            for arm, lam_e in ladder:
                r = gm[(gm["arm"] == arm) & (gm["head"] == head) & (gm["ckpt"] == ckpt)]
                v = float(r.gm.values[0]) if len(r) else np.nan
                xs.append(lam_e); ys.append(v)
                if arm in SWEEP_ARMS:
                    cr = ci_vs_359[(ci_vs_359["arm"] == arm) & (ci_vs_359["head"] == head) & (ci_vs_359["ckpt"] == ckpt)]
                    if len(cr) and not np.isnan(cr.gm_delta_lo.values[0]):
                        point = float(cr.gm_delta_abs.values[0])
                        lo.append(v - (point - float(cr.gm_delta_lo.values[0])))
                        hi.append(v + (float(cr.gm_delta_hi.values[0]) - point))
                    else:
                        lo.append(v); hi.append(v)
                else:
                    lo.append(v); hi.append(v)
            label = f"{head} / {ckpt}-ckpt"
            ls = "-" if ckpt == "best" else "--"
            col = "#1f77b4" if head == "2L" else "#d62728"
            ax.plot(xs, ys, marker="o", lw=1.4, ls=ls, color=col, label=label)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.10)
        ax.set_xscale("log")
        if lam_h == 0.1:
            ax.set_xticks([0.1, 1.0, 10.0, 100.0])
            ax.set_xticklabels(["0.1\n(#355)", "1.0\n(#359)", "10.0\n(arm 1)", "100.0\n(arm 5)"])
        else:
            ax.set_xticks([1.0, 10.0, 1000.0])
            ax.set_xticklabels(["1.0\n(arm 4)", "10.0\n(arm 2)", "1000.0\n(arm 6)"])
        ax.set_xlabel(f"λ_e  (at fixed λ_h={lam_h:g})")
        ax.set_title(f"λ_h = {lam_h:g}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("GM-Rel MASE (lower = better)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("λ_e ladders at λ_h=0.1 (left) and λ_h=1.0 (right) — bands = paired-bootstrap 95% CI vs #359")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_best_vs_last_drift(gm: pd.DataFrame, out: Path):
    """For each arm, plot (last − best) drift on 2L and 6L. Positive drift means
    `last` is worse than `best`, i.e. the model stopped improving by step 12 500
    (the desirable direction is negative drift: last ≤ best, model still
    learning at the end)."""
    arms = ANCHOR_ORDER + SWEEP_ARMS
    rows = []
    for arm in arms:
        for head in ("2L", "6L"):
            b = gm[(gm["arm"] == arm) & (gm["head"] == head) & (gm["ckpt"] == "best")]
            l = gm[(gm["arm"] == arm) & (gm["head"] == head) & (gm["ckpt"] == "last")]
            if len(b) and len(l):
                rows.append((arm, head, float(l.gm.values[0]) - float(b.gm.values[0])))
    df = pd.DataFrame(rows, columns=["arm", "head", "drift"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(arms))
    w = 0.4
    for i, head in enumerate(("2L", "6L")):
        vals = [df[(df["arm"] == a) & (df["head"] == head)].drift.values[0]
                if len(df[(df["arm"] == a) & (df["head"] == head)]) else np.nan for a in arms]
        ax.bar(x + (i - 0.5) * w, vals, w,
               label=f"{head} q-head",
               color="#1f77b4" if head == "2L" else "#d62728")
        for xi, vi in zip(x + (i - 0.5) * w, vals):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.001 if vi >= 0 else vi - 0.003, f"{vi:+.3f}",
                        ha="center", va="bottom" if vi >= 0 else "top", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms], rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("last − best GM-Rel MASE  (positive = drift)")
    ax.set_title(
        "Drift = last − best GM-Rel MASE per arm\n"
        "positive = last worse than best (stopped improving); "
        "negative = last better than best (still improving)"
    )
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _smoothed(d: pd.DataFrame, col: str, start_step: int) -> pd.DataFrame:
    """Apply 50-step rolling mean on the full curve, then trim to `step ≥ start_step`.
    Smoothing on the full curve keeps the rolling window populated; trimming after
    cuts the warm-up regime out of the plot range without re-introducing
    edge-effect bias at `start_step`."""
    out = d.copy()
    out[f"{col}_sm"] = out[col].rolling(50, min_periods=1).mean()
    return out[out["step"] >= start_step]


def plot_loss_curves(arm_runs: Path, sigreg_runs: Path, sigreg10_runs: Path, out: Path):
    """Log-log total-loss curve over the 12 500 steps. `PLOT_START_STEP` cuts the
    warm-up: the first ~100 steps have loss > 10× the converged tail and would
    dominate the y-range on linear axes; log x also requires step > 0."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    overlays = {
        "sigreg01_enc3": sigreg_runs / f"bb_{BASE_TAG}_losses.csv",
        "sigreg10_enc3": sigreg10_runs / f"bb_{BASE_TAG}_emb10_losses.csv",
    }
    for arm, p in overlays.items():
        if p.exists():
            d = _smoothed(pd.read_csv(p), "loss", PLOT_START_STEP)
            ax.plot(d["step"], d["loss_sm"],
                    label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.0)
    for arm in SWEEP_ARMS:
        p = arm_runs / f"bb_{BASE_TAG}_{arm}_losses.csv"
        if p.exists():
            d = _smoothed(pd.read_csv(p), "loss", PLOT_START_STEP)
            ax.plot(d["step"], d["loss_sm"],
                    label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"step (log; start = {PLOT_START_STEP})")
    ax.set_ylabel("loss (50-step rolling mean, log)")
    ax.set_title("Total training loss — log-log; anchors (#355, #359) and the 6 sweep arms")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_sigreg_inspection(arm_runs: Path, sigreg_runs: Path, sigreg10_runs: Path, out: Path):
    """4-panel log-y trajectories of L_SIGReg(e_t) / L_SIGReg(h_t) / U_batch(e_t) /
    U_temporal(e_t). Bottom row = dimension-usage U on the embedding side `e_t`,
    clipped to `[1/K, 1]`. U = 1 ⇔ isotropic / all K dims used; U = 1/K ⇔ rank-1
    collapse (one effective dim); K · U ≈ effective number of dims in use. The
    floor at 1/K marks rank-1 collapse. All panels use log y so the bottom row's
    tiny values (1/K ≈ 0.0026, u_batch_e ≈ 0.02–0.06) are readable instead of
    crammed at the baseline of a [0,1] linear axis; matches the parent #359
    report's sigreg_e_inspection convention. `PLOT_START_STEP` cuts the warm-up
    regime."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    K = 384
    panels = [
        ("sigreg_e",     "L_SIGReg(e_t)"),
        ("sigreg_h",     "L_SIGReg(h_t)"),
        ("u_batch_e",    "U_batch (e_t) — cross-batch dimension usage"),
        ("u_temporal_e", "U_temporal (e_t) — cross-time dimension usage"),
    ]
    overlays = {
        "sigreg01_enc3": sigreg_runs / f"bb_{BASE_TAG}_losses.csv",
        "sigreg10_enc3": sigreg10_runs / f"bb_{BASE_TAG}_emb10_losses.csv",
    }
    for ax, (col, title) in zip(axes.ravel(), panels):
        for arm, p in overlays.items():
            if p.exists():
                d = pd.read_csv(p)
                if col in d.columns:
                    d = _smoothed(d, col, PLOT_START_STEP)
                    ax.plot(d["step"], d[f"{col}_sm"],
                            label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.0)
        for arm in SWEEP_ARMS:
            p = arm_runs / f"bb_{BASE_TAG}_{arm}_losses.csv"
            if p.exists():
                d = pd.read_csv(p)
                if col in d.columns:
                    d = _smoothed(d, col, PLOT_START_STEP)
                    ax.plot(d["step"], d[f"{col}_sm"],
                            label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.6)
        if col in ("u_batch_e", "u_temporal_e"):
            ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5,
                       label=f"1/K = 1/{K} ≈ {1/K:.4f} (rank-1 collapse, 1 effective dim)")
        ax.set_yscale("log")
        ax.set_xlabel(f"step  (start = {PLOT_START_STEP})")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="best"); ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        f"SIGReg trajectories — log y; the 6 sweep arms vs the two prior λ_h=0.1 anchors "
        f"(start step = {PLOT_START_STEP}, 50-step rolling mean)"
    )
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_dim_usage(arm_runs: Path, sigreg_runs: Path, sigreg10_runs: Path,
                   retro_csv: Path, traj_csv: Path | None, out: Path):
    """Dimension usage U for sweep arms and the 2 anchors. U = participation-ratio
    fraction in [1/K, 1] with K=384: U=1 ⇔ isotropic / all K dims used,
    U=1/K ⇔ rank-1 collapse (one effective dim). Effective dims ≈ K · U.

    Two stacked panels with shared x-axis and per-row y-scales:
      top   — e_t (embedding side): u_batch_e, u_temporal_e trajectories;
              u_batchtime_e all-checkpoint trajectory (dotted) for the 4 prior
              sweep arms + 2 anchors, terminal ★ at step 12 500. y ∈ [1/K, 0.1] (log).
      bottom — h_t (encoder side): u_batch, u_temporal trajectories;
              u_batchtime all-checkpoint trajectory (dotted) for the 4 prior
              sweep arms + 2 anchors, terminal ★ at step 12 500. y ∈ [0.05, 1] (log).

    Linestyle = pooling axis (solid = u_batch, dashed = u_temporal,
    dotted+★ = u_batchtime checkpoint trajectory). Colour = arm.
    For arms 4 and 6 (training post-dates the trajectory snapshot),
    only the FINAL retroactive value (from `retro_csv`) is plotted as a ★."""
    K = 384
    fig, (ax_e, ax_h) = plt.subplots(2, 1, sharex=True, figsize=(13, 9.5))
    overlays = {
        "sigreg01_enc3": sigreg_runs / f"bb_{BASE_TAG}_losses.csv",
        "sigreg10_enc3": sigreg10_runs / f"bb_{BASE_TAG}_emb10_losses.csv",
    }

    def _draw_panel(ax, latent_suffix):
        for arm, p in overlays.items():
            if not p.exists():
                continue
            d = pd.read_csv(p)
            for base_col, ls in (("u_batch", "-"), ("u_temporal", "--")):
                col = f"{base_col}{latent_suffix}"
                if col in d.columns:
                    ds = _smoothed(d, col, PLOT_START_STEP)
                    ax.plot(ds["step"], ds[f"{col}_sm"],
                            color=ARM_COLOR[arm], lw=1.0, ls=ls)
        for arm in SWEEP_ARMS:
            p = arm_runs / f"bb_{BASE_TAG}_{arm}_losses.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            for base_col, ls in (("u_batch", "-"), ("u_temporal", "--")):
                col = f"{base_col}{latent_suffix}"
                if col in d.columns:
                    ds = _smoothed(d, col, PLOT_START_STEP)
                    ax.plot(ds["step"], ds[f"{col}_sm"],
                            color=ARM_COLOR[arm], lw=1.6, ls=ls)

    _draw_panel(ax_e, "_e")
    _draw_panel(ax_h, "")

    retro_arm_map = {
        "anchor_emb01":   "sigreg01_enc3",
        "anchor_emb10":   "sigreg10_enc3",
        "emb100_enc01":   "emb100_enc01",
        "emb100_enc10":   "emb100_enc10",
        "emb100_enc100":  "emb100_enc100",
        "emb10_enc10":    "emb10_enc10",
        "emb1000_enc01":  "emb1000_enc01",
        "emb10000_enc10": "emb10000_enc10",
    }

    if traj_csv is not None and traj_csv.exists():
        traj = pd.read_csv(traj_csv)
        traj_step = traj[traj["ckpt_kind"].str.startswith("step_", na=False)].copy()
        traj_best = traj[traj["ckpt_kind"] == "best_loss"].copy()
        for arm_key, arm in retro_arm_map.items():
            t = traj_step[traj_step["arm"] == arm_key].sort_values("step")
            if not len(t):
                continue
            ax_e.plot(t["step"], t["u_batchtime_e"],
                      color=ARM_COLOR[arm], lw=1.0, ls=":", marker="o",
                      markersize=4, alpha=0.85, zorder=3)
            ax_h.plot(t["step"], t["u_batchtime"],
                      color=ARM_COLOR[arm], lw=1.0, ls=":", marker="o",
                      markersize=4, alpha=0.85, zorder=3)
            # best_loss / FINAL.pth: ★ at the actual best-loss step
            tb = traj_best[traj_best["arm"] == arm_key]
            if len(tb):
                br = tb.iloc[0]
                ax_e.scatter([int(br["step"])], [float(br["u_batchtime_e"])],
                             color=ARM_COLOR[arm], marker="*", s=130,
                             edgecolor="black", linewidths=0.5, zorder=5)
                ax_h.scatter([int(br["step"])], [float(br["u_batchtime"])],
                             color=ARM_COLOR[arm], marker="*", s=130,
                             edgecolor="black", linewidths=0.5, zorder=5)

    for ax, (lo, hi), title in (
        (ax_e, (1.0 / K, 0.1),  "e_t (embedding-side) — u_batch_e (solid), u_temporal_e (dashed), u_batchtime_e trajectory (dotted+●), best_loss=FINAL.pth (★)"),
        (ax_h, (0.05, 1.0),     "h_t (encoder-side)  — u_batch (solid), u_temporal (dashed), u_batchtime trajectory (dotted+●), best_loss=FINAL.pth (★)"),
    ):
        ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5,
                   label=f"1/K = 1/{K} ≈ {1/K:.4f}  (rank-1 collapse: 1 effective dim)")
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_ylabel("U (dimension usage; higher = more effective dims)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")

    ax_h.set_xlabel(f"step  (start = {PLOT_START_STEP})")

    plotted_arms = ["sigreg01_enc3", "sigreg10_enc3",
                    "emb100_enc01", "emb100_enc10", "emb100_enc100",
                    "emb10_enc10", "emb1000_enc01", "emb10000_enc10"]
    arm_handles = [plt.Line2D([0], [0], color=ARM_COLOR[a], lw=2.0,
                              label=ARM_LABEL[a]) for a in plotted_arms]
    style_handles = [
        plt.Line2D([0], [0], color="black", ls="-",  lw=1.5, label="u_batch (cross-batch)"),
        plt.Line2D([0], [0], color="black", ls="--", lw=1.5, label="u_temporal (cross-time)"),
        plt.Line2D([0], [0], color="black", ls=":", marker="o", markersize=5,
                   lw=1.0, label="u_batchtime — checkpoint trajectory"),
        plt.Line2D([0], [0], color="black", marker="*", markersize=10,
                   markerfacecolor="white", markeredgecolor="black",
                   linestyle="", label="best_loss = FINAL.pth (★ at best-loss step)"),
        plt.Line2D([0], [0], color="k", ls=":", alpha=0.5,
                   label=f"1/K ≈ {1/K:.4f}  (rank-1 collapse)"),
    ]
    leg1 = ax_e.legend(handles=arm_handles, fontsize=6.5, loc="upper left",
                       ncol=2, title="arm", title_fontsize=7)
    ax_e.add_artist(leg1)
    ax_h.legend(handles=style_handles, fontsize=6.8, loc="lower right",
                title="pooling axis / marker", title_fontsize=7)

    fig.suptitle(
        "Dimension usage U — split by latent. K·U ≈ effective number of "
        "dimensions in use (K=384). Higher = more dims used.")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm-runs", type=Path, required=True)
    p.add_argument("--sigreg01-runs", type=Path, required=True)
    p.add_argument("--sigreg10-runs", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args(argv)

    plots = args.report_dir / "plots"
    results = args.report_dir / "results"
    plots.mkdir(parents=True, exist_ok=True)

    gm = load_gm_table(results / "gm_table.csv")
    ci = load_ci(results / "bootstrap_ci_vs_359.csv")

    plot_gm_bars(gm, ci, plots / "gm_rel_mase.png")
    plot_heatmap_grid(gm, plots / "heatmap.png")
    plot_lambda_e_ladder(gm, ci, plots / "lambda_e_ladder.png")
    plot_best_vs_last_drift(gm, plots / "best_vs_last_drift.png")
    plot_loss_curves(args.arm_runs, args.sigreg01_runs, args.sigreg10_runs,
                     plots / "loss_curve.png")
    plot_sigreg_inspection(args.arm_runs, args.sigreg01_runs, args.sigreg10_runs,
                           plots / "sigreg_e_inspection.png")
    traj_path = results / "u_batchtime_trajectory.csv"
    plot_dim_usage(args.arm_runs, args.sigreg01_runs, args.sigreg10_runs,
                   results / "u_batchtime_retro.csv",
                   traj_path if traj_path.exists() else None,
                   plots / "dim_usage.png")
    print(f"wrote 7 plots under {plots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
