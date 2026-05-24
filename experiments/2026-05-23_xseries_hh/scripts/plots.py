#!/usr/bin/env python3
"""#318 plots — generates every figure for RESULTS.md from the eval
summaries + training losses CSVs. Each figure auto-skips if its inputs are
missing, so this is safe to run incrementally as cells land.

Figures:
  gm_summary.png      final full-97 & triage GM-Relative MASE: xshh {2L,6L}
                      vs β {2L,6L}, with v11c + seasonal-naive references.
  gm_vs_step.png      full-97 GM vs backbone training step (2L head),
                      xshh vs β — does the "more training stops helping"
                      decoupling shrink?
  training_curves.png contrastive training dynamics (loss / gap / AUC / Top1)
                      vs step, xshh vs β.
  perdomain.png       per-domain GM-Relative MASE (2L), xshh vs β vs v11c —
                      flags collateral damage on strongly-seasonal domains.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh"
RES = f"{OUT}/results"
PLOTS = "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh/experiments/2026-05-23_xseries_hh/plots"
os.makedirs(PLOTS, exist_ok=True)
BETA_DIR = "/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound"
V11C = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"

XSHH_CSV = f"{OUT}/runs/bb_xshh_50k_losses.csv"
BETA_CSV = f"{BETA_DIR}/runs/bb_beta_50k_losses.csv"

C_XSHH, C_BETA, C_V11C = "#1f77b4", "#d62728", "#9467bd"


def agg_gm(sum_txt):
    """Aggregate GM-Relative MASE printed in a summary.txt (or None)."""
    if not os.path.exists(sum_txt):
        return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                for t in reversed(line.replace(":", " ").split()):
                    try:
                        return float(t)
                    except ValueError:
                        continue
    return None


def per_config_relative(sum_txt):
    """{config: relative_mase} parsed from a summary.txt body (or {})."""
    out = {}
    if not os.path.exists(sum_txt):
        return out
    with open(sum_txt) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and "/" in p[0]:
                try:
                    out[p[0]] = float(p[-1])
                except ValueError:
                    pass
    return out


def config_domain(csv_path):
    """{config: domain} from an all_results.csv (or {})."""
    out = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["dataset"]] = row.get("domain", "?")
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# ---------------------------------------------------------------- summary bars
def plot_gm_summary():
    # label -> summary.txt. β 2L is the #309 backbone-of-record eval.
    cells = [
        ("xshh · 2L head", f"{RES}/gift_eval_full_xshh_50k_2L/summary.txt",
         f"{RES}/gift_eval_triage_xshh_50k_2L/summary.txt", C_XSHH),
        ("xshh · 6L head", f"{RES}/gift_eval_full_xshh_50k_6L/summary.txt",
         f"{RES}/gift_eval_triage_xshh_50k_6L/summary.txt", C_XSHH),
        ("β · 2L head", f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt",
         f"{BETA_DIR}/results/gift_eval_triage_bb_beta_50k/summary.txt", C_BETA),
        ("β · 6L head", f"{RES}/gift_eval_full_beta_50k_6L/summary.txt",
         f"{RES}/gift_eval_triage_beta_50k_6L/summary.txt", C_BETA),
    ]
    rows = []
    for lab, fs, ts, c in cells:
        rows.append((lab, agg_gm(fs), agg_gm(ts), c))
    if not any(r[1] or r[2] for r in rows):
        print("gm_summary: no data yet — skip")
        return
    v = agg_gm(f"{V11C}/summary.txt") or 1.292
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, idx, title in ((axes[0], 1, "full-97"), (axes[1], 2, "triage-11")):
        labs = [r[0] for r in rows]
        vals = [r[idx] for r in rows]
        cols = [r[3] for r in rows]
        y = range(len(rows))
        ax.barh(list(y), [x if x else 0 for x in vals], color=cols, alpha=0.85)
        for i, val in enumerate(vals):
            if val:
                ax.text(val + 0.004, i, f"{val:.4f}", va="center", fontsize=9)
        ax.set_yticks(list(y)); ax.set_yticklabels(labs, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"{title} GM-Relative MASE (lower = better)")
        good = [x for x in vals if x]
        if good:
            ax.set_xlim(min(good + [v]) * 0.98, max(good) * 1.06)
        if idx == 1:
            ax.axvline(v, color=C_V11C, ls="--", lw=1.4, label=f"v11c = {v:.3f}")
        ax.axvline(1.0, color="k", ls=":", lw=1.0, alpha=0.5)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("#318 — frozen-backbone GM-Relative MASE: cross-series same-step h↔h vs β", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/gm_summary.png", dpi=130, bbox_inches="tight")
    print("gm_summary.png written")


# --------------------------------------------------------------- GM vs step
def plot_gm_vs_step():
    steps = [20000, 35000, 50000]
    xshh = {20000: f"{RES}/gift_eval_full_xshh_20k_2L/summary.txt",
            35000: f"{RES}/gift_eval_full_xshh_35k_2L/summary.txt",
            50000: f"{RES}/gift_eval_full_xshh_50k_2L/summary.txt"}
    beta = {20000: f"{RES}/gift_eval_full_beta_20k_2L/summary.txt",
            35000: f"{RES}/gift_eval_full_beta_35k_2L/summary.txt",
            50000: f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt"}
    xs = [(s, agg_gm(xshh[s])) for s in steps if agg_gm(xshh[s])]
    bs = [(s, agg_gm(beta[s])) for s in steps if agg_gm(beta[s])]
    if not xs and not bs:
        print("gm_vs_step: no data yet — skip")
        return
    v = agg_gm(f"{V11C}/summary.txt") or 1.292
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if xs:
        ax.plot([s/1000 for s, _ in xs], [g for _, g in xs], "o-", color=C_XSHH, lw=2, label="xshh (this card)")
    if bs:
        ax.plot([s/1000 for s, _ in bs], [g for _, g in bs], "s-", color=C_BETA, lw=2, label="β (#309)")
    ax.axhline(v, color=C_V11C, ls="--", lw=1.4, label=f"v11c = {v:.3f}")
    ax.set_xlabel("backbone training step (k)")
    ax.set_ylabel("full-97 GM-Relative MASE (2L head, lower = better)")
    ax.set_title("#318 — does more contrastive training stop helping?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/gm_vs_step.png", dpi=130, bbox_inches="tight")
    print("gm_vs_step.png written")


# --------------------------------------------------------- training curves
def _load_curve(csv_path):
    if not os.path.exists(csv_path):
        return None
    cols = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k, val in row.items():
                try:
                    cols.setdefault(k, []).append(float(val))
                except (ValueError, TypeError):
                    pass
    return cols


def _smooth(y, w=200):
    if len(y) < w:
        return y
    out, acc = [], 0.0
    from collections import deque
    q = deque()
    for v in y:
        q.append(v); acc += v
        if len(q) > w:
            acc -= q.popleft()
        out.append(acc/len(q))
    return out


def plot_training_curves():
    a = _load_curve(XSHH_CSV)
    b = _load_curve(BETA_CSV)
    if not a and not b:
        print("training_curves: no data yet — skip")
        return
    panels = [("loss", "contrastive loss"), ("gap", "pos−neg gap"),
              ("auc", "AUC (pos vs neg)"), ("top1", "Top-1 retrieval")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (key, title) in zip(axes.flat, panels):
        for cols, c, lab in ((a, C_XSHH, "xshh"), (b, C_BETA, "β")):
            if cols and key in cols and "step" in cols:
                ax.plot(cols["step"], _smooth(cols[key]), color=c, lw=1.4, label=lab)
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("#318 — contrastive training dynamics (200-step MA): xshh vs β", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/training_curves.png", dpi=130, bbox_inches="tight")
    print("training_curves.png written")


# ------------------------------------------------------------- per-domain
def plot_perdomain():
    dom_map = (config_domain(f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/all_results.csv")
               or config_domain(f"{RES}/gift_eval_full_xshh_50k_2L/all_results.csv"))
    srcs = [("xshh 2L", f"{RES}/gift_eval_full_xshh_50k_2L/summary.txt", C_XSHH),
            ("β 2L", f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt", C_BETA),
            ("v11c", f"{V11C}/summary.txt", C_V11C)]
    series = []
    for lab, st, c in srcs:
        rel = per_config_relative(st)
        if not rel:
            continue
        byd = {}
        for cfg, r in rel.items():
            byd.setdefault(dom_map.get(cfg, "?"), []).append(r)
        series.append((lab, {d: gm(v) for d, v in byd.items()}, c))
    if len(series) < 1 or not dom_map:
        print("perdomain: no data yet — skip")
        return
    doms = sorted({d for _, m, _ in series for d in m})
    import numpy as np
    x = np.arange(len(doms)); w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, (lab, m, c) in enumerate(series):
        ax.bar(x + i*w, [m.get(d, 0) for d in doms], w, color=c, alpha=0.85, label=lab)
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x + w*(len(series)-1)/2); ax.set_xticklabels(doms, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("per-domain GM-Relative MASE (lower=better)")
    ax.set_title("#318 — per-domain transfer (full-97, 2L): cross-series same-step h↔h vs β")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(f"{PLOTS}/perdomain.png", dpi=130, bbox_inches="tight")
    print("perdomain.png written")


if __name__ == "__main__":
    plot_gm_summary()
    plot_gm_vs_step()
    plot_training_curves()
    plot_perdomain()
