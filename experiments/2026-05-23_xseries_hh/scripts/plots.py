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
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# For the InfoNCE floor (re-bases the contrastive loss so arms with different
# negative counts N are comparable). Training config: τ=0.10, B=256, C=1, and
# latent T=64 — the HF streamer crops to T_RAW=1024 (it ignores --t-raw 4096),
# and 1024//W(=16) = 64. (Verified by a model forward.)
sys.path.insert(0, "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh")
from src.loss import infonce_floor, _effective_negative_count

TAU_TRAIN, B_TRAIN, T_TRAIN, C_TRAIN = 0.10, 256, 64, 1
_SHAPE = {"xshh": "cosine_similarity_batch_full_hh_negs_xshh",
          "allt": "cosine_similarity_batch_full_hh_negs_xshh_allt",
          "beta": "cosine_similarity_batch_full_hh_negs"}
FLOOR = {k: infonce_floor(TAU_TRAIN, _effective_negative_count(s, B_TRAIN, T_TRAIN, C_TRAIN))
         for k, s in _SHAPE.items()}

OUT = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh"
RES = f"{OUT}/results"
PLOTS = "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh/experiments/2026-05-23_xseries_hh/plots"
os.makedirs(PLOTS, exist_ok=True)
BETA_DIR = "/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound"
V11C = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"

XSHH_CSV = f"{OUT}/runs/bb_xshh_50k_losses.csv"
ALLT_CSV = f"{OUT}/runs/bb_xshh_allt_50k_losses.csv"
BETA_CSV = f"{BETA_DIR}/runs/bb_beta_50k_losses.csv"

C_XSHH, C_ALLT, C_BETA, C_V11C = "#1f77b4", "#2ca02c", "#d62728", "#9467bd"


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
        ("xshh same-step · 2L", f"{RES}/gift_eval_full_xshh_50k_2L/summary.txt",
         f"{RES}/gift_eval_triage_xshh_50k_2L/summary.txt", C_XSHH),
        ("xshh same-step · 6L", f"{RES}/gift_eval_full_xshh_50k_6L/summary.txt",
         f"{RES}/gift_eval_triage_xshh_50k_6L/summary.txt", C_XSHH),
        ("xshh all-time · 2L", f"{RES}/gift_eval_full_xshh_allt_50k_2L/summary.txt",
         f"{RES}/gift_eval_triage_xshh_allt_50k_2L/summary.txt", C_ALLT),
        ("xshh all-time · 6L", f"{RES}/gift_eval_full_xshh_allt_50k_6L/summary.txt",
         f"{RES}/gift_eval_triage_xshh_allt_50k_6L/summary.txt", C_ALLT),
        ("β · 2L", f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt",
         f"{BETA_DIR}/results/gift_eval_triage_bb_beta_50k/summary.txt", C_BETA),
        ("β · 6L", f"{RES}/gift_eval_full_beta_50k_6L/summary.txt",
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
    # (label, curve, colour, floor-key)
    arms = [("xshh same-step", _load_curve(XSHH_CSV), C_XSHH, "xshh"),
            ("xshh all-time",  _load_curve(ALLT_CSV), C_ALLT, "allt"),
            ("β",              _load_curve(BETA_CSV), C_BETA, "beta")]
    arms = [a for a in arms if a[1]]
    if not arms:
        print("training_curves: no data yet — skip")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # Panel 1: contrastive loss MINUS each arm's InfoNCE floor. The floor
    # log(1+N·e^(−1/τ)) grows with the negative count N, which differs ~52×
    # across arms, so RAW losses are not comparable — subtracting each arm's
    # own floor is. Genuine log–log: every arm's floor-excess stays strictly
    # positive at all steps (measured ≈ +0.5–0.6 at 50k, larger earlier), so
    # log-y is well-defined and exposes the slow power-law approach to the floor.
    ax = axes[0, 0]
    for lab, cols, col, fk in arms:
        if "loss" in cols and "step" in cols:
            y = _smooth([v - FLOOR[fk] for v in cols["loss"]], 100)
            ax.plot(cols["step"], y, color=col, lw=1.4, label=f"{lab}  (floor {FLOOR[fk]:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("contrastive loss − InfoNCE floor   (log–log)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # Panels 2–3: retrieval error 1−AUC and 1−Top1 (→0), genuine log-log.
    for ax, key, title in ((axes[0, 1], "auc", "1 − AUC"),
                           (axes[1, 0], "top1", "1 − Top-1 retrieval")):
        for lab, cols, col, _ in arms:
            if key in cols and "step" in cols:
                err = _smooth([max(1.0 - v, 1e-6) for v in cols[key]], 100)
                ax.plot(cols["step"], err, color=col, lw=1.4, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{title}   (log-log)"); ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # Panel 4: pos−neg gap (bounded, saturates) — log-x, linear-y.
    ax = axes[1, 1]
    for lab, cols, col, _ in arms:
        if "gap" in cols and "step" in cols:
            ax.plot(cols["step"], _smooth(cols["gap"], 100), color=col, lw=1.4, label=lab)
    ax.set_xscale("log"); ax.set_title("pos−neg gap   (log x)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    fig.suptitle("#318 — contrastive training dynamics (100-step MA), floor-subtracted loss", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/training_curves.png", dpi=130, bbox_inches="tight")
    print("training_curves.png written")


# ------------------------------------------------------------- per-domain
def plot_perdomain():
    dom_map = (config_domain(f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/all_results.csv")
               or config_domain(f"{RES}/gift_eval_full_xshh_50k_2L/all_results.csv"))
    srcs = [("xshh same-step 2L", f"{RES}/gift_eval_full_xshh_50k_2L/summary.txt", C_XSHH),
            ("xshh all-time 2L", f"{RES}/gift_eval_full_xshh_allt_50k_2L/summary.txt", C_ALLT),
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
    # plot_gm_vs_step() — deferred: only the 50k endpoints exist, so the curve
    # would be two lone points. Re-enable once a step-resolved sweep is run.
    plot_training_curves()
    plot_perdomain()
