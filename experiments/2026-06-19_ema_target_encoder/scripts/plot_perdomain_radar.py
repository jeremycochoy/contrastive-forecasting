#!/usr/bin/env python3
"""Per-domain radar for the EMA-target arm vs the stop-grad-positive
baseline. Output in plots/perdomain_radar.png — per-domain geometric-mean
relative MASE on GIFT-Eval full-97, log-scale radial axis, one panel per
q-head depth (2L | 6L), curves for each (arm × checkpoint).
"""
import csv
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
BASE_DIR = f"{W}/2026-06-13_cpc_infonce_aux/results"
EMA_DIR = f"{W}/2026-06-19_ema_target_encoder/results"
BASE_TAG = "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc"
EMA_TAG = "allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc"
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS, exist_ok=True)
GREY, BLUE = "0.4", "C0"

CELLS = [
    ("2L best", "2L", ""),
    ("2L last", "2L", "_last"),
    ("6L best", "6L", ""),
    ("6L last", "6L", "_last"),
]


def dataset_domain(all_csv):
    m = {}
    if not os.path.exists(all_csv):
        return m
    for r in csv.DictReader(open(all_csv)):
        m[r["dataset"]] = r.get("domain", "Other")
    return m


def relatives(sum_txt):
    out = {}
    if not os.path.exists(sum_txt):
        return out
    for line in open(sum_txt):
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try:
                out[p[0]] = float(p[3])
            except ValueError:
                pass
    return out


def gm_by_domain(rels, dmap):
    acc = {}
    for cfg, rel in rels.items():
        if rel <= 0 or cfg not in dmap:
            continue
        acc.setdefault(dmap[cfg], []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def load_cell(eval_dir, tag, head, ckpt_suffix):
    sum_t = f"{eval_dir}/gift_eval_full_{tag}{ckpt_suffix}_{head}/summary.txt"
    all_c = f"{eval_dir}/gift_eval_full_{tag}{ckpt_suffix}_{head}/all_results.csv"
    return relatives(sum_t), dataset_domain(all_c)


# --- radar: one panel per head, all 4 (arm × checkpoint) curves overlaid ---
HEAD_PANELS = ["2L", "6L"]
# (arm-label, eval-dir, eval-tag, ckpt-suffix, colour, linestyle)
CURVES = [
    ("baseline · best",  BASE_DIR, BASE_TAG, "",       GREY, "-"),
    ("baseline · last",  BASE_DIR, BASE_TAG, "_last",  GREY, "--"),
    ("EMA-target · best", EMA_DIR, EMA_TAG,  "",       BLUE, "-"),
    ("EMA-target · last", EMA_DIR, EMA_TAG,  "_last",  BLUE, "--"),
]
fig, axes = plt.subplots(1, 2, figsize=(15, 8),
                         subplot_kw=dict(polar=True))
for ax, head in zip(axes, HEAD_PANELS):
    cells = []
    for lab, edir, tag, suf, col, ls in CURVES:
        rel, dmap = load_cell(edir, tag, head, suf)
        gm = gm_by_domain(rel, dmap)
        if gm:
            cells.append((lab, gm, col, ls))
    if not cells:
        ax.text(0.5, 0.5, "no eval", transform=ax.transAxes); continue
    domains = sorted(set().union(*(g for _, g, _, _ in cells)))
    N = len(domains)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    vals = [v for _, g, _, _ in cells for v in g.values()]
    lo, hi = max(0.5, min(vals) * 0.92), max(vals) * 1.06
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(theta); ax.set_xticklabels(domains, fontsize=8)
    ax.set_rscale("log"); ax.set_ylim(lo, hi)
    rticks = [t for t in (0.8, 1.0, 1.2, 1.5, 2.0) if lo < t < hi]
    ax.set_yticks(rticks); ax.set_yticklabels(
        [f"{t:g}" for t in rticks], fontsize=7, color="0.4")
    ax.set_rlabel_position(90)
    ax.plot(theta_closed, [1.0] * len(theta_closed),
            color="k", ls=(0, (2, 2)), lw=0.8, alpha=0.6, zorder=1)
    for lab, g, col, ls in cells:
        v = np.array([g.get(d, np.nan) for d in domains]
                     + [g.get(domains[0], np.nan)])
        ax.plot(theta_closed, v, color=col, ls=ls, lw=1.6, zorder=3,
                marker="o", markersize=3, label=lab)
    ax.set_title(f"{head} q-head", fontsize=11, pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.06),
              fontsize=9, frameon=False, ncol=2)
fig.suptitle("Per-domain GM relative MASE on GIFT-Eval full-97 "
             "(grey = stop-grad on positive, blue = EMA-target; "
             "solid = best-loss, dashed = last; radial = log; "
             "ring at 1.0 = seasonal-naive; lower = better)", fontsize=11)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig(f"{PLOTS}/perdomain_radar.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {PLOTS}/perdomain_radar.png")

