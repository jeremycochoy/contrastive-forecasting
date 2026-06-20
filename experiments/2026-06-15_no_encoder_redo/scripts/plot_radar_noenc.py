#!/usr/bin/env python3
"""Per-domain forecast-error radar (house style): GM-Relative MASE by GIFT-Eval
domain, both heads, best-loss checkpoint. Profiles isolate the encoder vs CPC
question: the best plain-encoder backbone (enc-3 base, dashed reference) against
the no-encoder + CPC arm and the encoder'd + CPC arm (enc-3 + CPC), plus the
seasonal-naive ring (1.0). Closer to centre is better; log radial axis. Per-task
relative error from each arm's summary.txt; dataset->domain from all_results.csv.
Writes plots/perdomain_radar.png.
"""
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E348 = f"{W}/2026-06-15_no_encoder_redo/results"
E339 = f"{W}/2026-06-10_stopgrad_positive/results"
E344 = f"{W}/2026-06-13_cpc_infonce_aux/results"
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "perdomain_radar.png")
DOMAINS = ["Econ/Fin", "Energy", "Healthcare", "Nature", "Sales", "Transport", "Web/CloudOps"]
# (label, results_dir, tag, colour, linestyle)
ARMS = [
    ("enc-3 base (best encoder, no CPC)", E339, "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024",       "0.45",    "--"),
    ("no-encoder + CPC",                  E348, "allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc",  "#d62728", "-"),
    ("enc-3 + CPC",                       E344, "allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc",   "#1f77b4", "-"),
]


def relatives(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try:
                out[p[0]] = float(p[3])
            except ValueError:
                pass
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def domain_map():
    p = f"{E348}/gift_eval_full_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_base_2L/all_results.csv"
    return {r["dataset"]: r["domain"] for r in csv.DictReader(open(p))}


def per_domain(rel, dmap):
    by = {d: [] for d in DOMAINS}
    for cfg, v in rel.items():
        if dmap.get(cfg) in by:
            by[dmap[cfg]].append(v)
    return [gm(by[d]) for d in DOMAINS]


def main():
    dmap = domain_map()
    ang = np.linspace(0, 2 * np.pi, len(DOMAINS), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    ring_t = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), subplot_kw=dict(polar=True))
    allvals = []
    for ax, head, title in zip(axes, ["2L", "6L"], ["2-layer head", "6-layer head"]):
        ax.plot(ring_t, [1.0] * 200, ls="--", color="k", lw=1, alpha=0.55, label="seasonal-naive (1.0)")
        for label, rdir, tag, col, ls in ARMS:
            vals = per_domain(relatives(f"{rdir}/gift_eval_full_{tag}_{head}/summary.txt"), dmap)
            allvals += [v for v in vals if v == v]
            lw = 1.8 if ls == "--" else 2.2
            ax.plot(ang_c, vals + vals[:1], ls=ls, color=col, lw=lw, marker="o", ms=3, label=label)
        ax.set_xticks(ang)
        ax.set_xticklabels(DOMAINS, fontsize=9)
        ax.set_title(title, fontsize=12, pad=24)
        ax.grid(alpha=0.3)
        handles = ax.get_legend_handles_labels()
    lo, hi = min(allvals), max(allvals)
    for ax in axes:
        ax.set_rscale("log")
        ax.set_rlim(lo * 0.95, hi * 1.05)
        ax.yaxis.set_major_locator(FixedLocator([round(lo, 1), 1.0, round(hi, 1)]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_rlabel_position(88)
        ax.tick_params(labelsize=8)
    fig.legend(handles[0], handles[1], loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("GM-Relative MASE by GIFT-Eval domain (best-loss): no-encoder + CPC vs encoder'd backbones "
                 "(closer to centre is better, log radial)", fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))
    for head in ["2L", "6L"]:
        print(f"\n{head}  {'domain':13s}" + "".join(f"{a[0]:>16s}" for a in ARMS))
        cols = [per_domain(relatives(f"{r}/gift_eval_full_{t}_{head}/summary.txt"), dmap) for _, r, t, _, _ in ARMS]
        for i, d in enumerate(DOMAINS):
            print(f"   {d:13s}" + "".join(f"{c[i]:16.3f}" for c in cols))


if __name__ == "__main__":
    main()
