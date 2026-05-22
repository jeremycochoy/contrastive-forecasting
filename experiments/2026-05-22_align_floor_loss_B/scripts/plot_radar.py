#!/usr/bin/env python3
"""#313 — per-domain radar: (B)+align+floor vs (B) vs v11c.

Radial axis = per-domain geometric-mean relative MASE on full GIFT-Eval
(97 configs), log scale; dashed ring at 1.0 = seasonal-naive parity;
lower = better. Adapted from #309 plot_results.py:draw_radar.

Robust to a missing new-arm eval (skipped) so it can be validated on the
two baselines before training finishes.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B"
CL_ABL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
ENC = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster"
OUT = f"{MAIN}/plots"; os.makedirs(OUT, exist_ok=True)

C_NEW, C_B, C_V = "#ff7f0e", "#7f7f7f", "#9467bd"
# (label, colour, full_eval_dir, ls, lw)
ARMS = [
    ("(B)+L_align+floor", C_NEW, f"{MAIN}/results/gift_eval_full_bb_alignfloor_50k", "-", 2.4),
    ("(B) bneck τ0.1 β2.95", C_B, f"{CL_ABL}/results/gift_eval_full_cl_hh_50k", "-", 2.0),
    ("v11c (target)", C_V, f"{ENC}/results/gift_eval_full_v11c", (0, (4, 2)), 1.8),
]


def dom_map(ar_csv):
    m = {}
    if not os.path.exists(ar_csv): return m
    with open(ar_csv) as f:
        for r in csv.DictReader(f):
            m[r["dataset"]] = r.get("domain", "?")
    return m


def rel_by_domain(sum_txt, dmap):
    if not os.path.exists(sum_txt) or not dmap: return {}
    acc = {}
    with open(sum_txt) as f:
        for line in f:
            p = line.split()
            if len(p) < 4: continue
            cfg = p[0]
            try: rel = float(p[-1])
            except ValueError: continue
            if cfg not in dmap or rel <= 0: continue
            acc.setdefault(dmap[cfg], []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def agg_gm(sum_txt):
    if not os.path.exists(sum_txt): return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                for t in reversed(line.replace(":", " ").split()):
                    try: return float(t)
                    except ValueError: continue
    return None


radar = []
for lab, col, edir, ls, lw in ARMS:
    g = rel_by_domain(f"{edir}/summary.txt", dom_map(f"{edir}/all_results.csv"))
    if g:
        radar.append((lab, col, g, agg_gm(f"{edir}/summary.txt"), ls, lw))
    else:
        print(f"radar: skipping {lab} (no eval yet at {edir})")

if not radar:
    print("radar: nothing to plot")
    raise SystemExit

domains = sorted({d for _, _, g, *_ in radar for d in g})
N = len(domains)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])
vals = [v for _, _, g, *_ in radar for v in g.values()]
lo, hi = max(0.5, min(vals) * 0.92), max(vals) * 1.06
fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_xticks(theta); ax.set_xticklabels(domains, fontsize=10)
ax.set_rscale("log"); ax.set_ylim(lo, hi)
rticks = [t for t in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0) if lo < t < hi]
ax.set_yticks(rticks); ax.set_yticklabels([f"{t:g}" for t in rticks], fontsize=8, color="0.4")
ax.set_rlabel_position(90)
ax.plot(theta_closed, [1.0] * len(theta_closed), color="k", ls=(0, (2, 2)), lw=1.0, alpha=0.6, zorder=1)
for lab, col, g, gm, ls, lw in radar:
    v = np.array([g.get(d, np.nan) for d in domains] + [g.get(domains[0], np.nan)])
    ax.plot(theta_closed, v, color=col, ls=ls, lw=lw, zorder=3,
            label=f"{lab}   GM={gm:.3f}" if gm else lab, marker="o", markersize=3)
ax.set_title("#313 — full GIFT-Eval (97 cfg) per domain: (B)+align+floor vs (B) vs v11c\n"
             "(radial = per-domain GM rel-MASE, log; dashed ring = seasonal naive 1.0; lower = better)",
             fontsize=11, pad=24)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/perdomain_star.png", dpi=120, bbox_inches="tight"); plt.close()
print(f"wrote {OUT}/perdomain_star.png — arms={len(radar)} domains={N} rlim=({lo:.2f},{hi:.2f})")
