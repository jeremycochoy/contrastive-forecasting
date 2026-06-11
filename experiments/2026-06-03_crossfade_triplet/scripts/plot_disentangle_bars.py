#!/usr/bin/env python3
"""#328 disentanglement bar chart -> plots/gm_summary.png.

One panel per head (2L, 6L). Per arm: two bars = Δ GM-Relative MASE vs base at the
best-loss and last checkpoints, with 90% paired-bootstrap CIs. Bars below the zero
(base) line are improvements; green = CI clear of zero (reliable), grey = straddles
zero, red = reliably worse. Arms with no eval yet are skipped (so re-running picks
up L6+nobn+triplet / base+triplet once they land)."""
import math, os, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CF = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
BASE_RES = f"{CF}/2026-05-29_forked_6Lf_b1024/results"
TRIP_RES = f"{CF}/2026-06-03_crossfade_triplet/results"
BASE_TAG = "xshh_allt_forked2_qk_aon_b1024"
OUT = os.environ.get("OUT", "/tmp/cf-328/experiments/2026-06-03_crossfade_triplet/plots/gm_summary.png")
ARMS = [  # display, tag
    ("L3", "allt08_L3_qk_aon_b1024"),
    ("L6+nobn", "allt08_nobn_qk_aon_b1024"),
    ("L3+nobn", "allt08_L3_nobn_qk_aon_b1024"),
    ("L3+nobn+triplet", "allt08_xftrip_nobn_enc3_qk_aon_b1024"),
    ("L6+nobn+triplet", "allt08_xftrip_nobn_enc6_qk_aon_b1024"),
    ("base+triplet", "allt08_xftrip_bn_enc6_qk_aon_b1024"),
]


def relatives(p):
    d = {}
    if not os.path.exists(p):
        return d
    for line in open(p):
        f = line.split()
        if len(f) == 4 and "/" in f[0]:
            try: d[f[0]] = float(f[3])
            except ValueError: pass
    return d


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def delta_ci(base, arm, n=2000, seed=0):
    common = sorted(set(base) & set(arm))
    if len(common) < 2: return None
    a = [base[c] for c in common]; b = [arm[c] for c in common]
    d = gm(b) - gm(a); rng = random.Random(seed); ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([b[i] for i in idx]) - gm([a[i] for i in idx]))
    ds.sort()
    return d, ds[int(0.05 * n)], ds[int(0.95 * n)]


def rels(tag, head): return relatives(f"{TRIP_RES}/gift_eval_full_{tag}_{head}/summary.txt")


def color(lo, hi):
    if hi < 0: return "#2ca02c"      # reliably better
    if lo > 0: return "#d62728"      # reliably worse
    return "#9e9e9e"                  # straddles zero


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, head in zip(axes, ["2L", "6L"]):
    base = relatives(f"{BASE_RES}/gift_eval_full_{BASE_TAG}_{head}/summary.txt")
    labels, xs, i = [], [], 0
    for disp, tag in ARMS:
        bars = [(off, hatch, delta_ci(base, rels(tag + ck, head)))
                for off, ck, hatch in [(-0.2, "", None), (0.2, "_last", "//")]
                if rels(tag + ck, head)]
        # only show an arm once BOTH best and last evals exist (clean, no half rows)
        if len([b for b in bars if b[2]]) < 2:
            continue
        for off, hatch, res in bars:
            d, lo, hi = res
            ax.bar(i + off, d, 0.38, color=color(lo, hi), hatch=hatch, edgecolor="white",
                   yerr=[[d - lo], [hi - d]], capsize=3, error_kw=dict(lw=1, alpha=0.7))
        labels.append(disp); xs.append(i); i += 1
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"{head} head", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Δ GM-Relative MASE vs base  (↓ = better)")
fig.suptitle("Disentanglement: each change vs base, best-loss (solid) and last (hatched) checkpoints.\n"
             "Green = 90% CI clear of zero (reliable); grey = straddles zero; red = reliably worse.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130); print("wrote", OUT)
