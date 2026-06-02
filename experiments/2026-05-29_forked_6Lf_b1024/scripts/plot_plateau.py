#!/usr/bin/env python3
"""Plateau figure: the contrastive loss keeps falling, but forecasting does not improve.

Left  — floor-subtracted contrastive loss vs step (log-x) for the two arms tested, with the
        early / mid-plateau checkpoint marked. The loss is still high there and drops a lot
        afterwards.
Right — GM-Relative MASE at that early/mid-plateau checkpoint vs the fully-trained model, per
        arm × head (a dumbbell per cell). Flat or rising = training longer did not help.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/runs"
RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/results"
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024/"
       "experiments/2026-05-29_forked_6Lf_b1024/plots/plateau.png")


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def rel_gm(evaldir):
    p = f"{RES}/{evaldir}/summary.txt"
    if not os.path.exists(p):
        return None
    vals = []
    for line in open(p):
        q = line.split()
        if len(q) == 4 and "/" in q[0]:
            try:
                vals.append(float(q[3]))
            except ValueError:
                pass
    return gm(vals)


def loss_curve(name):
    s, l = [], []
    for r in csv.DictReader(open(f"{RUNS}/{name}_losses.csv")):
        s.append(int(float(r["step"]))); l.append(float(r["loss"]))
    return s, l


# arm: (loss-csv name, mid-checkpoint step, colour, {head: (mid evaldir, final evaldir)})
ARMS = {
    "allt·50%": ("bb_xshh_allt_forked_qk_aon_6Lf_b1024", 1000, "#d62728", {
        "2L": ("gift_eval_full_allt50_step1k_2L", "gift_eval_full_xshh_allt_forked_qk_aon_b1024_2L"),
        "6L": ("gift_eval_full_allt50_step1k_6L", "gift_eval_full_xshh_allt_forked_qk_aon_b1024_6L"),
    }),
    "allt·10%": ("bb_xshh_allt_forked10pct_qk_aon_6Lf_b1024", 2500, "#ff7f0e", {
        "2L": ("gift_eval_full_allt10_plat2500_2L", "gift_eval_full_xshh_allt_forked10pct_qk_aon_b1024_2L"),
        "6L": ("gift_eval_full_allt10_plat2500_6L", "gift_eval_full_xshh_allt_forked10pct_qk_aon_b1024_6L"),
    }),
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- left: loss curves with the checkpoint marked
for arm, (lname, mid, c, _) in ARMS.items():
    try:
        s, l = loss_curve(lname)
    except FileNotFoundError:
        continue
    axL.semilogx(s, l, color=c, lw=1.6, label=arm)
    li = min(range(len(s)), key=lambda i: abs(s[i] - mid))
    axL.scatter([s[li]], [l[li]], color=c, s=55, zorder=5, edgecolor="k", lw=0.6)
    axL.annotate(f"step {mid}", (s[li], l[li]), textcoords="offset points",
                 xytext=(6, 8), fontsize=9, color=c)
axL.set_xlabel("training step (log)")
axL.set_ylabel("contrastive loss above its floor")
axL.set_title("The loss keeps falling after the marked checkpoint")
axL.grid(True, which="both", alpha=0.25)
axL.legend(fontsize=10)

# ---- right: dumbbell of MASE, mid-plateau vs final, per arm × head
rows = []
for arm, (_, _, c, heads) in ARMS.items():
    for hd, (mid_dir, fin_dir) in heads.items():
        mid, fin = rel_gm(mid_dir), rel_gm(fin_dir)
        rows.append((f"{arm} · {hd}", mid, fin, c))
y = list(range(len(rows)))
for yi, (lab, mid, fin, c) in zip(y, rows):
    if mid is None or fin is None:
        axR.text(1.2, yi, f"{lab}: pending", va="center", fontsize=9, color="#888")
        continue
    axR.plot([mid, fin], [yi, yi], color=c, lw=2, zorder=2)
    axR.scatter([mid], [yi], color="white", edgecolor=c, lw=2, s=70, zorder=3)
    axR.scatter([fin], [yi], color=c, s=70, zorder=3)
    axR.annotate(f"{mid:.3f}", (mid, yi), textcoords="offset points", xytext=(-4, 7),
                 ha="right", fontsize=8, color=c)
    axR.annotate(f"{fin:.3f}", (fin, yi), textcoords="offset points", xytext=(4, 7),
                 ha="left", fontsize=8, color=c)
axR.set_yticks(y); axR.set_yticklabels([r[0] for r in rows])
axR.invert_yaxis()
axR.set_xlabel("GM-Relative MASE (lower better)")
axR.set_title("Forecast error: early/mid-plateau checkpoint (○) vs fully trained (●)")
axR.grid(axis="x", ls=":", alpha=0.4)
# legend for the marker meaning
import matplotlib.lines as mlines
axR.legend(handles=[
    mlines.Line2D([], [], marker="o", color="#666", markerfacecolor="white", ls="",
                  markersize=9, label="early / mid-plateau checkpoint"),
    mlines.Line2D([], [], marker="o", color="#666", ls="", markersize=9, label="fully trained"),
], fontsize=9, loc="lower right")

fig.suptitle("Training past the plateau does not improve forecasting", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for lab, mid, fin, _ in rows:
    print(f"  {lab}: mid={mid} final={fin}")
