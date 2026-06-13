"""Per-domain GM-Relative MASE: where the full-4096 MOIRAI-HP backbone wins and
loses on GIFT-Eval. Computed entirely from this experiment's own committed eval
outputs:
  results/gift_eval_resume50k_local/all_results.csv  -> dataset -> domain
  results/gift_eval_resume50k_local/summary.txt       -> dataset -> Relative MASE
joined on dataset, grouped by domain (geometric mean). 1.0 = seasonal-naive.

Adapted from experiments/2026-04-13_gift-eval/scripts/plot_domain_breakdown.py.
No per-domain leaderboard markers are drawn: this experiment commits no per-domain
reference numbers, and inventing them would violate the facts-only rule."""
import csv, math, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "gift_eval_resume50k_local")

# dataset -> Relative MASE (from the summary table)
rel = {}
for line in open(os.path.join(RES, "summary.txt")):
    p = line.split()
    if len(p) >= 4 and "/" in p[0]:
        try:
            rel[p[0]] = float(p[-1])
        except ValueError:
            pass

# dataset -> domain (from the per-config CSV)
dom = {r["dataset"]: r["domain"] for r in csv.DictReader(open(os.path.join(RES, "all_results.csv")))}

g = defaultdict(list)
for ds, d in dom.items():
    if ds in rel:
        g[d].append(rel[ds])
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
ours = {d: gm(v) for d, v in g.items()}
ncfg = {d: len(v) for d, v in g.items()}
overall = gm([rel[ds] for ds in dom if ds in rel])

order = sorted(ours, key=lambda d: -ours[d])
labels = [f"{d}\n({ncfg[d]} cfg)" for d in order]
ov = [ours[d] for d in order]
colors = ["#c0392b" if o > 1.0 else "#27ae60" for o in ov]

fig, ax = plt.subplots(figsize=(9, 4.6))
y = range(len(order))
ax.barh(y, ov, color=colors, zorder=3)
ax.axvline(1.0, color="k", ls="--", lw=1.2, zorder=2, label="seasonal-naive (1.0)")
for i, o in enumerate(ov):
    ax.text(o + 0.03, i, f"{o:.2f}", va="center", fontsize=9)
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("GM-Relative MASE  (lower = better; 1.0 = seasonal-naive)")
ax.set_xlim(0, 2.3)
ax.set_title(f"Per-domain GM-Relative MASE — overall {overall:.3f}\n"
             f"red = worse than seasonal-naive, green = better")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
out = os.path.join(HERE, "..", "plots", "domain_breakdown.png")
plt.savefig(out, dpi=130)
print("per-domain:", {d: round(ours[d], 3) for d in order})
print("overall", round(overall, 4))
print("wrote", os.path.abspath(out))
