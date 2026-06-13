"""Per-domain GM-Relative MASE: where the Tiny backbone underperforms on GIFT-Eval.
Computed from this experiment's own eval outputs:
  results/v2_pair_30k_all_results.csv  -> dataset -> domain (and raw MASE)
  results/v2_pair_30k_summary.txt      -> dataset -> Relative MASE (= MASE / seasonal-naive)
joined on dataset, grouped by domain (geometric mean). 1.0 = seasonal-naive.
`best` = best <50M GIFT-Eval leaderboard model per domain (GIFT-Eval leaderboard, tabulated
in notes/DOMAIN_COMPARISON.md) — published numbers, not our eval."""
import csv, math, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
rel = {}
for line in open(os.path.join(RES, "v2_pair_30k_summary.txt")):
    p = line.split()
    if len(p) >= 4 and "/" in p[0]:
        try: rel[p[0]] = float(p[-1])
        except ValueError: pass
dom = {r["dataset"]: r["domain"] for r in csv.DictReader(open(os.path.join(RES, "v2_pair_30k_all_results.csv")))}
g = defaultdict(list)
for ds, d in dom.items():
    if ds in rel: g[d].append(rel[ds])
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
ours = {d: gm(v) for d, v in g.items()}; ncfg = {d: len(v) for d, v in g.items()}
overall = gm([rel[ds] for ds in dom if ds in rel])
best = {"Sales": 0.689, "Transport": 0.610, "Nature": 0.704, "Energy": 0.827,
        "Web/CloudOps": 0.636, "Econ/Fin": 0.760, "Healthcare": 0.600}
order = sorted(ours, key=lambda d: -ours[d])
labels = [f"{d}\n({ncfg[d]} cfg)" for d in order]
ov = [ours[d] for d in order]; bv = [best[d] for d in order]
colors = ["#c0392b" if o > 1.0 else "#27ae60" for o in ov]
fig, ax = plt.subplots(figsize=(9, 4.6)); y = range(len(order))
ax.barh(y, ov, color=colors, zorder=3)
ax.scatter(bv, y, marker="D", color="#2c3e50", zorder=4, label="best leaderboard model (<50M)")
ax.axvline(1.0, color="k", ls="--", lw=1.2, zorder=2, label="seasonal-naive (1.0)")
for i, o in enumerate(ov): ax.text(o + 0.02, i, f"{o:.2f}", va="center", fontsize=9)
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("GM-Relative MASE  (lower = better; 1.0 = seasonal-naive)"); ax.set_xlim(0, 2.0)
ax.set_title(f"Where the Tiny backbone underperforms — overall {overall:.3f} (worse than seasonal-naive)")
ax.legend(loc="lower right", fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "plots", "domain_breakdown.png"), dpi=130)
print("per-domain:", {d: round(ours[d], 3) for d in order}, "overall", round(overall, 4))
