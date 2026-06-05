"""GIFT-Eval GM-Relative MASE progression across the three runs on this HP/data axis:
  #6  default-HP, 30k steps    -> 1.804  (cited cross-experiment value)
  #9  MOIRAI-HP,  30k steps    -> 1.639  (cited cross-experiment value)
  this run: full epoch (167k) + quantile head -> recomputed from committed summary.txt

The #6 / #9 numbers are NOT produced by committed eval data in this experiment dir;
they are the head-to-head values recorded in notes/REPORT_PLAN.md (#9 1.6391 won 30k
vs #6 1.8043), kept as-reported and labelled as cited. Only the final bar is
recomputed from results/gift_eval_resume50k_local/summary.txt."""
import math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "..", "results", "gift_eval_resume50k_local", "summary.txt")

rel = []
for line in open(SUMMARY):
    p = line.split()
    if len(p) >= 4 and "/" in p[0]:
        try:
            rel.append(float(p[-1]))
        except ValueError:
            pass
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
ours = gm(rel)
assert abs(ours - 1.1828) < 5e-4, f"recomputed ours={ours:.4f} != committed 1.1828"

labels = ["#6\ndefault HP\n30k",
          "#9\nMOIRAI HP\n30k",
          "this run\nMOIRAI HP\n167k + qhead"]
vals = [1.804, 1.639, ours]
cited = [True, True, False]
colors = ["#bdc3c7", "#bdc3c7", "#c0392b"]

fig, ax = plt.subplots(figsize=(6.6, 4.4))
x = range(len(vals))
ax.bar(x, vals, color=colors, zorder=3)
ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=2, label="seasonal-naive (1.0)")
for i, v in enumerate(vals):
    tag = f"{v:.3f}" + ("*" if cited[i] else "")
    ax.text(i, v + 0.02, tag, ha="center", va="bottom", fontsize=10,
            fontweight="bold" if not cited[i] else "normal")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("GM-Relative MASE  (lower = better)")
ax.set_ylim(0, 2.0)
ax.set_title("Progression on the MOIRAI-HP / real-data axis\n"
             "step budget + quantile head cut GM-Rel MASE 1.804 -> 1.183 (-34%)")
ax.legend(loc="upper right", fontsize=8)
ax.text(0.01, -0.30, "* cited cross-experiment value (notes/REPORT_PLAN.md); final bar recomputed from committed summary.txt",
        transform=ax.transAxes, fontsize=7, color="#555")
plt.tight_layout()
out = os.path.join(HERE, "..", "plots", "progression.png")
plt.savefig(out, dpi=130)
print(f"ours={ours:.4f}  bars={[round(v,4) for v in vals]}")
print("wrote", os.path.abspath(out))
