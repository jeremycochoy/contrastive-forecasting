"""GM-MASE: does adding cross-time negatives (cosine_similarity_batch)
to the span=512 best arm help on synth? It does not — it regresses.

Bars read directly from the shared, committed eval table
  ../../2026-04-27__aggregate/results/synth_eval.csv
keyed on the `arm` column (gm_mase + sn_gm_mase):
  - "fe+mu @ 30k span=512"                          -> no_time_neg baseline
  - "fe+mu @ 30k span=512 +cosine_similarity_batch" -> this experiment
Seasonal-naive (0.497) is the `sn_gm_mase` reference. Single seed;
1024 held-out synth samples. Lower = better.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "2026-04-27__aggregate", "results", "synth_eval.csv")

BASE = "fe+mu @ 30k span=512"
CSB = "fe+mu @ 30k span=512 +cosine_similarity_batch"

rows = {r["arm"]: r for r in csv.DictReader(open(CSV))}
base = float(rows[BASE]["gm_mase"])
csb = float(rows[CSB]["gm_mase"])
sn = float(rows[BASE]["sn_gm_mase"])

labels = ["no_time_neg\nbaseline", "+cosine_similarity_batch\n(this)"]
vals = [base, csb]
colors = ["#27ae60", "#c0392b"]  # baseline good (green), the regression (red)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = range(len(vals))
ax.bar(x, vals, color=colors, zorder=3, width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.axhline(sn, color="#2c3e50", ls="--", lw=1.4, zorder=2,
           label=f"seasonal-naive = {sn:.3f}")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("GM-MASE  (lower = better)")
ax.set_ylim(0, max(vals) * 1.18)
pct = (csb - base) / base * 100
ax.set_title(f"Cross-time negatives hurt the span=512 arm on synth: +{pct:.1f}% GM-MASE\n"
             "(single-axis loss change; single seed; 1024 held-out synth samples)", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
OUT = os.path.join(HERE, "..", "plots", "csb_eval.png")
plt.savefig(OUT, dpi=130)
print("wrote", os.path.relpath(OUT, HERE))
print(f"  {BASE:48s} gm_mase={base:.5f}")
print(f"  {CSB:48s} gm_mase={csb:.5f}")
print(f"  seasonal-naive (sn_gm_mase) = {sn:.5f}")
print(f"  delta = +{pct:.2f}% GM-MASE")
