"""GM-MASE across the 5 original synth arms — RevIN vs fe+mu vs fe+mu+pstats.

Every bar is read directly from the shared, committed eval table
  ../../2026-04-27__aggregate/results/synth_eval.csv
keyed on the `arm` column (gm_mase + sn_gm_mase). Seasonal-naive (0.497)
is the same `sn_gm_mase` value on every row and is drawn as a reference
line. Lower GM-MASE = better. Single seed; 1024 held-out synth samples.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "2026-04-27__aggregate", "results", "synth_eval.csv")

# (csv arm key, short label) in the order we want them plotted (best -> worst).
ARMS = [
    ("RevIN-synth @ 60k", "RevIN-synth\n@ 60k"),
    ("fe+mu @ 60k", "fe+mu\n@ 60k"),
    ("fe+mu @ 30k", "fe+mu\n@ 30k"),
    ("fe+mu+pstats @ 60k", "fe+mu+pstats\n@ 60k"),
    ("fe+mu+pstats @ 30k", "fe+mu+pstats\n@ 30k"),
]

rows = {r["arm"]: r for r in csv.DictReader(open(CSV))}
mase = [float(rows[a]["gm_mase"]) for a, _ in ARMS]
sn = float(rows[ARMS[0][0]]["sn_gm_mase"])  # identical across rows
labels = [lbl for _, lbl in ARMS]
# RevIN (the best, index 0) highlighted; the rest grey.
colors = ["#2e86de"] + ["#95a5a6"] * (len(ARMS) - 1)

fig, ax = plt.subplots(figsize=(8, 4.6))
x = range(len(ARMS))
ax.bar(x, mase, color=colors, zorder=3)
for i, v in enumerate(mase):
    ax.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.axhline(sn, color="#c0392b", ls="--", lw=1.4, zorder=2,
           label=f"seasonal-naive = {sn:.3f}")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("GM-MASE  (lower = better)")
ax.set_ylim(0, max(mase) * 1.22)
ax.set_title("Synth GM-MASE — RevIN best of the 4 original arms, all far above seasonal-naive\n"
             "(span=32 normaliser; single seed; 1024 held-out synth samples)", fontsize=10)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
OUT = os.path.join(HERE, "..", "plots", "revin_eval.png")
plt.savefig(OUT, dpi=130)
print("wrote", os.path.relpath(OUT, HERE))
for (a, _), v in zip(ARMS, mase):
    print(f"  {a:28s} gm_mase={v:.5f}")
print(f"  seasonal-naive (sn_gm_mase) = {sn:.5f}")
