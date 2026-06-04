"""GIFT-Eval GM-Relative MASE vs backbone training steps (v2 shuffled data).
Computed from this experiment's own results/v2_pair_{30k,60k}_summary.txt (geometric mean
of the per-config Relative = MASE/seasonal-naive column). Lower is better; 1.0 = seasonal-naive.
Only the two checkpoints with committed eval outputs are shown."""
import math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
def overall_gm(summ):
    rel = []
    for line in open(os.path.join(RES, summ)):
        p = line.split()
        if len(p) >= 4 and "/" in p[0]:
            try: rel.append(float(p[-1]))
            except ValueError: pass
    return math.exp(sum(map(math.log, rel)) / len(rel))
steps = [30_000, 60_000]
mase = [overall_gm("v2_pair_30k_summary.txt"), overall_gm("v2_pair_60k_summary.txt")]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(steps, mase, "o-", color="#2196F3", lw=2.5, ms=12, label="Ours (v2, shuffled data)", zorder=3)
for s, m in zip(steps, mase):
    ax.annotate(f"{m:.3f}", (s, m), textcoords="offset points", xytext=(0, 13), ha="center", fontsize=11, fontweight="bold")
ax.axhline(1.0, color="gray", ls="--", lw=1.4, alpha=0.8, label="seasonal-naive (1.0)")
for name, val in {"Sundial": 0.673, "TimesFM": 0.680, "PatchTST": 0.762, "Chronos": 0.786, "Moirai": 0.809}.items():
    ax.axhline(val, color="#FF9800", ls=":", lw=1, alpha=0.45)
    ax.text(63_500, val, name, fontsize=7.5, color="#e67e22", va="center")
ax.set_xlabel("Backbone training steps"); ax.set_ylabel("GM-Relative MASE (lower = better)")
ax.set_title("Doubling backbone training does not help — if anything, slightly worse")
ax.legend(loc="center left", fontsize=9); ax.set_xlim(15_000, 73_000); ax.set_ylim(0.5, 1.45); ax.grid(True, alpha=0.3)
ax.set_xticks(steps); ax.set_xticklabels(["30k", "60k"])
plt.tight_layout(); plt.savefig(os.path.join(HERE, "..", "plots", "fig_v2_scaling_curve.png"), dpi=140); print("saved", mase)
