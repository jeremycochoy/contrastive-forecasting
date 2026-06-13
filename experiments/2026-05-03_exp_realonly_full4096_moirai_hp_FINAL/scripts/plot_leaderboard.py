"""GIFT-Eval leaderboard bar chart: our full-4096 MOIRAI-HP backbone vs the
published GIFT-Eval reference models.

`ours` is recomputed here (not hard-coded): geometric mean over the 97 configs
of the Relative column in results/gift_eval_resume50k_local/summary.txt. It is
asserted to equal the committed aggregate (1.1828) so the figure can never drift
from the data. The reference models (Sundial / TimesFM / PatchTST / Chronos /
Moirai / seasonal-naive) are the published GIFT-Eval leaderboard numbers, read
from the `Leaderboard comparison:` block in the same committed summary.txt."""
import math, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "..", "results", "gift_eval_resume50k_local", "summary.txt")

# --- ours: geomean of the per-config Relative column ---
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
print(f"ours = {ours:.4f}  (n={len(rel)} configs)")

# --- reference models: parse the committed leaderboard block ---
refs = {}
for line in open(SUMMARY):
    m = re.match(r"\s*([A-Za-z]+):\s*([0-9.]+)\s*$", line)
    if m and m.group(1) not in ("Config",):
        refs[m.group(1)] = float(m.group(2))
# Drop the self row; keep published references only.
refs.pop("Ours", None)
print("refs:", refs)

models = ["Sundial", "TimesFM", "PatchTST", "Chronos", "Moirai", "Naive", "Ours"]
vals = [refs["Sundial"], refs["TimesFM"], refs["PatchTST"], refs["Chronos"],
        refs["Moirai"], refs["Naive"], ours]
labels = ["Sundial", "TimesFM", "PatchTST", "Chronos", "Moirai",
          "seasonal-naive", "ours\n(full-4096, 167k)"]
colors = ["#7f8c8d"] * 5 + ["#000000", "#c0392b"]  # refs grey, naive black, ours red

fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = range(len(models))
ax.bar(x, vals, color=colors, zorder=3)
ax.axhline(refs["Naive"], color="k", ls="--", lw=1.0, zorder=2)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold" if models[i] == "Ours" else "normal")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("GM-Relative MASE  (lower = better)")
ax.set_ylim(0, 1.32)
ax.set_title("GIFT-Eval: our backbone (1.183) vs published leaderboard — best at the time,\n"
             "but still above the seasonal-naive 1.0 line that every reference model clears")
plt.tight_layout()
out = os.path.join(HERE, "..", "plots", "leaderboard.png")
plt.savefig(out, dpi=130)
print("wrote", os.path.abspath(out))
