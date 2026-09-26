from pathlib import Path
STUDY = Path(__file__).resolve().parent.parent
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv, math
from collections import defaultdict

R = f"{STUDY}/results/per_config/radar"
SN = f"{STUDY}/results/per_config/seasonal_naive.csv"
COL = "eval_metrics/MASE[0.5]"

def read(path):
    out = {}
    for row in csv.DictReader(open(path)):
        try: out[row["dataset"]] = float(row[COL])
        except (ValueError, TypeError, KeyError): pass
    return out

sn = read(SN)
# The flat run shows its best stop and its last one, the Moirai-schedule run
# its latest; the curves figure holds the rest.
ARMS = [("cyan665",   "contrastive, cosine to 1e-6 by 200k, 665k   1.1369", "#17becf"),
        ("value25k",  "value space, flat 1e-3, 25k at batch 256   1.2951", "#7f7f7f"),
        ("value90k",  "value space, flat 1e-3, 90k at batch 256   1.4482", "#d62728"),
        ("moirai25k", "value space, Moirai schedule, 25k at batch 256   1.5283", "#8c564b")]

per_ds = {}
for name, _, _ in ARMS:
    m = read(f"{R}/{name}.csv")
    g = defaultdict(list)
    for cfg, v in m.items():
        if cfg in sn and sn[cfg] > 0 and v > 0:
            g[cfg.split("/")[0]].append(math.log(v / sn[cfg]))
    per_ds[name] = {d: math.exp(sum(x) / len(x)) for d, x in g.items()}

labels = sorted(set().union(*[set(v) for v in per_ds.values()]))
N = len(labels)
ang = [n / N * 2 * math.pi for n in range(N)] + [0.0]

fig, ax = plt.subplots(figsize=(11.5, 10.5), subplot_kw=dict(polar=True))
for name, lab, col in ARMS:
    vals = [per_ds[name].get(d, float("nan")) for d in labels]
    vals += vals[:1]
    ax.plot(ang, vals, "-o", color=col, lw=2.4, ms=5, label=lab, zorder=3)

ax.plot(ang, [1.0] * (N + 1), color="#2ca02c", ls="--", lw=1.8, zorder=4)
ax.set_xticks(ang[:-1])
ax.set_xticklabels(labels, fontsize=9.5)
ax.tick_params(axis="x", pad=12)
ax.set_rscale("log")
ticks = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
ax.minorticks_off()
ax.set_yticks(ticks)
ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=8, color="dimgrey")
ax.set_ylim(0.55, 9.0)
ax.set_rlabel_position(97)
ax.set_title("Issues #414 and #415: the value-space reference against the best contrastive model\n"
             "Relative MASE per GIFT-Eval dataset, geometric mean over its configs.\n"
             "The green ring is 1.0, the seasonal-naive level. Inside it is better.",
             fontsize=12, pad=28)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=10,
          title="arm, stop, and its GM-Relative MASE")
fig.tight_layout()
fig.savefig(f"{STUDY}/plots/gm_mase_radar_value.png", dpi=130)

print("radar written")

