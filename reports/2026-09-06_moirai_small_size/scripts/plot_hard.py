from pathlib import Path
STUDY = Path(__file__).resolve().parent.parent
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, math, glob, os
from collections import defaultdict

T = f"{STUDY}/results/per_config/traj"
SN = f"{STUDY}/results/per_config/seasonal_naive.csv"
C = "eval_metrics/MASE[0.5]"

def read(p):
    return {r["dataset"]: float(r[C]) for r in csv.DictReader(open(p))
            if r.get(C) not in (None, "")}

sn = read(SN)

def per_ds(p):
    m = read(p); g = defaultdict(list)
    for k, v in m.items():
        if k in sn and sn[k] > 0 and v > 0:
            g[k.split("/")[0]].append(math.log(v / sn[k]))
    return {d: math.exp(sum(x) / len(x)) for d, x in g.items()}

ARMS = [("blue", "5.6e-5", "#1f77b4"), ("yellow", "5.6e-6", "#ff9f40")]
curves = {}
for arm, _, _ in ARMS:
    stops = sorted(int(os.path.basename(f).split("_")[1].split(".")[0])
                   for f in glob.glob(f"{T}/{arm}_*.csv"))
    curves[arm] = (stops, [per_ds(f"{T}/{arm}_{s}.csv") for s in stops])

PANELS = ["m4_hourly", "bizitobs_service", "bizitobs_application",
          "solar", "electricity", "m4_yearly"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, ds in zip(axes.ravel(), PANELS):
    for arm, lab, col in ARMS:
        stops, rows = curves[arm]
        x = [s * 1000 for s in stops]
        y = [r.get(ds, float("nan")) for r in rows]
        ax.plot(x, y, "-o", color=col, lw=2.6, ms=5, label=lab)
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.2)
    ax.set_title(ds, fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks([40000, 100000, 300000, 600000])
    ax.set_xticklabels(["40k", "100k", "300k", "600k"], fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
axes[0][0].legend(fontsize=9, title="backbone rate")
fig.suptitle("Issue #414: the five hardest GIFT-Eval datasets, and m4_yearly\n"
             "Relative MASE against seasonal naive. The green line is 1.0.\n"
             "At 5.6e-6 every hard dataset falls with more data. At 5.6e-5 they stall.",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(f"{STUDY}/plots/gm_mase_hard_datasets.png", dpi=130)
print("done")
