"""Radar of the relative MASE per GIFT-Eval dataset for MPE at 200k and at its
best checkpoint, grey at its best checkpoint and yellow at its best checkpoint.

A dataset's value is the geometric mean, over its configs, of the MASE of the
control head relative to seasonal naive (radar_lib.py of
2026-09-06_moirai_small_size). MPE's per-config MASE comes from that report
(results/per_config/radar). Grey's and yellow's comes from
results/config_mase.tsv. The x positions are batch-64 steps: MPE trains at
batch 256, so its 200k is its 50k checkpoint and its best (664k) is its 166k.

Usage: python3 plot_radar_best.py
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent.parent
MOIRAI = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(MOIRAI / "scripts"))
from radar_lib import RADAR, SN, bell_order, per_dataset, read_mase, style_axes  # noqa: E402

OUT = STUDY / "plots" / "radar_best.png"
# (per-config source, legend label, colour, line style). A source is a CSV name
# in the radar folder of the Moirai report, or (run, stop in k) in config_mase.tsv.
ARMS = [
    ("moirai421ew_native50k", "MPE, 200k (its 50k at batch 256)   0.9831", "#6495ed", "--"),
    ("moirai421ew_native166k", "MPE, best: 664k (its 166k at batch 256)   0.9362", "#000080", "--"),
    (("grn_ema099995_f6", 240), "Grey, best: 240k   1.0341", "#7f7f7f", "-"),
    (("grn_ema099995_f6_rep01", 200), "Yellow, best: 200k   1.0467", "#ffb000", "-"),
]


def control_mase(run, stop_k):
    """{config: MASE} of the control head of ``run`` at ``stop_k``."""
    with open(STUDY / "results" / "config_mase.tsv") as fh:
        return {r["config"]: float(r["mase"]) for r in csv.DictReader(fh, delimiter="\t")
                if r["run"] == run and r["stop_k"] == str(stop_k)
                and r["arm"] == "control" and not r["member"]}


def arm_mase(source):
    return read_mase(RADAR / f"{source}.csv") if isinstance(source, str) else control_mase(*source)


def main():
    sn = read_mase(SN)
    per_ds = {label: per_dataset(arm_mase(source), sn) for source, label, _, _ in ARMS}
    names = bell_order(per_ds)
    ang = [n / len(names) * 2 * math.pi for n in range(len(names))] + [0.0]
    fig, ax = plt.subplots(figsize=(11.5, 10.5), subplot_kw=dict(polar=True))
    for _, label, colour, style in ARMS:
        vals = [per_ds[label].get(d, float("nan")) for d in names]
        ax.plot(ang, vals + vals[:1], style, marker="o", color=colour, lw=2.4, ms=5,
                label=label, zorder=3)
    ax.plot(ang, [1.0] * len(ang), color="#2ca02c", ls="--", lw=1.8, zorder=4)
    style_axes(ax, names, ang)
    ax.set_title("Relative MASE per GIFT-Eval dataset (lower is better)\n"
                 "Green ring: seasonal naive (1.0). The hardest datasets are at the top.",
                 fontsize=12, pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1, fontsize=10,
              title="run, checkpoint (batch-64 steps), GM-Relative MASE")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
