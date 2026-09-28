"""Radar of relative MASE per GIFT-Eval dataset, shared by the radar figures.

Each arm is a per-config CSV of eval_metrics/MASE[0.5] in results/per_config/radar.
A dataset's value is the geometric mean, over its configs, of the MASE relative
to seasonal naive.
"""
from pathlib import Path
import csv, math
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent.parent
RADAR = STUDY / "results/per_config/radar"
SN = STUDY / "results/per_config/seasonal_naive.csv"
COL = "eval_metrics/MASE[0.5]"
TICKS = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


def read_mase(path):
    """{config: MASE} from one per-config CSV. Rows without a number are skipped."""
    out = {}
    for row in csv.DictReader(open(path)):
        try: out[row["dataset"]] = float(row[COL])
        except (ValueError, TypeError, KeyError): pass
    return out


def per_dataset(mase, sn):
    """{dataset: geometric mean over its configs of MASE / seasonal-naive MASE}."""
    logs = defaultdict(list)
    for cfg, v in mase.items():
        if cfg in sn and sn[cfg] > 0 and v > 0:
            logs[cfg.split("/")[0]].append(math.log(v / sn[cfg]))
    return {d: math.exp(sum(x) / len(x)) for d, x in logs.items()}


def bell_order(per_ds):
    """Datasets by difficulty (the mean log value over the arms). The hardest
    sits at the top, the next ones alternate right and left of it, and the
    easiest meet at the bottom."""
    names = set().union(*per_ds.values())
    mean_log = {d: sum(math.log(v[d]) for v in per_ds.values() if d in v)
                   / sum(d in v for v in per_ds.values()) for d in names}
    ranked = sorted(names, key=mean_log.get, reverse=True)
    return ranked[:1] + ranked[1::2] + ranked[2::2][::-1]


def style_axes(ax, labels, ang):
    """Dataset names round the rim, clockwise from the top, and a log radius
    from 0.55 to 9."""
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="x", pad=12)
    ax.set_rscale("log")
    ax.minorticks_off()
    ax.set_yticks(TICKS)
    ax.set_yticklabels([f"{t:g}" for t in TICKS], fontsize=8, color="dimgrey")
    ax.set_ylim(0.55, 9.0)
    ax.set_rlabel_position(97)


def load(arms):
    """{arm: {dataset: value}} for arms given as (CSV name in RADAR, ...)."""
    sn = read_mase(SN)
    return {arm[0]: per_dataset(read_mase(RADAR / f"{arm[0]}.csv"), sn) for arm in arms}


def draw(arms, title, out_png):
    """arms: [(CSV name in RADAR, legend label, colour)]. Returns {arm: {dataset: value}}."""
    per_ds = load(arms)
    labels = bell_order(per_ds)
    ang = [n / len(labels) * 2 * math.pi for n in range(len(labels))] + [0.0]
    fig, ax = plt.subplots(figsize=(11.5, 10.5), subplot_kw=dict(polar=True))
    for name, lab, col in arms:
        vals = [per_ds[name].get(d, float("nan")) for d in labels]
        ax.plot(ang, vals + vals[:1], "-o", color=col, lw=2.4, ms=5, label=lab, zorder=3)
    ax.plot(ang, [1.0] * len(ang), color="#2ca02c", ls="--", lw=1.8, zorder=4)
    style_axes(ax, labels, ang)
    ax.set_title(title, fontsize=12, pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=10,
              title="run, checkpoint, GM-Relative MASE")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    return per_ds
