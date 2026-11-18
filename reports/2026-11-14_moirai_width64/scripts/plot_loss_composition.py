"""The loss parts of MPE at width 64 (#462): the loss term of each rollout
depth (val_err_d0 to val_err_d3, the pinball loss of the forecast 1 to 4
patches ahead) as stacked areas, and the total loss as a line. The total is
the sum of the parts.

Each value is the mean over log-spaced bins of steps (BINS bins from
LOSS_FIRST_STEP, default 1,000, to 166,000). A partial last bin centres on
its logged steps. The script reads the losses CSVs of the run in $R/ckpt:
the first start, then each resume (r2, r3, ...). It writes
results/loss_parts_binned.tsv and plots/loss_composition.png.

Usage: python3 plot_loss_composition.py [run folder] [out.png] [table.tsv]
  The run folder is CF462_ROOT, default /home/jupyter/cf_runs/cf462.
"""
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY.parent.parent))
from src.loss_parts import depth_columns, log_binned_means, read_loss_legs  # noqa: E402

ROOT = Path(os.environ.get("CF462_ROOT", "/home/jupyter/cf_runs/cf462"))
OUT = STUDY / "plots/loss_composition.png"
TABLE = STUDY / "results/loss_parts_binned.tsv"
NAME = "cf462_mpe_w64"
FIRST, TOTAL, BINS = int(os.environ.get("LOSS_FIRST_STEP", 1000)), 166_000, 60
TICKS_K = [1, 10, 25, 50, 100, 166]
# Depth 0 (the next patch) at the bottom, darkest: one hue, the hue of the run
DEPTH_HUES = ["#7a0146", "#b0186f", "#d1478f", "#e88fbf"]


def loss_files(folder):
    """The losses CSVs of the run in leg order: the first start, r2, r3, ..."""
    def leg(path):
        m = re.search(r"_r(\d+)_losses\.csv$", path.name)
        return int(m.group(1)) if m else 1
    return sorted(Path(folder).glob(f"{NAME}*_losses.csv"), key=leg)


def binned_parts(files):
    """(centres, {column: means}) of the depth parts and the total."""
    with open(files[0]) as f:
        parts = depth_columns(next(csv.reader(f)))
    steps, values = read_loss_legs(files, parts + ["loss"])
    centres, means = log_binned_means(steps, values, FIRST, TOTAL, BINS)
    return centres, means, parts


def write_table(path, centres, means):
    with open(path, "w") as f:
        f.write("bin_centre_step\t" + "\t".join(means) + "\n")
        for i, centre in enumerate(centres):
            f.write(f"{centre:.0f}\t" + "\t".join(f"{means[c][i]:.6f}" for c in means) + "\n")


def plot(out, centres, means, parts):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.stackplot(centres, *[means[p] for p in parts], colors=DEPTH_HUES[:len(parts)],
                 labels=[f"depth {p[-1]}" for p in parts], edgecolor="white", linewidth=0.8)
    ax.plot(centres, means["loss"], color="black", lw=1.6, label="total")
    ax.set_xscale("log")
    ax.set_xlim(FIRST * 0.9, TOTAL * 1.1)
    ticks = [k for k in TICKS_K if FIRST <= k * 1000 <= TOTAL]
    ax.set_xticks([k * 1000 for k in ticks], [f"{k}k" for k in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05 * np.nanmax(means["loss"]))
    ax.grid(color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Training step (batch 256, log scale)")
    ax.set_ylabel("Loss, mean over log-spaced bins of steps")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def main(folder=ROOT, out=OUT, table=TABLE):
    files = loss_files(Path(folder) / "ckpt")
    if not files:
        raise SystemExit(f"no {NAME}*_losses.csv in {folder}/ckpt")
    centres, means, parts = binned_parts(files)
    write_table(table, centres, means)
    plot(out, centres, means, parts)
    print("wrote", out, "and", table, "| legs:", [f.name for f in files])


if __name__ == "__main__":
    main(*sys.argv[1:])
