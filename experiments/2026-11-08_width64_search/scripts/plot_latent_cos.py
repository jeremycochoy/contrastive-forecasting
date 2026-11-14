"""One plot: how well each run forecasts its own latents, by distance.

The y value is cos(f, h): f = the forecast latent k patches ahead, h = the
encoder latent of that patch (1 - the cos_err_d<k-1> column of the losses
CSV). All runs are read over the same training steps: the last 500 steps of
the slower of the two runs of 11-14 (the L2 run and the 1/d run). Usage: plot_latent_cos.py <png>
"""
import os
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HP = Path(os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64"))
BASE = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere"
WINDOW = 500
# suffix -> (legend label, colour, marker)
RUNS = {
    "_sigf_align10pw2_rollfix": ("red: L_align x 10 (cosine) at 1, 2, 4, 8, 16, 32 patches + SIGReg on f",
                                 "#d62728", "*"),
    "_sigf_align10pw2_l2raw_rollfix": ("L2 run: red with mean (f - h)^2 of the raw vectors in place of the cosine",
                                       "#17becf", "X"),
    "_sigf_align10harm_rollfix": ("1/d run: red with the cosine at each distance d = 1 to 32, weight 1/d",
                                  "#8c564b", "h"),
}
NEW_RUNS = ("_sigf_align10pw2_l2raw_rollfix", "_sigf_align10harm_rollfix")


def read_rows(suffix):
    """The rows of a run, its losses files chained (a resume writes <name>_r2_losses.csv. The later file wins a step)."""
    name = BASE + suffix
    out = {}
    for f in sorted((HP / name / "ckpt").glob(f"{name}*_losses.csv")):
        for r in csv.DictReader(open(f)):
            out[int(float(r["step"]))] = r
    return [out[s] for s in sorted(out)]


def cos_by_distance(rows):
    """{patches ahead: (mean cos, standard error of the mean)} over the rows."""
    out = {}
    for col in rows[0]:
        if not col.startswith("cos_err_d"):
            continue
        vals = [1.0 - float(r[col]) for r in rows if r[col] not in ("", None)]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        out[int(col[len("cos_err_d"):]) + 1] = (mean, sd / math.sqrt(len(vals)))
    return dict(sorted(out.items()))


def main(png):
    hi = min(int(read_rows(suffix)[-1]["step"]) for suffix in NEW_RUNS)
    lo = hi - WINDOW + 1
    fig, ax = plt.subplots(figsize=(10, 6.2))
    top = 0.0
    for suffix, (label, colour, marker) in RUNS.items():
        rows = [r for r in read_rows(suffix) if lo <= int(r["step"]) <= hi]
        points = cos_by_distance(rows)
        xs = list(points)
        ys = [points[x][0] for x in xs]
        top = max(top, *ys)
        print(suffix, len(rows), {x: (round(m, 4), round(se, 4)) for x, (m, se) in points.items()})
        ms = 13 if marker == "*" else (5 if len(xs) > 8 else 8)  # the 1/d run has a point at each of 32 distances
        ax.plot(xs, ys, color=colour, marker=marker, ms=ms, lw=2, label=label)
    ax.axhline(0.0, color="grey", ls="--", lw=1.2)
    ax.text(1.03, 0.006, "0 = the forecast carries no information (a random direction)", color="grey", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 8, 16, 32], ["1", "2", "3", "4", "8", "16", "32"])
    ax.set_xlim(0.9, 36)
    ax.set_ylim(-0.02, round(top * 1.3, 2))  # room above the highest point for the legend
    ax.set_xlabel("How far ahead the latent is forecast (patches; 1 patch = 16 values)")
    ax.set_ylabel("cos(forecast latent, true latent)\nhigher is better: 1 = exact, 0 = no information")
    ax.set_title(f"Latent forecast quality at the same training steps ({lo / 1000:.1f}k to {hi / 1000:.1f}k)")
    ax.grid(color="#e6e6e6")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)  # the empty part: the far distances are low
    fig.tight_layout()
    fig.savefig(png, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1])
