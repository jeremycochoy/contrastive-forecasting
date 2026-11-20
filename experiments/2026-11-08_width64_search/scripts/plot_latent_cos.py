"""One plot: how well each run forecasts its own latents, by distance.

The y value is cos(f, h): f = the forecast latent k patches ahead, h = the
encoder latent of that patch (1 - the cos_err_d<k-1> column of the losses
CSV). The script reads each run over its own last 500 steps with finite
values (the user, Nov 17: 'plot the latest of each run ... so that we can sort
of compare'). The runs can be at different training steps: the legend gives the
steps of each run. Usage: plot_latent_cos.py <png>
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
    "_align10pw2_rollfix": ("red without SIGReg-F (the variant; NaN at 13.7k)", "#e377c2", "P"),
    "_sigf_align10pw2_l2raw_rollfix": ("L2 run + SIGReg-F: red with mean (f - h)^2 of the raw vectors in place of "
                                       "the cosine", "#17becf", "X"),
    "_align10pw2_l2raw_rollfix": ("L2 run: the same without SIGReg on f", "#bcbd22", "X"),
    "_sigf_align10harm_rollfix": ("1/d run + SIGReg-F: red with the cosine at each distance d = 1 to 32, weight 1/d",
                                  "#8c564b", "h"),
    "_align10harm_rollfix": ("1/d run: the same without SIGReg on f", "#7f7f7f", "h"),
}


def read_rows(suffix):
    """The rows of a run, its losses files chained (a resume writes <name>_r2_losses.csv. The later file wins a step)."""
    name = BASE + suffix
    out = {}
    for f in sorted((HP / name / "ckpt").glob(f"{name}*_losses.csv")):
        # a line with NUL bytes: the end of a file cut by the reboot of elisa (21:56 UTC Nov 15)
        rows = list(csv.DictReader(line for line in open(f) if "\0" not in line))
        rows = [r for r in rows if None not in r.values() and r.get("step")]  # a half-written last row (a pulled copy)
        if rows:  # a resume replaces every step from its first one: the rows of the dead branch go
            first = int(float(rows[0]["step"]))
            out = {s: r for s, r in out.items() if s < first}
        for r in rows:
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


def is_finite(row):
    """All the cos_err_d columns of the row hold a finite value (a NaN step gives nan)."""
    vals = [v for c, v in row.items() if c.startswith("cos_err_d")]
    return bool(vals) and all(v not in ("", None) and math.isfinite(float(v)) for v in vals)


def last_window(rows):
    """The finite rows of the last WINDOW steps up to the last finite step of the run."""
    finite = [r for r in rows if is_finite(r)]
    hi = int(finite[-1]["step"])
    return [r for r in finite if int(r["step"]) > hi - WINDOW]


def draw_run(ax, suffix, rows):
    """The line of one run over its window, with its steps in the legend. Returns its highest point."""
    label, colour, marker = RUNS[suffix]
    points = cos_by_distance(rows)
    xs, ys = list(points), [mean for mean, _ in points.values()]
    lo, hi = int(rows[0]["step"]), int(rows[-1]["step"])
    label = f"{label}\n    steps {lo / 1000:.1f}k to {hi / 1000:.1f}k"
    print(suffix, f"{lo}-{hi}", len(rows), {x: (round(m, 4), round(se, 4)) for x, (m, se) in points.items()})
    plus = "+ SIGReg-F" in label  # larger markers: the run without SIGReg on f lies on them and both show
    ms = 13 if marker == "*" else ((7 if plus else 5) if len(xs) > 8 else (11 if plus else 7))  # 1/d: 32 points
    # red on top: its star stays whole, and the arms of an X at the same value show around it
    ax.plot(xs, ys, color=colour, marker=marker, ms=ms, lw=2, label=label, zorder=3 if marker == "*" else 2)
    return max(ys)


def main(png):
    fig, ax = plt.subplots(figsize=(11, 7.4))
    top = 0.0
    for suffix in RUNS:
        top = max(top, draw_run(ax, suffix, last_window(read_rows(suffix))))
    ax.axhline(0.0, color="grey", ls="--", lw=1.2)
    ax.text(1.03, 0.006, "0 = the forecast carries no information (a random direction)", color="grey", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 8, 16, 32], ["1", "2", "3", "4", "8", "16", "32"])
    ax.set_xlim(0.9, 36)
    ax.set_ylim(-0.02, round(top * 1.75, 2))  # room above the highest point for the legend (2 lines a run)
    ax.set_xlabel("How far ahead the latent is forecast (patches; 1 patch = 16 values)")
    ax.set_ylabel("cos(forecast latent, true latent)\nhigher is better: 1 = exact, 0 = no information")
    ax.set_title("Latent forecast quality: the last 500 steps of each run (the runs are at different steps)")
    ax.grid(color="#e6e6e6")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)  # the empty part: the far distances are low
    fig.tight_layout()
    fig.savefig(png, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1])
