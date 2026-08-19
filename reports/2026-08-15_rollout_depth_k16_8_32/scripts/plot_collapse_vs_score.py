#!/usr/bin/env python3
"""#401 — does the collapse probe explain the scores?

Two panels, from results/diag/collapse_vs_score.csv:

  left    GM-Relative MASE against the effective rank of the encoder latent.
          Every SUMMED cell sits at rank near 1 and scores 2.0 to 12.5. Every
          MEAN cell sits at rank 4 to 8, beside the k = 0 parent at 7.2, and
          scores 1.16 to 1.29. So the reduction, not the depth, decides
          whether the encoder keeps more than one direction, and rank near 1
          costs the score an order of magnitude.

          The panel also carries the answer's limit: the mean arm has healthy
          rank and still does not reach the k = 0 anchor. Rank explains the
          summed arm's numbers. It does not explain the mean arm's.

  right   GM-Relative MASE against `readout_r`, the correlation between the
          input series and the ONE direction the encoder keeps most of its
          variance in.

          A cell is on this panel only when that direction carries at least
          half of the variance (`top_dir_share` >= 0.5). Below that the
          quantity summarises nothing: the head reads the other directions
          too, and a single number about the first one is not a bound on
          what it gets. That rule keeps the rank-7 parent and every mean cell
          off the panel, on the measurement rather than on the arm's name.

Marker by arm, so no point is identified by colour alone: a star is the
k = 0 parent, a circle the summed arm, a square the mean arm.

Usage:  python3 plot_collapse_vs_score.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RES = STUDY / "results"
PLOTS = STUDY / "plots"

# The k = 0 anchor, measured by this study's own path: control c2.
K0_SCORE = float(
    (RES / "diag/score_c2_k0anchor_a4parent_bb40k_h30k_student.txt")
    .read_text().strip())

COLOUR = {0: "#2f6f4e", 8: "#c9772a", 16: "#a63a3a", 32: "#3a5ba6"}
MARKER = {"sum": "o", "mean": "s"}
ARM_NAME = {"sum": "summed", "mean": "mean"}

# The share of the latent variance the top direction must carry before
# `readout_r` says anything about what the head can read through it.
TOP_SHARE_FLOOR = 0.5


def load():
    with (RES / "diag/collapse_vs_score.csv").open() as f:
        return list(csv.DictReader(f))


def spearman(xs, ys):
    """Rank correlation, with the mean rank for ties. Same as the table's."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


# Where a label may sit, relative to its point, in display points. Tried in
# this order, and the first free slot wins — see `place_labels`.
LABEL_SLOTS = [(7, 4), (-7, 4), (7, -11), (-7, -11), (7, 13), (-7, 13),
               (7, -20), (-7, -20), (7, 21), (-7, 21), (0, 15), (0, -20)]
CHAR_W, LINE_H = 4.3, 9.5          # a label's size at fontsize 7.5, in points


def place_labels(ax, points):
    """One label per point, at the first slot that hits no placed label.

    A fixed cycle of offsets was enough for 8 points in one cluster. It is
    not enough for 14 in two: two points close in BOTH axes can draw the same
    slot and print one label over the other, and a point near an edge can
    send its label out of the panel. So each label takes the first slot that
    is free AND inside the panel, and a label with no free slot is dropped
    rather than printed on top of another.

    `points` are (x, y, text, colour) in data coordinates.
    """
    fig = ax.figure
    fig.canvas.draw()                       # fixes the transform and the DPI
    scale = fig.dpi / 72.0                  # display pixels per point
    x0, x1 = sorted(ax.transAxes.transform([(0, 0), (1, 1)])[:, 0])
    y0, y1 = sorted(ax.transAxes.transform([(0, 0), (1, 1)])[:, 1])
    placed = []
    for x, y, text, colour in sorted(points, key=lambda p: (p[0], p[1])):
        px, py = ax.transData.transform((x, y))
        w, h = len(text) * CHAR_W * scale, LINE_H * scale
        for dx, dy in LABEL_SLOTS:
            cx = px + dx * scale + (w / 2 if dx > 0 else
                                    -w / 2 if dx < 0 else 0)
            cy = py + dy * scale
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if box[0] < x0 or box[2] > x1 or box[1] < y0 or box[3] > y1:
                continue
            if any(box[0] < b[2] and b[0] < box[2]
                   and box[1] < b[3] and b[1] < box[3] for b in placed):
                continue
            placed.append(box)
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7.5, color=colour,
                        ha="left" if dx > 0 else
                           "right" if dx < 0 else "center")
            break


def draw(ax, rows, xk, xlabel):
    for r in rows:
        k, arm = int(r["k"]), r["reduce"]
        ax.scatter([float(r[xk])], [float(r["score"])], s=58,
                   color=COLOUR[k], marker=MARKER[arm], zorder=3)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("GM-Relative MASE (lower is better)", fontsize=9)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.tick_params(labelsize=8)


def main():
    rows = load()
    scored = [r for r in rows if r["score"] and int(r["k"]) > 0]
    # the parent at bb40k is the checkpoint control c2 scored
    anchor = [r for r in rows if int(r["k"]) == 0 and int(r["step_k"]) == 40
              and "393" in r["label"]][0]
    readable = [r for r in scored
                if float(r["top_dir_share"]) >= TOP_SHARE_FLOOR]

    PLOTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for a in ax:
        a.axhline(K0_SCORE, color="#888", lw=1, ls=":", zorder=1)
        a.text(0.02, K0_SCORE, f"  k = 0 anchor {K0_SCORE:.4f}",
               transform=a.get_yaxis_transform(), va="bottom",
               fontsize=8, color="#666")

    ax[0].scatter([float(anchor["eff_rank"])], [K0_SCORE], s=130, marker="*",
                  color=COLOUR[0], zorder=4)
    draw(ax[0], scored, "eff_rank",
         "effective rank of encoder latent, across series")
    draw(ax[1], readable, "readout_r",
         "readout r: |corr| of input with the surviving direction")

    # Room for the labels on both axes, set BEFORE they are placed: the
    # placer works in display coordinates, so a limit changed after it ran
    # would move every label off its point.
    for a in ax:
        lo, hi = a.get_xlim()
        a.set_xlim(lo - 0.09 * (hi - lo), hi + 0.12 * (hi - lo))
        lo, hi = a.get_ylim()
        a.set_ylim(lo * 0.78, hi * 1.28)

    def place():
        for a, rs, xk in ((ax[0], scored, "eff_rank"),
                          (ax[1], readable, "readout_r")):
            place_labels(a, [(float(r[xk]), float(r["score"]),
                              f"k={r['k']} {r['step_k']}k",
                              COLOUR[int(r["k"])]) for r in rs])

    rho = spearman([float(r["readout_r"]) for r in readable],
                   [float(r["score"]) for r in readable])
    ax[0].set_title("Rank separates the two reductions. It does not put the "
                    "mean arm under the k = 0 anchor.", fontsize=9.5)
    ax[1].set_title(f"Inside the collapsed set, the score follows the "
                    f"readout  (Spearman {rho:+.2f}, n = {len(readable)})",
                    fontsize=9.5)

    handles = [plt.Line2D([], [], marker="*", ls="", color=COLOUR[0],
                          markersize=12, label="k = 0 parent")]
    handles += [plt.Line2D([], [], marker=MARKER[arm], ls="", color="#555",
                           label=f"{ARM_NAME[arm]} arm")
                for arm in ("sum", "mean")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=COLOUR[k],
                           label=f"k = {k}") for k in (8, 16, 32)]
    ax[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=2)

    fig.suptitle("#401 — the summed arm collapsed to one direction. The mean "
                 "arm did not, and still does not beat k = 0.", fontsize=11)
    # The layout settles BEFORE the labels are placed. `place_labels` measures
    # in display coordinates, and `tight_layout` moves the axes box, so a
    # placement made first would be checked against a panel of another size.
    fig.tight_layout()
    place()
    out = PLOTS / "collapse_vs_score.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}  ({len(scored)} scored cell(s), "
          f"{len(readable)} on the readout panel)")


if __name__ == "__main__":
    main()
