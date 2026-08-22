#!/usr/bin/env python3
"""#407 — the card's deliverable: A4's score against backbone train step.

One axes, because the card asks one question: does the score keep falling
when A4 sees the rest of the data? A second panel would ask the reader to
hold two questions.

Four channels, four facts, none of them carrying two:

  line         the PER-STOP MEAN of the head-seed draws, one line per head.
               A single draw is its own mean, so every stop is on the same
               footing. The mean, not the protocol draw, is what the tables
               compare, so the picture and the text agree.
  colour       the HEAD. Blue is the student encoder, orange the teacher.
               Both hues clear the lightness band, the chroma floor, the
               all-pairs CVD separation and the 3:1 contrast floor against
               white, and they are the same two the parent study used for
               the same two heads.
  small dots   EVERY head-seed draw, at its own stop, in its head colour.
               `replicate_heads.sh` draws each head again under two more
               head seeds on the SAME backbone. The dots are the measured
               spread. No summary stands between the reader and the data,
               so the figure asks the reader to take nothing on trust.
  hollow ring  the parent study's published point at 40k, 100k and 200k.
               At 40k and 100k that point is the only draw, so the ring is
               the node. At 200k this card drew the same head seed again
               and added two more, so the ring marks which of the three
               dots the parent study published, and the line passes through
               their mean.
  grey rule    1.0660, the best score before this run — A4's student head
               at 200,000 steps. Read off the parent study's own score
               file, not typed here.

The three hollow points do not all come from one round of the parent study.
40k comes from `cf373_r2`, 100k and 200k come from `cf373_r3`, and the 200k
file carries an `_r2_` infix from the round that wrote it. `provenance()`
prints that line. It stays off the axes: the figure carries the measurement
and nothing else.

Direct labels on every mean, not a value axis the reader has to trace: the
whole figure is six means per head, and the differences it has to carry are
in the third decimal. For the same reason the axis does not reach
seasonal-naive parity at 1.0. A rule 0.07 below every point would take four
fifths of the axis and leave the card's question in a band too thin to read.

Usage:
  plot_full_pass.py [--results DIR] [--parent DIR] --out plots/full_pass.png
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                       # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_pass as FP                                    # noqa: E402
import head_band as HB                                    # noqa: E402

# The parent study's head palette, unchanged, so a reader who learned it
# there keeps it.
HEAD_COLOUR = {"student": "#2a78d6", "teacher": "#eb6834"}
INK, INK_SOFT, GRID, RULE = "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e8a"

# The caption, word for word. It names every mark on the axes and nothing
# else. `main` writes it beside the PNG so the report and the figure cannot
# drift apart.
CAPTION = (
    "GM-Relative MASE against backbone train step, student and teacher "
    "heads, lower is better. Lines join the per-stop means. Small dots are "
    "single head-seed draws. Hollow marks are the published points from "
    "the rollout-depth study. The dashed line is the prior best, 1.0660.")


def means(head, results, parent):
    """`{step: mean of the head-seed draws}` for one head.

    A stop with one draw keeps that draw. A stop with three keeps their
    mean. A stop with none is absent, so a head that has not scored yet
    draws a shorter line rather than a made-up point.
    """
    out = {}
    for stop in FP.PARENT_STOPS + FP.STOPS:
        got = HB.local_draws(stop, head, results, parent)
        if got:
            out[stop] = statistics.fmean(got.values())
    return out


def published(head, parent):
    """`{step: score}` for the parent study's three stops."""
    out = {}
    for stop in FP.PARENT_STOPS:
        value = FP.score(stop, head, parent)
        if value is not None:
            out[stop] = value
    return out


def provenance() -> str:
    """Which checkpoint tree each hollow point came from.

    The figure puts all three on one axis, so the study records the two
    trees. This line stays off the axes and goes to stdout and to
    `results/figure_provenance.txt`.
    """
    return ("The three hollow points come from the rollout-depth study. 40k "
            "comes from the cf373_r2 tree, 100k and 200k come from cf373_r3, "
            "and the 200k checkpoint file carries an _r2_ infix from the "
            "round that wrote it.")


# One label, in fractions of the y-axis span: how far its near edge sits
# from the mark it names, and how tall it is. `LABEL_OFFSET` matches the
# 12-point offset `draw` applies, and `LABEL_HEIGHT` an 8-point line.
LABEL_OFFSET, LABEL_HEIGHT = 0.039, 0.030


def label_box(anchor, side, span):
    """`(low, high)` of the label's own y range, in data units."""
    near = anchor + side * LABEL_OFFSET * span
    far = near + side * LABEL_HEIGHT * span
    return (min(near, far), max(near, far))


def label_side(curves, draws, best, span):
    """`{(head, step): +1 above / -1 below}`, so no label lands on a mark.

    At a step where both heads scored, the higher point takes the space
    above and the lower one the space below. The two heads land within
    0.001 of each other at 40k, which is closer than one label is tall.

    A label that covers the 1.0660 rule then flips to the other side. The
    student runs below the teacher at every stop, so its label goes under
    its lowest draw, and at 450k that is where the rule is. A number
    sitting on the rule reads as the rule's own label.
    """
    sides = {}
    for head, points in curves.items():
        for step, value in points.items():
            others = [c[step] for h, c in curves.items()
                      if h != head and step in c]
            side = -1 if any(o > value for o in others) else 1
            cloud = list(draws.get(head, {}).get(step, {}).values()) + [value]
            anchor = max(cloud) if side > 0 else min(cloud)
            low, high = label_box(anchor, side, span)
            if low <= best <= high:
                other = -side
                flip = max(cloud) if other > 0 else min(cloud)
                lo2, hi2 = label_box(flip, other, span)
                if not (lo2 <= best <= hi2) and \
                        not any(lo2 <= o <= hi2 for o in others):
                    side = other
            sides[(head, step)] = side
    return sides


def draw(ax, head, points, pub, sides, draws):
    """One head: the mean line, every draw as a dot, the parent's rings.

    `points` is `{step: mean}`, `pub` is `{step: published score}` and
    `draws` is `{step: {seed: score}}`.
    """
    colour = HEAD_COLOUR[head]
    xs = sorted(points)
    ax.plot([x / 1000 for x in xs], [points[x] for x in xs],
            lw=2.0, color=colour, zorder=2, solid_capstyle="round")

    # The dots go UNDER the nodes. A draw that lands on the mean would
    # otherwise put its white edge through the middle of the node, and a
    # filled node would then read as a hollow one.
    for step, got in draws.items():
        for value in got.values():
            ax.plot(step / 1000, value, marker="o", ms=3.6, ls="none",
                    color=colour, mfc=colour, mec="white", mew=0.6,
                    zorder=3, clip_on=False)

    # The parent's ring, with a transparent face, so the draw it rings
    # stays visible inside it.
    for step, value in pub.items():
        ax.plot(step / 1000, value, marker="o", ms=11.0, ls="none",
                mfc="none", mec=colour, mew=2.0, zorder=4, clip_on=False)

    # This card's stops carry a filled node. The parent's stops carry the
    # ring instead, so no mark means two things at one step.
    for x in xs:
        if x in pub:
            continue
        ax.plot(x / 1000, points[x], marker="o", ms=7.0, ls="none",
                color=colour, mfc=colour, mec=colour, mew=2.0,
                zorder=4, clip_on=False)

    for x in xs:
        side = sides[(head, x)]
        # Clear the dot cloud, not the mean. The label carries a white
        # stroke, and a label placed off the mean would rub out whichever
        # draw sits between the two.
        got = list(draws.get(x, {}).values()) + [points[x]]
        anchor = max(got) if side > 0 else min(got)
        ax.annotate(f"{points[x]:.4f}", (x / 1000, anchor),
                    textcoords="offset points", xytext=(0, 9 * side - 3),
                    ha="center", va="bottom" if side > 0 else "top",
                    fontsize=8, color=INK_SOFT, zorder=6,
                    path_effects=[pe.withStroke(linewidth=3.0,
                                                foreground="white")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=FP.RESULTS,
                    help="this study's results directory")
    ap.add_argument("--parent", default=FP.PARENT_RESULTS,
                    help="the rollout-depth study's results directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    curves = {h: means(h, a.results, a.parent) for h in FP.HEADS}
    pubs = {h: published(h, a.parent) for h in FP.HEADS}
    drawn = {h: {s: HB.local_draws(s, h, a.results, a.parent)
                 for s in sorted(curves[h])}
             for h in FP.HEADS}
    best = FP.best_before(a.parent)

    values = [v for c in curves.values() for v in c.values()] + [best]
    values += [v for p in pubs.values() for v in p.values()]
    values += [v for d in drawn.values() for g in d.values()
               for v in g.values()]

    # `or 0.01` covers the first figure of the study, where the only value
    # is the rule itself and the range is zero.
    span = (max(values) - min(values)) or 0.01
    ylo, yhi = min(values) - 0.16 * span, max(values) + 0.16 * span

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axhline(best, color=RULE, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.annotate(f"prior best, {best:.4f}", (0.995, best),
                xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 6),
                ha="right", fontsize=8.5, color=RULE, zorder=6,
                path_effects=[pe.withStroke(linewidth=3.0,
                                            foreground="white")])

    sides = label_side(curves, drawn, best, yhi - ylo)
    for head in FP.HEADS:
        draw(ax, head, curves[head], pubs[head], sides, drawn[head])

    ax.set_xlabel("backbone train step (thousands)", fontsize=10, color=INK)
    ax.set_ylabel("GM-Relative MASE, 97 GIFT-Eval configs (lower is better)",
                  fontsize=10, color=INK)
    ax.set_title("A4 to one full pass over small_v1",
                 fontsize=12.5, color=INK, loc="left", pad=12)
    ticks = [s // 1000 for s in FP.PARENT_STOPS + FP.STOPS]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}k" for t in ticks])
    ax.set_xlim(0, FP.STOPS[-1] / 1000 * 1.06)
    ax.set_ylim(ylo, yhi)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)

    handles = [Line2D([], [], color=HEAD_COLOUR[h], lw=2.0, marker="o",
                      ms=7, mec=HEAD_COLOUR[h], label=f"{h} head")
               for h in FP.HEADS]
    handles.append(Line2D([], [], color=INK_SOFT, lw=0, marker="o", ms=9,
                          mfc="none", mec=INK_SOFT, mew=1.8,
                          label="rollout-depth study point"))
    handles.append(Line2D([], [], color=INK_SOFT, lw=0, marker="o", ms=3.6,
                          mfc=INK_SOFT, mec="white", mew=0.6,
                          label="one head-seed draw"))
    # Top left. The curve's lowest point is at 200k and the rule sits under
    # it, so the bottom left holds the deepest part of the figure. The two
    # heads start near 1.086 and climb only after 200k, which leaves the
    # top left corner empty.
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_SOFT,
              loc="upper left", handletextpad=0.6, borderaxespad=0.6)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    res = Path(a.results)
    if res.is_dir():
        (res / "figure_caption.txt").write_text(CAPTION + "\n")
        print(f"wrote {res / 'figure_caption.txt'}")
        (res / "figure_provenance.txt").write_text(provenance() + "\n")
        print(f"wrote {res / 'figure_provenance.txt'}")
    print(f"  caption: {CAPTION}")
    print(f"  provenance: {provenance()}")
    for head in FP.HEADS:
        row = "  ".join(f"bb{s // 1000}k={v:.4f}"
                        for s, v in sorted(curves[head].items()))
        print(f"  {head:<8} mean  {row}")
        counts = "  ".join(f"bb{s // 1000}k n={len(g)}"
                           for s, g in sorted(drawn[head].items()))
        print(f"  {head:<8} draws {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
