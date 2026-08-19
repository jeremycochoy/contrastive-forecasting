#!/usr/bin/env python3
"""#401 deliverable 2 — GM-Relative MASE against backbone train step.

#373's `ladder.png`, on this study's entity. #373 drew 14 cells at one depth.
This study draws one cell at two depths, and it draws the two phases side
by side, because the card's second question is exactly the difference between
the two panels.

  left    phase 1, the fixed 30,000-step head on every stop.
  right   phase 2, the head budget matched to the backbone stop.

Horizontal axis: the backbone train step, 40k / 100k / 200k. Vertical axis:
GM-Relative MASE over the 97 GIFT-Eval configs, lower better. One line per
depth, direct-labelled at its end, so no line is identified by colour alone.

Three references, on the SAME cell, so the depth is the only thing that
changes across them:

  dashed  #373's k = 3, read out of its own score files.
  dotted  the k = 0 published, from `published.PUBLISHED`.
  diamond the k = 0 anchor, the same parent checkpoints re-scored on THIS
          study's path at THIS study's phase-1 head budget. It stands at
          every stop that has one: control c2 wrote bb40k, and the k = 0
          parent row of `scores.csv` writes bb100k. The report reads its
          differences off this number and not off the published one, so the
          panel has to hold it.

  I-bar   the head-seed repeats. A cell scored under more than one head seed
          draws the range of its draws at its stop, so the panel shows the
          spread of the head seed beside the differences it has to bound.
  grey    #373's best on this cell, 1.0660 at bb200k. It is the number this
          study has to beat, and it is drawn on both panels.

The dashed, the dotted and the grey mark are all the parent study's numbers,
so they carry ITS head budget: 15,000 steps at bb40k and 30,000 at bb100k and
bb200k. The right panel's own cells train 40,000 and 100,000 head steps, so
that panel is not head-matched. `REF_HEAD_NOTE` puts this in the legend.

A shaded band of +/-0.0384 rides on the k = 3 reference. That is the pooled
head-seed band of `ema_sched_ladder.md`, which `noise_band.py` pools, and it
is the rule the report reads
every difference against: a gap inside the band is not a measured difference.
It rides on k = 3 because k = 3 is what every depth of this study is read
against.

A cell that runs the card's depth and stop on ANOTHER training schedule draws
an open marker at its stop, named on the point. It takes no place on a line:
a line is one depth over the stops, and a variant sits at a stop the base cell
already holds, so it would replace that point rather than add one.

Reads `results/scores.csv`, written by `collect.sh`.

Usage: plot_depth_ladder.py [--scores results/scores.csv] \\
           --out plots/depth_ladder.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402
from matplotlib.patches import Patch                      # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PARENT = STUDY.parent / "2026-08-08_rollout_depth"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT / "scripts"))
import depth_colours as D                                 # noqa: E402
import published                                          # noqa: E402

plt.rcParams.update(D.rc())

# The cell this study runs, as #373 names it, and the head it trains.
CELL = "A4"
HEAD = "student"
STOPS_K = [40, 100, 200]
PHASE_TITLE = {1: "phase 1 — head at 30k steps on every stop",
               2: "phase 2 — head budget = backbone steps"}

# The pooled head-seed band of `ema_sched_ladder.md`, from `noise_band.py`.
# It bounds the head seed
# alone. Every difference this report reads is read against it.
HEAD_SEED_BAND = 0.0384

# The k = 0 anchor: control c2's score file. The tag names the stop and the
# head budget, so the panel it belongs on is read off the tag and not held in
# a second place.
K0_ANCHOR_FILES = {
    40: STUDY / ("results/diag/"
                 "score_c2_k0anchor_a4parent_bb40k_h30k_student.txt"),
}
K0_ANCHOR_PHASE = 1

# The head this study trains. A repeat carries the same head under another
# seed, and its encoder field is `student_s<seed>`.
SEED_PREFIX = HEAD + "_s"

# How far LEFT of its stop the anchor marker sits. The axis is categorical, so
# this is a dodge and not a claim about the step count. It is needed: the
# published k = 0 at bb40k is 1.1603 against the anchor's 1.1600, which is
# 0.1% of the panel height, so the two would print one over the other. The
# variant marker dodges right, so the anchor dodges left and the three stay
# apart.
ANCHOR_DODGE = 0.055

# The head budget the k = 3, the k = 0 published and the grey reference all
# carry. `rollout_depth.md` line 858: "Every bb40k head trains 15,000 steps
# and every bb100k and bb200k head trains 30,000."
REF_HEAD_NOTE = ("reference head budget: 15,000 steps at bb40k, "
                 "30,000 at bb100k and bb200k")


def read_scores(path):
    """The lines, the variant cells, the k = 0 anchors and the seed repeats.

    The stop is kept in thousands, the unit the axis and #373's own file
    names use. A trial stop below 1000 keeps its own step count, so a trial's
    table draws too.

    Four returns, because the panel draws four different marks:

      `out`       `{phase: {k: {stop_k: score}}}`, the card's own schedule
                  under the card's own head seed. Only these make a line.
      `variants`  `(phase, k, stop_k, variant, score)`. A variant cell sits
                  at a stop the base cell already holds, so it would replace
                  that point rather than add one.
      `anchors`   `{stop_k: score}`, the k = 0 parent on this study's path.
                  It is a reference, not a depth this study trained.
      `repeats`   `{(phase, k, stop_k): [score, ...]}`, every draw of a
                  base cell that ran under more than one head seed. A
                  variant is another schedule, not another draw of the same
                  cell, so it never joins one.
    """
    out, variants, anchors, repeats = {}, [], {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            enc = r["encoder"]
            if enc != HEAD and not enc.startswith(SEED_PREFIX):
                continue
            stop = int(r["stop"])
            stop_k = stop // 1000 if stop % 1000 == 0 else stop
            variant = r.get("variant") or "base"
            phase, k, score = int(r["phase"]), int(r["k"]), float(r["score"])
            if k == 0:
                anchors[stop_k] = score
                continue
            if variant == "base":
                repeats.setdefault((phase, k, stop_k), []).append(score)
            if enc != HEAD:
                continue
            if variant == "base":
                out.setdefault(phase, {}).setdefault(k, {})[stop_k] = score
            else:
                variants.append((phase, k, stop_k, variant, score))
    repeats = {key: sorted(v) for key, v in repeats.items() if len(v) > 1}
    return out, variants, anchors, repeats


def axis_stops(scores):
    """The stops the table holds, in order. The card's three when it has them."""
    seen = sorted({s for arms in scores.values()
                   for pts in arms.values() for s in pts})
    return [s for s in STOPS_K if s in seen] or seen


def parent_k3():
    """#373's k = 3 on this cell, per stop, out of its own score files."""
    out = {}
    for s in STOPS_K:
        f = PARENT / "results" / f"score_{CELL}_k3_bb{s}k_{HEAD}.txt"
        if f.is_file() and f.read_text().strip():
            out[s] = float(f.read_text().strip())
    return out


def k0_anchors(from_table):
    """`{stop_k: score}` — the k = 0 anchor at every stop that has one.

    Two sources. The controls wrote their own score files, and the k = 0
    parent rows of `scores.csv` carry the rest.
    """
    out = dict(from_table)
    for stop_k, f in K0_ANCHOR_FILES.items():
        if f.is_file() and f.read_text().strip():
            out.setdefault(stop_k, float(f.read_text().strip()))
    return out


def draw_anchor(ax, xs, anchors):
    """The k = 0 anchor, one marker at each stop that has one.

    Markers and not a line: the anchor stands at the stops the controls
    scored, and a line between them would read as a trend over the stops in
    between, which the study did not measure.
    """
    drawn = []
    for stop_k, score in sorted(anchors.items()):
        if stop_k not in xs:
            continue
        ax.plot([xs[stop_k] - ANCHOR_DODGE], [score], marker="D",
                ms=7.0, lw=0, color=D.REF_K0_INK, mec="white", mew=0.9,
                zorder=6)
        drawn.append(score)
    return drawn


def spread(ys, gap):
    """Push overlapping end labels apart, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for n, i in enumerate(order[1:], start=1):
        prev = out[order[n - 1]]
        if out[i] - prev < gap:
            out[i] = prev + gap
    return out


def draw_band(ax, xs, pts, ink):
    """The head-seed band, as a shaded ribbon on the k = 3 reference.

    Drawn under everything, so it reads as the floor of a difference and not
    as a subject. It returns its own edges, so the panel's vertical limits
    hold the whole band and a line that sits inside it stays visible.
    """
    if not pts:
        return []
    ss = sorted(pts)
    x = [xs[s] for s in ss]
    lo = [pts[s] - HEAD_SEED_BAND for s in ss]
    hi = [pts[s] + HEAD_SEED_BAND for s in ss]
    ax.fill_between(x, lo, hi, color=ink, alpha=0.14, lw=0, zorder=0)
    return lo + hi


def draw_reference(ax, xs, pts, ink, style, width=1.7):
    if not pts:
        return []
    ss = sorted(pts)
    ax.plot([xs[s] for s in ss], [pts[s] for s in ss], color=ink,
            linestyle=style, lw=width, marker="o", ms=4.0, mec="white",
            mew=0.8, zorder=2)
    return list(pts.values())


# How far right of its stop a variant marker sits, in axis units. The axis is
# categorical, so this is a dodge and not a claim about the step count.
VARIANT_DODGE = 0.055


def draw_variants(ax, cells, xs):
    """One open marker per variant cell, named on the point.

    Open, so it never reads as a stop of the solid line it sits on. The depth
    keeps its hue, because the cell is that depth on another schedule.

    Dodged right of its stop, because a variant lands near the scores of the
    same stop by construction: `ema30k` at 1.2385 sits 0.0048 from the k = 8
    point at the same stop, which is 1.6% of the axis, so the two markers
    touch. The dodge separates them and the label names which is which.
    """
    values = []
    for _, k, stop_k, variant, score in cells:
        if stop_k not in xs:
            continue
        x = xs[stop_k] + VARIANT_DODGE
        ax.plot([x], [score], marker="o", ms=8.0, lw=0,
                mfc="white", mec=D.colour(k), mew=2.0, zorder=5)
        ax.annotate(variant, (x, score), xytext=(9, -3),
                    textcoords="offset points", fontsize=8, color=D.INK,
                    va="center", ha="left")
        values.append(score)
    return values


# How far left of its stop the seed I-bar sits. Left, because the variant
# marker already holds the right side, and the anchor holds bb40k on the far
# left of its own stop only.
SEED_DODGE = 0.115


def draw_seeds(ax, repeats, phase, xs):
    """One I-bar per cell that ran under more than one head seed.

    It spans the lowest and the highest draw and caps both ends. The depth
    keeps its hue: the draws are that depth's own cell, under another head
    seed and nothing else.
    """
    values = []
    for (ph, k, stop_k), draws in sorted(repeats.items()):
        if ph != phase or stop_k not in xs:
            continue
        x = xs[stop_k] - SEED_DODGE
        lo, hi = draws[0], draws[-1]
        ax.plot([x, x], [lo, hi], color=D.colour(k), lw=1.6, zorder=5,
                solid_capstyle="butt")
        for y in (lo, hi):
            ax.plot([x - 0.028, x + 0.028], [y, y], color=D.colour(k),
                    lw=1.6, zorder=5)
        ax.annotate(f"{len(draws)} head seeds", (x, lo), xytext=(0, -11),
                    textcoords="offset points", fontsize=8, color=D.INK,
                    va="top", ha="center")
        values += [lo, hi]
    return values


def draw_panel(ax, phase, arms, k3, k0, xs, frontier, stops_k, variants=(),
               anchor=None, repeats=None):
    values = []
    ends = []
    for k in D.DEPTHS_DRAWN:
        pts = {s: v for s, v in arms.get(k, {}).items() if s in xs}
        if not pts:
            continue
        ss = sorted(pts)
        col = D.colour(k)
        ax.plot([xs[s] for s in ss], [pts[s] for s in ss], color=col,
                lw=2.0, marker="o", ms=5.0, mec="white", mew=0.9, zorder=4)
        values += list(pts.values())
        ends.append((xs[ss[-1]], pts[ss[-1]], k, col))

    values += draw_variants(ax, [c for c in variants if c[0] == phase], xs)
    values += draw_seeds(ax, repeats or {}, phase, xs)
    values += draw_band(ax, xs, k3, D.REF_K3_INK)
    values += draw_reference(ax, xs, k3, D.REF_K3_INK, D.STYLE_K3)
    values += draw_reference(ax, xs, k0, D.REF_K0_INK, D.STYLE_K0)
    if phase == K0_ANCHOR_PHASE:
        values += draw_anchor(ax, xs, anchor or {})

    ax.axhline(frontier, color=D.PRIOR_INK, lw=2.0, zorder=1)
    ax.set_title(PHASE_TITLE[phase], loc="left", fontsize=10.5, color=D.INK)
    ax.set_xticks([xs[s] for s in stops_k])
    ax.set_xticklabels([f"{s}k" if s in STOPS_K else str(s) for s in stops_k])
    ax.set_xlim(-0.14, len(stops_k) - 1 + 0.62)
    ax.set_xlabel("backbone train step")
    ax.grid(axis="y", color=D.GRID, lw=0.8)
    ax.set_axisbelow(True)
    return values, ends


def label_ends(ax, ends, lo, hi):
    """Direct labels: a coloured leader carries the identity, the text is ink.

    The aqua slot sits below 3:1 against white, so no figure here leaves an
    identity on colour alone. See depth_colours.py.
    """
    gap = (hi - lo) * 0.034
    ys = spread([e[1] for e in ends], gap)
    for (x, y, k, col), yy in zip(ends, ys):
        ax.annotate(D.label(k), (x, y), xytext=(x + 0.13, yy),
                    textcoords="data", fontsize=9, color=D.INK,
                    va="center", ha="left", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.4,
                                    shrinkA=2, shrinkB=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(STUDY / "results" / "scores.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    scores, variants, table_anchors, repeats = read_scores(a.scores)
    if not scores:
        raise SystemExit(f"ABORT: {a.scores} holds no {HEAD}-head score")
    k3 = parent_k3()
    k0 = published.PUBLISHED.get(CELL, {}).get(HEAD, {})
    anchor = k0_anchors(table_anchors)
    frontier = min(k3.values()) if k3 else min(k0.values())

    stops_k = axis_stops(scores)
    xs = {s: i for i, s in enumerate(stops_k)}
    # Both references live on the card's stops, so they are drawn only where
    # this table shares one. A trial table shares none, and draws neither.
    k3 = {s: v for s, v in k3.items() if s in xs}
    k0 = {s: v for s, v in k0.items() if s in xs}
    phases = [p for p in (1, 2) if scores.get(p)]
    fig, axes = plt.subplots(1, len(phases), figsize=(6.6 * len(phases), 5.6),
                             squeeze=False)
    axes = axes[0]

    values, per_panel = [], []
    for ax, phase in zip(axes, phases):
        v, ends = draw_panel(ax, phase, scores[phase], k3, k0, xs, frontier,
                             stops_k, variants, anchor, repeats)
        values += v
        per_panel.append((ax, ends))

    lo, hi = min(values), max(values)
    bot, top = lo - (hi - lo) * 0.08 - 0.005, hi + (hi - lo) * 0.10 + 0.005
    for ax, ends in per_panel:
        ax.set_ylim(bot, top)
        label_ends(ax, ends, bot, top)
    axes[0].set_ylabel("GM-Relative MASE, 97 GIFT-Eval configs "
                       "(lower is better)")
    handles = [
        Line2D([], [], color=D.INK_SOFT, linestyle=D.STYLE_K3, lw=1.7,
               label="k = 3, same cell and same head"),
        Patch(facecolor=D.REF_K3_INK, alpha=0.14, lw=0,
              label=f"head-seed band ±{HEAD_SEED_BAND:.4f}"),
        Line2D([], [], color=D.REF_K0_INK, linestyle=D.STYLE_K0, lw=1.7,
               label="k = 0 published, same cell and same head"),
        Line2D([], [], color=D.PRIOR_INK, lw=2.0,
               label=f"the best this cell reached before this study, "
                     f"{frontier:.4f}")]
    if any(s in xs for s in anchor):
        handles.insert(3, Line2D([], [], marker="D", ms=7.0, lw=0,
                                 color=D.REF_K0_INK, mec="white", mew=0.9,
                                 label="k = 0 anchor, this study's path, "
                                       "left panel only"))
    if repeats:
        handles.append(Line2D([], [], marker="_", ms=9.0, lw=1.6,
                              color=D.INK,
                              label="I-bar: the range over the head seeds"))
    if any(c[2] in xs for c in variants):
        handles.append(
            Line2D([], [], marker="o", ms=8.0, lw=0, mfc="white",
                   mec=D.INK, mew=2.0,
                   label="open marker: the same depth and stop on another "
                         "training schedule"))
    # The head budget of the three reference marks. They are the parent
    # study's own numbers, so they carry that study's head budget and not
    # this one's. The right panel draws cells at 40,000 and 100,000 head
    # steps against them, so the panel is not head-matched and the figure
    # has to say so.
    handles.append(Line2D([], [], lw=0, label=REF_HEAD_NOTE))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(3, len(handles)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("GM-Relative MASE against backbone train step, "
                 "rollout depth k = 8 and 32", fontsize=12.5, color=D.INK)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945))
    fig.savefig(a.out)
    drawn = sum(len(e) for _, e in per_panel)
    shown = [c for c in variants if c[2] in xs]
    extra = "".join(f", {v} k = {k} at bb{s}k" for _, k, s, v, _ in shown)
    anc = "".join(f", k = 0 anchor {v:.4f} at bb{s}k"
                  for s, v in sorted(anchor.items()) if s in xs)
    rep = "".join(f", {len(v)} head seeds on k = {k} at bb{s}k"
                  for (_, k, s), v in sorted(repeats.items()) if s in xs)
    print(f"wrote {a.out}  ({drawn} line(s), frontier {frontier:.4f}"
          f"{anc}{rep}{extra})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
