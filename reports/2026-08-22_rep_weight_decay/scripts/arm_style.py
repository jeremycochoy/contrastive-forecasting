#!/usr/bin/env python3
"""One name, one color and one role for each run of #409.

WHY THIS MODULE EXISTS. Three figures draw the same runs. A color that means
one thing in one figure and another thing in the next makes the set
unreadable. So the mapping lives here, and every figure imports it.

THE ENCODING. Every arm of this card is the SAME treatment: one decay of the
weight on L_rep, over repeat backbone seeds. So the arms are not categories.
They are replicates, and they take ONE series color.

  series color   a run that held the contrastive task
  alarm color    a run that lost it. That is a state, not a series, so it
                 takes the status palette
  muted ink      a reference the sweep already published. A reference is
                 recessive: it is not one of this card's runs
  direct label   the backbone seed, at the right end of each curve

That gives two data colors. The pair passes the six checks of the data-viz
standard on all pairs, in light mode and in dark mode: CVD delta-E 23.8,
normal-vision delta-E 31.6, and both clear 3:1 against each surface.

Identity is never color alone. Each curve carries its seed as a direct label,
and each figure names the two colors in a legend.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

# Categorical slot 1 of the data-viz standard. Every run of this card.
SERIES = "#2a78d6"
# The `critical` step of the status palette. A run that lost the contrastive
# task, which is a state and not a series.
LOST = "#d03b3b"
# Ink, grid and the references. Text never wears a series color.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
SURFACE = "#fcfcfb"
# The reference lines are the sweep's published scores, so they are recessive.
REFERENCE = MUTED

# The two scores `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md`
# publishes for this cell, at the same 40,000-step stop and the same
# 30,000-step head. This card measures no control, so these are what an arm is
# read against.
SWEEP_SCORES = {"20260524": 1.1491, "20260520": 1.1507}


def seed_label(row):
    """What the reader sees beside a curve: the backbone seed."""
    return f"seed {row['seed']}"


def read_arms(path):
    """The arms table, in the card's order, as a list of dicts.

    Two columns: the arm and its backbone seed. The decay is the card's, not
    the arm's, so it is in `study.sh` and not here.
    """
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] == "arm":
                continue
            out.append({"arm": parts[0], "seed": parts[1]})
    return out


def read_csv_column(path, column, every=1):
    """`[(step, value), ...]` for one column of one losses CSV.

    A blank cell is a term the loss skipped that step, and a non-finite value
    is a diagnostic that did not compute. Both are dropped, so a caller never
    plots a gap as a zero.
    """
    out = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or column not in reader.fieldnames:
            return out
        for n, row in enumerate(reader):
            if n % every:
                continue
            try:
                step = int(row["step"])
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                out.append((step, value))
    return out


def smooth(series, window):
    """A trailing mean over `window` points. One step of this trainer is one
    batch, so a raw curve hides its own trend."""
    if window <= 1 or not series:
        return series
    steps = [s for s, _ in series]
    values = [v for _, v in series]
    out, run = [], []
    for step, value in zip(steps, values):
        run.append(value)
        if len(run) > window:
            run.pop(0)
        out.append((step, sum(run) / len(run)))
    return out


def label_right(ax, items, fontsize=8, pad=1.5):
    """Name every curve at its right end, without letting two labels collide.

    `items` is `[(series, text, color), ...]`. Six replicates share one color,
    so no figure leaves identity to the color alone, and repeats of one arm can
    land on the same value. This stacks the labels in PIXEL space, which is
    where a collision happens: a gap in data units means a different number of
    lines of text on a short panel and on a tall one. A label pushed off its
    curve keeps a hairline back to it, and the stack slides down if it would
    leave the top of the panel.

    Call it once per panel, after `figure.canvas.draw()`, so the transform
    reads the limits the panel ends with.
    """
    ends = [(s[-1][0], s[-1][1], t, c) for s, t, c in items if s]
    if not ends:
        return
    dpi = ax.figure.dpi
    gap = fontsize * pad * dpi / 72.0
    rows = sorted(((ax.transData.transform((step, value))[1], step, value,
                    text, color) for step, value, text, color in ends),
                  key=lambda r: r[0])
    placed = []
    for row in rows:
        y = row[0]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
    try:
        box = ax.get_window_extent()
        over = placed[-1] - box.y1
        if over > 0:
            placed = [y - min(over, placed[0] - box.y0) for y in placed]
    except (AttributeError, RuntimeError):
        pass

    for (y0, step, value, text, color), y in zip(rows, placed):
        dy = (y - y0) * 72.0 / dpi
        ax.annotate(text, (step, value), xytext=(6, dy),
                    textcoords="offset points", fontsize=fontsize,
                    color=color, va="center", ha="left",
                    annotation_clip=False,
                    arrowprops=(dict(arrowstyle="-", color=color,
                                     linewidth=0.6, alpha=0.7,
                                     shrinkA=0, shrinkB=2)
                                if abs(dy) > 0.5 else None))


def tidy(ax):
    """The recessive frame every figure of this card uses."""
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def read_verdicts(path):
    """`{run name: verdict}` from `results/auc_verdicts.tsv`.

    A run the AUC gate stopped takes the alarm color in every figure, so the
    reader sees the collapse and the curve as one fact. A missing table gives
    an empty dict, and every run then draws as held.
    """
    out = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                run = row.get("run")
                if run:
                    out[run] = (row.get("verdict") or "").strip()
    except OSError:
        return {}
    return out


def run_colour(path, verdicts):
    """The color of one losses CSV: alarm if that run lost the contrastive
    task, the series color otherwise."""
    name = Path(path).name
    for run, verdict in verdicts.items():
        if verdict == "lost" and (run == name or Path(run).name == name):
            return LOST
    return SERIES


def study_paths(root, arms, cell="arm6_v2_combab_alignT", k=32, stop=40000):
    """`{arm: [losses.csv, ...]}` under one checkpoint root.

    A leg re-fired after a crash writes a second CSV under a `_rN` run name,
    and the report reads both.
    """
    out = {}
    root = Path(root)
    leg = f"leg_{stop // 1000}k"
    for row in arms:
        arm = row["arm"]
        d = root / arm / cell / leg
        name = f"cf393_{cell}_cf373k{k}_cf409_{arm}"
        found = sorted(d.glob(f"{name}_losses.csv")) + \
            sorted(d.glob(f"{name}_r[0-9]*_losses.csv"))
        if found:
            out[arm] = found
    return out
