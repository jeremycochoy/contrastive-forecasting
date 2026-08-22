#!/usr/bin/env python3
"""One name, one colour and one line style for each arm of #409.

WHY THIS MODULE EXISTS. Three figures draw the same eight arms. A colour that
means "the 0.5 floor" in one figure and "seed 20260524" in another makes the
set unreadable. So the mapping lives here, and every figure imports it.

THE ENCODING. Two things separate the arms, so two channels carry them:

  colour       the L_rep floor. It is an ORDERED quantity — 1.0, 0.5, 0.2,
               0.0 — so it takes one hue, light to dark. A deeper decay is a
               darker line. The one arm whose L_align pulls toward the EMA
               teacher is a different object, not a fifth floor, so it takes
               the accent hue.
  line style   the backbone seed. Solid is 20260520, dashed is 20260524.

The ramp passes the ordinal checks of the data-viz standard: one hue, a
monotone lightness, an adjacent gap of 0.06 or more, and a light end that
clears the surface. Every figure also labels its curves directly, because two
adjacent steps of one ramp are close under simulated colour blindness.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

# Light to dark: the more of L_rep the arm removes, the darker its line.
FLOOR_COLOUR = {
    "1.0": "#6baed6",
    "0.5": "#3182bd",
    "0.2": "#08519c",
    "0.0": "#08306b",
}
# The arm whose L_align pulls toward the EMA teacher. Past its ramp that arm
# holds no L_rep, so it is BYOL and not a floor of the walk.
TEACHER_COLOUR = "#d95f02"
# Ink, grid and the AUC gate line. Text never wears a series colour.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
ALARM = "#b2182b"

SEED_STYLE = {"20260520": "-", "20260524": "--"}
SEED_SHORT = {"20260520": "s20", "20260524": "s24"}


def floor_key(rep_end, default="1.0"):
    """The floor of one arm, as a key of FLOOR_COLOUR."""
    if rep_end in (None, "", "-"):
        return default
    return f"{float(rep_end):.1f}"


def arm_colour(row):
    """The colour of one arms.tsv row."""
    if row["align_target"] == "teacher":
        return TEACHER_COLOUR
    return FLOOR_COLOUR[floor_key(row["rep_end"])]


def arm_style(row):
    return SEED_STYLE.get(row["seed"], "-")


def arm_label(row):
    """What the reader sees. The floor, the seed, and the target when it is
    not the cell's own student."""
    floor = floor_key(row["rep_end"])
    if row["rep_end"] in (None, "", "-"):
        head = "no decay"
    else:
        head = f"floor {floor}"
    tail = SEED_SHORT.get(row["seed"], row["seed"])
    if row["align_target"] == "teacher":
        return f"{head}, {tail}, teacher"
    return f"{head}, {tail}"


def read_arms(path):
    """The arms table, in the card's order, as a list of dicts."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] == "arm":
                continue
            out.append({"arm": parts[0], "rep_end": parts[1],
                        "ramp": parts[2], "seed": parts[3],
                        "align_target": parts[4]})
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


def label_right(ax, series, text, colour, fontsize=8):
    """Name one curve at its right end. Two adjacent steps of one ramp are
    close under simulated colour blindness, so no figure leaves identity to
    the colour alone."""
    if not series:
        return
    step, value = series[-1]
    ax.annotate(text, (step, value), xytext=(4, 0),
                textcoords="offset points", fontsize=fontsize,
                color=colour, va="center", ha="left")


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


def study_paths(root, arms, cell="arm6_v2_combab_alignS", stop=40000):
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
        name = f"cf393_{cell}_cf373k3_cf409_{arm}"
        found = sorted(d.glob(f"{name}_losses.csv")) + \
            sorted(d.glob(f"{name}_r[0-9]*_losses.csv"))
        if found:
            out[arm] = found
    return out
