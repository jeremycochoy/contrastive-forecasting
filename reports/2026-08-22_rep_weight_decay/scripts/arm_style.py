#!/usr/bin/env python3
"""One name, one color and one role for each run of #409.

WHY THIS MODULE EXISTS. Three figures draw the same runs. A color that means
one thing in one figure and another thing in the next makes the set
unreadable. So the mapping lives here, and every figure imports it.

THE ENCODING. Every arm of this card carries the SAME decay and differs in the
EMA schedule. The schedules form an ORDER, not a set of categories: each one is
read by the momentum it holds at the stop, from 0.840 to 0.990. An order does
not need eight hues, and eight hues would say "eight unrelated things".

  series color   a run that held the contrastive task
  alarm color    a run that lost it. That is a state, not a series, so it
                 takes the status palette
  muted ink      a reference the sweep already published. A reference is
                 recessive: it is not one of this card's runs
  direct label   the arm's schedule, at the right end of each curve

That gives two data colors. The pair passes the six checks of the data-viz
standard on all pairs, in light mode and in dark mode: CVD delta-E 23.8,
normal-vision delta-E 31.6, and both clear 3:1 against each surface.

Identity is never color alone. Each curve carries its schedule as a direct
label, and each figure names the two colors in a legend.
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

# What `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md` published for
# each of this card's schedules, on this same cell, at the same 40,000-step
# stop and the same 30,000-step head, with NO decay. The key is the schedule as
# `arms.tsv` writes it. This card measures no control, so these are what an arm
# is read against. `0.99 fixed` is absent: the sweep never ran it.
SWEEP_SCORES = {
    ("0.9", "1.0", "100000"): 1.1507,
    ("0.9", "1.0", "60000"): 1.1873,
    ("0.9", "1.0", "200000"): 1.1784,
    ("0.9", "-", "-"): 1.1819,
    ("0.95", "-", "-"): 1.1907,
    ("0.95", "1.0", "100000"): 1.2130,
    ("0.8", "1.0", "200000"): 1.1782,
}
# The best of them, which is the number the card asks an arm to beat.
SWEEP_BEST = 1.1491
STOP = 40000


def schedule(row):
    """One arm's EMA schedule, as the key that identifies its row."""
    return (row["tau"], row["end"], row["ramp"])


def momentum_at(row, step=STOP):
    """The momentum an arm HOLDS at a step, not the one its flags name.

    Two arms can name 0.9 and hold 0.967 and 0.920 at 40,000 steps. The held
    value is what ranks them. This repeats `src.models.ema_tau_at_step`, which
    `cf409_momentum_at` in `study.sh` also repeats.
    """
    tau = float(row["tau"])
    if row["end"] == "-":
        return tau
    end, ramp = float(row["end"]), int(row["ramp"])
    if ramp <= 0:
        return end
    return tau + min(max(step / ramp, 0.0), 1.0) * (end - tau)


def arm_label(row):
    """What the reader sees beside a curve: the arm's EMA schedule.

    The seed rides along only where a schedule has more than one, which is arm
    1. Every other schedule carries one seed, and a seed on every label would
    be noise.
    """
    if row["end"] == "-":
        text = f"{row['tau']} fixed"
    else:
        text = f"{row['tau']} to {row['end']} at {int(row['ramp']) // 1000}k"
    if row.get("ambiguous"):
        # Two rows on one schedule AND one seed. The schedule names neither.
        return f"{text}, {row['arm']}"
    if row.get("repeat"):
        text = f"{text}, seed {row['seed']}"
    return text


def schedule_label(row):
    """One arm's EMA schedule, for a CSV cell. No seed and no comma.

    `arm_label` puts the seed after a comma, which reads well beside a curve
    and badly inside a comma-separated field. Every table that carries this
    also carries a `seed` column of its own.
    """
    if row["end"] == "-":
        return f"{row['tau']} fixed"
    return f"{row['tau']} to {row['end']} at {int(row['ramp']) // 1000}k"


def curve_label(row):
    """What the reader sees beside a CURVE: the momentum, then the arm.

    A curve figure holds nine of these at the right margin, and
    "0.9 to 1.0 at 100k, seed 20260524" is 34 characters of it. The momentum
    the arm HOLDS at the stop is what orders the arms, and the arm name is the
    key every table of this study joins on. `arm_label` keeps the long form
    for the score figure, whose labels sit on a y axis with room.
    """
    return f"{momentum_at(row):.3f}  {row['arm']}"


def read_arms(path):
    """The arms table, in the card's order, as a list of dicts.

    Five columns: the arm, the three EMA momentum columns and the backbone
    seed. A `-` is a flag the arm does not pass. The decay is the card's, not
    the arm's, so it is in `study.sh` and not here.

    Each row also gets two flags:

      repeat     another row shares its schedule at a DIFFERENT seed. Those
                 rows are a seed spread, and `plot_scores.py` draws their
                 range as the bar that says whether a gap is a rank.
      ambiguous  another row shares its schedule AND its seed. Those rows are
                 NOT a seed spread: `dec_m080_r200` and `dec_ramp30k_m080`
                 differ in the DECAY ramp, which is not a column here. The
                 schedule cannot name such a curve, so only the arm can.
    """
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] == "arm":
                continue
            out.append({"arm": parts[0], "tau": parts[1], "end": parts[2],
                        "ramp": parts[3], "seed": parts[4]})
    pairs = [(schedule(r), r["seed"]) for r in out]
    for row in out:
        key = schedule(row)
        row["repeat"] = any(s == key and seed != row["seed"]
                            for s, seed in pairs)
        row["ambiguous"] = pairs.count((key, row["seed"])) > 1
    return out


def read_run(paths, columns, every=1):
    """One RUN's whole trajectory, over every losses CSV that run wrote.

    A run of this card can write its steps more than once. A leg re-fired
    after a crash resumes under a `_rN` name and opens a SECOND CSV, and a leg
    that starts again from step 0 APPENDS to the first one. `dec_m080_r200`
    holds 59,900 rows over 40,000 steps for that reason, and `dec_m099_fix`
    holds two files that overlap from step 15,001 to 19,900.

    TWO RULES SETTLE AN OVERLAP.

    Inside ONE file, the LAST row of a step wins. A later row of a file is a
    later attempt at that step.

    Between two FILES, the file that reached the FURTHEST step wins. That is
    the attempt the report reads to the stop, and it is not always the last
    name: `dec_m099_fix` ran on in `_r2` to 40,000, so `_r2` wins, while
    `dec_s23` ran on in its BASE file to 22,900 and its `_r2` gave up at
    20,300, so the base wins. Taking the `_rN` file every time would splice
    100 steps of a dead attempt into the middle of a live one.

    A reader that took one ROW in `every` before it looked at the step column
    would interleave two attempts and give a curve that walks backwards, and
    its last row would be the last row of the file rather than the last step
    of the run.

    Returns `{column: [(step, value), ...]}`, each list sorted by step. A
    blank cell is a term the loss skipped that step and a non-finite value is
    a diagnostic that did not compute. Both are dropped, so a caller never
    reads a gap as a zero.
    """
    columns = list(columns)
    per_file = []
    for path in paths:
        rows = {}
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                continue
            index = {name: n for n, name in enumerate(header)}
            if "step" not in index:
                continue
            want = [(c, index[c]) for c in columns if c in index]
            step_at = index["step"]
            for row in reader:
                try:
                    step = int(row[step_at])
                except (IndexError, ValueError):
                    continue
                cell = rows.setdefault(step, {})
                for column, n in want:
                    try:
                        value = float(row[n])
                    except (IndexError, ValueError):
                        cell.pop(column, None)
                        continue
                    if math.isfinite(value):
                        cell[column] = value
                    else:
                        cell.pop(column, None)
        if rows:
            per_file.append((max(rows), rows))

    data = {}
    for _, rows in sorted(per_file, key=lambda t: t[0]):
        for step, cell in rows.items():
            data.setdefault(step, {}).update(cell)
    steps = sorted(data)[::max(1, int(every))]
    return {c: [(s, data[s][c]) for s in steps if c in data[s]]
            for c in columns}


def read_csv_column(path, column, every=1):
    """`[(step, value), ...]` for one column of one losses CSV."""
    return read_run([path], [column], every)[column]


def window_mean(series, hi, span=1000):
    """The mean of one term over the `span` steps that end at `hi`.

    One step of this trainer is one batch, so a term read at a single step is
    noise. A window states which steps it covers, which a trailing mean over a
    subsampled curve does not.
    """
    values = [v for s, v in series if hi - span < s <= hi]
    return sum(values) / len(values) if values else None


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


def run_colour(paths, verdicts):
    """The color of one run: alarm if it lost the contrastive task, the series
    color otherwise.

    Takes one path or the whole list of CSVs one arm wrote. An arm that lost
    the task in ANY of its legs is a lost arm, whichever leg holds the last
    step.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    names = {Path(p).name for p in paths}
    for run, verdict in verdicts.items():
        if verdict == "lost" and (run in names or Path(run).name in names):
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
