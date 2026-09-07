#!/usr/bin/env python3
"""One color and one role for each run of #412, and the 1.1M references.

WHY THIS MODULE EXISTS. Four figures draw the same arms. A color that means
one thing in one figure and another thing in the next makes the set
unreadable. So the mapping lives here, and every figure imports it.

THE ENCODING. This card moves ONE thing, the width. So the size is the top
split, and it takes the two roles the data-viz standard already gives:

  series color   an 11.4M run of THIS card
  muted ink      a 1.1M score a parent study already published. A reference is
                 recessive: it is not one of this card's runs
  alarm color    a run that lost the contrastive task. That is a state, not a
                 series, so it takes the `critical` step of the status palette
  a light ramp   inside one figure that draws several arms, the rollout depth
                 k orders the curves. Depth is a magnitude, so it takes one
                 hue light to dark, never one hue for each arm

Identity is never color alone. Every curve carries its arm as a direct label,
and every figure with two or more colors names them in a legend.

The generic readers (`read_run`, `label_right`, `tidy`, `smooth`,
`window_mean`) come from #409's `arm_style.py`. `study.sh` already takes the
leg runner from a parent study, and this is the same reuse.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent

# The generic plot helpers of #409, by path. They read a losses CSV, stack
# labels without a collision and draw the recessive frame this project uses.
_SRC = REPO / "reports" / "2026-08-22_rep_weight_decay" / "scripts" / "arm_style.py"
_spec = importlib.util.spec_from_file_location("cf409_arm_style", _SRC)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cf409_arm_style"] = _mod
_spec.loader.exec_module(_mod)

read_run = _mod.read_run
label_right = _mod.label_right
tidy = _mod.tidy
smooth = _mod.smooth
window_mean = _mod.window_mean
read_verdicts = _mod.read_verdicts

# ---- The colors -------------------------------------------------------------
# Categorical slot 1. Every 11.4M run of this card.
SERIES = "#2a78d6"
# The `critical` step of the status palette. A run that lost the contrastive
# task, which is a state and not a series.
LOST = "#d03b3b"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
SURFACE = "#fcfcfb"
# A published 1.1M score. Recessive: it is not one of this card's runs.
REFERENCE = MUTED
# Held arms that the headline does not name, in a curve figure.
HELD = "#b0b0b0"
# The rollout depth k, light to dark on the series hue. Depth is a magnitude.
DEPTH_RAMP = {3: "#8fbce8", 8: "#5b9ade", 32: "#1d5599"}

STOP = 40000
CELL = "arm6_v2_combab_alignT"
# THE SEED BAND, and this card measured it at 11.4M parameters: `k3_r100_09`
# scores 1.3495 and `k3_r100_09b` scores 1.2927 at the 40,000-step stop. The
# two runs differ in the backbone seed alone. Two numbers closer than this are
# not ranked, at every stop of this report.
#
# It replaces #409's 0.0471, which is a 1.1M number. A band measured at the
# width the report ranks at is the one the report uses.
BAND = 0.0568
BAND_SOURCE = "this card measured it at 11.4M"

# ---- The 1.1M references ----------------------------------------------------
#
# What the parent studies published for the SAME cell at 1.1M parameters, on
# the same 97 GIFT-Eval configs under strategy B4 at horizon 16, with the
# student-encoder head. `{arm: {stop: (score, head steps, source)}}`.
#
# THE HEAD BUDGET IS PART OF THE KEY. #373 trained a 15,000-step head at its
# 40,000-step stop and a 30,000-step head at 100,000 and 200,000
# (`reports/2026-08-08_rollout_depth/rollout_depth.md`, "The head budget
# differs by column"). #404 and #409 trained a 30,000-step head at 40,000
# steps. This card trains 30,000 at every stop. So a comparison is
# head-matched at some cells and stops and not at others, and every figure
# says which.
#
# `k3_r100_09` DOES have a 1.1M twin: cell A3 of #373,
# `arm6_v2_combab_alignT_sched`, which is this cell at k = 3 under the sum
# reduction, the same 0.9-to-1.0-at-100k momentum and the same seed 20260520
# (`reports/2026-08-08_rollout_depth/scripts/cells.tsv`, `scripts/run_leg_k.sh`).
# `arms.tsv` of this card calls configuration 1 "never run", and that is wrong.
A3 = "reports/2026-08-08_rollout_depth (cell A3)"
K32 = "reports/2026-08-19_ema_momentum_k32"
DEC = "reports/2026-08-22_rep_weight_decay"
REF_1M1 = {
    "k3_r100_09": {40000: (1.3618, 15000, A3), 100000: (1.3010, 30000, A3),
                   200000: (1.3998, 30000, A3)},
    "k3_r100_09b": {40000: (1.3618, 15000, A3), 100000: (1.3010, 30000, A3),
                    200000: (1.3998, 30000, A3)},
    # SEED-MATCHED. #404 ran this schedule at two backbone seeds, 1.1507 at
    # 20260520 and 1.1491 at 20260524. This card runs 20260520, so 1.1507 is
    # the twin and the pair is the reference's own spread.
    "k32_r100_09": {40000: (1.1507, 30000, K32)},
    "k32_r200_08": {40000: (1.1782, 30000, K32)},
    "k32_r100_09_dec": {40000: (1.2295, 30000, DEC)},
}
# The whole measured spread of a 1.1M reference, where the parent ran a repeat.
REF_1M1_SEEDS = {("k32_r100_09", 40000): (1.1491, 1.1507)}


# The two seeds of `k3_r100_09` at 11.4M parameters. They are what `BAND`
# above measures.
BAND_SEEDS = ("k3_r100_09", "k3_r100_09b")


def effective_band(scores):
    """The band a gap is read against, and where it came from.

    `scores` is `{arm: score}` at ONE stop. `BAND` is this card's own
    measurement, and it holds at every stop: the two seeds ran at 40,000
    steps, and no other stop of this card carries a repeat. The pair is read
    again here, so a re-collected score that widens the spread widens the
    band with it.
    """
    pair = [scores[a] for a in BAND_SEEDS if a in scores]
    if len(pair) == 2:
        return max(BAND, abs(pair[0] - pair[1])), BAND_SOURCE
    return BAND, BAND_SOURCE


# The contrastive AUC the parents published for the SAME cell, seed and
# 40,000-step stop at 1.1M parameters. `{arm: (last AUC, floor, source)}`.
#
# It is the reference the AUC table needs, because the card's k = 32 arms read
# far under it and its k = 3 arms do not. #373 published no AUC column for
# cells A3 and A4, so the k = 3 rows have no anchor.
#
#   k32_r100_09      #404, `ema_momentum_k32.md`, the 20260520 row of "the
#                    fourteen runs": AUC at the stop 0.978, score 1.1507.
#   k32_r200_08      the same table, momentum 0.8 to 1.0 at 200k: 0.957.
#   k32_r100_09_dec  #409, `rep_weight_decay.md`, run `dec_m090r100_ramp2k`:
#                    floor 0.9092 at step 3,209, last 0.9833 at 40,000, HELD.
#                    Same k, same momentum, same 2,000-step decay ramp, same
#                    seed. It is the exact twin of the arm that collapsed here.
REF_1M1_AUC = {
    "k32_r100_09": (0.978, None, "#404"),
    "k32_r200_08": (0.957, None, "#404"),
    "k32_r100_09_dec": (0.9833, 0.9092, "#409"),
}


def reference_auc(arm):
    """(last AUC, floor, source) of the 1.1M twin at 40,000 steps, or None."""
    return REF_1M1_AUC.get(arm)


def reference_spread(arm, stop):
    """(low, high) of the 1.1M reference over the parent's seeds, or None."""
    return REF_1M1_SEEDS.get((arm, int(stop)))
# The project best, at 1.1M. A DIFFERENT align target: cell A4 targets the
# STUDENT. It is the number the card asks an 11.4M model to beat.
PROJECT_BEST = 1.0651
PROJECT_BEST_LABEL = "1.1M, align student, k = 3, 200,000 steps"
# Moirai-2-Small, the size this card matches, on the same 97 configs.
MOIRAI_SMALL = 0.728


def reference(arm, stop):
    """(score, head steps, source) of the 1.1M twin, or None."""
    return REF_1M1.get(arm, {}).get(int(stop))


# The columns of `arms.tsv`, in its order.
ARMS_COLUMNS = ("arm", "k", "reduce", "tau", "end", "ramp", "seed", "decay",
                "lr")
# The rate every published run of this cell trained at. Two arms bracket it.
LR_DEFAULT = "1e-3"


def read_arms(path):
    """The arms table of this card, in its order, as a list of dicts."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or parts[0] == "arm":
                continue
            row = dict(zip(ARMS_COLUMNS, parts))
            if row.get("lr", "-") == "-":
                row["lr"] = LR_DEFAULT
            out.append(row)
    return out


def read_scores(path, stop=None):
    """`{(arm, stop): score}` from `results/scores.csv`.

    An arm carried past 40,000 steps holds one measured score at each stop, so
    the stop is part of the key. `stop=<n>` gives `{arm: score}` at that stop.
    """
    rows = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows[(row["arm"], int(row["stop"]))] = float(row["score"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return {}
    if stop is None:
        return rows
    return {a: v for (a, s), v in rows.items() if s == stop}


def arm_label(row):
    """What a reader sees beside a curve or a bar: the depth, then the
    treatment that separates this arm from the others at that depth."""
    text = f"k = {row['k']}"
    if row["end"] != "-":
        text += f", {row['tau']} to {row['end']} at {int(row['ramp']) // 1000}k"
    else:
        text += f", {row['tau']} fixed"
    if row["decay"] != "-":
        text += f", decay to 0 by {int(row['decay']) // 1000}k"
    # The bracket arms differ from configuration 1 in the rate alone, so a
    # label that dropped it would name three curves the same.
    if row["lr"] != LR_DEFAULT:
        text += f", lr {row['lr']}"
    return text


def run_name(arm, k):
    """The run name every artefact of one arm carries."""
    return f"cf393_{CELL}_cf373k{k}_cf412_{arm}"


def losses_csvs(root, arm, k, stop=STOP):
    """Every losses CSV one arm wrote, over every leg, oldest stop first.

    A leg re-fired after a crash resumes under a `_rN` name and opens a second
    CSV. A higher stop opens a `leg_<N>k` directory of its own. The report
    reads them all, and `read_run` settles the overlaps.
    """
    root = Path(root) / arm / CELL
    name = run_name(arm, k)
    found = []
    for leg in sorted(root.glob("leg_*k"),
                      key=lambda p: int(p.name[4:-1] or 0)):
        found += sorted(leg.glob(f"{name}_losses.csv"))
        found += sorted(leg.glob(f"{name}_r[0-9]*_losses.csv"))
    return found


def depth_colour(k):
    """The color of one arm's curve, by its rollout depth."""
    return DEPTH_RAMP.get(int(k), SERIES)


def lost(verdicts, paths):
    """True when any leg of this arm lost the contrastive task."""
    names = {Path(p).name for p in paths}
    return any(v == "lost" and (r in names or Path(r).name in names)
               for r, v in verdicts.items())
