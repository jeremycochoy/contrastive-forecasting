#!/usr/bin/env python3
"""The training loss to 40,000 steps, by term, for every run.

WHY THIS FIGURE EXISTS. The card asks what happens when the weight on L_rep
falls to 0.0. L_rep holds 92 to 93 percent of the total loss and reaches its
level near step 100, so the answer is not in the total. It is in the terms.

THE PANELS. One panel per term, all on the same x axis:

  loss              the total the trainer optimizes, as the CSV logs it.
                    On a LOG axis: it falls from 14 to under 1 over the first
                    12,000 steps, and a linear axis squashes the other 28,000.
  rep_w             the live weight on L_rep. It is the treatment.
  l_rep             the UNWEIGHTED L_rep the loss computed. The trainer leaves
                    the cell blank at weight 0.0, where it computes no L_rep,
                    so every line ends at step 9,999.
  L_align, reduced  the align term as the loss holds it. See below.
  mean cos_err      the forecast error over the rollout depths.

ONE LINE PER ARM, NOT PER FILE. A leg re-fired after a crash resumes under a
`_rN` name and opens a second CSV, and a leg that starts again from step 0
appends to the first one. `arm_style.read_run` keys every row by its step and
lets the last row of that step win, so one arm gives one trajectory.

THE ALIGN PANEL IS A RESIDUAL, AND HAS TO BE. This card runs k = 32 under the
`mean` reduction against the EMA TEACHER, and two things follow. The `l_align`
column is the depth-0 copy alone, while the loss holds the MEAN of 33 copies.
And `l_align` is NOT `2 * cos_err_d0` here, because `cos_err_dj` reads the
student's next latent and the teacher target reads the teacher's. So the
`cos_err_d*` columns cannot rebuild it. `notes/loss_decomposition.md` gives
the formula this panel uses:

  L_align, reduced = (loss - rep_w * l_rep - sigreg_e - sigreg_h) / align_w

That is exact on this cell, whose CPC weight is 0.0 and whose SIGReg weights
are 1.0.

THE SLOPE. The card's second question asks which backbone can improve more
with longer training. A term that still falls at the stop is headroom. So
`results/loss_terms_at_stop.csv` gives, per arm and per term:

  value_at_30k         the mean over steps 29,001 to 30,000
  value_at_stop        the mean over the last 1,000 steps the run reached
  change_30k_to_40k    the second minus the first, and BLANK for a run that
                       stopped before 40,000 steps or a term that ended early

A window states which steps it covers. A single step of this trainer is one
batch, so a term read at one step is noise.

`results/loss_terms_trajectory.csv` reads the same windows on a 5,000-step
grid, so a reader can tell a trend at the stop from one spike.

Usage:
  plot_loss_terms.py --root /home/jupyter/checkpoints_backup/cf-409 \
      --arms scripts/arms.tsv --out plots/loss_terms.png \
      --table results/loss_terms_at_stop.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

PANELS = [
    ("loss", "the total loss", True),
    ("rep_w", "weight on L_rep", False),
    ("l_rep", "L_rep, unweighted", False),
    ("align_reduced", "L_align, reduced", True),
    ("cos_err", "mean cos_err", True),
]
# The rollout depth of this card's cell. `cos_err_d0` thru `cos_err_dk`.
DEPTH = 32
# The columns the five panels need, in one pass over each CSV.
COLUMNS = (["loss", "rep_w", "l_rep", "sigreg_e", "sigreg_h"]
           + [f"cos_err_d{j}" for j in range(DEPTH + 1)])
# The step the card stops at, and the step the slope window starts from.
STOP = 40000
SLOPE_FROM = 30000
# The width of every window of both tables, in steps.
SPAN = 1000
# `results/loss_terms_trajectory.csv` reads each term on this grid. The card
# asks for the training loss TO 40,000 steps, and two windows give two points.
# A grid says whether a slope at the stop is a trend or one spike.
GRID = list(range(5000, STOP + 1, 5000))


def mean_cos_err(run, depth=DEPTH):
    """The mean of `cos_err_d0` thru `cos_err_d<depth>`, step by step."""
    columns = [run.get(f"cos_err_d{j}") or [] for j in range(depth + 1)]
    columns = [c for c in columns if c]
    if not columns:
        return []
    by_step = {}
    for column in columns:
        for step, value in column:
            by_step.setdefault(step, []).append(value)
    want = len(columns)
    return [(step, sum(v) / len(v)) for step, v in sorted(by_step.items())
            if len(v) == want]


def align_reduced(run, align_weight=1.0):
    """The align term as the loss holds it, from the residual of the total.

    Returns `[(step, value), ...]`, and skips a step it cannot close: one
    where `rep_w` is non-zero and `l_rep` is blank. At weight 0.0 the term is
    off and blank is correct, so those steps close with `rep_w * l_rep = 0`.
    """
    total = run.get("loss") or []
    if not total or align_weight == 0.0:
        return []
    rep_w = dict(run.get("rep_w") or [])
    l_rep = dict(run.get("l_rep") or [])
    sig_e = dict(run.get("sigreg_e") or [])
    sig_h = dict(run.get("sigreg_h") or [])
    out = []
    for step, value in total:
        w = rep_w.get(step, 0.0)
        if w and step not in l_rep:
            continue
        rep = w * l_rep.get(step, 0.0)
        out.append((step, (value - rep - sig_e.get(step, 0.0)
                           - sig_h.get(step, 0.0)) / align_weight))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="/home/jupyter/checkpoints_backup/cf-409")
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "loss_terms.png"))
    p.add_argument("--verdicts",
                   default=str(HERE.parent / "results" / "auc_verdicts.tsv"))
    p.add_argument("--table",
                   default=str(HERE.parent / "results" / "loss_terms_at_stop.csv"))
    p.add_argument("--trajectory",
                   default=str(HERE.parent / "results"
                               / "loss_terms_trajectory.csv"))
    p.add_argument("--align-weight", type=float, default=1.0,
                   help="the cell's --align-loss-weight, for the residual")
    p.add_argument("--smooth", type=int, default=500)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--min-steps", type=int, default=1000,
                   help="a run that reached fewer steps is named, not drawn")
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    paths = S.study_paths(args.root, arms)
    if not paths:
        print(f"no losses CSV under {args.root}", file=sys.stderr)
        return 2
    verdicts = S.read_verdicts(args.verdicts)

    # ONE ARM AT A TIME, AT FULL RESOLUTION. The two windows of the table are
    # 1,000 raw steps wide, and `--every` would move the last step of a run off
    # 40,000 and onto 39,991. So each arm is read whole, reduced to its five
    # term series, measured, and only then thinned for the curves. Reading all
    # nine arms first would hold 38 columns of 40,000 rows nine times over.
    curves, rows, track, skipped = {}, [], [], []
    for row in arms:
        files = paths.get(row["arm"], [])
        if not files:
            continue
        run = S.read_run(files, COLUMNS, 1)
        terms = {"align_reduced": align_reduced(run, args.align_weight),
                 "cos_err": mean_cos_err(run)}
        for column, _, _ in PANELS:
            terms.setdefault(column, run.get(column) or [])
        del run
        reached = max((series[-1][0] for series in terms.values() if series),
                      default=0)
        if reached < args.min_steps:
            skipped.append((row["arm"], reached))
            continue
        drawn = {}
        for column, _, _ in PANELS:
            raw = terms[column]
            if not raw:
                continue
            window = 1 if column == "rep_w" else max(1, args.smooth // args.every)
            drawn[column] = S.smooth(raw[::max(1, args.every)], window)
            # The table reads the RAW series. A window mean over 1,000 steps
            # already smooths, and a trailing mean would carry the steps
            # before the window into it.
            last_step = raw[-1][0]
            at_stop = S.window_mean(raw, last_step, SPAN)
            at_30k = (S.window_mean(raw, SLOPE_FROM, SPAN)
                      if reached >= STOP else None)
            change = (None if at_30k is None or at_stop is None
                      else at_stop - at_30k)
            rows.append({
                "arm": row["arm"], "ema": S.schedule_label(row),
                "ema_at_stop": f"{S.momentum_at(row):.3f}",
                "seed": row["seed"], "term": column,
                "reached": reached, "last_step": last_step,
                "value_at_30k": "" if at_30k is None else f"{at_30k:.6f}",
                "value_at_stop": f"{at_stop:.6f}",
                "change_30k_to_40k": ("" if change is None
                                      else f"{change:+.6f}"),
            })
            for step in GRID:
                value = S.window_mean(raw, step, SPAN)
                if value is None:
                    continue
                track.append({
                    "arm": row["arm"], "ema": S.schedule_label(row),
                    "ema_at_stop": f"{S.momentum_at(row):.3f}",
                    "seed": row["seed"], "term": column, "step": step,
                    "value": f"{value:.6f}",
                })
        curves[row["arm"]] = (drawn, S.curve_colour(row["arm"], files, verdicts))
    # No silent cap. A run this figure leaves out is named on stdout, and
    # `results/auc_verdicts.tsv` carries the same runs as `error`.
    for arm, reached in skipped:
        print(f"skipped {arm}: reached step {reached}, under --min-steps "
              f"{args.min_steps}")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.6, 13.5), sharex=True)
    panel_labels = []
    for ax, (column, title, log_y) in zip(axes, PANELS):
        labels = []
        for row in arms:
            if row["arm"] not in curves:
                continue
            drawn, colour = curves[row["arm"]]
            series = drawn.get(column)
            if not series:
                continue
            ax.plot([s for s, _ in series], [v for _, v in series],
                    color=colour, linewidth=1.5)
            labels.append((series, S.curve_label(row), colour))
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(title, fontsize=9)
        S.tidy(ax)
        panel_labels.append((ax, labels))
    # The right labels are laid out per panel, in pixel space, after the draw
    # settles every panel's limits. Two runs that end on one value would print
    # on top of each other otherwise.
    fig.canvas.draw()
    for ax, labels in panel_labels:
        S.label_right(ax, labels, fontsize=7)
    axes[-1].set_xlabel("backbone step")
    axes[0].set_title("Loss by term to the 40,000-step stop",
                      color=S.INK, fontsize=11, loc="left", pad=14)
    # Three colors, three meanings. The arm is the direct label on each line.
    axes[0].plot([], [], color=S.SERIES, linewidth=2.0,
                 label=f"{S.HIGHLIGHT_ARM}, the best arm")
    axes[0].plot([], [], color=S.HELD, linewidth=2.0, label="held the task")
    axes[0].plot([], [], color=S.LOST, linewidth=2.0, label="lost the task")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(right=0.78, hspace=0.18)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(curves)} arm(s)")

    if rows:
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        with open(args.table, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{args.table}: {len(rows)} row(s)")
    if track:
        with open(args.trajectory, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(track[0]))
            w.writeheader()
            w.writerows(track)
        print(f"{args.trajectory}: {len(track)} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
