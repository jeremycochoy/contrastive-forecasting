#!/usr/bin/env python3
"""The contrastive AUC of every run, against the gate that stops a run.

WHY THIS FIGURE EXISTS. L_rep carries the negatives of this objective. Past
step 10,000 its weight is 0.0, so nothing pushes the representations apart.
The card asks one question of every run: did it keep the contrastive task, and
if not, at which step did it stop?

WHAT IT SHOWS. One line per run. The x axis is the backbone step. The y axis
is the `auc` column of that run's losses CSV, over a trailing mean, because
one step of this trainer is one batch and the raw column is noisy. The dotted
line at 0.55 is the gate `auc_guard.sh` reads: a run whose rolling median
falls under it, and stays under it, lost the task.

Every run carries the same decay and differs in the EMA schedule, so every run
takes one color and a run that lost the task takes the alarm color. The
schedule is a direct label at the right end of each line.

The grey band from step 0 to the end of the decay ramp is where the weight
falls. A run that leaves the band with its AUC held did not lose the task to
the decay.

Usage:
  plot_auc.py --root /home/jupyter/checkpoints_backup/cf-409 \
      --arms scripts/arms.tsv --out plots/auc.png
"""
from __future__ import annotations

import argparse
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="/home/jupyter/checkpoints_backup/cf-409")
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "auc.png"))
    p.add_argument("--verdicts",
                   default=str(HERE.parent / "results" / "auc_verdicts.tsv"))
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--ramp", type=int, default=10000,
                   help="end of the decay ramp, in steps")
    p.add_argument("--smooth", type=int, default=200)
    p.add_argument("--every", type=int, default=10,
                   help="read one row in N. A 40,000-row CSV needs no more")
    p.add_argument("--min-steps", type=int, default=1000,
                   help="a run that reached fewer steps is named, not drawn")
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    paths = S.study_paths(args.root, arms)
    if not paths:
        print(f"no losses CSV under {args.root}", file=sys.stderr)
        return 2
    verdicts = S.read_verdicts(args.verdicts)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axvspan(0, args.ramp, color="#000000", alpha=0.045, linewidth=0)
    ax.axhline(args.threshold, color=S.LOST, linestyle=":", linewidth=1.2)
    ax.annotate(f"the gate, AUC {args.threshold}", (0, args.threshold),
                xytext=(4, -11), textcoords="offset points",
                fontsize=8, color=S.LOST)

    # ONE LINE PER ARM, NOT PER FILE. A leg re-fired after a crash resumes
    # under a `_rN` name and opens a second CSV. Two lines for one arm read as
    # two arms, and both carry the same label. `read_run` stitches them: it
    # keys every row by its step and lets the last row of that step win.
    # `results/auc_verdicts.tsv` stays per FILE, because the gate reads a file.
    drawn = lost = 0
    labels, skipped = [], []
    for row in arms:
        files = paths.get(row["arm"], [])
        if not files:
            continue
        raw = S.read_run(files, ["auc"], args.every)["auc"]
        reached = raw[-1][0] if raw else 0
        if reached < args.min_steps:
            skipped.append((row["arm"], reached))
            continue
        series = S.smooth(raw, max(1, args.smooth // args.every))
        colour = S.run_colour(files, verdicts)
        lost += colour == S.LOST
        ax.plot([s for s, _ in series], [v for _, v in series],
                color=colour, linewidth=1.6)
        labels.append((series, S.curve_label(row), colour))
        drawn += 1
    # No silent cap. A run this figure leaves out is named on stdout, and
    # `results/auc_verdicts.tsv` carries the same runs.
    for arm, reached in skipped:
        print(f"skipped {arm}: reached step {reached}, under --min-steps "
              f"{args.min_steps}")

    if not drawn:
        print("no readable `auc` column", file=sys.stderr)
        return 2

    ax.set_xlabel("backbone step")
    ax.set_ylabel("contrastive AUC (trailing mean)")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(left=0)
    ax.set_title("Does the L_rep decay lose the contrastive task?",
                 color=S.INK, fontsize=11, loc="left")
    ax.annotate("the grey band is the decay ramp", (args.ramp / 2, 0.995),
                xytext=(0, -4), textcoords="offset points",
                fontsize=8, color=S.MUTED, ha="center", va="top")
    S.tidy(ax)
    # The right labels are laid out together, in pixel space, so two runs that
    # end on one value do not print on top of each other. The draw settles the
    # limits the layout reads.
    fig.canvas.draw()
    S.label_right(ax, labels)
    # Two colors, two meanings. The seed is the direct label on each line.
    ax.plot([], [], color=S.SERIES, linewidth=2.0, label="held the task")
    ax.plot([], [], color=S.LOST, linewidth=2.0, label="lost the task")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(right=0.78)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight",
                facecolor=S.SURFACE)
    print(f"{args.out}: {drawn} run(s), {lost} lost the task")
    return 0


if __name__ == "__main__":
    sys.exit(main())
