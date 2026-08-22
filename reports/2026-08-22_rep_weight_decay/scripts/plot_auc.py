#!/usr/bin/env python3
"""The contrastive AUC of every run, against the gate that stops a run.

WHY THIS FIGURE EXISTS. L_rep carries the negatives of this objective. At
weight 0.0 nothing pushes the representations apart. The card asks one
question of every run: did it keep the contrastive task, and if not, at which
step did it stop?

WHAT IT SHOWS. One line per run. The x axis is the backbone step. The y axis
is the `auc` column of that run's losses CSV, over a trailing mean, because
one step of this trainer is one batch and the raw column is noisy. The dotted
line at 0.55 is the gate `auc_guard.sh` reads: a run whose rolling median
falls under it, and stays under it, lost the task.

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
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--ramp", type=int, default=10000,
                   help="end of the decay ramp, in steps")
    p.add_argument("--smooth", type=int, default=200)
    p.add_argument("--every", type=int, default=10,
                   help="read one row in N. A 40,000-row CSV needs no more")
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    paths = S.study_paths(args.root, arms)
    if not paths:
        print(f"no losses CSV under {args.root}", file=sys.stderr)
        return 2

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axvspan(0, args.ramp, color="#000000", alpha=0.045, linewidth=0)
    ax.axhline(args.threshold, color=S.ALARM, linestyle=":", linewidth=1.2)
    ax.annotate(f"the gate, AUC {args.threshold}", (0, args.threshold),
                xytext=(4, -11), textcoords="offset points",
                fontsize=8, color=S.ALARM)

    drawn = 0
    for row in arms:
        for path in paths.get(row["arm"], []):
            series = S.smooth(S.read_csv_column(path, "auc", args.every),
                              max(1, args.smooth // args.every))
            if not series:
                continue
            ax.plot([s for s, _ in series], [v for _, v in series],
                    color=S.arm_colour(row), linestyle=S.arm_style(row),
                    linewidth=1.6, label=S.arm_label(row))
            S.label_right(ax, series, S.arm_label(row), S.arm_colour(row))
            drawn += 1

    if not drawn:
        print("no readable `auc` column", file=sys.stderr)
        return 2

    ax.set_xlabel("backbone step")
    ax.set_ylabel("contrastive AUC (trailing mean)")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(left=0)
    ax.set_title("Does the decay lose the contrastive task?",
                 color=S.INK, fontsize=11, loc="left")
    ax.annotate("the grey band is the decay ramp", (args.ramp, 0.47),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=S.MUTED)
    S.tidy(ax)
    # Under the axis, not on it: a legend of eight arms placed inside covers
    # the curves it names.
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(right=0.80)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight",
                facecolor="#fcfcfb")
    print(f"{args.out}: {drawn} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
