#!/usr/bin/env python3
"""The training loss to 40,000 steps, by term, for every arm.

WHY THIS FIGURE EXISTS. The card asks what happens when the weight on L_rep
falls to a floor. L_rep holds 92 to 93 percent of the total loss and reaches
its level near step 100, so the answer is not in the total. It is in the terms.

THE PANELS. One panel per term, all on the same x axis:

  rep_w                the live weight on L_rep. It is the treatment, so the
                       figure shows it first: a reader sees which arm decays
                       and how far before reading any loss.
  l_rep                the UNWEIGHTED L_rep the loss computed. The trainer
                       leaves the cell blank at weight 0.0, where it computes
                       no L_rep, so those arms end with no line.
  l_align              the UNWEIGHTED L_align, which is the depth-0 copy.
  mean cos_err_d0..d3  the forecast error over the four rollout depths. This
                       card runs k = 3, so `l_align` alone is one depth of
                       four. A reader who wants the forecast error reads this
                       panel, not the `l_align` one.

THE SLOPE. The card's second question asks which backbone can improve more
with longer training. A term that still falls at the stop is headroom. So each
panel prints the slope of its own last 10,000 steps, per arm, in the table
`results/loss_terms_at_stop.csv`.

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
    ("rep_w", "the weight on L_rep", False),
    ("l_rep", "L_rep, unweighted", False),
    ("l_align", "L_align, unweighted", True),
    ("cos_err", "mean forecast error, d0 to d3", True),
]


def mean_cos_err(path, every):
    """The mean of `cos_err_d0` thru `cos_err_d3`, step by step."""
    depths = [S.read_csv_column(path, f"cos_err_d{j}", every) for j in range(4)]
    depths = [d for d in depths if d]
    if not depths:
        return []
    n = min(len(d) for d in depths)
    out = []
    for i in range(n):
        step = depths[0][i][0]
        out.append((step, sum(d[i][1] for d in depths) / len(depths)))
    return out


def slope_per_10k(series, span=10000):
    """The change of one term over its last `span` steps.

    A negative value is a term that still falls at the stop, which is
    headroom. A value near zero is a term that stopped moving.
    """
    if len(series) < 4:
        return None
    last_step = series[-1][0]
    window = [(s, v) for s, v in series if s >= last_step - span]
    if len(window) < 4:
        return None
    head = window[:max(2, len(window) // 5)]
    tail = window[-max(2, len(window) // 5):]
    a = sum(v for _, v in head) / len(head)
    b = sum(v for _, v in tail) / len(tail)
    return b - a


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="/home/jupyter/checkpoints_backup/cf-409")
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "loss_terms.png"))
    p.add_argument("--table",
                   default=str(HERE.parent / "results" / "loss_terms_at_stop.csv"))
    p.add_argument("--smooth", type=int, default=500)
    p.add_argument("--every", type=int, default=10)
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    paths = S.study_paths(args.root, arms)
    if not paths:
        print(f"no losses CSV under {args.root}", file=sys.stderr)
        return 2

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.6, 10.0), sharex=True)
    rows = []
    for ax, (column, title, log_y) in zip(axes, PANELS):
        for row in arms:
            for path in paths.get(row["arm"], []):
                if column == "cos_err":
                    raw = mean_cos_err(path, args.every)
                else:
                    raw = S.read_csv_column(path, column, args.every)
                if not raw:
                    continue
                window = 1 if column == "rep_w" else max(1, args.smooth // args.every)
                series = S.smooth(raw, window)
                ax.plot([s for s, _ in series], [v for _, v in series],
                        color=S.arm_colour(row), linestyle=S.arm_style(row),
                        linewidth=1.5, label=S.arm_label(row))
                S.label_right(ax, series, S.arm_label(row), S.arm_colour(row),
                              fontsize=7)
                rows.append({
                    "arm": row["arm"], "rep_end": row["rep_end"],
                    "seed": row["seed"], "align_target": row["align_target"],
                    "term": column, "last_step": series[-1][0],
                    "value_at_stop": f"{series[-1][1]:.6f}",
                    "change_over_last_10k": ("" if slope_per_10k(series) is None
                                             else f"{slope_per_10k(series):+.6f}"),
                })
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(title, fontsize=9)
        S.tidy(ax)
    axes[-1].set_xlabel("backbone step")
    axes[0].set_title("The loss by term, to the 40,000-step stop",
                      color=S.INK, fontsize=11, loc="left")
    # Under the panels, not inside one: a legend of eight arms placed on an
    # axis covers the curves it names.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(right=0.80, hspace=0.16)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"{args.out}: {len(paths)} arm(s)")

    if rows:
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        with open(args.table, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{args.table}: {len(rows)} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
