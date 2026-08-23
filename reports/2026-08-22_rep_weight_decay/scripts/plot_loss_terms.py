#!/usr/bin/env python3
"""The training loss to 40,000 steps, by term, for every run.

WHY THIS FIGURE EXISTS. The card asks what happens when the weight on L_rep
falls to 0.0. L_rep holds 92 to 93 percent of the total loss and reaches its
level near step 100, so the answer is not in the total. It is in the terms.

THE PANELS. One panel per term, all on the same x axis:

  loss              the total the trainer optimizes, as the CSV logs it.
  rep_w             the live weight on L_rep. It is the treatment.
  l_rep             the UNWEIGHTED L_rep the loss computed. The trainer leaves
                    the cell blank at weight 0.0, where it computes no L_rep,
                    so every run ends with no line.
  L_align, reduced  the align term as the loss holds it. See below.
  mean cos_err      the forecast error over the rollout depths.

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
with longer training. A term that still falls at the stop is headroom. So each
panel prints the change of its own last 10,000 steps, per run, in the table
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
    ("loss", "the total loss", False),
    ("rep_w", "weight on L_rep", False),
    ("l_rep", "L_rep, unweighted", False),
    ("align_reduced", "L_align, reduced", True),
    ("cos_err", "mean cos_err", True),
]
# The rollout depth of this card's cell. `cos_err_d0` thru `cos_err_dk`.
DEPTH = 32


def mean_cos_err(path, every, depth=DEPTH):
    """The mean of `cos_err_d0` thru `cos_err_d<depth>`, step by step."""
    depths = [S.read_csv_column(path, f"cos_err_d{j}", every)
              for j in range(depth + 1)]
    depths = [d for d in depths if d]
    if not depths:
        return []
    n = min(len(d) for d in depths)
    out = []
    for i in range(n):
        step = depths[0][i][0]
        out.append((step, sum(d[i][1] for d in depths) / len(depths)))
    return out


def align_reduced(path, every, align_weight=1.0):
    """The align term as the loss holds it, from the residual of the total.

    Returns `[(step, value), ...]`, and skips a step it cannot close: one
    where `rep_w` is non-zero and `l_rep` is blank. At weight 0.0 the term is
    off and blank is correct, so those steps close with `rep_w * l_rep = 0`.
    """
    def column(name):
        return dict(S.read_csv_column(path, name, every))

    total = S.read_csv_column(path, "loss", every)
    rep_w, l_rep = column("rep_w"), column("l_rep")
    sig_e, sig_h = column("sigreg_e"), column("sigreg_h")
    if not total or align_weight == 0.0:
        return []
    out = []
    for step, value in total:
        w = rep_w.get(step, 0.0)
        if w and step not in l_rep:
            continue
        rep = w * l_rep.get(step, 0.0)
        out.append((step, (value - rep - sig_e.get(step, 0.0)
                           - sig_h.get(step, 0.0)) / align_weight))
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
    p.add_argument("--verdicts",
                   default=str(HERE.parent / "results" / "auc_verdicts.tsv"))
    p.add_argument("--table",
                   default=str(HERE.parent / "results" / "loss_terms_at_stop.csv"))
    p.add_argument("--align-weight", type=float, default=1.0,
                   help="the cell's --align-loss-weight, for the residual")
    p.add_argument("--smooth", type=int, default=500)
    p.add_argument("--every", type=int, default=10)
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    paths = S.study_paths(args.root, arms)
    if not paths:
        print(f"no losses CSV under {args.root}", file=sys.stderr)
        return 2
    verdicts = S.read_verdicts(args.verdicts)

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.6, 12.0), sharex=True)
    rows, panel_labels = [], []
    for ax, (column, title, log_y) in zip(axes, PANELS):
        labels = []
        for row in arms:
            for path in paths.get(row["arm"], []):
                if column == "cos_err":
                    raw = mean_cos_err(path, args.every)
                elif column == "align_reduced":
                    raw = align_reduced(path, args.every, args.align_weight)
                else:
                    raw = S.read_csv_column(path, column, args.every)
                if not raw:
                    continue
                window = 1 if column == "rep_w" else max(
                    1, args.smooth // args.every)
                series = S.smooth(raw, window)
                colour = S.run_colour(path, verdicts)
                ax.plot([s for s, _ in series], [v for _, v in series],
                        color=colour, linewidth=1.5)
                labels.append((series, S.seed_label(row), colour))
                change = slope_per_10k(series)
                rows.append({
                    "arm": row["arm"], "seed": row["seed"],
                    "term": column, "last_step": series[-1][0],
                    "value_at_stop": f"{series[-1][1]:.6f}",
                    "change_over_last_10k": ("" if change is None
                                             else f"{change:+.6f}"),
                })
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
    axes[0].set_title("The loss by term, to the 40,000-step stop",
                      color=S.INK, fontsize=11, loc="left")
    # Two colors, two meanings. The seed is the direct label on each line.
    axes[0].plot([], [], color=S.SERIES, linewidth=2.0, label="held the task")
    axes[0].plot([], [], color=S.LOST, linewidth=2.0, label="lost the task")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(right=0.78, hspace=0.18)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
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
