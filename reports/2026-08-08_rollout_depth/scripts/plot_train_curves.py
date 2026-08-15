#!/usr/bin/env python3
"""#373 — the three training-curve diagnostics of `lalign_teacher.md`.

Rebuilt on this study's cells, k = 3 against k = 0:

  per_run_loss        the training loss. It is NOT comparable across the two
                      depths — a k = 3 loss is the k = 0 objective plus
                      three added terms — so the panel exists to show the
                      shape and to catch a divergence, not to rank the arms.
  cos_error_per_arm   `1 − ff`, the depth-0 forecast error. This one IS
                      comparable: it is the same quantity on both runs.
  dim_usage_per_arm   `u_batchtime` on h_t and on e_t. The card names these
                      as the collapse watch: at k = 3 the f side carries
                      four times its baseline weight against the f-free
                      L_rep and SIGReg terms, and a model can win the deeper
                      terms by flattening f.

Usage: plot_train_curves.py --out-dir plots \\
           --run <cell>:<k>=<losses.csv> [--run ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
from losses_csv import read_by_step, series             # noqa: E402

plt.rcParams.update(cc.rc())
PANELS = [
    ("per_run_loss", [("loss", "training loss")],
     "Training loss"),
    ("cos_error_per_arm", [("ff", "1 − ff")],
     "Depth-0 forecast error, 1 − cos(f_t, h_{t+1})"),
    ("dim_usage_per_arm", [("u_batchtime", "u_batchtime on h_t"),
                           ("u_batchtime_e", "u_batchtime on e_t")],
     "Dimension usage over (batch × time)"),
]

# The collapse floor. `u_batchtime` is capped at 1 and a latent that points
# one way reads `1 / H`. Every backbone here is d_model = 64, so the floor is
# the same number on both latents. This is the collapse-watch figure, so it
# carries the collapse threshold.
D_MODEL = 64
FLOOR = 1.0 / D_MODEL


def smooth(ys, window=50):
    out, run, n = [], 0.0, 0
    for i, v in enumerate(ys):
        run += v
        n += 1
        if n > window:
            run -= ys[i - window]
            n = window
        out.append(run / n)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   metavar="CELL:K=CSV")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    runs = []
    for spec in args.run:
        head, path = spec.split("=", 1)
        cell, k = head.split(":")
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        runs.append((cell, int(k), path))
    if not runs:
        raise SystemExit("ABORT: no run had a losses CSV")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for name, panel, title in PANELS:
        cols = [c for c, _lab in panel]
        # The legend goes in a reserved band under the panels, so the figure
        # grows by the band rather than the band covering the curves.
        ncol = 2 if len(panel) > 1 else 1
        band = 0.21 * -(-len(runs) // ncol) + 0.15
        fig, axes = plt.subplots(1, len(panel),
                                 figsize=(6.2 * len(panel), 4.2 + band),
                                 squeeze=False)
        drew = False
        for ax, (col, ylab) in zip(axes[0], panel):
            for cell, k, path in runs:
                xs, ys = series(read_by_step(path, [col]), col)
                if not xs:
                    continue
                if col == "ff":
                    ys = [1.0 - v for v in ys]
                ax.plot(xs, smooth(ys), color=cc.arm_colour(cell),
                        linestyle=cc.style(k), linewidth=cc.width(cell, 1.7))
                drew = True
            if name == "dim_usage_per_arm":
                ax.axhline(FLOOR, color=cc.PARITY, linewidth=1.2,
                           linestyle=(0, (4, 3)), zorder=0)
                ax.annotate(f"1/H = {FLOOR:.4f}, one direction",
                            (0.99, FLOOR), xycoords=("axes fraction", "data"),
                            ha="right", va="bottom", fontsize=8.5,
                            color=cc.INK_SOFT,
                            bbox=dict(fc="#ffffff", ec="none", pad=0.8))
                # Room under the rule for its own label.
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, FLOOR - 0.06 * (hi - lo)), hi)
            ax.set_xlabel("backbone step")
            ax.set_ylabel(ylab)
        if not drew:
            plt.close(fig)
            print(f"  skip {name}: no run carries {cols}")
            continue
        handles = [Line2D([], [], color=cc.arm_colour(c), linestyle=cc.style(k),
                          linewidth=cc.width(c, 1.7),
                          label=f"{cc.label(c)}  k = {k}"
                                + ("  ✗ retracted" if cc.retracted(c) else ""))
                   for c, k, _p in runs]
        # The legend sits in that band, not inside a panel: eleven entries
        # drawn over a single panel covered the curves they name.
        frac = band / (4.2 + band)
        fig.legend(handles=handles, frameon=False, fontsize=8, ncol=ncol,
                   loc="upper center", bbox_to_anchor=(0.5, frac))
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, frac, 1, 0.96))
        path = out_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
        print(f"wrote {path}")

    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
