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
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402

plt.rcParams.update(cc.rc())
PANELS = [
    ("per_run_loss", [("loss", "training loss")],
     "Training loss (k = 3 is the k = 0 objective plus three added terms)"),
    ("cos_error_per_arm", [("ff", "1 − ff")],
     "Depth-0 forecast error, 1 − cos(f_t, h_{t+1})"),
    ("dim_usage_per_arm", [("u_batchtime", "u_batchtime on h_t"),
                           ("u_batchtime_e", "u_batchtime on e_t")],
     "Dimension usage over (batch × time)"),
]


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


def read(path, cols):
    got = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                step = float(r["step"])
            except (KeyError, ValueError):
                continue
            for c in cols:
                v = r.get(c)
                if v not in (None, ""):
                    try:
                        got[c].append((step, float(v)))
                    except ValueError:
                        pass
    return got


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

    for name, series, title in PANELS:
        cols = [c for c, _lab in series]
        fig, axes = plt.subplots(1, len(series),
                                 figsize=(6.2 * len(series), 4.2),
                                 squeeze=False)
        drew = False
        for ax, (col, ylab) in zip(axes[0], series):
            for cell, k, path in runs:
                pts = read(path, [col]).get(col, [])
                if not pts:
                    continue
                xs = [s for s, _v in pts]
                ys = [v for _s, v in pts]
                if col == "ff":
                    ys = [1.0 - v for v in ys]
                ax.plot(xs, smooth(ys), color=cc.colour(cell),
                        linestyle=cc.style(k), linewidth=1.7)
                drew = True
            ax.set_xlabel("backbone step")
            ax.set_ylabel(ylab)
        if not drew:
            plt.close(fig)
            print(f"  skip {name}: no run carries {cols}")
            continue
        handles = [Line2D([], [], color=cc.colour(c), linestyle=cc.style(k),
                          label=f"{cc.label(c)}  k = {k}")
                   for c, k, _p in runs]
        axes[0][-1].legend(handles=handles, frameon=False, fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
        print(f"wrote {path}")

    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
