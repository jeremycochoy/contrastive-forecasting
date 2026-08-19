#!/usr/bin/env python3
"""#404 deliverable 1 — GM-Relative MASE against the EMA momentum.

One point per arm, at the arm's alpha at step 0. A fixed arm and a scheduled
arm can share that alpha, so the marker carries the schedule: a filled circle
holds alpha for the whole run, an open square raises it to 1.0 at step
200,000.

The figure also carries what the card compares against.

  grey point   #401's k = 32 arm at bb40k, 1.2082. That arm is already
               computed, and it is where this sweep starts.
  grey band    the k = 3 score at bb40k, 1.0862, plus the repeat spread of
               #373 (0.6% to 1.3%). An arm inside the band is an arm no
               repeat of k = 3 would separate from k = 3.
  dotted       1.0660, the best score of the project, and 1.1637, the best
               score at k = 32.

CAPTION: GM-Relative MASE at 40,000 backbone steps, against the EMA momentum.
The grey point is #401's k = 32 arm at the same stop. The grey band holds the
k = 3 score at bb40k and the repeat spread of #373. The two dotted lines come
from runs that trained to 200,000 steps, and the arms here stop at 40,000, so
they are a reminder of the target and not a fair comparison.

Usage:
  plot_momentum.py --scores results/scores.csv --out plots/momentum.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def _references():
    spec = importlib.util.spec_from_file_location(
        "cf404_refs", HERE / "references.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _references()

# One colour per arm, stable across every figure of this study.
COLOURS = {"a08": "#1f77b4", "a09": "#d62728",
           "s08": "#2ca02c", "s09": "#9467bd"}
FALLBACK = "#7f7f7f"

# `fixed` holds alpha for the whole run, `ramp` raises it to 1.0 at 200k.
MARKERS = {"fixed": "o", "ramp": "s"}
SCHEDULE_LABEL = {"fixed": "fixed", "ramp": "ramp to 1.0 at 200k"}


def read_scores(path) -> list[dict]:
    """The rows of `collect.sh`'s scores.csv, typed."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({"arm": r["arm"], "alpha": float(r["alpha"]),
                             "schedule": r["schedule"],
                             "score": float(r["score"])})
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(rows, key=lambda r: (r["alpha"], r["schedule"], r["arm"]))


def draw(rows: list[dict], out):
    """Draw the figure and write it to `out`. Returns (figure, axes)."""
    if not rows:
        raise SystemExit("ABORT: no arm is scored yet — nothing to draw")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # The band first, so every point sits on top of it.
    lo, hi = REF.band_bounds()
    ilo, ihi = REF.inner_band_bounds()
    ax.axhspan(lo, hi, color="0.85", zorder=0,
               label=f"k = 3 at bb40k, repeat spread {max(REF.SPREAD):.1%}")
    ax.axhspan(ilo, ihi, color="0.72", zorder=0)
    ax.axhline(REF.K3_BB40K, color="0.45", lw=1.2, zorder=1)

    # #401's k = 32 arm at the same stop: where this sweep starts.
    ax.axhline(REF.K32_BB40K, color="0.55", lw=1.6, zorder=1,
               label=f"k = 32, mean, student, bb40k ({REF.K32_BB40K:.4f})")

    # The two 200,000-step scores. Dotted, because the arms stop at 40,000.
    for label, value in REF.dotted_lines():
        ax.axhline(value, color="0.30", lw=1.1, ls=":", zorder=1,
                   label=f"{label} ({value:.4f})")

    seen_schedules = set()
    for r in rows:
        marker = MARKERS.get(r["schedule"], "^")
        label = None
        if r["schedule"] not in seen_schedules:
            seen_schedules.add(r["schedule"])
            label = SCHEDULE_LABEL.get(r["schedule"], r["schedule"])
        ax.plot([r["alpha"]], [r["score"]], marker=marker, ms=11, ls="none",
                color=COLOURS.get(r["arm"], FALLBACK),
                mfc=COLOURS.get(r["arm"], FALLBACK) if r["schedule"] == "fixed"
                else "white",
                mew=2.0, zorder=3, label=label)
        ax.annotate(f"{r['arm']}  {r['score']:.4f}",
                    (r["alpha"], r["score"]), textcoords="offset points",
                    xytext=(10, 4), fontsize=9,
                    color=COLOURS.get(r["arm"], FALLBACK))

    alphas = sorted({r["alpha"] for r in rows})
    pad = 0.02 if len(alphas) < 2 else 0.4 * (alphas[-1] - alphas[0])
    ax.set_xlim(alphas[0] - pad, alphas[-1] + pad)
    ax.set_xticks(alphas)
    ax.set_xlabel("EMA momentum α at step 0")
    ax.set_ylabel("GM-Relative MASE (97 configs)")
    ax.set_title("#404 — EMA momentum at k = 32, L_align on the teacher,\n"
                 "40,000 backbone steps")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(rows)} arm(s)")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if not Path(args.scores).is_file():
        raise SystemExit(f"ABORT: no scores table at {args.scores}")
    draw(read_scores(args.scores), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
