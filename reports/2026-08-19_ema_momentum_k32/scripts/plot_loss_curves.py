#!/usr/bin/env python3
"""#404 deliverable 2 — the training loss of each arm, log scale on both axes.

One curve per arm, from that arm's `<run>_losses.csv`. The four arms train one
objective at four EMA momenta, so a curve that separates early says the
momentum changed what the model optimises, and not only where it ended.

A step-0 row has no place on a log axis, so it is dropped. That row is the
trainer's own first line on some runs, and a crash at redraw time is worse
than one missing point.

Usage:
  plot_loss_curves.py --root <checkpoint root> --out plots/loss_curves.png
  plot_loss_curves.py --curve a08=<path to losses.csv> --out ...
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY_SH = HERE / "study.sh"

# The palette of plot_momentum.py, so one arm is one colour everywhere.
COLOURS = {"a08": "#1f77b4", "a09": "#d62728",
           "s08": "#2ca02c", "s09": "#9467bd"}
FALLBACK = "#7f7f7f"


def read_losses(path) -> list[tuple[int, float]]:
    """`(step, loss)` for the rows a log axis can hold."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                step, loss = int(r["step"]), float(r["loss"])
            except (KeyError, ValueError, TypeError):
                continue
            if step > 0 and loss > 0:
                out.append((step, loss))
    return sorted(out)


def study_arms() -> list[str]:
    """The arm list, from the one place that holds it."""
    out = subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && printf "%s" "$CF404_ARMS"'],
        capture_output=True, text=True)
    return out.stdout.split()


def find_curves(root) -> list[tuple[str, list[tuple[int, float]]]]:
    """One arm's losses CSV under `root`, for every arm that has one.

    A re-fired leg writes a second CSV under train.py's `_rN` infix
    (`safe_run_name`), so an arm can hold more than one. The LONGEST file
    wins: it is the run that trained furthest, and sort order would return
    the leg that died at step 200 just as readily.
    """
    series = []
    for arm in study_arms():
        found = sorted(Path(root, arm).rglob("*_losses.csv"),
                       key=lambda p: p.stat().st_size, reverse=True)
        if found:
            series.append((arm, read_losses(found[0])))
    return series


def draw(series, out):
    """Draw the curves and write the figure. Returns (figure, axes)."""
    series = [(arm, points) for arm, points in series if points]
    if not series:
        raise SystemExit("ABORT: no arm has a losses CSV yet — nothing to draw")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for arm, points in series:
        steps = [s for s, _ in points]
        losses = [v for _, v in points]
        ax.plot(steps, losses, lw=1.4, label=arm,
                color=COLOURS.get(arm, FALLBACK))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("backbone step")
    ax.set_ylabel("training loss")
    ax.set_title("#404 — training loss per arm, k = 32, mean over the depth copies")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(series)} curve(s)")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", help="the checkpoint root, one directory per arm")
    p.add_argument("--curve", action="append", default=[],
                   metavar="ARM=CSV", help="one arm's losses CSV, explicitly")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    series = []
    for spec in args.curve:
        if "=" not in spec:
            raise SystemExit(f"ABORT: --curve wants ARM=CSV, got {spec!r}")
        arm, path = spec.split("=", 1)
        series.append((arm, read_losses(path)))
    if args.root:
        series += find_curves(args.root)
    if not series:
        raise SystemExit("ABORT: pass --root or at least one --curve")
    draw(series, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
