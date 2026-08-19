#!/usr/bin/env python3
"""#404 deliverable 2 — the training loss of each arm, log scale on both axes.

One curve per arm, from that arm's `<run>_losses.csv`. The four arms train one
objective at four EMA momenta, so a curve that separates early says the
momentum changed what the model optimises, and not only where it ended.

A step-0 row has no place on a log axis, so it is dropped. That row is the
trainer's own first line on some runs, and a crash at redraw time is worse
than one missing point.

The raw curve is drawn faint, and a median over log-spaced bins is drawn on
top of it. At 40,000 rows per arm the four raw traces overlap into one band
and the figure says nothing. The bins are log-spaced because the x axis is:
a fixed-width window would smooth the last decade and leave the first raw.
The median, not the mean, so one diverged step does not move the line.

Usage:
  plot_loss_curves.py --root <checkpoint root> --out plots/loss_curves.png
  plot_loss_curves.py --curve a08=<path to losses.csv> --out ...
  plot_loss_curves.py --root ... --bins 0 --out ...   # raw only
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY_SH = HERE / "study.sh"

import importlib.util


# One colour per arm, shared with every other figure of this study.
def _colours():
    spec = importlib.util.spec_from_file_location(
        "cf404_arm_colours", HERE / "arm_colours.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.colours


arm_colours = _colours()


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


def binned(points, bins):
    """The median of `points` in each of `bins` log-spaced step bins.

    Returns the bin centres and their medians, empty bins dropped. `bins` of
    0 or fewer returns the points unchanged, so `--bins 0` draws the raw
    curve alone.
    """
    if bins <= 0 or len(points) <= bins:
        return points
    lo, hi = points[0][0], points[-1][0]
    span = math.log10(hi) - math.log10(lo)
    if span <= 0:
        return points
    buckets: dict[int, list[float]] = {}
    for step, loss in points:
        i = min(bins - 1, int(bins * (math.log10(step) - math.log10(lo)) / span))
        buckets.setdefault(i, []).append(loss)
    out = []
    for i in sorted(buckets):
        centre = 10 ** (math.log10(lo) + span * (i + 0.5) / bins)
        out.append((centre, statistics.median(buckets[i])))
    return out


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


def draw(series, out, bins=320):
    """Draw the curves and write the figure. Returns (figure, axes)."""
    series = [(arm, points) for arm, points in series if points]
    if not series:
        raise SystemExit("ABORT: no arm has a losses CSV yet — nothing to draw")

    palette = arm_colours([arm for arm, _ in series])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for arm, points in series:
        ax.plot([s for s, _ in points], [v for _, v in points],
                lw=0.7, alpha=0.18, color=palette[arm], zorder=1)
        smooth = binned(points, bins)
        ax.plot([s for s, _ in smooth], [v for _, v in smooth],
                lw=1.6, label=arm, color=palette[arm], zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("backbone step")
    ax.set_ylabel("training loss")
    ax.set_title("#404 — training loss per arm, k = 32, mean over the depth copies")
    if bins > 0:
        ax.text(0.015, 0.02, f"median over {bins} log-spaced bins, raw behind",
                transform=ax.transAxes, fontsize=8, color="0.35")
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
    p.add_argument("--bins", type=int, default=320,
                   help="log-spaced median bins; 0 draws the raw curve alone")
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
    draw(series, args.out, bins=args.bins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
