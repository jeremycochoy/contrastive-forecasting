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
import matplotlib.ticker

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

# The reader sees the momentum and the schedule, never the internal arm code.
# The map is NOT written here. A hand-written dict named eight arms, and every
# arm added after it fell through to its own internal code in the legend.
# `plot_backbone_health.arms()` builds the label from `arms.tsv`, so one arm
# carries one label on every figure of this study.
sys.path.insert(0, str(HERE))
from plot_backbone_health import arm_rows as _arm_rows  # noqa: E402
from plot_backbone_health import arms as _arms  # noqa: E402

ARM_LABEL = dict(_arms())

# The L_align weight of each arm, from the same table the labels come from.
ALIGN_W = {r["arm"]: r["align_w"] for r in _arm_rows()}


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


def study_schedules() -> dict[str, str]:
    """`{arm: schedule}`, out of study.sh. `{}` when study.sh does not answer.

    A fixed arm and a ramped arm of one alpha run the same momentum until the
    ramp bites. Below step 500 the pair differs by 0.005 of a loss near 13, so
    one curve hides the other. The schedule picks the DASH pattern, so both
    stay visible where they agree, and the figure keeps the momentum figure's
    own language: fixed and ramped read apart at a glance.
    """
    out = subprocess.run(
        ["bash", "-c",
         f'. "{STUDY_SH}" >/dev/null && for a in $CF404_ARMS; do '
         f'printf "%s %s\n" "$a" "$(cf404_schedule "$a")"; done'],
        capture_output=True, text=True)
    pairs = (line.split() for line in out.stdout.splitlines())
    return {p[0]: p[1] for p in pairs if len(p) == 2}


# The dash pattern of each schedule, and what the legend calls it.
DASHES = {"fixed": (4, 2), "ramp": ()}


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


def sync_curves(sync_root, stop=40000) -> list[tuple[str, list]]:
    """One arm's BACKBONE losses CSV out of the sync tree, for every arm.

    `find_curves` walks one checkpoint root, and the arms of this study
    trained on five boxes. It also has to reach past each arm's own eval
    directory, which holds a head losses CSV under the same name pattern.
    `seed_report.auc_series` is the study's one locator of the backbone curve,
    so this figure reads exactly the file the health figure reads.
    """
    kk = stop // 1000
    root = Path(sync_root).expanduser()
    series = []
    for arm in study_arms():
        hits = [h for h in root.glob(f"*/sync/{arm}/*/leg_{kk}k/*_losses.csv")
                if not h.name.endswith(".prev")]
        if hits:
            series.append((arm, read_losses(max(
                hits, key=lambda h: h.stat().st_size))))
    return series


def align_weight(arm: str) -> float:
    """The L_align weight of one arm, out of `arms.tsv`."""
    return ALIGN_W.get(arm, 1.0)


def _panel(ax, series, palette, schedules, bins):
    """Every curve of `series` on one axes. Returns `{arm: line}`."""
    # The solid curves go down first and the dashed ones on top of them. Two
    # arms of one alpha agree to 0.005 below step 500, so whichever goes last
    # hides the other. A dash on top leaves the curve under it in view.
    series = sorted(series, key=lambda s: bool(DASHES.get(
        schedules.get(s[0], ""), ())))
    handles = {}
    for arm, points in series:
        dashes = DASHES.get(schedules.get(arm, ""), ())
        ax.plot([s for s, _ in points], [v for _, v in points],
                lw=0.7, alpha=0.18, color=palette[arm], zorder=1)
        smooth = binned(points, bins)
        line, = ax.plot([s for s, _ in smooth], [v for _, v in smooth],
                        lw=1.6, label=ARM_LABEL.get(arm, arm),
                        color=palette[arm], zorder=2)
        if dashes:
            line.set_dashes(dashes)
        handles[arm] = line
    ax.set_xscale("log")
    ax.set_yscale("log")
    # The loss spans well under one decade, so matplotlib prints every y tick
    # as `1.25 x 10^1`. A reader reads `12.5` faster, so print the number.
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=tuple(x / 10 for x in range(11, 100, 5))))
    ax.set_xlabel("backbone step")
    ax.grid(alpha=0.25, which="both")
    return handles


def draw(series, out, bins=320):
    """Draw the curves and write the figure. Returns (figure, axes).

    TWO PANELS, ONE PER L_align WEIGHT. One arm multiplies the align term by
    3, so its loss sits on another scale and read as the worst run of the
    study on a shared axes — the worst run is the collapsed backbone. A panel
    of its own keeps the arm in the figure and off the other panel's scale.

    The legend sits BELOW the axes. Inside, it covered steps 1 to about 500 on
    every curve, which is the decade the log x axis exists to show.
    """
    series = [(arm, points) for arm, points in series if points]
    if not series:
        raise SystemExit("ABORT: no arm has a losses CSV yet — nothing to draw")

    palette = arm_colours([arm for arm, _ in series])
    schedules = study_schedules()
    base = [s for s in series if align_weight(s[0]) == 1.0]
    other = [s for s in series if align_weight(s[0]) != 1.0]
    panels = [(base, "align weight 1")] + \
             ([(other, "align weight 3")] if other else [])

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.6 * len(panels), 5.2), squeeze=False)
    axes = list(axes[0])
    handles = {}
    for ax, (rows, name) in zip(axes, panels):
        handles.update(_panel(ax, rows, palette, schedules, bins))
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("training loss")
    if bins > 0:
        axes[0].text(0.015, 0.02,
                     f"median over {bins} log-spaced bins, raw behind",
                     transform=axes[0].transAxes, fontsize=8, color="0.35")
    fig.suptitle("The training loss of every arm, at rollout depth 32,\n"
                 "as a mean over the depth copies")
    # The legend keeps the arm order of the study, not the draw order.
    order = sorted(handles)
    fig.legend([handles[a] for a in order],
               [ARM_LABEL.get(a, a) for a in order],
               fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out} — {len(series)} curve(s), {len(panels)} panel(s)")
    return fig, axes[0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", help="the checkpoint root, one directory per arm")
    p.add_argument("--sync-root", help="the sync tree, one directory per box")
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
    if args.sync_root:
        series += sync_curves(args.sync_root)
    if not series:
        raise SystemExit("ABORT: pass --root, --sync-root or one --curve")
    draw(series, args.out, bins=args.bins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
