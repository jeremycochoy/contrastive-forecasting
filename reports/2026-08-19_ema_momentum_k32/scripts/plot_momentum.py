#!/usr/bin/env python3
"""#404 deliverable 1 — GM-Relative MASE against the EMA momentum.

One point per arm, at the arm's alpha at step 0. A fixed arm and a scheduled
arm can share that alpha, so the marker carries the schedule: a filled circle
holds alpha for the whole run, an open square raises it to 1.0 at step
200,000.

Three markers share alpha = 0.9 — a09, s09 and #401's arm — so the marker
alone is not enough. Under one tick the markers take a small x offset, in the
order of the LENGTH of the ramp: the fixed arm first, then #401's 100,000
steps, then this card's 200,000. So the offset carries the schedule instead of
hiding two arms behind one. The tick stays at the true alpha, and no marker
leaves its own tick. Each arm's own label steps up and down as well, because
two arms of one alpha can score within a hair of one another.

The figure also carries what the card compares against.

  grey point   #401's k = 32 arm at bb40k, 1.2082. That arm is already
               computed, and it is where this sweep starts. It is also a cell
               of the sweep: it ran alpha = 0.9 raised to 1.0 at step 100,000,
               between this card's 0.9 fixed and 0.9 raised at 200,000. So it
               takes an x position as well as a level.
  grey band    the k = 3 score at bb40k, 1.0862, plus the repeat spread of
               #373 (0.6% to 1.3%). An arm inside the band is an arm no
               repeat of k = 3 would separate from k = 3.
  dotted       1.0660, the best score of the project, and 1.1637, the best
               score at k = 32.

CAPTION: GM-Relative MASE at 40,000 backbone steps, against the EMA momentum.
Under one momentum the markers step to the right by the length of the ramp:
fixed, then #401's 100,000 steps, then 200,000. The grey point is #401's
k = 32 arm at the same stop. The grey band holds the k = 3 score at bb40k and
the repeat spread of #373. The two dotted lines come from runs that trained to
200,000 steps, and the arms here stop at 40,000, so they are a reminder of the
target and not a fair comparison.

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


# One colour per arm, shared with every other figure of this study.
def _colours():
    spec = importlib.util.spec_from_file_location(
        "cf404_arm_colours", HERE / "arm_colours.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.colours


arm_colours = _colours()

# `fixed` holds alpha for the whole run, `ramp` raises it to 1.0 at 200k.
MARKERS = {"fixed": "o", "ramp": "s"}
SCHEDULE_LABEL = {"fixed": "fixed", "ramp": "ramp to 1.0 at 200k"}

# What `x_positions` calls #401's arm. It is a reference and not an arm of the
# table, so it takes a key no arm name can take.
REF_KEY = "#401"

# The widest a group of markers gets, as a share of the gap to the next
# momentum. Under half, so no marker crosses the midpoint between two ticks
# and reads as the momentum next door.
GROUP_SPAN = 0.40

# The fallback when every arm holds ONE momentum, so there is no gap to take
# a share of.
LONE_DX = 0.012

# Where an arm's own label goes, by its rank under its own momentum. The
# offsets step up and down, because two arms of one momentum can score within
# a hair of one another and one offset then prints two labels on top of each
# other.
LABEL_DY = (6, -14, 20, -28)

# How far the label sits from its marker, in points. A marker in the right
# half of the figure takes the label on its LEFT, or the text runs past the
# frame.
LABEL_DX = 10


def read_scores(path) -> list[dict]:
    """The rows of `collect.sh`'s scores.csv, typed.

    `ramp` is the length of the arm's ramp in steps, and 0 for a fixed arm. A
    table written before that column existed still draws: the arm reads as
    fixed, which is what a missing ramp means.
    """
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({"arm": r["arm"], "alpha": float(r["alpha"]),
                             "schedule": r["schedule"],
                             "ramp": int(float(r.get("ramp") or 0)),
                             "score": float(r["score"])})
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(rows, key=lambda r: (r["alpha"], r["ramp"], r["arm"]))


def _members(rows: list[dict]) -> dict[float, list[str]]:
    """`{alpha: [key, ...]}`, ordered by the length of the ramp.

    #401's arm is a member as well as a reference: it ran alpha = 0.9 raised
    to 1.0 at step 100,000, so it belongs between this card's 0.9 fixed and
    its 0.9 raised at 200,000.
    """
    every = [(r["alpha"], r["ramp"], r["arm"]) for r in rows]
    every.append((REF.K32_BB40K_ALPHA, REF.K32_BB40K_RAMP, REF_KEY))
    out: dict[float, list[str]] = {}
    for alpha, ramp, key in sorted(every):
        out.setdefault(alpha, []).append(key)
    return out


def x_positions(rows: list[dict]) -> dict[str, float]:
    """The x of every marker, by arm name and by `REF_KEY`.

    Every marker of one alpha sits around that alpha, in the order of the
    length of its ramp. The markers keep ONE pitch across the figure, so the
    pitch comes from the widest group. A group of one keeps the alpha itself,
    so #401's arm is at 0.9 whenever it is alone there.
    """
    groups = _members(rows)
    alphas = sorted(groups)
    gaps = [b - a for a, b in zip(alphas, alphas[1:])]
    widest = max(len(v) for v in groups.values())
    dx = (GROUP_SPAN * min(gaps) / max(widest - 1, 1)) if gaps else LONE_DX
    out = {}
    for alpha, keys in groups.items():
        for i, key in enumerate(keys):
            out[key] = alpha + (i - (len(keys) - 1) / 2) * dx
    return out


def label_ranks(rows: list[dict]) -> dict[str, int]:
    """Each arm's rank under its own momentum, which picks its label offset.

    `rows` arrives ordered by (alpha, ramp, arm), so the rank is the order the
    markers take on the x axis.
    """
    out: dict[str, int] = {}
    seen: dict[float, int] = {}
    for r in rows:
        n = seen.get(r["alpha"], 0)
        out[r["arm"]] = n
        seen[r["alpha"]] = n + 1
    return out


def draw(rows: list[dict], out):
    """Draw the figure and write it to `out`. Returns (figure, axes)."""
    if not rows:
        raise SystemExit("ABORT: no arm is scored yet — nothing to draw")

    fig, ax = plt.subplots(figsize=(7.6, 6.4))

    # The band first, so every point sits on top of it.
    lo, hi = REF.band_bounds()
    ilo, ihi = REF.inner_band_bounds()
    ax.axhspan(lo, hi, color="0.85", zorder=0,
               label=f"k = 3 repeat spread, {max(REF.SPREAD):.1%}")
    ax.axhspan(ilo, ihi, color="0.72", zorder=0,
               label=f"k = 3 repeat spread, {min(REF.SPREAD):.1%}")
    ax.axhline(REF.K3_BB40K, color="0.45", lw=1.2, zorder=1,
               label=f"k = 3 at bb40k ({REF.K3_BB40K:.4f})")

    # Every marker's x. Under one tick they spread by the length of the ramp,
    # so a09, s09 and #401's arm, which all hold α = 0.9 at step 0, read as
    # three markers and not as one.
    xs = x_positions(rows)

    # #401's k = 32 arm at the same stop: where this sweep starts. The level
    # spans the figure, so every arm reads against it, and the marker sits at
    # the momentum that arm actually ran.
    ax.axhline(REF.K32_BB40K, color="0.55", lw=1.6, zorder=1,
               label=f"k = 32, mean, student, bb40k ({REF.K32_BB40K:.4f})")
    ax.plot([xs[REF_KEY]], [REF.K32_BB40K], marker="s", ms=11,
            ls="none", color="0.45", mfc="white", mew=2.0, zorder=2,
            label=f"#401, α {REF.K32_BB40K_ALPHA:g} to 1.0 at "
                  f"{REF.K32_BB40K_RAMP // 1000}k")

    # The two 200,000-step scores. Dotted, because the arms stop at 40,000.
    for label, value in REF.dotted_lines():
        ax.axhline(value, color="0.30", lw=1.1, ls=":", zorder=1,
                   label=f"{label} ({value:.4f})")

    palette = arm_colours([r["arm"] for r in rows])
    ranks = label_ranks(rows)
    # The middle of the drawn range. A label to the right of it points left.
    centre = (min(xs.values()) + max(xs.values())) / 2
    seen_schedules = set()
    for r in rows:
        marker = MARKERS.get(r["schedule"], "^")
        label = None
        if r["schedule"] not in seen_schedules:
            seen_schedules.add(r["schedule"])
            label = SCHEDULE_LABEL.get(r["schedule"], r["schedule"])
        x = xs[r["arm"]]
        ax.plot([x], [r["score"]], marker=marker, ms=11, ls="none",
                color=palette[r["arm"]],
                mfc=palette[r["arm"]] if r["schedule"] == "fixed"
                else "white",
                mew=2.0, zorder=3, label=label)
        side = LABEL_DX if x <= centre else -LABEL_DX
        ax.annotate(f"{r['arm']}  {r['score']:.4f}", (x, r["score"]),
                    textcoords="offset points",
                    xytext=(side, LABEL_DY[ranks[r["arm"]] % len(LABEL_DY)]),
                    fontsize=9, ha="left" if side > 0 else "right",
                    color=palette[r["arm"]])

    alphas = sorted({r["alpha"] for r in rows} | {REF.K32_BB40K_ALPHA})
    pad = 0.02 if len(alphas) < 2 else 0.4 * (alphas[-1] - alphas[0])
    ax.set_xlim(min(alphas[0], min(xs.values())) - pad,
                max(alphas[-1], max(xs.values())) + pad)
    ax.set_xticks(alphas)
    ax.set_xlabel("EMA momentum α at step 0")
    ax.set_ylabel("GM-Relative MASE (97 configs)")
    ax.set_title("#404 — EMA momentum at k = 32, L_align on the teacher,\n"
                 "40,000 backbone steps")
    ax.grid(alpha=0.25, zorder=0)
    # The PNG travels without its caption, so it carries the one warning a
    # reader needs: the dotted lines are a target, not a comparison.
    ax.annotate("the dotted lines come from 200,000-step runs\n"
                "the arms here stop at 40,000 steps",
                xy=(0.02, 0.34), xycoords="axes fraction",
                fontsize=8, color="0.30", va="bottom", ha="left")
    # Under the axes, in two columns. Seven entries inside the frame cover an
    # arm's own label whichever corner matplotlib picks.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, framealpha=0.9)
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
