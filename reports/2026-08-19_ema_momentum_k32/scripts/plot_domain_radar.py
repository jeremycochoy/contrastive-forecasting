#!/usr/bin/env python3
"""#404 deliverable 3 — GM-Relative MASE per domain, one polygon per arm.

The same figure #373 and #401 publish, with the same two references.

  black ring   GM-Relative MASE = 1.0, parity with seasonal naive. A polygon
               inside it beats seasonal naive on that family.
  grey polygon k = 3 at bb40k, the score this card has to beat. It is the
               1.0862 of the card's reference table, split by domain. Four
               arms of one cell that differ in one hyperparameter draw four
               near-equal polygons, so without it a reader cannot see where
               they sit against k = 3.
  break        A domain the arm has no row for. A hole drawn at 1.0 would sit
               exactly on the parity ring, so a reader would take it for a
               real score.

The radial axis is log2, as in #373, so equal multiplicative steps are equal
distances and the parity ring sits at 0.

The k = 32 arm of #401, which is where this sweep starts, has no per-domain
table in this repository. It carries the aggregate row of the table and of
the momentum figure instead.

It reads `results/splits.csv`, which `collect.sh` builds with #373's
`split_scores.py` from each eval's own 97-config CSV, normalised by #379's
committed seasonal-naive denominator. The reference polygon comes from #373's
own committed `results/splits.csv`, written by that same script against that
same denominator, so the arms and the reference are on one scale.

The radius is GM-Relative MASE, so a SMALLER polygon is a better arm. The
domains come from the eval's own `domain` column, never from a list here: a
list would silently drop a domain the benchmark adds.

Usage:
  plot_domain_radar.py --splits results/splits.csv --out plots/domain_radar.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util

HERE = Path(__file__).resolve().parent


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One colour per arm, shared with every other figure of this study.
arm_colours = _module("cf404_arm_colours", "arm_colours.py").colours
REF = _module("cf404_refs", "references.py")

# #373's committed per-domain table, and the row this card compares against.
# `A4_k3_bb40k_student` is the run behind the card's `k = 3, bb40k | 1.0862`,
# so the reader checks the reference polygon against a number the card states.
REFERENCE_SPLITS = HERE.parent.parent / "2026-08-08_rollout_depth" \
    / "results" / "splits.csv"
REFERENCE_KEY = "A4_k3_bb40k_student"

PARITY_INK = "#0b0b0b"
REFERENCE_INK = "#9a9a96"

# The ticks the radial axis can take, in GM-Relative MASE. The figure keeps
# the ones its own data reaches.
TICKS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2)


def arm_of_tag(tag: str) -> str:
    """`a08_bb40k_h30k_student` -> `a08`."""
    return tag.split("_bb", 1)[0]


def _rows(path, key=None):
    """`{name: value}` of the `domain` rows, and the `all` row, for one key."""
    domains, whole = {}, None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if key is not None and r.get("stop") != key:
                continue
            try:
                value = float(r["gm_rel_mase"])
            except (KeyError, ValueError, TypeError):
                continue
            if r.get("split") == "domain":
                domains[r["name"]] = value
            elif r.get("split") == "all":
                whole = value
    return domains, whole


def read_splits(path) -> dict[str, dict[str, float]]:
    """`{arm: {domain: GM-Relative MASE}}` out of collect.sh's splits.csv."""
    out: dict[str, dict[str, float]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "domain":
                continue
            try:
                value = float(r["gm_rel_mase"])
            except (KeyError, ValueError, TypeError):
                continue
            out.setdefault(arm_of_tag(r["stop"]), {})[r["name"]] = value
    return out


def read_reference(path=REFERENCE_SPLITS, key=REFERENCE_KEY) \
        -> dict[str, float]:
    """The per-domain k = 3 score at bb40k, out of #373's own table.

    Returns `{}`, and says why, when the table is missing or when its
    aggregate row is not the card's `K3_BB40K`. A polygon under the wrong key
    is a reference a reader would take for k = 3, so the figure drops it
    rather than draws it.
    """
    if not Path(path).is_file():
        print(f"WARN: no reference table at {path} — the figure carries the "
              f"parity ring alone", file=sys.stderr)
        return {}
    domains, whole = _rows(path, key)
    if not domains:
        print(f"WARN: {path} holds no `domain` row for '{key}'",
              file=sys.stderr)
        return {}
    if whole is None or round(whole, 4) != round(REF.K3_BB40K, 4):
        print(f"WARN: '{key}' scores {whole} over the 97 configs, and this "
              f"card's k = 3 at bb40k is {REF.K3_BB40K:.4f}. The reference "
              f"polygon is dropped.", file=sys.stderr)
        return {}
    return domains


def ticks_for(values) -> list[float]:
    """The tick values the data reaches, always at least two."""
    lo, hi = min(values), max(values)
    kept = [t for t in TICKS if lo / 1.05 <= t <= hi * 1.05]
    if not kept or kept[0] > lo:
        kept.insert(0, round(lo, 3))
    if kept[-1] < hi:
        kept.append(round(hi, 3))
    return kept


def tick_labels(ticks, gap: float = 0.08) -> list[str]:
    """The tick text, with a crowded end tick blanked.

    `ticks_for` puts a tick at the lowest and at the highest value the data
    reaches. That tick can land beside a standard one: 0.772 beside 0.8 draws
    two numbers 4% of the radius apart, and they touch. The end tick then
    keeps its ring and drops its text.
    """
    span = math.log2(ticks[-1]) - math.log2(ticks[0])
    out = [f"{t:g}" for t in ticks]
    if span > 0 and len(ticks) > 2:
        if math.log2(ticks[1] / ticks[0]) < gap * span:
            out[0] = ""
        if math.log2(ticks[-1] / ticks[-2]) < gap * span:
            out[-1] = ""
    return out


def polygon_of(values: dict[str, float], domains) -> list[float]:
    """One arm's radii in log2, over `domains`, and closed on the first one.

    A domain the arm has no row for takes a NaN, which breaks the line there.
    The value that a hole took before was 1.0, and 1.0 is the parity ring, so
    a hole and a score at parity drew the same point.
    """
    radii = [math.log2(values[d]) if d in values else math.nan
             for d in domains]
    return radii + radii[:1]


def draw(by_arm: dict[str, dict[str, float]], out, reference=None):
    """Draw the radar and write the figure. Returns (figure, axes)."""
    if not by_arm:
        raise SystemExit("ABORT: no per-domain row yet — nothing to draw")
    reference = read_reference() if reference is None else reference
    domains = sorted({d for values in by_arm.values() for d in values})
    if not domains:
        raise SystemExit("ABORT: splits.csv holds no `domain` row")

    angles = [2 * math.pi * i / len(domains) for i in range(len(domains))]
    angles.append(angles[0])

    palette = arm_colours(sorted(by_arm))
    fig, ax = plt.subplots(figsize=(6.8, 6.4),
                           subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # The radial axis holds the DATA range, not 0 to max. Every arm of this
    # card sits between about 0.8 and 1.4, so an axis from 0 draws four
    # polygons on top of one another. #373 draws the same figure the same way.
    every = [v for values in by_arm.values() for v in values.values()]
    every += [reference[d] for d in domains if d in reference]
    ticks = ticks_for(every)
    ax.set_ylim(math.log2(ticks[0]) - 0.02, math.log2(ticks[-1]) + 0.02)
    ax.set_yticks([math.log2(t) for t in ticks])
    ax.set_yticklabels(tick_labels(ticks), fontsize=8)

    # Parity with seasonal naive. A polygon inside this ring beats it.
    ax.plot(angles, [0.0] * len(angles), color=PARITY_INK, lw=1.7, zorder=5,
            label="seasonal-naive parity, 1.0")

    # k = 3 at bb40k, the score the card has to beat.
    if all(d in reference for d in domains):
        values = [math.log2(reference[d]) for d in domains]
        ax.plot(angles, values + values[:1], color=REFERENCE_INK, lw=3.4,
                solid_capstyle="round", zorder=2,
                label=f"k = 3 at bb40k ({REF.K3_BB40K:.4f})")
    elif reference:
        print("WARN: the reference table covers only "
              f"{sorted(set(reference) & set(domains))} of {domains} — no "
              f"reference polygon", file=sys.stderr)

    for arm in sorted(by_arm):
        values = polygon_of(by_arm[arm], domains)
        colour = palette[arm]
        ax.plot(angles, values, lw=1.8, label=arm, color=colour, zorder=4)
        holes = [d for d in domains if d not in by_arm[arm]]
        if holes:
            # No fill: a polygon with a break encloses no area. The line alone
            # shows the domains the arm does have.
            print(f"WARN: arm {arm} has no row for {holes} — the polygon "
                  f"breaks there", file=sys.stderr)
        else:
            ax.fill(angles, values, alpha=0.06, color=colour, zorder=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_title("#404 — GM-Relative MASE per domain, bb40k\n"
                 "(radial axis log2, smaller is better)", pad=22)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.24, 1.12))
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out} — {len(by_arm)} arm(s), {len(domains)} domain(s)")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference-splits", default=REFERENCE_SPLITS)
    p.add_argument("--reference-key", default=REFERENCE_KEY)
    args = p.parse_args(argv)
    if not Path(args.splits).is_file():
        raise SystemExit(f"ABORT: no per-domain table at {args.splits}")
    draw(read_splits(args.splits), args.out,
         read_reference(args.reference_splits, args.reference_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
