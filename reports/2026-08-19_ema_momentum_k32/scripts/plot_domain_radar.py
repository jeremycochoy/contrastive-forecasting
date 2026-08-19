#!/usr/bin/env python3
"""#404 deliverable 3 — GM-Relative MASE per domain, one polygon per arm.

The same figure #373 and #401 publish. It reads `results/splits.csv`, which
`collect.sh` builds with #373's `split_scores.py` from each eval's own
97-config CSV, normalised by #379's committed seasonal-naive denominator.

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

# One colour per arm, shared with every other figure of this study.
def _colours():
    spec = importlib.util.spec_from_file_location(
        "cf404_arm_colours", HERE / "arm_colours.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.colours


arm_colours = _colours()

# How much room to leave outside the data, as a share of its own range.
PAD = 0.15


def arm_of_tag(tag: str) -> str:
    """`a08_bb40k_h30k_student` -> `a08`."""
    return tag.split("_bb", 1)[0]


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


def draw(by_arm: dict[str, dict[str, float]], out):
    """Draw the radar and write the figure. Returns (figure, axes)."""
    if not by_arm:
        raise SystemExit("ABORT: no per-domain row yet — nothing to draw")
    domains = sorted({d for values in by_arm.values() for d in values})
    if not domains:
        raise SystemExit("ABORT: splits.csv holds no `domain` row")

    angles = [2 * math.pi * i / len(domains) for i in range(len(domains))]
    angles.append(angles[0])

    palette = arm_colours(sorted(by_arm))
    fig, ax = plt.subplots(figsize=(6.8, 6.4),
                           subplot_kw={"projection": "polar"})
    for arm in sorted(by_arm):
        values = [by_arm[arm].get(d, float("nan")) for d in domains]
        values.append(values[0])
        colour = palette[arm]
        ax.plot(angles, values, lw=1.6, label=arm, color=colour)
        ax.fill(angles, values, alpha=0.08, color=colour)

    # The radial axis holds the DATA range, not 0 to max. Every arm of this
    # card sits between about 1.0 and 1.3, so an axis from 0 draws four
    # polygons on top of one another. #373 and #401 draw the same figure the
    # same way. The innermost ring carries its own value, so a reader sees
    # where the axis starts.
    every = [v for values in by_arm.values() for v in values.values()]
    lo, hi = min(every), max(every)
    pad = PAD * (hi - lo) if hi > lo else 0.05 * hi
    ax.set_ylim(max(0.0, lo - pad), hi + pad)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_title("#404 — GM-Relative MASE per domain, bb40k\n"
                 "(smaller is better)", pad=22)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.18, 1.10))
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(by_arm)} arm(s), {len(domains)} domain(s)")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if not Path(args.splits).is_file():
        raise SystemExit(f"ABORT: no per-domain table at {args.splits}")
    draw(read_splits(args.splits), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
