#!/usr/bin/env python3
"""GM-Relative MASE per domain, one row per arm.

A RADAR CAME FIRST AND IT DID NOT WORK. Fourteen arms drew fourteen
near-equal polygons in seven colours that repeated, over a sixteen-row legend,
and no reader could map a polygon to a row. A grid gives every arm its own row
and its own printed number.

  columns    the domains the benchmark reports, then the whole 97 configs
  rows       every arm, best first, under the k = 3 score at the same stop
  colour     diverging around 1.0, which is parity with seasonal naive. Blue
             beats seasonal naive on that domain, red does not.

It reads `results/splits.csv`, which `collect.sh` builds with #373's
`split_scores.py` from each eval's own 97-config CSV, normalised by #379's
committed seasonal-naive denominator. The reference row comes from #373's own
committed `results/splits.csv`, written by that same script against that same
denominator, so the arms and the reference are on one scale.

Usage:
  plot_domain_grid.py --splits results/splits.csv --out plots/domain_grid.png
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


# The label the reader sees for one arm. It names the momentum and the
# schedule, never the internal arm code. `plot_backbone_health.arms()` builds
# it from `arms.tsv`, so one arm carries one label on every figure.
sys.path.insert(0, str(HERE))
from plot_backbone_health import arms as _arms  # noqa: E402

ARM_LABEL = dict(_arms())
REF = _module("cf404_refs", "references.py")

# #373's committed per-domain table, and the row this card compares against.
# `A4_k3_bb40k_student` is the run behind the card's `k = 3, bb40k | 1.0862`,
# so the reader checks the reference polygon against a number the card states.
REFERENCE_SPLITS = HERE.parent.parent / "2026-08-08_rollout_depth" \
    / "results" / "splits.csv"
REFERENCE_KEY = "A4_k3_bb40k_student"



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


ALL = "all 97 configs"


def read_splits(path) -> dict[str, dict[str, float]]:
    """`{arm: {domain: GM-Relative MASE}}` out of collect.sh's splits.csv.

    The `all` row comes in under `ALL`, so the grid can carry the aggregate
    beside the domains and a reader can tie a row to the ranking figure.
    """
    out: dict[str, dict[str, float]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            split = r.get("split")
            if split not in ("domain", "all"):
                continue
            try:
                value = float(r["gm_rel_mase"])
            except (KeyError, ValueError, TypeError):
                continue
            key = ALL if split == "all" else r["name"]
            out.setdefault(arm_of_tag(r["stop"]), {})[key] = value
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
              f"row is dropped.", file=sys.stderr)
        return {}
    domains[ALL] = whole
    return domains


def draw(by_arm: dict[str, dict[str, float]], out, reference=None):
    """Draw the grid and write the figure. Returns (figure, axes)."""
    if not by_arm:
        raise SystemExit("ABORT: no per-domain row yet — nothing to draw")
    reference = read_reference() if reference is None else reference
    domains = sorted({d for values in by_arm.values() for d in values
                      if d != ALL})
    if not domains:
        raise SystemExit("ABORT: splits.csv holds no `domain` row")
    columns = domains + [ALL]

    # Best arm at the top, by the score the ranking figure orders on. An arm
    # with no aggregate row sorts last rather than crashing the redraw.
    order = sorted(by_arm, key=lambda a: by_arm[a].get(ALL, math.inf))
    rows = [(ARM_LABEL.get(a, a), by_arm[a]) for a in order]
    if reference:
        rows.insert(0, (f"k = 3, same 40,000 steps", reference))

    grid = [[values.get(c, math.nan) for c in columns] for _, values in rows]

    fig, ax = plt.subplots(figsize=(1.15 * len(columns) + 5.4,
                                    0.42 * len(rows) + 2.2))
    # The colour is symmetric in log2 around parity, so 0.5x and 2x sit the
    # same distance from the middle and the middle IS 1.0.
    #
    # The SPAN IS A PERCENTILE, NOT THE MAXIMUM. One cell of the collapsed
    # backbone reaches 3.90, and a span set by it washed the other 111 cells
    # to near white. Cells past the span take the end colour, and every cell
    # prints its own number, so nothing is lost.
    finite = sorted(abs(math.log2(v)) for row in grid for v in row if v == v)
    span = max(finite[int(0.99 * (len(finite) - 1))], 0.3)
    image = [[math.log2(v) if v == v else math.nan for v in row]
             for row in grid]
    mesh = ax.imshow(image, cmap="RdBu_r", vmin=-span, vmax=span,
                     aspect="auto")

    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v != v:
                continue
            shade = "white" if abs(math.log2(v)) > 0.62 * span else "0.10"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=shade)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows], fontsize=9)
    # The aggregate is not a domain, so a line separates it.
    ax.axvline(len(domains) - 0.5, color="white", linewidth=3)
    if reference:
        ax.axhline(0.5, color="white", linewidth=3)
    ax.set_xticks([x - 0.5 for x in range(1, len(columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_title("GM-Relative MASE per domain, at 40,000 backbone steps")
    shown = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.7,
                         1.9)
             if abs(math.log2(t)) <= span]
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, extend="both",
                       ticks=[math.log2(t) for t in shown])
    bar.ax.set_yticklabels([f"{t:g}" for t in shown])
    bar.set_label("GM-Relative MASE, 1.0 is seasonal-naive parity",
                  fontsize=9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out} — {len(rows)} row(s), {len(columns)} column(s)")
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
