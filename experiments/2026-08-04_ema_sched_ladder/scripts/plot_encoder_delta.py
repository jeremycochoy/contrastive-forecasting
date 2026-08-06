#!/usr/bin/env python3
"""#393 — the second question: student encoder or teacher encoder?

Every checkpoint carries two heads, each trained and evaluated on its own
encoder. This draws the paired difference at every stop that carries both:

    x = teacher-encoder GM-Relative MASE - student-encoder GM-Relative MASE

Left of 0 means the teacher encoder scores lower, which is better. The grey
band is the pooled head-seed range (scripts/noise_band.py), so a dot inside
it is a difference the head seed alone can produce.

One row per run, in the report's run order, each in its run colour
(`scripts/run_colours.py`). The marker shape is the backbone stop.

Reads `results/ladder_all.csv`. Writes `plots/encoder_delta.png`.

Usage:  python3 scripts/plot_encoder_delta.py [--ladder FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import noise_band  # noqa: E402
import run_colours  # noqa: E402
from cell_label import label as cell_label  # noqa: E402

INK = run_colours.INK
INK_SOFT = run_colours.INK_SOFT
STOP_MARKER = {40000: "o", 100000: "s", 200000: "^"}


def read_pairs(path: str) -> list[tuple[str, int, float]]:
    """(run slug, stop, teacher - student) for every stop carrying both."""
    by_key: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("gm_rel_mase") or "").strip():
                continue
            by_key.setdefault((r["cell"], int(r["stop"])), {})[r["head"]] = \
                float(r["gm_rel_mase"])
    out = []
    for (cell, stop), heads in by_key.items():
        if "student" in heads and "teacher" in heads:
            out.append((cell, stop, heads["teacher"] - heads["student"]))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def draw(pairs: list[tuple[str, int, float]], path: str, band: float) -> None:
    runs = run_colours.in_order({c for c, _, _ in pairs})
    ypos = {c: i for i, c in enumerate(runs)}

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.axvspan(-band, band, color=INK_SOFT, alpha=0.15, zorder=0)
    for edge in (-band, band):
        ax.axvline(edge, color=INK_SOFT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)

    for cell, stop, d in pairs:
        ax.plot([d], [ypos[cell]], STOP_MARKER.get(stop, "o"), ms=8,
                color=run_colours.colour(cell),
                mec="white", mew=1.1, zorder=3)

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([cell_label(c) for c in runs], fontsize=8)
    for tick, cell in zip(ax.get_yticklabels(), runs):
        tick.set_color(run_colours.colour(cell))
    ax.invert_yaxis()
    lim = max(abs(d) for _, _, d in pairs) * 1.3
    ax.set_xlim(-lim, max(lim, band * 1.3))
    ax.set_xlabel("GM-Relative MASE, teacher encoder minus student encoder")
    ax.set_title("Teacher encoder against student encoder, paired at every "
                 "stop that carries both heads", fontsize=11)
    ax.grid(alpha=0.22, axis="x")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.01, -0.16, "← teacher encoder lower", transform=ax.transAxes,
            fontsize=8, color=INK)
    ax.text(0.99, -0.16, "student encoder lower →", transform=ax.transAxes,
            fontsize=8, color=INK, ha="right")

    handles = [Line2D([], [], marker=STOP_MARKER[s], lw=0, ms=7, color=INK,
                      mec="white", mew=1.1, label=f"backbone {s // 1000}k")
               for s in sorted(STOP_MARKER) if any(st == s for _, st, _ in pairs)]
    handles.append(Line2D([], [], color=INK_SOFT, lw=7, alpha=0.35,
                          label=f"±{band:.4f}, {noise_band.LABEL}"))
    fig.legend(handles=handles, fontsize=8, ncol=4, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    res = os.path.join(EXP_DIR, "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--ladder", default=os.path.join(res, "ladder_all.csv"))
    p.add_argument("--out", default=os.path.join(EXP_DIR, "plots",
                                                 "encoder_delta.png"))
    a = p.parse_args()

    pairs = read_pairs(a.ladder)
    if not pairs:
        print(f"plot_encoder_delta: no paired stop in {a.ladder}",
              file=sys.stderr)
        return 1
    band = noise_band.pooled_band()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    draw(pairs, a.out, band)
    lower = sum(1 for _, _, d in pairs if d < 0)
    print(f"[out] {a.out}  ({len(pairs)} paired stops, teacher lower at "
          f"{lower}, max |delta| {max(abs(d) for _, _, d in pairs):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
