#!/usr/bin/env python3
"""#393 — the headline comparison: scheduled EMA momentum against fixed 0.9.

The parent reports trained the same ten cells with the EMA momentum held at
0.9 for the whole run (`results/union_parents.csv`). This study rebuilt them
with the momentum raised linearly to 1.0 by step 100k
(`results/ladder_all.csv`). Both halves are matched three ways:

  cell            same recipe and setting;
  align target    a student-align row reads the parent sweep that ran
                  L_align on the student, a teacher-align row reads the
                  retrain that ran it on the teacher;
  stop            bb40k against bb40k, bb100k against bb100k, bb200k
                  against bb200k. Never best against best: the two halves
                  hold different numbers of stops, so best-of-N is biased
                  toward whichever side has more of them.

The parent reports evaluate one head per row and train it on the student
encoder, so the matched column here is this study's student-encoder head.

Delta = scheduled - fixed. Negative means the schedule scores lower, which
is better. The grey band is the pooled head-seed range (scripts/noise_band.py).

Writes `results/schedule_vs_fixed.csv` and `plots/schedule_vs_fixed.png`.

Usage:  python3 scripts/schedule_vs_fixed.py [--out-csv F] [--out-png F]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
RES = os.path.join(EXP_DIR, "results")
sys.path.insert(0, SCRIPTS_DIR)

import noise_band  # noqa: E402

STOPS = [40000, 100000]
BETTER = "#1f4e79"   # schedule lower
WORSE = "#c0504d"    # schedule higher


def cell_name(arm: str) -> str:
    """`arm6_v2_combab` -> `arm6_v2 combab`, the parent table's cell name."""
    head, _, tail = arm.rpartition("_")
    return f"{head} {tail}" if head else arm


def read_scheduled(path: str) -> dict:
    """{(cell, align, stop): student-encoder score} for this study."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["head"] != "student" or not (r.get("gm_rel_mase") or "").strip():
                continue
            align = (r.get("align") or "").strip() or "n/a"
            out[(cell_name(r["arm"]), align, int(r["stop"]))] = float(
                r["gm_rel_mase"])
    return out


def read_fixed(path: str) -> dict:
    """{(cell, align, stop): score} for the fixed-0.9 parent runs."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["cell"], r["align"], int(r["stop"]))] = float(
                r["gm_rel_mase"])
    return out


def pairs(sched: dict, fixed: dict) -> list[dict]:
    rows = []
    for key in sorted(set(sched) & set(fixed), key=lambda k: (k[2], k[0], k[1])):
        cell, align, stop = key
        rows.append({
            "cell": cell,
            "align_target": align,
            "stop": stop,
            "fixed_0p9": f"{fixed[key]:.4f}",
            "scheduled": f"{sched[key]:.4f}",
            "delta": f"{sched[key] - fixed[key]:+.4f}",
        })
    return rows


# α at each stop under the schedule; the parent runs hold 0.9 at every stop.
ALPHA = {40000: 0.94, 100000: 1.00, 200000: 1.00}


def label(row: dict) -> str:
    a = row["align_target"]
    return row["cell"] if a == "n/a" else f"{row['cell']}  (align {a})"


def draw(rows: list[dict], out: str, band: float) -> None:
    panels = [s for s in STOPS if any(r["stop"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.6, 4.6), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    lim = max(abs(float(r["delta"])) for r in rows) * 1.25

    for ax, stop in zip(axes, panels):
        sub = sorted((r for r in rows if r["stop"] == stop),
                     key=lambda r: float(r["delta"]))
        ys = range(len(sub))
        vals = [float(r["delta"]) for r in sub]
        ax.axvspan(-band, band, color="0.85", zorder=0)
        ax.axvline(0, color="0.3", lw=1, zorder=1)
        ax.barh(list(ys), vals, height=0.62, zorder=2,
                color=[BETTER if v < 0 else WORSE for v in vals],
                edgecolor="white", linewidth=0)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([label(r) for r in sub], fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.invert_yaxis()
        ax.set_title(f"backbone {stop // 1000}k, "
                     f"α {ALPHA[stop]:.2f} against 0.90", fontsize=9)
        ax.set_xlabel("GM-Relative MASE, scheduled minus fixed 0.9")
        ax.grid(axis="x", color="0.9", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].text(0.02, 0.02, "← schedule better", transform=axes[0].transAxes,
                 fontsize=8, color=BETTER)
    axes[-1].text(0.98, 0.02, "schedule worse →", transform=axes[-1].transAxes,
                  fontsize=8, color=WORSE, ha="right")
    fig.suptitle("Scheduled EMA momentum against fixed 0.9, matched stop, "
                 f"student encoder (grey = head-seed band ±{band:.4f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--ladder", default=os.path.join(RES, "ladder_all.csv"))
    p.add_argument("--parents", default=os.path.join(RES, "union_parents.csv"))
    p.add_argument("--out-csv", default=os.path.join(RES,
                                                     "schedule_vs_fixed.csv"))
    p.add_argument("--out-png", default=os.path.join(EXP_DIR, "plots",
                                                     "schedule_vs_fixed.png"))
    a = p.parse_args()

    rows = pairs(read_scheduled(a.ladder), read_fixed(a.parents))
    if not rows:
        print("schedule_vs_fixed: no matched pair", file=sys.stderr)
        return 1
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {a.out_csv} ({len(rows)} matched pairs)")

    band = noise_band.pooled_band()
    draw([r for r in rows if r["stop"] in STOPS], a.out_png, band)
    print(f"[out] {a.out_png}")
    for stop in STOPS + [200000]:
        sub = [float(r["delta"]) for r in rows if r["stop"] == stop]
        if sub:
            print(f"bb{stop // 1000}k  n={len(sub)}  "
                  f"mean {sum(sub) / len(sub):+.4f}  "
                  f"lower {sum(1 for v in sub if v < 0)}  "
                  f"outside band {sum(1 for v in sub if abs(v) > band)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
