#!/usr/bin/env python3
"""#393 — how far the latent moves between adjacent checkpoints, per encoder.

Same measure and same figure shape as the parent studies' latent-movement
panel: `1 - cos(h(prev), h(next))` averaged over `(b, t, c)`, against the
training step of the later checkpoint, one line per run. Left panel is the
student encoder, right panel the teacher encoder. Colour is the run and line
style is the `L_align` target, both from `scripts/run_colours.py`.

Reads `results/latent_drift.csv` (written by `collect_latent_drift.py`).
Writes `plots/latent_movement.png`.

Usage: python3 plot_latent_movement.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import run_colours  # noqa: E402
from cell_label import label  # noqa: E402

SRC = EXP / "results" / "latent_drift.csv"
OUT = EXP / "plots" / "latent_movement.png"
RAMP_END = 100_000
PANEL = [("student_h", "student encoder"), ("teacher_h", "teacher encoder")]


def read() -> dict[tuple[str, str], list[tuple[int, float]]]:
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    with open(SRC, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["kind"] != "adjacent":
                continue
            key = (row["cell"], row["latent"])
            series.setdefault(key, []).append(
                (int(row["step"]), float(row["drift_cos"])))
    for pts in series.values():
        pts.sort()
    return series


def main() -> int:
    series = read()
    cells = run_colours.in_order({c for c, _ in series})

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, (latent, title) in zip(axes, PANEL):
        for cell in cells:
            pts = series.get((cell, latent))
            if not pts:
                continue
            ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts],
                    lw=1.8, marker="o", ms=4, alpha=0.9, label=label(cell),
                    **run_colours.line_style(cell))
        ax.axvspan(RAMP_END / 1000, 200, color=run_colours.INK_SOFT,
                   alpha=0.14, zorder=0)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("training step of the later checkpoint (thousands)")
        ax.set_title(f"{title}\nshaded: α = 1.0", fontsize=11)
        ax.grid(True, color=run_colours.GRID, alpha=0.8)
    axes[0].set_ylabel("1 − cos(h previous, h next)")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, fontsize=8, frameon=False, ncol=4,
                     loc="lower center", bbox_to_anchor=(0.5, 0.012),
                     title=run_colours.LEGEND_KEY)
    leg.get_title().set_fontsize(8)
    fig.suptitle("Latent movement between adjacent checkpoints, "
                 "20k steps apart", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.155, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for cell in cells:
        for latent, _ in PANEL:
            pts = [p for p in series.get((cell, latent), []) if p[0] > RAMP_END]
            if pts:
                mean = sum(p[1] for p in pts) / len(pts)
                print(f"  {label(cell):46s} {latent:9s} "
                      f"mean drift after 100k = {mean:.6f}  (n={len(pts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
