#!/usr/bin/env python3
# #357 τ-sweep plot: x = EMA-target τ, y = mean of (2L/last, 6L/last) GM-Rel MASE.
# no-EMA placed at x=0 on a broken x-axis (gap between 0 and 0.80) so the τ
# discontinuity is visible; the line continues across the break.
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


ARM_TAU = {
    "sigreg_enc3_noema":  0.00,
    "sigreg_enc3_tau080": 0.80,
    "sigreg_enc3_tau090": 0.90,
    "sigreg_enc3_tau098": 0.98,
    "sigreg_enc3_tau099": 0.99,
}


def load_last_avg(gm_csv: Path) -> dict[str, tuple[float, float]]:
    cells: dict[tuple[str, str, str], float] = {}
    with gm_csv.open() as fh:
        for r in csv.DictReader(fh):
            cells[(r["arm"], r["head"], r["ckpt"])] = float(r["gm"])
    out: dict[str, tuple[float, float]] = {}
    for arm in ARM_TAU:
        v2 = cells.get((arm, "2L", "last"))
        v6 = cells.get((arm, "6L", "last"))
        if v2 is None or v6 is None:
            continue
        out[arm] = (ARM_TAU[arm], 0.5 * (v2 + v6))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True, type=Path)
    args = p.parse_args()
    gm = args.report_dir / "results" / "gm_table.csv"
    plots = args.report_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    data = load_last_avg(gm)
    left_pt = (0.0, data["sigreg_enc3_noema"][1])
    right_pts = sorted(
        (data[a] for a in data if a != "sigreg_enc3_noema"),
        key=lambda xy: xy[0],
    )

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.5, 4.5), sharey=True,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    line_color = "#d62728"

    axL.plot([left_pt[0]], [left_pt[1]], marker="o", color=line_color,
             markersize=8, zorder=3)
    axL.annotate(f"{left_pt[1]:.4f}", left_pt, textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8)
    axL.set_xlim(-0.05, 0.05)
    axL.set_xticks([0.0])
    axL.set_xticklabels(["0 (no EMA)"])
    axL.set_ylabel("GM-Rel MASE  (mean of 2L/last, 6L/last)")
    axL.grid(axis="y", alpha=0.3)
    axL.spines["right"].set_visible(False)

    xs = [x for x, _ in right_pts]
    ys = [y for _, y in right_pts]
    axR.plot(xs, ys, marker="o", color=line_color, lw=1.6,
             markersize=8, zorder=3)
    label_offsets = {0.80: (-12, 9), 0.90: (0, 9), 0.98: (-14, 9), 0.99: (14, 9)}
    for x, y in right_pts:
        dx, dy = label_offsets.get(x, (0, 9))
        axR.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(dx, dy), ha="center", fontsize=8)
    axR.set_xlim(0.78, 1.01)
    axR.set_xticks([0.80, 0.90, 0.98, 0.99])
    axR.set_xticklabels(["0.80", "0.90", "0.98", "0.99"], rotation=45, ha="right")
    axR.set_xlabel("EMA-target τ")
    axR.grid(axis="y", alpha=0.3)
    axR.grid(axis="x", alpha=0.3, ls=":")
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", which="both", left=False)
    axR.tick_params(axis="x", labelsize=8)

    # connect no-EMA point across the break to the τ=0.80 point
    first_right = right_pts[0]
    conn = ConnectionPatch(xyA=left_pt, coordsA=axL.transData,
                           xyB=first_right, coordsB=axR.transData,
                           color=line_color, lw=1.6, ls=(0, (3, 3)), zorder=2)
    fig.add_artist(conn)

    # axis break marks
    d = 0.012
    kw = dict(transform=axL.transAxes, color="k", clip_on=False, lw=1)
    axL.plot((1 - d, 1 + d), (-d, +d), **kw)
    axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw = dict(transform=axR.transAxes, color="k", clip_on=False, lw=1)
    axR.plot((-d, +d), (-d, +d), **kw)
    axR.plot((-d, +d), (1 - d, 1 + d), **kw)

    fig.suptitle("τ-sweep: mean GM-Rel MASE over (2L/last, 6L/last) head-matched cells")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.18, wspace=0.05)
    out = plots / "tau_sweep_last_avg.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for arm in ARM_TAU:
        if arm in data:
            print(f"  {arm:25s}  τ={data[arm][0]:.2f}  avg={data[arm][1]:.4f}")


if __name__ == "__main__":
    main()
