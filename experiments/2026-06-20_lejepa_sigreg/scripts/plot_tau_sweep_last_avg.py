#!/usr/bin/env python3
# #357 τ-sweep plot: x = EMA-target τ (with no-EMA at τ=0), y = mean of
# (2L/last, 6L/last) GM-Rel MASE. Five B=512 SIGReg arms on one connected line.
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    pts = sorted((data[a] for a in data), key=lambda xy: xy[0])
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(xs, ys, marker="o", color="#d62728", lw=1.6, markersize=8, zorder=3)
    label_offsets = {0.00: (10, 9), 0.80: (-10, 9), 0.90: (0, 9), 0.98: (-14, 9), 0.99: (14, 9)}
    for x, y in pts:
        dx, dy = label_offsets.get(x, (0, 9))
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=8)
    ax.set_xticks([0.00, 0.80, 0.90, 0.98, 0.99])
    ax.set_xticklabels(["0 (no EMA)", "0.80", "0.90", "0.98", "0.99"], rotation=45, ha="right")
    ax.set_xlabel("EMA-target τ")
    ax.set_ylabel("GM-Rel MASE  (mean of 2L/last, 6L/last)")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.3, ls=":")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("τ-sweep: mean GM-Rel MASE over (2L/last, 6L/last) head-matched cells")
    fig.tight_layout()
    out = plots / "tau_sweep_last_avg.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for arm in ARM_TAU:
        if arm in data:
            print(f"  {arm:25s}  τ={data[arm][0]:.2f}  avg={data[arm][1]:.4f}")


if __name__ == "__main__":
    main()
