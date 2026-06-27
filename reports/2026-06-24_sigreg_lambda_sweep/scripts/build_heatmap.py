#!/usr/bin/env python3
# #363 SIGReg λ-sweep — 2D heatmap of GM-Rel MASE over (λ_e, λ_h).
#
# 4 panels (2L/best, 2L/last, 6L/best, 6L/last). X axis = λ_e on a log grid
# {0.1, 1.0, 10.0, 100.0}; Y axis = λ_h on a log grid {0.1, 1.0, 10.0}.
# Cell colour = GM-Rel MASE on a diverging colormap centred on the per-cell
# #359 anchor (1.0, 0.1) so red = worse than #359, blue = better. Cell text
# = GM value. Un-run grid points are drawn with a hatched grey background.
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

LAMBDA_E = [0.1, 1.0, 10.0, 100.0]
LAMBDA_H = [0.1, 1.0, 10.0]
CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]

# (λ_e, λ_h) -> arm key in gm_table.csv
GRID_ARM = {
    (0.1,   0.1):  ("sigreg01_enc3", "#355 anchor"),
    (1.0,   0.1):  ("sigreg10_enc3", "#359 anchor"),
    (10.0,  0.1):  ("emb100_enc01",  "arm 1"),
    (10.0,  1.0):  ("emb100_enc10",  "arm 2"),
    (10.0,  10.0): ("emb100_enc100", "arm 3"),
    (100.0, 0.1):  ("emb1000_enc01", "arm 5"),
}
# (λ_e, λ_h) cells whose GM is not yet known: drawn as hatched gaps.
UNRUN = [
    (0.1,   1.0),
    (0.1,   10.0),
    (1.0,   1.0),
    (1.0,   10.0),
    (100.0, 1.0),
    (100.0, 10.0),
]
ANCHOR_KEY = "sigreg10_enc3"  # #359, (1.0, 0.1) — diverging-colormap centre


def heatmap(gm: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), constrained_layout=True)
    nx, ny = len(LAMBDA_E), len(LAMBDA_H)

    all_vals = []
    for arm_key, _ in GRID_ARM.values():
        for head, ckpt in CELLS:
            r = gm[(gm["arm"] == arm_key) & (gm["head"] == head) & (gm["ckpt"] == ckpt)]
            if len(r):
                all_vals.append(float(r.gm.values[0]))
    vmin_global = min(all_vals)
    vmax_global = max(all_vals)

    last_im = None
    for ax, (head, ckpt) in zip(axes.ravel(), CELLS):
        r_anchor = gm[(gm["arm"] == ANCHOR_KEY) & (gm["head"] == head) & (gm["ckpt"] == ckpt)]
        anchor_val = float(r_anchor.gm.values[0])
        half = max(anchor_val - vmin_global, vmax_global - anchor_val)
        norm = TwoSlopeNorm(vmin=anchor_val - half, vcenter=anchor_val, vmax=anchor_val + half)

        grid = np.full((ny, nx), np.nan)
        for (le, lh), (arm_key, _) in GRID_ARM.items():
            r = gm[(gm["arm"] == arm_key) & (gm["head"] == head) & (gm["ckpt"] == ckpt)]
            if len(r):
                grid[LAMBDA_H.index(lh), LAMBDA_E.index(le)] = float(r.gm.values[0])

        im = ax.imshow(grid, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
        last_im = im

        # hatch the un-run cells
        for (le, lh) in UNRUN:
            xi = LAMBDA_E.index(le)
            yi = LAMBDA_H.index(lh)
            ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1,
                                   facecolor="#dddddd", edgecolor="#888888",
                                   hatch="///", lw=0.5, zorder=2))
            ax.text(xi, yi, "n/a", ha="center", va="center",
                    fontsize=7.0, color="#555555", zorder=3)

        # value overlay + arm tag on each populated cell
        for (le, lh), (arm_key, tag) in GRID_ARM.items():
            xi = LAMBDA_E.index(le)
            yi = LAMBDA_H.index(lh)
            v = grid[yi, xi]
            if np.isnan(v):
                continue
            dist = abs(v - anchor_val) / (half if half > 0 else 1.0)
            txt_col = "white" if dist > 0.55 else "black"
            ax.text(xi, yi + 0.15, f"{v:.3f}", ha="center", va="center",
                    fontsize=10.0, color=txt_col, fontweight="bold", zorder=3)
            ax.text(xi, yi - 0.22, tag, ha="center", va="center",
                    fontsize=6.8, color=txt_col, zorder=3)

        ax.set_xticks(range(nx))
        ax.set_xticklabels([f"{v:g}" for v in LAMBDA_E])
        ax.set_yticks(range(ny))
        ax.set_yticklabels([f"{v:g}" for v in LAMBDA_H])
        ax.set_xlabel("λ_e (log)")
        ax.set_ylabel("λ_h (log)")
        ax.set_title(f"{head} q-head / {ckpt}-ckpt  —  centre = #359 ({anchor_val:.3f})")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("GM-Rel MASE", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "GM-Rel MASE across (λ_e, λ_h) — diverging colormap centred on per-cell #359 anchor "
        "(red = worse than #359, blue = better); hatched = not run",
        fontsize=10.5,
    )
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args(argv)

    results = args.report_dir / "results"
    plots = args.report_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    gm = pd.read_csv(results / "gm_table.csv")
    heatmap(gm, plots / "heatmap.png")
    print(f"wrote {plots / 'heatmap.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
