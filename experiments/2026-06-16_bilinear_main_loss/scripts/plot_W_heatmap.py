#!/usr/bin/env python3
"""#350 — heatmap of W at best_loss for each arm, plus W − Wᵀ (the antisymmetric
part) on the right. Makes the off-diagonal / antisymmetric structure reported
in the W table visible at a glance.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RUNS = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss/runs"
ABORT = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss/_run2_aborted"
ARMS = [
    ("run-1 (W on h), best_loss step 3,500", f"{RUNS}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_best_loss.pth"),
    ("run-2 (W on f), best_loss step 3,700", f"{ABORT}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_best_loss.pth"),
]
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "W_heatmap.png")


def main():
    n = len(ARMS)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.4 * n))
    if n == 1:
        axes = np.array([axes])

    for row, (label, path) in enumerate(ARMS):
        W = torch.load(path, map_location="cpu", weights_only=True)["main_w.weight"].float().numpy()
        Wt = W.T
        sym = 0.5 * (W + Wt)
        asym = 0.5 * (W - Wt)

        # Left: full W. Centre at 0 with diverging colormap (init was 10·I, mean diag ~9.3).
        ax = axes[row, 0]
        vmax = float(np.percentile(np.abs(W - np.diag(np.diag(W))), 99.5)) or 1.0
        im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"W  ({label})\n‖W‖_F={np.linalg.norm(W):.1f}  mean(diag)={W.diagonal().mean():.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Middle: W with diagonal removed (so off-diagonal structure is visible).
        ax = axes[row, 1]
        Woff = W - np.diag(np.diag(W))
        vmax2 = float(np.percentile(np.abs(Woff), 99.5)) or 1.0
        im = ax.imshow(Woff, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
        of_frac = np.linalg.norm(Woff) / np.linalg.norm(W)
        ax.set_title(f"W − diag(W)  (off-diagonal only)\n‖W_off‖_F / ‖W‖_F = {of_frac:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Right: antisymmetric part ½(W − Wᵀ).
        ax = axes[row, 2]
        vmax3 = float(np.percentile(np.abs(asym), 99.5)) or 1.0
        im = ax.imshow(asym, cmap="RdBu_r", vmin=-vmax3, vmax=vmax3, aspect="auto")
        asym_ratio = np.linalg.norm(W - Wt) / np.linalg.norm(W)
        ax.set_title(f"½(W − Wᵀ)  (antisymmetric part)\n‖W − Wᵀ‖_F / ‖W‖_F = {asym_ratio:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Learned W at best_loss — full matrix (colour clipped to 99.5%-ile off-diag for visibility), "
                 "off-diagonal only, antisymmetric part", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
