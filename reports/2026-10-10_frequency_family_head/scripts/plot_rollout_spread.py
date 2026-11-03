"""The iterated forecast (B4 rollout) on electricity/H/long, per future patch.

Reads results_rollout_c4096.tsv (rollout_spread.py, 200 test windows, the
4096-value context the backbones train on). Panels: the cosine of the rolled
latent with the encoder latent of the true future (dashed: the one-step
prediction from the true past); the norm of the rolled latent over the norm of
the true one; the relative change of the latent from one patch to the next
(dashed black: the true future).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RES = "/home/jupyter/cf_runs/qkgain_test/results_rollout_c4096.tsv"
MODELS = {
    "grey_cd288k": ("grey cooldown 288k (best score)", "#000000"),
    "grey_440k": ("grey 440k", "#7f7f7f"),
    "violet_460k": ("violet 460k (µP)", "#8a2be2"),
    "ema098_240k": ("EMA from 0.98, 240k", "#17becf"),
    "ema098_460k": ("EMA from 0.98, 460k (features collapsed)", "#0b6e78"),
}


def panel(ax, t, col, title, ylabel, dashed=None, truth=None, log=False):
    for key, (legend, colour) in MODELS.items():
        d = t[t.label == key]
        ax.plot(d.k, d[col], color=colour, lw=2, label=legend)
        if dashed:
            ax.plot(d.k, d[dashed], color=colour, lw=1.2, ls="--")
    if truth:
        d = t[t.label == "grey_cd288k"]
        ax.plot(d.k, d[truth], color="#d62728", lw=1.5, ls="--", label="true future (grey cooldown latents)")
    ax.set_title(title)
    ax.set_xlabel("future patch k (16 hours each)")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def main():
    t = pd.read_csv(RES, sep="\t")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panel(axes[0], t, "cos_rolled", "Rolled latent vs true future",
          "cosine (dashed: one step from the true past)", dashed="cos_onestep")
    panel(axes[1], t, "norm_ratio", "Norm of the rolled latent", "norm rolled / norm true", log=True)
    panel(axes[2], t, "change_rolled", "Change from one patch to the next",
          "relative change of the latent", truth="change_true", log=True)
    fig.suptitle("Iterated forecast on electricity/H/long: 200 test windows, 4096-value context")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130)


if __name__ == "__main__":
    main()
