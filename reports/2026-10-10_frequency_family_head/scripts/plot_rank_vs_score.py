"""Forecast GM-MASE and the effective rank of the forecaster features per stop.

Left: the forecast GM-MASE of each scored stop (control head). Right: the
effective rank of the forecaster output f (the features the forecast head
reads) on 62 real GIFT-Eval windows (latent_spectrum.py).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SCORES = ("/tmp/contrastive-forecasting-freq-family/reports/"
          "2026-10-10_frequency_family_head/results/scores.tsv")
RANKS = "/home/jupyter/cf_runs/qkgain_test/results_latent_spectrum_stops.tsv"
# label in the rank file: (run name in the score file, legend, colour)
RUNS = {
    "ema098": ("grn_ema098", "EMA from 0.98, 3 layers", "#17becf"),
    "ema099": ("grn_ema099", "EMA from 0.99, 3 layers", "#800000"),
    "grey": ("grn_ema099995_f6", "grey", "#7f7f7f"),
    "yellow": ("grn_ema099995_f6_rep01", "yellow", "#ffb000"),
    "violet": ("grn_ema099995_f6_rep01_mup", "violet (µP)", "#8a2be2"),
    "lime": ("grn_ema0999975_f6_rep01_mup", "lime (µP, slower EMA)", "#7fff00"),
}


def score_curves():
    s = pd.read_csv(SCORES, sep="\t")
    s = s[(s.arm == "control") & (s.stop_k >= 40)]
    return {k: s[s.run == run].set_index("stop_k").gm_rel_mase.sort_index() for k, (run, _, _) in RUNS.items()}


def rank_curves():
    t = pd.read_csv(RANKS, sep="\t")
    t = t[t.latent == "f"]
    t["run"] = t.label.str.rsplit("_", n=1).str[0]
    t["k"] = t.label.str.rsplit("_", n=1).str[1].str[:-1].astype(int)
    return {k: t[t.run == k].set_index("k").eff_rank.sort_index() for k in RUNS}


def draw(ax, curves, title, ylabel):
    for key, (_, legend, colour) in RUNS.items():
        c = curves[key]
        ax.plot(c.index, c.values, "-o", color=colour, lw=2, ms=5, label=legend,
                mec="#333333" if key == "lime" else colour, mew=0.6)
    ax.set_title(title)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)


def main():
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    draw(left, score_curves(), "Forecast score", "GM-MASE (lower is better)")
    draw(right, rank_curves(), "Spread of the features the head reads",
         "effective rank of the forecaster output")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130)


if __name__ == "__main__":
    main()
