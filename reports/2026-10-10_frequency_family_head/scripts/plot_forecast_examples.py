"""Forecasts of a few GIFT-Eval test windows: our best forecast so far (BLK, the
cooldown of grey at 288k, head B4) next to Moirai small 1.1-R and Seasonal
Naive (the user, 11-03). The forecasts come from forecast_examples_ours.py and
forecast_examples_moirai.py (results/forecast_examples/*.npz). Each panel shows
the end of the context (grey), the true future (black) and the median forecast
of each model; its title gives the MASE of each model on the window (the mean
absolute error over the mean absolute seasonal difference of the context,
season 24 for hourly data, as the GIFT-Eval MASE; missing values skipped).

Writes plots/forecast_examples.png.

Usage: python3 plot_forecast_examples.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parent.parent
DATA = STUDY / "results" / "forecast_examples"
OUT = STUDY / "plots" / "forecast_examples.png"
SEASON = 24
SHOWN_CONTEXT = {"m4_hourly/H/short": 240, "electricity/H/long": 720}
MODELS = {"ours": ("ours: BLK, grey cooldown 288k", "#d62728"),
          "moirai": ("Moirai small 1.1-R", "#1f77b4"),
          "naive": ("Seasonal Naive", "#7f7f7f")}


def windows():
    """[(config, index)] of the saved windows, in file order."""
    ours = np.load(DATA / "ours_blk288k.npz")
    keys = [k.rsplit("|", 1)[0] for k in ours.files if k.endswith("|truth")]
    return [(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])) for k in keys]


def seasonal_naive(context, horizon):
    return np.resize(context[-SEASON:], horizon)


def mase(truth, forecast, context):
    """MASE on the window; missing values of the context or the future are skipped."""
    scale = np.nanmean(np.abs(context[SEASON:] - context[:-SEASON]))
    return np.nanmean(np.abs(truth - forecast)) / scale


def window_data(config, index):
    ours, moirai = np.load(DATA / "ours_blk288k.npz"), np.load(DATA / "moirai_small.npz")
    key = f"{config}|{index}"
    context, truth = ours[f"{key}|context"], ours[f"{key}|truth"]
    forecasts = {"ours": ours[f"{key}|median"], "moirai": moirai[f"{key}|median"],
                 "naive": seasonal_naive(context, len(truth))}
    return context, truth, forecasts


def draw_window(ax, config, index):
    context, truth, forecasts = window_data(config, index)
    shown = context[-SHOWN_CONTEXT[config]:]
    x_context = np.arange(-len(shown), 0)
    x_future = np.arange(len(truth))
    ax.plot(x_context, shown, color="#999999", lw=1)
    ax.plot(x_future, truth, color="black", lw=1.5, label="true future")
    scores = []
    for name, (label, colour) in MODELS.items():
        style = "--" if name == "naive" else "-"
        ax.plot(x_future, forecasts[name], color=colour, ls=style, lw=1.3, label=label)
        scores.append(f"{name} {mase(truth, forecasts[name], context):.2f}")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(f"{config}, window {index}\nMASE: " + ", ".join(scores), fontsize=10)
    ax.set_xlabel("hours from the forecast start")


def main():
    shown = windows()
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    for ax, (config, index) in zip(axes.flat, shown):
        draw_window(ax, config, index)
    for ax in axes.flat[len(shown):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.12), fontsize=11)
    fig.suptitle("GIFT-Eval test windows: the end of the context (grey), the true future and the median "
                 "forecasts", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)


if __name__ == "__main__":
    main()
