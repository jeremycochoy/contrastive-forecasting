"""Examples of the synthetic series of the Data section.

(a) A forked ARMA pair: one shared ARIMA(p, 1, q) past, then two futures.
(b) A crossfade C of two series A and B, each at zero mean and unit
    variance. In training A and B are real windows of the batch; here they
    are two ARIMA series, so that the figure needs no data file.

The draws follow the text of the Data section: orders uniform in 1..8,
standard normal coefficients rescaled to an l1 norm of at most 0.95, a
burn-in of 200 steps, a fork uniform in [T/4, 3T/4], per-branch coefficients
plus N(0, 0.05) noise clipped to the same bound, then a cumulative sum.

Writes paper/figures/synthetic_forked_arma.pdf and
paper/figures/synthetic_crossfade.pdf.
Usage: python3 paper/scripts/plot_synthetic_series.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent.parent
OUT_DIR = PAPER / "figures"
T = 1024
BOUND = 0.95
BURN_IN = 200
BLUE, VERMILLION, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#888888"


def clip_l1(c):
    norm = np.abs(c).sum()
    return c * (BOUND / norm) if norm > BOUND else c


def arma_step(phi, theta, x_hist, e_hist, e):
    ar = sum(phi[i] * x_hist[-1 - i] for i in range(len(phi)))
    ma = sum(theta[j] * e_hist[-1 - j] for j in range(len(theta)))
    return ar + ma + e


def forked_arma(rng):
    p, q = rng.integers(1, 9, size=2)
    phi, theta = clip_l1(rng.standard_normal(p)), clip_l1(rng.standard_normal(q))
    fork = int(rng.integers(T // 4, 3 * T // 4 + 1))
    lag = max(p, q)
    x, e = [0.0] * lag, [0.0] * lag
    for _ in range(BURN_IN + fork):
        eps = rng.standard_normal()
        x.append(arma_step(phi, theta, x, e, eps))
        e.append(eps)
    branches = []
    for _ in range(2):
        ph = clip_l1(phi + 0.05 * rng.standard_normal(p))
        th = clip_l1(theta + 0.05 * rng.standard_normal(q))
        xb, eb = list(x), list(e)
        for _ in range(T - fork):
            eps = rng.standard_normal()
            xb.append(arma_step(ph, th, xb, eb, eps))
            eb.append(eps)
        branches.append(np.cumsum(xb[lag + BURN_IN:]))
    return branches, fork


def zscore(x):
    return (x - x.mean()) / x.std()


def crossfade(rng, a, b):
    mid = rng.uniform(0, T)
    width = T * np.exp(rng.uniform(np.log(1 / 128), 0))
    s = np.clip((np.arange(T) - (mid - width / 2)) / width, 0.0, 1.0)
    return (1 - s) * a + s * b


def _wide(rng):
    mid = rng.uniform(0, T)
    width = T * np.exp(rng.uniform(np.log(1 / 128), 0))
    return 0.35 * T < mid < 0.65 * T and width > 0.4 * T


def main():
    rng = np.random.default_rng(7)
    (s1, s2), fork = forked_arma(rng)
    # A is the first future of panel (a); B is one more draw of the same
    # generator. Both at zero mean and unit variance, as in training.
    a = zscore(s1)
    b = zscore(forked_arma(np.random.default_rng(11))[0][0])
    # A draw with a wide transition near the middle, so that the figure
    # shows the blend (the training draw is the same distribution).
    seed = next(k for k in range(1000)
                if _wide(np.random.default_rng(k)))
    c = crossfade(np.random.default_rng(seed), a, b)
    t = np.arange(T)

    plt.rcParams.update({"font.size": 8, "legend.handlelength": 1.6,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(draw, name):
        fig, ax = plt.subplots(figsize=(3.9, 1.7))
        draw(ax)
        ax.set_xlim(0, T)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=6, colors=MUTED)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        fig.tight_layout()
        fig.savefig(OUT_DIR / name)
        plt.close(fig)
        print(f"wrote {OUT_DIR / name}")

    def forked(ax):
        ax.plot(t[fork:], s2[fork:], color=VERMILLION, lw=1.0, label="future 2")
        ax.plot(t[fork:], s1[fork:], color=BLUE, lw=1.0, label="future 1")
        ax.plot(t[:fork + 1], s1[:fork + 1], color=INK, lw=1.0, label="shared past")
        ax.axvline(fork, color=MUTED, lw=0.6, ls=":")
        ax.legend(frameon=False, fontsize=7, loc="lower left")

    def blend(ax):
        ax.plot(t, a, color=BLUE, lw=0.8, alpha=0.7, label="A")
        ax.plot(t, b, color=VERMILLION, lw=0.8, alpha=0.7, label="B")
        ax.plot(t, c, color=INK, lw=1.0, label="crossfade C")
        ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=3)

    save(forked, "synthetic_forked_arma.pdf")
    save(blend, "synthetic_crossfade.pdf")

if __name__ == "__main__":
    main()
