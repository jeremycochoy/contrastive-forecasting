#!/usr/bin/env python3
"""Plot the RevEWMNorm span sweep from the committed training logs.

Reads the authoritative per-span logs in ../logs/span_*.log (each a single
3k-step run, one run per span) and emits two figures into ../plots/:

  1. span_sweep_headline.png  -- best contrastive gap @3k vs span (log-x),
     one marker per span, no-norm baseline as a reference line, span=8 flagged
     as diverged (NaN).
  2. span_sweep_curves.png    -- gap-vs-step, one line per span.

The gap is FF - FP: how much more a window's forecast resembles its own
future (FF) than its present (FP). Higher = better.

Run from anywhere; paths are resolved relative to this file.

    python scripts/plot_span_sweep.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE.parent / "logs"
PLOT_DIR = HERE.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# span -> (log filename stem, EMA half-life in timesteps as reported in setup)
# half-life = ceil(span/2 * ln(2)) per pandas EWM span->alpha; values match
# the report's setup table. None == no normalization baseline.
SPANS = [16, 32, 45, 64, 91, 128, 512]
HALFLIFE = {16: 6, 32: 11, 45: 16, 64: 22, 91: 32, 128: 44, 512: 176}
PATCH_W = 16  # patch width; half-life / W = half-life in patches

STEP_RE = re.compile(
    r"\[\s*(\d+)\]\s+loss=(\S+)\s+FF=(\S+)\s+FP=(\S+)\s+gap=(\S+)\s+best=(\S+)"
)


def parse_log(stem):
    """Return (steps, gaps, best_final) from logs/span_<stem>.log.

    Robust to the carriage-return / line-buffering mangling present in the
    span=128 and span=512 logs (a couple of intermediate ticks were
    overwritten mid-line); we extract every well-formed step record.
    A run whose gaps are all NaN (span=8) returns best_final = nan.
    """
    path = LOG_DIR / f"span_{stem}.log"
    txt = path.read_text()
    steps, gaps = [], []
    for m in STEP_RE.finditer(txt):
        step = int(m.group(1))
        gap = float(m.group(5))  # 'nan' parses to float('nan')
        steps.append(step)
        gaps.append(gap)
    steps = np.array(steps, dtype=float)
    gaps = np.array(gaps, dtype=float)
    finite = gaps[np.isfinite(gaps)]
    best_final = float(np.max(finite)) if finite.size else float("nan")
    return steps, gaps, best_final


def main():
    # ---- gather data -----------------------------------------------------
    curves = {sp: parse_log(str(sp)) for sp in SPANS}
    _, _, none_best = parse_log("none")
    _, gaps8, best8 = parse_log("8")  # all-NaN diverged run
    none_steps, none_gaps, _ = parse_log("none")

    best_gap = {sp: curves[sp][2] for sp in SPANS}

    # ---- figure 1: headline best-gap vs span -----------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    xs = np.array(SPANS, dtype=float)
    ys = np.array([best_gap[sp] for sp in SPANS])

    # no-norm baseline as a horizontal reference band
    ax.axhline(none_best, color="0.55", ls="--", lw=1.4, zorder=1)
    ax.text(
        SPANS[-1], none_best + 0.004,
        f"no-norm baseline = {none_best:.3f}",
        ha="right", va="bottom", color="0.35", fontsize=9,
    )

    ax.plot(xs, ys, "-o", color="#1f77b4", lw=1.8, ms=8, zorder=3,
            label="RevEWMNorm (best gap @3k)")

    # mark span=8 as diverged at the bottom of the axis
    ax.scatter([8], [none_best], marker="x", s=130, color="#d62728",
               zorder=4, linewidths=2.5)
    ax.annotate(
        "span=8\ndiverged (NaN)",
        xy=(8, none_best), xytext=(8, none_best + 0.045),
        ha="center", va="bottom", color="#d62728", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
    )

    # annotate the (marginal) peak at span=32
    ax.annotate(
        f"peak: span=32, gap={best_gap[32]:.3f}",
        xy=(32, best_gap[32]), xytext=(70, best_gap[32] + 0.012),
        ha="left", va="bottom", fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
    )
    # call out the near-tie at span=16
    ax.annotate(
        f"span=16\n{best_gap[16]:.3f}",
        xy=(16, best_gap[16]), xytext=(16, best_gap[16] - 0.05),
        ha="center", va="top", fontsize=8.5, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0),
    )

    ax.set_xscale("log")
    ax.set_xticks([8] + SPANS)
    ax.set_xticklabels([str(s) for s in ([8] + SPANS)])
    ax.minorticks_off()
    ax.set_xlabel("RevEWMNorm span  (smaller = faster EMA adaptation)")
    ax.set_ylabel("best contrastive gap @ 3k steps  (FF − FP)")
    ax.set_title(
        "RevEWMNorm span sweep on ARIMA(1,8,8) — sub-patch span wins\n"
        "single run per span, 3k steps, Tiny backbone (W=16)",
        fontsize=11,
    )
    ax.set_ylim(-0.01, 0.30)
    ax.grid(True, which="major", axis="both", alpha=0.25)

    # secondary x-axis: half-life in patches
    secax = ax.secondary_xaxis("top")
    secax.set_xscale("log")
    secax.set_xticks(SPANS)
    secax.set_xticklabels([f"{HALFLIFE[s] / PATCH_W:.1f}" for s in SPANS])
    secax.set_xlabel("EMA half-life in patches (W=16)", fontsize=9)
    secax.minorticks_off()

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out1 = PLOT_DIR / "span_sweep_headline.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    print(f"wrote {out1}")

    # ---- figure 2: gap-vs-step curves ------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    cmap = plt.get_cmap("viridis")
    order = SPANS  # ascending span
    for i, sp in enumerate(order):
        steps, gaps, _ = curves[sp]
        color = cmap(i / (len(order) - 1))
        ax.plot(steps, gaps, "-o", color=color, ms=4, lw=1.6,
                label=f"span={sp}  ({best_gap[sp]:.3f})")

    # no-norm baseline curve
    ax.plot(none_steps, none_gaps, marker="s", color="0.45", ms=4, lw=1.6,
            ls="--", label=f"no-norm  ({none_best:.3f})")

    # span=8 collapsed to NaN immediately: show as a flat marker at zero
    ax.plot([500, 3000], [0, 0], color="#d62728", lw=1.0, ls=":")
    ax.text(3000, 0.004, "span=8: NaN at step 500", ha="right", va="bottom",
            color="#d62728", fontsize=8.5)

    ax.set_xlabel("training step")
    ax.set_ylabel("contrastive gap  (FF − FP)")
    ax.set_title(
        "Contrastive gap vs step, per span (one run each)\n"
        "ARIMA(1,8,8), Tiny backbone, 3k steps",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 3150)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9,
              title="span  (best gap @3k)")
    fig.tight_layout()
    out2 = PLOT_DIR / "span_sweep_curves.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    print(f"wrote {out2}")

    # ---- echo the parsed best gaps for cross-checking --------------------
    print("\nbest gap @3k (parsed from logs):")
    print(f"  span=8   : NaN (diverged; best={best8})")
    for sp in SPANS:
        print(f"  span={sp:<4}: {best_gap[sp]:.4f}")
    print(f"  no-norm  : {none_best:.4f}")


if __name__ == "__main__":
    main()
