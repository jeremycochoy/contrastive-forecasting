"""Resume50k continuity plot: visual check that the FRESH→RESUME50k handoff
shows no discontinuity (no jump in mean, no change in std).

Panels:
  (a) Full loss trajectory: FRESH steps 0..52400 + RESUME50k 50001..current,
      raw + EMA, with a vertical marker at step 50000.
  (b) Zoom on [40000, current_max]: same data, narrower window.
  (c) Rolling std (window=500) of raw loss for both runs across the boundary.
  (d) Histograms of raw loss in matching [50000, 52400] windows for both runs.

Color scheme follows CLAUDE.md convention (FRESH=blue, RESUME50k=orange).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRESH_CSV = ROOT / (
    "sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/"
    "checkpoints/tiny_full4096_moirai_hp_FRESH_losses.csv"
)
RESUME_CSV = ROOT / (
    "sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/"
    "moirai_hp_FRESH_RESUME50k/checkpoints/"
    "tiny_full4096_moirai_hp_FRESH_RESUME50k_losses.csv"
)
OUT = ROOT / (
    "experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/"
    "resume50k_continuity.png"
)

RESUME_BOUNDARY = 50000  # step at which RESUME50k picks up
COLOR_FRESH = "#1f77b4"   # blue
COLOR_RESUME = "#ff7f0e"  # orange
ROLLING_WINDOW = 500


EMA_SPAN = 200  # visual smoothing only (CSV doesn't store an ema_loss column)


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("step").reset_index(drop=True)
    df["ema_loss"] = df.loss.ewm(span=EMA_SPAN, adjust=False).mean()
    return df


def main() -> None:
    fresh = load(FRESH_CSV)
    resume = load(RESUME_CSV)

    fresh_max = int(fresh.step.max())
    resume_max = int(resume.step.max())
    print(f"FRESH:    {len(fresh)} rows, steps [{fresh.step.min()}, {fresh_max}]")
    print(f"RESUME:   {len(resume)} rows, steps [{resume.step.min()}, {resume_max}]")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (a) Full trajectory
    ax = axes[0, 0]
    ax.plot(fresh.step, fresh.loss, color=COLOR_FRESH, alpha=0.18,
            linewidth=0.4, label="_FRESH raw")
    ax.plot(fresh.step, fresh.ema_loss, color=COLOR_FRESH, linewidth=1.2,
            label=f"FRESH EMA  (n={len(fresh)})")
    ax.plot(resume.step, resume.loss, color=COLOR_RESUME, alpha=0.25,
            linewidth=0.4, label="_RESUME raw")
    ax.plot(resume.step, resume.ema_loss, color=COLOR_RESUME, linewidth=1.2,
            label=f"RESUME50k EMA  (n={len(resume)})")
    ax.axvline(RESUME_BOUNDARY, color="black", linestyle="--", linewidth=0.8,
               label=f"resume @ step {RESUME_BOUNDARY}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("(a) Full loss trajectory — FRESH 0..52,400 + RESUME50k 50,001..now")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Zoom [40k, current]
    ax = axes[0, 1]
    zoom_lo, zoom_hi = 40000, max(resume_max, fresh_max)
    f_zoom = fresh[fresh.step >= zoom_lo]
    r_zoom = resume[resume.step >= zoom_lo]
    ax.plot(f_zoom.step, f_zoom.loss, color=COLOR_FRESH, alpha=0.30,
            linewidth=0.6, label="FRESH raw")
    ax.plot(f_zoom.step, f_zoom.ema_loss, color=COLOR_FRESH, linewidth=1.5,
            label="FRESH EMA")
    ax.plot(r_zoom.step, r_zoom.loss, color=COLOR_RESUME, alpha=0.40,
            linewidth=0.6, label="RESUME50k raw")
    ax.plot(r_zoom.step, r_zoom.ema_loss, color=COLOR_RESUME, linewidth=1.5,
            label="RESUME50k EMA")
    ax.axvline(RESUME_BOUNDARY, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"(b) Zoom [{zoom_lo:,}, {zoom_hi:,}] — boundary inspection")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Rolling std (window=500) of raw loss
    ax = axes[1, 0]
    f_rstd = fresh.loss.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
    r_rstd = resume.loss.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
    ax.plot(fresh.step, f_rstd, color=COLOR_FRESH, linewidth=1.2,
            label=f"FRESH rolling std (w={ROLLING_WINDOW})")
    ax.plot(resume.step, r_rstd, color=COLOR_RESUME, linewidth=1.2,
            label=f"RESUME50k rolling std (w={ROLLING_WINDOW})")
    ax.axvline(RESUME_BOUNDARY, color="black", linestyle="--", linewidth=0.8,
               label=f"resume @ step {RESUME_BOUNDARY}")
    # Reference bands for prior #9 baseline (0.23) and v1/v2 corrupted (0.35)
    ax.axhline(0.23, color="green", linestyle=":", linewidth=0.7,
               label="#9 baseline std=0.23")
    ax.axhline(0.35, color="red", linestyle=":", linewidth=0.7,
               label="v1/v2 corrupted std=0.35")
    ax.set_xlabel("step")
    ax.set_ylabel("rolling std of raw loss")
    # Clamp y-axis to the diagnostic range so the std-jump signal isn't
    # swamped by the early-training warmup peak (std~3-4 over steps 0-1000
    # while loss decays from 18 to 6). 0.18..0.40 covers both the #9
    # baseline (0.23) and the v1/v2 corrupted level (0.35) with margin.
    ax.set_ylim(0.18, 0.40)
    # Skip the warmup window where the rolling-std hasn't stabilised:
    # x-axis starts at step 5000 (where steady-state training begins).
    ax.set_xlim(5000, max(resume_max, fresh_max) + 500)
    ax.set_title(f"(c) Rolling std (window={ROLLING_WINDOW}) — std-jump detector "
                 f"[y-axis clamped to diagnostic range]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Histogram comparing distributions over matched windows
    ax = axes[1, 1]
    # Match window: take FRESH last 2400 steps (50000..52400) vs RESUME first 2400
    # available, capped at min(resume_max - 50000, 52400 - 50000)
    win_hi = min(resume_max, 52400)
    if win_hi > RESUME_BOUNDARY:
        f_win = fresh[(fresh.step >= RESUME_BOUNDARY) & (fresh.step <= win_hi)].loss
        r_win = resume[(resume.step >= RESUME_BOUNDARY) & (resume.step <= win_hi)].loss
        bins = np.linspace(
            min(f_win.min(), r_win.min()),
            max(f_win.max(), r_win.max()),
            40,
        )
        ax.hist(f_win, bins=bins, alpha=0.55, color=COLOR_FRESH,
                label=f"FRESH [{RESUME_BOUNDARY:,}, {win_hi:,}] "
                      f"n={len(f_win)} μ={f_win.mean():.3f} σ={f_win.std():.3f}")
        ax.hist(r_win, bins=bins, alpha=0.55, color=COLOR_RESUME,
                label=f"RESUME50k [{RESUME_BOUNDARY:,}, {win_hi:,}] "
                      f"n={len(r_win)} μ={r_win.mean():.3f} σ={r_win.std():.3f}")
        ax.set_xlabel("raw loss")
        ax.set_ylabel("count")
        ax.set_title(f"(d) Distribution comparison on matched window "
                     f"[{RESUME_BOUNDARY:,}, {win_hi:,}]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5,
                "RESUME50k has not yet reached step 50000+;\n"
                "histogram comparison not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(d) Distribution comparison — not yet available")

    fig.suptitle(
        f"FRESH→RESUME50k continuity — resume @ step {RESUME_BOUNDARY:,};  "
        f"FRESH last={fresh_max:,}; RESUME50k last={resume_max:,}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
