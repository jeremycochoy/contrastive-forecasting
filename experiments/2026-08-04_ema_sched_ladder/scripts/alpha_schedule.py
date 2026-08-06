#!/usr/bin/env python3
"""Record and plot the EMA momentum α against training step (#393).

α is the weight the teacher keeps on its own previous value in
θ_T ← α·θ_T + (1−α)·θ_S. This experiment raises it linearly from 0.9 at
step 0 to 1.0 at step 100k, then holds it at 1.0: from 100k on the teacher
stops moving. The ramp is anchored to step 100k, not to a run's budget, so
all ten runs sit on one curve however far each of them gets.

The curve comes from `src.models.ema_tau_at_step`, the same function the
trainer calls every step, so the record cannot drift from what ran.

Writes `results/alpha_schedule.csv` and `plots/alpha_schedule.png`.

Usage: python3 alpha_schedule.py [--max-step 200000]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)


def load_ladder():
    spec = importlib.util.spec_from_file_location(
        "ladder_393", os.path.join(SCRIPTS_DIR, "ladder.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, series):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "ema_tau"])
        for step, alpha in series:
            writer.writerow([step, f"{alpha:.6f}"])


def write_plot(path, series, ladder, stops):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    steps = [s / 1000 for s, _ in series]
    alphas = [a for _, a in series]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(steps, alphas, color="#1f4e79", lw=2)
    ax.axvline(ladder.EMA_TAU_RAMP_STEPS / 1000, color="#999", ls="--", lw=1)
    for stop in stops:
        ax.plot([stop / 1000], [ladder.alpha_at(stop)],
                "o", color="#c0504d", ms=5, zorder=3)
    ax.set_xlabel("backbone step (thousands)")
    ax.set_ylabel("EMA momentum α")
    ax.set_title("EMA momentum against training step")
    ax.set_ylim(0.895, 1.005)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--max-step", type=int, default=200_000)
    p.add_argument("--every", type=int, default=1_000)
    p.add_argument("--csv-out", default=os.path.join(EXP_DIR, "results",
                                                     "alpha_schedule.csv"))
    p.add_argument("--plot-out", default=os.path.join(EXP_DIR, "plots",
                                                      "alpha_schedule.png"))
    args = p.parse_args()

    ladder = load_ladder()
    # Straight from the trainer's own schedule function, so the record
    # cannot drift from what ran.
    series = [(s, ladder.alpha_at(s))
              for s in range(0, args.max_step + 1, args.every)]
    write_csv(args.csv_out, series)

    stops, step = [], 0
    while True:
        step = ladder.next_stop(step)
        if step > args.max_step:
            break
        stops.append(step)
    write_plot(args.plot_out, series, ladder, stops)

    print(f"[out] {args.csv_out} ({len(series)} rows)")
    print(f"[out] {args.plot_out}")
    for stop in stops:
        print(f"  α({stop:>7d}) = {ladder.alpha_at(stop):.4f}")


if __name__ == "__main__":
    main()
