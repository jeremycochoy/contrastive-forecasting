#!/usr/bin/env python3
"""#373 — the headline: GM-Relative MASE over the full 97, k = 3 against k = 0.

The parent's `schedule_vs_fixed.png` in this study's terms. One row per
(cell, head). The bar is the change; the band behind it is
`ema_sched_ladder.md`'s pooled head-seed noise band, ±0.0384 GM-Relative
MASE.

Read the band as a floor on the noise, not as the noise. It bounds the head
seed alone. Each delta here is the difference of two independent backbone
trainings, and that spread is not measured anywhere in this study or its
parents.

Reads results/splits.csv (`all` rows), written by split_scores.py.

Usage: plot_k3_vs_k0.py --splits results/splits.csv --out plots/k3_vs_k0.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402

# ema_sched_ladder.md's pooled head-seed band.
NOISE_BAND = 0.0384
plt.rcParams.update(cc.rc())


def load(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "all":
                continue
            parts = r["stop"].split("_")
            if len(parts) < 4 or not parts[1].startswith("k"):
                continue
            out[(parts[0], int(parts[1][1:]), parts[3])] = float(r["gm_rel_mase"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = load(args.splits)
    rows = []
    for cell in cc.ORDER:
        for head in ("student", "teacher"):
            a, b = data.get((cell, 0, head)), data.get((cell, 3, head))
            if a is not None and b is not None:
                rows.append((cell, head, a, b))
    if not rows:
        raise SystemExit("ABORT: no (cell, head) has both depths in "
                         f"{args.splits}")

    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(rows) + 2.1))
    ys = range(len(rows))
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.PARITY, alpha=0.25, zorder=0)
    ax.axvline(0.0, color=cc.INK_SOFT, linewidth=1.0, zorder=1)

    for y, (cell, head, k0, k3) in zip(ys, rows):
        d = k3 - k0
        ax.barh(y, d, height=0.6, color=cc.colour(cell),
                alpha=1.0 if head == "student" else 0.5,
                edgecolor=cc.colour(cell), zorder=2)
        ax.text(d + (0.004 if d >= 0 else -0.004), y,
                f"{k0:.4f} → {k3:.4f}  ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right", fontsize=8,
                color=cc.INK)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{cc.label(c)}  [{h}]" for c, h, _a, _b in rows],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("GM-Relative MASE, k = 3 minus k = 0  "
                  "(97 configs, negative is better)")
    lim = max(abs(k3 - k0) for _c, _h, k0, k3 in rows) * 2.4 + NOISE_BAND
    ax.set_xlim(-lim, lim)
    ax.legend(handles=[Line2D([], [], color=cc.PARITY, linewidth=8, alpha=0.25,
                              label=f"head-seed band ±{NOISE_BAND}")],
              loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Rollout depth k = 3 against k = 0, at bb40k")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
