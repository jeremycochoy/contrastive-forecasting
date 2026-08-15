#!/usr/bin/env python3
"""#373 figure 1 — the depth response of every arm this study trained.

One bar per (arm, head, depth): GM-Relative MASE at that depth minus the
SAME arm's own k = 0, over the full 97 configs. Negative is better.

Each bar carries the 95% percentile interval of a paired dataset-cluster
bootstrap over the pair's 97 configs. It bounds the eval sample and not
run-to-run variance: both ends move with the datasets, not with the seed.

Two reference spans sit behind the bars, and they are not the same kind of
thing:

  head-seed band     ±0.0384, `ema_sched_ladder.md`'s pooled range. It
                     bounds the HEAD seed alone.
  machine span       what the box alone moved one cell's score: B5·s1
                     against B5·s3, same seed, same code, two machines.

A hatched bar is a delta whose two sides trained on different machines. Every
rented-box `k = 0` in this study missed its published value and every elisa
one hit it, and the machine span says the box alone is worth more than most
of these bars. So a hatched bar carries a term larger than the effect it
reports. Two bars per head are not hatched, and they are the ones a reader
can lean on.

Reads results/splits.csv (`all` rows) and results/bootstrap.csv (`all`
rows), and resolves tags through runs.py.

Usage: plot_depth_response.py --splits results/splits.csv \\
           --bootstrap results/bootstrap.csv --out plots/depth_response.png
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
from matplotlib.patches import Patch                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
import runs as R                                       # noqa: E402
from published import NOISE_BAND                       # noqa: E402

plt.rcParams.update(cc.rc())


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["split"] == "all":
            out[r["stop"]] = float(r["gm_rel_mase"])
    return out


def load_ci(path):
    """`{(arm, k, head): (lo, hi)}` over the full 97 configs.

    `find_artefacts.py --what pairs` labels a comparison `<arm>_k<k>_<head>`
    with the arm's middle dot written as an underscore, so this undoes that
    one substitution and parses nothing else.
    """
    out = {}
    if not path or not Path(path).is_file():
        return out
    for r in csv.DictReader(open(path)):
        if r["subset"] != "all":
            continue
        for arm in R.ARM_ORDER:
            for head in ("student", "teacher"):
                stem = arm.replace(chr(183), "_")
                if not r["label"].startswith(f"{stem}_k"):
                    continue
                if not r["label"].endswith(f"_{head}"):
                    continue
                ktxt = r["label"][len(stem) + 2:-len(head) - 1]
                if ktxt.isdigit():
                    out[(arm, int(ktxt), head)] = (float(r["ci_lo"]),
                                                   float(r["ci_hi"]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--bootstrap")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = load(args.splits)
    ci = load_ci(args.bootstrap)
    rows = []
    for arm, head, k, base, deep in R.pairs(data):
        a, b = data.get(base.tag), data.get(deep.tag)
        if a is not None and b is not None:
            rows.append((arm, head, k, a, b, R.machine_held(base, deep),
                         ci.get((arm, k, head))))
    if not rows:
        raise SystemExit(f"ABORT: no arm has two depths in {args.splits}")
    rows.sort(key=lambda r: (R.ARM_ORDER.index(r[0]), r[1], r[2]))

    # The machine span: one cell, one seed, one code, two machines. B5·s1
    # against B5·s3 measures it directly, so the band behind the bars is a
    # measured quantity rather than a mixed seed-and-machine gap. Read over
    # every pair of B5 backbones that holds the seed and changes the box, so
    # a fourth backbone enters it without an edit here.
    retrain_gap = 0.0
    b5 = [a for a in R.ARM_ORDER if a.startswith("B5·")]
    for k in (0, 3):
        seen = []
        for arm in b5:
            run = R.find_run(arm, k, "depth") or R.find_run(arm, k, "control")
            v = data.get(f"{run.stem}_bb40k_student") if run else None
            if v is not None:
                seen.append((arm, v))
        for i, (a1, v1) in enumerate(seen):
            for a2, v2 in seen[i + 1:]:
                same_seed = R.arm_seed(a1) == R.arm_seed(a2)
                same_box = R.arm_where(a1) == R.arm_where(a2)
                if same_seed and not same_box and abs(v2 - v1) > retrain_gap:
                    retrain_gap = abs(v2 - v1)

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(rows) + 2.2))
    ys = list(range(len(rows)))
    if retrain_gap:
        ax.axvspan(-retrain_gap, retrain_gap, color=cc.BAND, alpha=0.45,
                   zorder=0)
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.PARITY, alpha=0.30, zorder=1)
    ax.axvline(0.0, color=cc.INK, linewidth=1.1, zorder=3)

    for y, (arm, head, k, k0, kk, held, iv) in zip(ys, rows):
        d = kk - k0
        col = cc.colour(arm)
        ax.barh(y, d, height=0.62, color=cc.face(arm), edgecolor=col,
                linewidth=1.6, zorder=4)
        # The hatch goes on a second, unfilled bar. Matplotlib draws a
        # hatch in the patch's EDGE colour, and the edge here is the fill
        # colour, so hatching the first bar would draw it invisible.
        if cc.hatch(held):
            ax.barh(y, d, height=0.62, color="none", edgecolor=cc.INK,
                    linewidth=0.0, hatch=cc.CROSSED_HATCH, zorder=5)
        if iv:
            ax.errorbar(d, y, xerr=[[d - iv[0]], [iv[1] - d]], fmt="none",
                        ecolor=cc.INK, elinewidth=1.3, capsize=4.5,
                        capthick=1.3, zorder=6)
        # The value label sits beyond the whisker, not beyond the bar, or
        # a wide interval prints its cap through the text.
        edge = (max(d, iv[1]) if d >= 0 else min(d, iv[0])) if iv else d
        ax.text(edge + (0.010 if d >= 0 else -0.010), y,
                f"{k0:.4f} → {kk:.4f}   ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color=cc.INK, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{arm}  k = {k}  [{head}]{' ✗' if arm in R.RETRACTED else ''}"
         for arm, head, k, _a, _b, _m, _c in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("GM-Relative MASE, depth k minus the same arm's k = 0   "
                  "(97 configs, negative is better)")
    # Every bar carries its own value label outside the bar end, so the axis
    # has to hold the bar AND the text. 1.75x leaves room for the longest.
    lim = max(max(abs(b - a), *(abs(v) for v in (c or (0.0, 0.0))))
              for _r, _h, _k, a, b, _m, c in rows) * 1.75
    ax.set_xlim(-lim, lim)
    ax.set_title("Rollout depth against the arm's own k = 0, at bb40k",
                 loc="left", fontsize=12)

    handles = [Patch(facecolor=cc.COLOUR[c], label=cc.label(c))
               for c in ("B9", "B1", "B5", "A3") if c in cc.COLOUR]
    handles.append(Patch(facecolor="#ffffff", edgecolor=cc.INK_SOFT,
                         label="second backbone seed (20260521)"))
    handles.append(Patch(facecolor="#ffffff", edgecolor=cc.INK,
                         hatch=cc.CROSSED_HATCH, linewidth=0.6,
                         label="the two sides trained on different machines"))
    handles.append(Line2D([], [], color=cc.INK, linewidth=1.3,
                          marker="|", markersize=7,
                          label="95% CI, paired dataset-cluster bootstrap"))
    if any(a in R.RETRACTED for a, *_ in rows):
        handles.append(Patch(facecolor="#ffffff", edgecolor="#ffffff",
                             label="✗ retracted, see the reproduction table"))
    handles.append(Patch(facecolor=cc.PARITY, alpha=0.30,
                         label=f"head-seed band ±{NOISE_BAND}"))
    if retrain_gap:
        handles.append(Patch(facecolor=cc.BAND, alpha=0.45,
                             label=f"machine band ±{retrain_gap:.4f}, "
                                   "one seed"))
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, -0.09), borderaxespad=0.0)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
