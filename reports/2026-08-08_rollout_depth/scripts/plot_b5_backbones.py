#!/usr/bin/env python3
"""#373 figure 3 — B5 trained three times: two seeds, two machines.

One cell, one recipe, one head seed, one 97-config eval. Three backbones:

    B5·s1   seed 20260520, a rented RTX 5090
    B5·s2   seed 20260521, elisa
    B5·s3   seed 20260520, elisa

The first two disagree about the depth, and they differ by SEED and by
MACHINE at once, so neither of them can say which it was. B5·s3 is the third
corner: it holds the seed of s1 and the machine of s2. It answers the
question.

    s1 against s3   same seed, two machines   -> the machine, -0.1166
    s2 against s3   same machine, two seeds   -> the seed, +0.0035

The figure carries the two channels the study confounded, one each:

    marker shape   the backbone seed
    marker fill    the machine — filled is elisa, hollow is a rented box

The dashed rule is the parent report's published value for this cell.

The third panel draws those two contrasts on one zero axis with their 95%
paired dataset-cluster intervals. The 0.1166 gates every cross-machine delta
in the report, so it is drawn with its interval and not as a point.

Reads results/splits.csv (`all` rows), results/bootstrap.csv and the
registry.

Usage: plot_b5_backbones.py --splits results/splits.csv \\
           --bootstrap results/bootstrap.csv --out plots/b5_backbones.png
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
import runs as R                                       # noqa: E402
from published import PUBLISHED                        # noqa: E402

PUB = PUBLISHED["B5"]["student"][40]
KS = [0, 3]
SHAPE = {20260520: "o", 20260521: "s"}
# The two contrasts the third corner separates, and the bootstrap row that
# bounds each. Both are k = 0, student head: that is where all three
# backbones carry a score.
CONTRASTS = [("the machine\ns1 against s3", "B5_machine_k0_student"),
             ("the backbone seed\ns2 against s3", "B5_seed_k0_student")]
plt.rcParams.update(cc.rc())


def load(path):
    return {r["stop"]: float(r["gm_rel_mase"])
            for r in csv.DictReader(open(path)) if r["split"] == "all"}


def load_ci(path):
    """(delta, lo, hi) per bootstrap label, `all` rows only."""
    out = {}
    with open(path) as fh:
        for r in csv.reader(fh):
            if len(r) >= 6 and r[1] == "all":
                try:
                    out[r[0]] = (float(r[3]), float(r[4]), float(r[5]))
                except ValueError:
                    pass
    return out


def arm_points(data, arm, head):
    """[(k, value)] for one backbone, over the depths it has a score at."""
    out = []
    for k in KS:
        run = R.find_run(arm, k, "depth") or R.find_run(arm, k, "control")
        if run is None:
            continue
        v = data.get(f"{run.stem}_bb40k_{head}")
        if v is not None:
            out.append((k, v))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--bootstrap", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = load(args.splits)
    ci = load_ci(args.bootstrap)
    col = cc.COLOUR["B5"]

    # Every B5 backbone the registry knows, minus the control that swaps in
    # a published checkpoint: that one measures the head and the eval, not a
    # training of this cell.
    arms = [a for a in R.arms_of("B5") if a != "B5·pub"]

    fig, (ax_s, ax_t, dax) = plt.subplots(
        1, 3, figsize=(15.4, 4.9),
        gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.62], wspace=0.28))
    ax_s.sharey(ax_t)
    ax_t.tick_params(labelleft=False)
    axes = (ax_s, ax_t)
    drawn, have = 0, []
    # Where each value label goes. A fixed offset per backbone stacks two
    # labels on top of each other wherever two backbones land close at the
    # same depth, and B5's three backbones do exactly that: s2 and s3 sit
    # 0.0035 apart at k = 0. So the side is decided per COLUMN, from the
    # values: a label hangs below its marker unless another marker sits just
    # under it, and then it goes above.
    allv = [y for a in arms for h in ("student", "teacher")
            for _k, y in arm_points(data, a, h)]
    span = (max(allv) - min(allv)) if allv else 1.0
    for ax, head in zip(axes, ("student", "teacher")):
        # The published rule belongs on the student panel only: group B's
        # parent reports publish one head, trained on the student encoder,
        # so there is no published teacher number to draw against.
        if head == "student":
            ax.axhline(PUB, color=cc.PARITY, linewidth=1.2,
                       linestyle=(0, (4, 3)))
        for arm in arms:
            pts = arm_points(data, arm, head)
            if not pts:
                continue
            drawn += 1
            if arm not in have:
                have.append(arm)
            seed = R.arm_seed(arm)
            elisa = R.arm_where(arm) == "elisa"
            xs = [k for k, _v in pts]
            ys = [v for _k, v in pts]
            ax.plot(xs, ys, color=col, linewidth=2.2,
                    linestyle="solid" if elisa else (0, (5, 2)),
                    marker=SHAPE.get(seed, "^"), markersize=10,
                    markerfacecolor=col if elisa else "#ffffff",
                    markeredgecolor=col, markeredgewidth=2.0, zorder=3)
        column = {}
        for arm in arms:
            for k, y in arm_points(data, arm, head):
                column.setdefault(k, []).append(y)
        near = 0.085 * span
        for k, ys_k in column.items():
            for y in ys_k:
                below = [o for o in ys_k if 0 < y - o < near]
                ax.annotate(f"{y:.4f}", (k, y), textcoords="offset points",
                            xytext=(0, 15 if below else -19), ha="center",
                            va="bottom" if below else "top", fontsize=8.5,
                            color=cc.INK,
                            bbox=dict(fc="#ffffff", ec="none", pad=0.5))
        ax.set_xticks(KS)
        ax.set_xticklabels([f"k = {k}" for k in KS])
        ax.set_xlim(-0.55, 3.55)
        ax.set_title(f"{head} head", loc="left")
        ax.set_xlabel("training rollout depth")
    if not drawn:
        raise SystemExit(f"ABORT: no B5 backbone in {args.splits}")

    axes[0].set_ylabel("GM-Relative MASE, 97 configs  (lower is better)")
    # Room inside the axes for the labels that hang below the lowest marker.
    # Without it they land on the depth tick labels and hide one of them.
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.11 * (hi - lo), hi + 0.03 * (hi - lo))
    axes[0].annotate(f"the parent report publishes {PUB:.4f} for this cell",
                     (3.45, PUB), fontsize=8, color=cc.INK_SOFT,
                     ha="right", va="bottom",
                     bbox=dict(fc="#ffffff", ec="none", pad=0.8))
    # A reader who counts three lines on the left panel and two on the right
    # has to be told why, on the panel where the line is missing.
    missing = [a for a in have
               if arm_points(data, a, "teacher") == []]
    if missing:
        axes[1].annotate(f"{', '.join(missing)}: no teacher head "
                         "(see the annex)",
                         (0.03, 0.97), xycoords="axes fraction", fontsize=8.5,
                         color=cc.INK_SOFT, va="top")

    # ---- the third panel: the two contrasts, with their intervals ---------
    # The two score panels draw levels. What the report carries out of this
    # figure is a difference, so the difference gets its own zero axis and
    # its own 95% interval. The machine term gates every cross-machine delta
    # in the report and it is one run pair, so its width belongs on the page.
    rows = [(lab, ci[key]) for lab, key in CONTRASTS if key in ci]
    ys = list(range(len(rows)))[::-1]
    for yi, (lab, (d, lo, hi)) in zip(ys, rows):
        dax.plot([lo, hi], [yi, yi], color=col, linewidth=9.0, alpha=0.25,
                 solid_capstyle="butt", zorder=1)
        dax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", markersize=8,
                     color=col, ecolor=col, elinewidth=1.6, capsize=5,
                     zorder=3)
        dax.annotate(f"{d:+.4f}\n[{lo:+.4f}, {hi:+.4f}]", (d, yi),
                     textcoords="offset points", xytext=(0, 12), ha="center",
                     fontsize=8.5, color=cc.INK)
    dax.axvline(0, color=cc.INK, linewidth=1.1, zorder=2)
    dax.set_yticks(ys)
    dax.set_yticklabels([lab for lab, _ in rows], fontsize=9)
    dax.set_ylim(-0.7, len(rows) - 0.2)
    dax.set_xlabel("change against B5·s3, 97 configs\nbars are 95% intervals")
    dax.set_title("k = 0, student head", loc="left")
    ends = [v for _lab, (_d, lo, hi) in rows for v in (lo, hi)] + [0.0]
    pad = 0.18 * (max(ends) - min(ends))
    dax.set_xlim(min(ends) - pad, max(ends) + pad)
    dax.grid(axis="y", visible=False)

    handles = []
    for arm in have:
        seed, elisa = R.arm_seed(arm), R.arm_where(arm) == "elisa"
        handles.append(Line2D([], [], color=col, linewidth=2.2,
                              linestyle="solid" if elisa else (0, (5, 2)),
                              marker=SHAPE.get(seed, "^"), markersize=9,
                              markerfacecolor=col if elisa else "#ffffff",
                              markeredgecolor=col, markeredgewidth=2.0,
                              label=f"{arm}  seed {seed}, {R.arm_where(arm)}"))
    # Under both panels, in one row. Inside the teacher panel the legend sat
    # on the s2/s3 line, which crosses the lower right corner on its way to
    # k = 3, so the text and the curve it names were drawn on top of each
    # other.
    fig.legend(handles=handles, loc="lower center", fontsize=8.5, ncol=3,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("B5 arm4_combab_fix09, three backbones, bb40k",
                 x=0.005, ha="left", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({drawn} line(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
