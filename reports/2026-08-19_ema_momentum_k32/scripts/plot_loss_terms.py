#!/usr/bin/env python3
"""The total loss and each term that makes it, one panel per term.

WHY THIS FIGURE EXISTS. The total loss falls, turns near step 500 and rises for
the rest of the run. The total alone cannot say which term rises.

THE DECOMPOSITION. `train.py` builds the total in this order:

    loss = L_rep                                        the h-anchored term
         + align_w        * L_align                     line 1889
         + sigreg_emb_w   * sigreg_e                    line 1946
         + sigreg_enc_w   * sigreg_h                    line 1953
         + cpc_w          * cpc_aux                     line 1932, weight 0 here

Every weight of this card is 1.0, and the CPC weight is 0.

L_align IS NOT LOGGED. The CSV carries `cos_err_d0` thru `cos_err_dk`, and
`rollout_cos_error` in src/metrics.py defines each one as 1 - cos. The loss
term is 2 - 2*cos, which is twice that. The reduction over the depth copies is
a mean, so

    L_align = align_w * 2 * mean(cos_err_d0 .. cos_err_dk)

L_REP IS NOT LOGGED EITHER. It is what the total holds after the other three
terms come out:

    L_rep = loss - align_w*L_align - w_e*sigreg_e - w_h*sigreg_h

So the L_rep panel carries the error of every other term. It is a residual, and
the panel says so.

Usage:
  plot_loss_terms.py --curve <arm>=<losses.csv> [--curve ...] \
      --out plots/loss_terms.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Every weight of this card. `read_weights` takes them off a run log when one
# is given, so a run with other weights draws correctly.
DEFAULT_WEIGHTS = {"align": 1.0, "sigreg_e": 1.0, "sigreg_h": 1.0}

PANELS = (
    ("total", "the total loss"),
    ("align", "L_align, weighted"),
    ("rep", "L_rep, the residual"),
    ("sigreg_e", "SIGReg on the embedding"),
    ("sigreg_h", "SIGReg on the encoding"),
)


def read_terms(path, weights):
    """`{term: [(step, value), ...]}` for one run."""
    out = {name: [] for name, _ in PANELS}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        depth_cols = [c for c in (reader.fieldnames or [])
                      if c.startswith("cos_err_d")]
        for r in reader:
            try:
                step = int(r["step"])
                total = float(r["loss"])
            except (KeyError, TypeError, ValueError):
                continue
            errs = [float(r[c]) for c in depth_cols
                    if r.get(c) not in (None, "")]
            if not errs:
                continue
            # cos_err is 1 - cos, and the loss term is 2 - 2*cos.
            align = weights["align"] * 2.0 * (sum(errs) / len(errs))
            se = float(r["sigreg_e"]) if r.get("sigreg_e") else 0.0
            sh = float(r["sigreg_h"]) if r.get("sigreg_h") else 0.0
            rep = total - align - weights["sigreg_e"] * se \
                - weights["sigreg_h"] * sh
            out["total"].append((step, total))
            out["align"].append((step, align))
            out["rep"].append((step, rep))
            out["sigreg_e"].append((step, se))
            out["sigreg_h"].append((step, sh))
    return out


def binned(points, bins=260):
    """The median of each of `bins` log-spaced step bins, to stop over-draw."""
    if not points:
        return [], []
    import math
    lo = max(1, points[0][0])
    hi = max(points[-1][0], lo + 1)
    edges = [lo * (hi / lo) ** (i / bins) for i in range(bins + 1)]
    xs, ys, k = [], [], 0
    for i in range(bins):
        chunk = []
        while k < len(points) and points[k][0] <= edges[i + 1]:
            if points[k][0] >= edges[i]:
                chunk.append(points[k][1])
            k += 1
        if chunk:
            chunk.sort()
            xs.append(math.sqrt(edges[i] * edges[i + 1]))
            ys.append(chunk[len(chunk) // 2])
    return xs, ys


def draw(runs, out, turn=None, cols=3, grey_red=False):
    """One panel per term, over `cols` columns.

    `grey_red` draws every run that held in one grey and the run that fell
    in red, the encoding the other figures of this study use. It is what a
    figure with fourteen runs needs, because fourteen colours read as none.
    """
    rows_n = (len(PANELS) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(4.6 * cols, 3.9 * rows_n),
                             squeeze=False)
    flat = [axes[r][c] for r in range(rows_n) for c in range(cols)]
    colours = ("#1f77b4", "#d95f02", "#7570b3", "#66a61e", "#e7298a")
    for slot, (term, title) in enumerate(PANELS):
        ax = flat[slot]
        named = set()
        for i, (label, terms) in enumerate(runs):
            xs, ys = binned(terms[term])
            if grey_red:
                fell = label.startswith("!")
                clean = label.lstrip("!")
                first = clean not in named
                named.add(clean)
                ax.plot(xs, ys, linewidth=2.2 if fell else 1.0,
                        color="#d62728" if fell else "0.55",
                        alpha=1.0 if fell else 0.75, zorder=3 if fell else 2,
                        label=clean if first else None)
            else:
                ax.plot(xs, ys, linewidth=1.6,
                        color=colours[i % len(colours)], label=label)
        if turn:
            ax.axvline(turn, color="0.55", linestyle=":", linewidth=1.2)
            # The report keeps no sentence for the turn, so the two panels
            # that show it name the step themselves.
            if term in ("total", "align"):
                ax.annotate(f"step {turn}", xy=(turn, 0.97),
                            xycoords=("data", "axes fraction"),
                            xytext=(4, 0), textcoords="offset points",
                            ha="left", va="top", fontsize=9, color="0.35")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("backbone step")
        ax.grid(True, alpha=0.3)
        if slot % cols == 0:
            ax.set_ylabel("loss")
    # A slot with no term left carries no empty frame.
    for spare in flat[len(PANELS):]:
        spare.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if len(labels) > 1:
        fig.legend(handles, labels, fontsize=8, loc="lower center",
                   ncol=min(len(labels), 4), framealpha=0.9)
        fig.subplots_adjust(bottom=0.24)
    fig.suptitle("The total loss and each term that makes it", fontsize=12)
    fig.tight_layout(rect=(0, 0.10 if len(labels) > 1 else 0, 1, 0.96))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(runs)} run(s), {len(PANELS)} panels")
    return fig, axes


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--curve", action="append", default=[],
                   metavar="LABEL=CSV", help="one run's losses CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=3,
                   help="panels per row")
    p.add_argument("--grey-red", action="store_true",
                   help="one grey for every run, red for a label "
                        "that starts with !")
    p.add_argument("--turn", type=int, default=None,
                   help="draw a dotted line at this step")
    args = p.parse_args(argv)
    if not args.curve:
        raise SystemExit("ABORT: pass at least one --curve LABEL=CSV")
    runs = []
    for spec in args.curve:
        label, _, path = spec.partition("=")
        if not Path(path).is_file():
            raise SystemExit(f"ABORT: no losses CSV at {path}")
        runs.append((label, read_terms(path, DEFAULT_WEIGHTS)))
    draw(runs, args.out, args.turn, args.cols, args.grey_red)
    return 0


if __name__ == "__main__":
    sys.exit(main())
