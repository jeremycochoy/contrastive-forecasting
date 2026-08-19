#!/usr/bin/env python3
"""#401 deliverable 1 — per-domain GM-Relative MASE, one panel per depth.

#373's `domain_radar`, on this study's entity. GM-Relative MASE = 1.0 is
parity with seasonal naive, so a polygon inside the black ring beats it on
that family. Energy carries 32 of the 97 configs, so its spoke carries a
third of the aggregate: the aggregate figure cannot say WHERE a depth won,
and this one can.

One panel per depth this study trained. Each panel draws:

  solid   the depth, at the stop named in the panel title.
  dashed  #373's k = 3 on the SAME cell, the same head and the same stop.
          The depth is the only thing that changes between the two polygons.
  grey    the best per-domain value any k = 3 run of #373 reached, over
          every (cell, stop) in its splits table. The per-domain form of the
          rule the ladder figure draws.
  black   the 1.0 parity ring.

The radial axis is log2(ratio), so equal multiplicative steps are equal
distances.

Which stop a panel draws: `--stop best` (the default) takes the stop with
that depth's best AGGREGATE, which is the number deliverable 2 draws, so the
two deliverables name one stop per depth. `--stop 100000` pins one stop for
all of them.

Reads `results/splits.csv` (this study, written by `collect.sh` through
#373's `split_scores.py`) and #373's own `results/splits.csv`.

Usage: plot_domain_radar.py --phase 1 --out plots/domain_radar_phase1.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PARENT = STUDY.parent / "2026-08-08_rollout_depth"
sys.path.insert(0, str(HERE))
import depth_colours as D                                 # noqa: E402

plt.rcParams.update(D.rc())

CELL = "A4"
HEAD = "student"


def load_domains(path):
    """`{label: {domain: value}}`, `{label: aggregate}`, `{domain: count}`.

    The aggregate is the table's own `all,all` row, over all 97 configs. It
    is the quantity the ladder figure draws, and it is what a panel's stop is
    picked by — see pick_panels.
    """
    vals, aggs, counts = {}, {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "all":
                aggs[r["stop"]] = float(r["gm_rel_mase"])
            elif r["split"] == "domain":
                vals.setdefault(r["stop"], {})[r["name"]] = \
                    float(r["gm_rel_mase"])
                counts[r["name"]] = int(r["n"])
    return vals, aggs, counts


def steps_label(n):
    """`40000` -> `40k`. Mirrors cf401_steps_label in study.sh."""
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def study_rows(vals, phase):
    """`{(k, stop, variant): label}` for the rows of one phase.

    The tag carries the depth, the stop and the head budget, so the phase is
    read back off it the same way collect.sh reads it: a head budget equal to
    the backbone stop is phase 2.

    A tag can carry a fifth part, between the depth and the stop: the training
    schedule. `k32_ema30k_bb40k_h30k_student` is the card's cell at k = 32 with
    the EMA ramp shortened to 30,000 steps. It holds the same three numbers as
    its base cell, so the schedule is in the key. Without it the two rows join
    as one and the panel draws whichever the table lists second.
    """
    out = {}
    for label in vals:
        parts = label.split("_")
        if len(parts) == 4:
            parts.insert(1, "base")
        if len(parts) != 5:
            continue
        k_s, variant, bb_s, h_s, enc = parts
        if enc != HEAD or not k_s.startswith("k"):
            continue
        try:
            k = int(k_s[1:])
            stop = unlabel(bb_s[2:])
            head = unlabel(h_s[1:])
        except ValueError:
            continue
        if (2 if head == stop else 1) != phase:
            continue
        out[(k, stop, variant)] = label
    return out


def unlabel(text):
    """`40k` -> 40000, `400` -> 400. Mirrors cf401_steps_of in study.sh."""
    return int(text[:-1]) * 1000 if text.endswith("k") else int(text)


def pick_panels(vals, aggs, phase, stop_arg):
    """One (k, stop, variant, label) per depth, then one per variant cell.

    `best` is the best AGGREGATE, which is the number deliverable 2 draws.
    By the best single family, a depth whose aggregate is best at 200k draws
    its 40k panel, and the two deliverables then name two stops for one
    depth. A tie goes to the earlier stop.

    A variant cell takes its OWN panel, at its own stop. It is a second
    training schedule at a stop the base cell already holds, so it competes
    with that cell for no panel: the reader sees both polygons and can put one
    on the other.
    """
    rows = study_rows(vals, phase)
    panels = []
    for k in D.DEPTHS_DRAWN:
        mine = {s: lab for (kk, s, v), lab in rows.items()
                if kk == k and v == "base"}
        if not mine:
            continue
        if stop_arg == "best":
            missing = sorted(lab for lab in mine.values() if lab not in aggs)
            if missing:
                raise SystemExit(f"ABORT: no `all,all` row for {missing} — "
                                 f"a panel's stop has no aggregate to pick by")
            stop = min(mine, key=lambda s: (aggs[mine[s]], s))
        else:
            stop = int(stop_arg)
            if stop not in mine:
                continue
        panels.append((k, stop, "base", mine[stop]))

    for (k, stop, variant), lab in sorted(rows.items()):
        if variant == "base":
            continue
        if stop_arg != "best" and stop != int(stop_arg):
            continue
        panels.append((k, stop, variant, lab))
    return panels


def frontier_of(parent_vals):
    """The best per-domain value over every k = 3 row of #373."""
    best = {}
    for label, doms in parent_vals.items():
        if "_k3_" not in label or not label.endswith(HEAD):
            continue
        for d, v in doms.items():
            best[d] = min(v, best.get(d, v))
    return best


def ticks_for(values):
    """The ring labels. A ring is added at an end only when it earns its ink.

    An extra ring within 8% of its neighbour prints two numbers on top of one
    another and tells the reader nothing the neighbour did not.
    """
    lo, hi = min(values), max(values)
    out = [t for t in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)
           if lo / 1.1 <= t <= hi * 1.1] or [round(lo, 2), round(hi, 2)]
    if out[0] > lo * 1.08:
        out.insert(0, round(lo, 2))
    if out[-1] * 1.08 < hi:
        out.append(round(hi, 2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(STUDY / "results" / "splits.csv"))
    ap.add_argument("--splits-parent",
                    default=str(PARENT / "results" / "splits.csv"))
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2))
    ap.add_argument("--stop", default="best",
                    help="'best' (per depth) or a stop in steps, e.g. 100000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    vals, aggs, counts = load_domains(a.splits)
    pvals, _, pcounts = load_domains(a.splits_parent)
    counts.update({d: n for d, n in pcounts.items() if d not in counts})

    panels = pick_panels(vals, aggs, a.phase, a.stop)
    if not panels:
        raise SystemExit(f"ABORT: {a.splits} holds no phase-{a.phase} "
                         f"{HEAD}-head row at stop={a.stop}")
    prior = frontier_of(pvals)

    domains = sorted(counts, key=lambda d: -counts[d])
    n = len(domains)
    ang = [i / n * 2 * math.pi for i in range(n)] + [0.0]

    allv = [v for *_, lab in panels for v in vals[lab].values()]
    allv += list(prior.values()) + [1.0]
    for _, stop, _, _ in panels:
        ref = pvals.get(f"{CELL}_k3_bb{steps_label(stop)}_{HEAD}")
        if ref:
            allv += list(ref.values())
    ticks = ticks_for(allv)
    # The rings are the labels. The limits come from the DATA, so no polygon
    # is ever clipped by a ring that stopped short of it.
    rlo = min(math.log2(ticks[0]), math.log2(min(allv))) - 0.03
    rhi = max(math.log2(ticks[-1]), math.log2(max(allv))) + 0.03

    ncol = min(len(panels), 3)
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 5.8 * nrow),
                             subplot_kw={"projection": "polar"}, squeeze=False)

    for idx, (k, stop, variant, label) in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([f"{d}\n({counts[d]})" for d in domains],
                           fontsize=7)
        ax.set_ylim(rlo, rhi)
        ax.set_yticks([math.log2(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
        ax.plot(ang, [0.0] * len(ang), color=D.PARITY_INK, lw=1.7, zorder=5)

        pv = [math.log2(prior[d]) for d in domains if d in prior]
        if len(pv) == n:
            ax.plot(ang, pv + pv[:1], color=D.PRIOR_INK, lw=3.4,
                    solid_capstyle="round", zorder=2)

        ref = pvals.get(f"{CELL}_k3_bb{steps_label(stop)}_{HEAD}")
        if ref and all(d in ref for d in domains):
            rv = [math.log2(ref[d]) for d in domains]
            ax.plot(ang, rv + rv[:1], color=D.REF_K3_INK, lw=1.8,
                    linestyle=D.STYLE_K3, zorder=3)

        mine = vals[label]
        v = [math.log2(mine[d]) for d in domains if d in mine]
        # A variant keeps its depth's hue and takes a dash, so a reader can
        # tell the two schedules apart across two panels.
        style = D.STYLE_STUDY if variant == "base" else (0, (4, 1.6))
        if len(v) == n:
            ax.plot(ang, v + v[:1], color=D.colour(k), lw=2.2,
                    linestyle=style, zorder=4)
        else:
            # A subset eval (a trial) has fewer families than the protocol.
            ax.plot([ang[i] for i, d in enumerate(domains) if d in mine],
                    [math.log2(mine[d]) for d in domains if d in mine],
                    color=D.colour(k), marker="o", ms=7, lw=0, zorder=4)

        sched = "" if variant == "base" else f", {variant} schedule"
        ax.set_title(f"{D.label(k)}   at bb{steps_label(stop)}{sched}\n"
                     f"against #373's k = 3, same cell", fontsize=9, pad=18)

    for idx in range(len(panels), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    # The depths take one keyed swatch each, so the legend is also the colour
    # key and no polygon is identified by its hue alone.
    handles = [Line2D([], [], color=D.colour(k), lw=2.2,
                      linestyle=(D.STYLE_STUDY if v == "base" else (0, (4, 1.6))),
                      label=D.label(k) if v == "base"
                      else f"{D.label(k)}, {v} schedule")
               for k, _, v, _ in panels]
    handles += [
        Line2D([], [], color=D.REF_K3_INK, lw=1.8, linestyle=D.STYLE_K3,
               label="#373's k = 3, the depth this study extends"),
        Line2D([], [], color=D.PRIOR_INK, lw=3.4,
               label="the best per-domain result before this study"),
        Line2D([], [], color=D.PARITY_INK, lw=1.7,
               label="seasonal-naive parity, 1.0. Inside it the model wins")]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(3, len(handles)), frameon=False, fontsize=9)
    fig.suptitle(f"Per-domain GM-Relative MASE, phase {a.phase}, "
                 f"{HEAD}-encoder head (radial axis log2, lower is better)",
                 fontsize=11)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95), h_pad=4.5)
    fig.savefig(a.out, bbox_inches="tight")
    # The stop of each panel, so a reader of make_plots.sh's output can check
    # it against the ladder without opening the figure.
    drawn = ", ".join(
        f"k{k}@bb{steps_label(s)}" + ("" if v == "base" else f" [{v}]")
        for k, s, v, _ in panels)
    print(f"wrote {a.out} ({len(panels)} panel(s): {drawn})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
