#!/usr/bin/env python3
"""#407 review gap 1 — the head-seed band, from the draws on disk.

The card scores one head per (stop, encoder), with head seed 20260722. That
gives no scale for a difference between two stops. `replicate_heads.sh`
draws the same head again on the same backbone under seeds 20260723 and
20260724, and this reads the numbers back.

Round 3 of the review added a third draw, and it is not a new seed. Seed
20260722 runs AGAIN at 200,000 steps, here and on this code. #373 drew that
seed on another round and another box, so the published 1.0660 carries head
seed, machine and code version together. The re-draw holds the seed still
and moves only the machine and the code. Two numbers come out of it:

  re-draw delta   the re-draw minus #373's published anchor. Machine and
                  code drift at one head seed.
  local band      the spread over the three seeds measured HERE. The
                  re-draw replaces the published anchor in it, so one
                  machine and one code version carry the whole band.

Reported per (stop, encoder): every draw, the mean, the sample standard
deviation, and the range. The range is what the parent study quotes, so
this study can be read next to it.

This measures the HEAD seed only. It does not measure the backbone seed,
which no run in this study or its parents has replicated, and it does not
measure the config sampling, which `stop_bootstrap.py` covers.

Usage:
  head_band.py [--stop STEPS ...] [--results DIR] [--parent DIR] [--csv F]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402

PROTOCOL_SEED = 20260722
REPLICATE_SEEDS = [20260723, 20260724]

# #393's published head-seed band, and where it comes from. It is a RANGE,
# the largest one that study measured at either head budget, so a range is
# what this study must put beside it. `noise_band.py` prints it, and this
# calls that script rather than typing the number twice.
PUBLISHED_BAND_STUDY = "2026-08-04_ema_sched_ladder"
PUBLISHED_BAND_FALLBACK = 0.0384
# The cell this card continues. #393 measured its own head-seed range too,
# and that row is the closer comparison: same cell, same head, same trainer.
THIS_CELL = "arm6_v2_combab_alignS"


def read_score(path):
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def draw_path(directory, stop, head, seed, redraw=False):
    """Where one draw's score lives.

    The protocol seed's tag carries no seed, because every other artefact of
    this study and of #373 names it that way. A replicate tag carries one,
    and so does the protocol RE-draw, which is why the re-draw cannot
    overwrite the card's own number.
    """
    tag = full_pass.tag(stop, head)
    if seed != PROTOCOL_SEED or redraw:
        tag = f"{tag}_s{seed}"
    return os.path.join(str(directory), f"score_{tag}.txt")


def _find(directories, stop, head, seed, redraw=False):
    for directory in directories:
        if directory is None:
            continue
        value = read_score(draw_path(directory, stop, head, seed, redraw))
        if value is not None:
            return value
    return None


def draws(stop, head, results, parent, seeds=None):
    """`{seed: score}` for one (stop, head), over every seed on disk.

    Each score is looked for in this study's results first and in #373's
    second. #373's stops wrote only into #373's directory, and `collect.sh`
    copies this card's stops into this study, so both places are real.
    """
    seeds = [PROTOCOL_SEED] + REPLICATE_SEEDS if seeds is None else seeds
    out = {}
    for seed in seeds:
        value = _find((results, parent), stop, head, seed)
        if value is not None:
            out[seed] = value
    return out


def redraw(stop, head, results, parent):
    """The protocol seed drawn again HERE, or None when it has not run."""
    return _find((results, parent), stop, head, PROTOCOL_SEED, redraw=True)


def local_draws(stop, head, results, parent):
    """`{seed: score}`, with every draw measured on THIS machine.

    Same shape as `draws`, and the same keys. The one difference is the
    protocol seed: when the re-draw exists, its number replaces the
    published anchor, so machine and code version stay fixed across the
    whole band. A stop with no re-draw is unchanged, because its protocol
    draw already ran here.
    """
    out = draws(stop, head, results, parent)
    again = redraw(stop, head, results, parent)
    if again is not None:
        out = dict(out)
        out[PROTOCOL_SEED] = again
    return out


def band(values):
    """`(mean, sample std, range)` of a list of draws, or None below two."""
    if len(values) < 2:
        return None
    return (statistics.fmean(values),
            statistics.stdev(values),
            max(values) - min(values))


def rows(stops, results, parent, local=True):
    """One row per (stop, head) that has at least one draw."""
    pick = local_draws if local else draws
    out = []
    for stop in stops:
        for head in full_pass.HEADS:
            got = pick(stop, head, results, parent)
            if got:
                out.append((stop, head, got))
    return out


def pooled_std(table):
    """The head-seed std pooled over the rows that have two draws or more.

    Root mean square of the per-row sample standard deviations. Same
    quantity `ema_sched_ladder` pools, so the two bands compare.
    """
    parts = [statistics.stdev(list(got.values()))
             for _, _, got in table if len(got) >= 2]
    if not parts:
        return None
    return (sum(s * s for s in parts) / len(parts)) ** 0.5


def largest_range(table):
    """The widest head-seed range over the rows with two draws or more."""
    spans = [max(got.values()) - min(got.values())
             for _, _, got in table if len(got) >= 2]
    return max(spans) if spans else None


def redraw_table(stops, results, parent):
    """`[(stop, head, anchor, again, delta)]` for every re-draw on disk."""
    out = []
    for stop in stops:
        for head in full_pass.HEADS:
            again = redraw(stop, head, results, parent)
            anchor = _find((results, parent), stop, head, PROTOCOL_SEED)
            if again is None or anchor is None:
                continue
            out.append((stop, head, anchor, again, again - anchor))
    return out


def published_band():
    """#393's head-seed band: the pooled maximum, and this cell's own rows.

    Returns `(pooled, [(cell, head, end, range), ...])`. The pooled number
    falls back to the published literal when #393's CSV is not on this box.
    """
    path = os.path.join(full_pass.REPO_ROOT, "experiments",
                        PUBLISHED_BAND_STUDY, "scripts")
    try:
        sys.path.insert(0, path)
        import noise_band  # noqa: E402
        rs = noise_band.ranges()
        pooled = noise_band.pooled_band()
    except Exception:
        return PUBLISHED_BAND_FALLBACK, []
    finally:
        if sys.path and sys.path[0] == path:
            sys.path.pop(0)
    if pooled is None:
        return PUBLISHED_BAND_FALLBACK, []
    return pooled, [r for r in rs if r[0] == THIS_CELL]


def selection_gap(results):
    """The gap that made 1.0660 the project's best, out of its own file."""
    path = os.path.join(str(results), "selection_context.json")
    try:
        with open(path) as fh:
            got = json.load(fh)
        return float(got["gap_to_runner_up"]), got
    except (OSError, ValueError, KeyError):
        return None, {}


def compare(table, results):
    """Review gap 6 — read this card's band against the published one.

    The question the card asks is whether a move between two stops means
    anything. The answer is a comparison of three numbers, and only one of
    them is new: this run's own head-seed range.
    """
    lines = []
    here = largest_range(table)
    pooled_here = pooled_std(table)
    published, cell_rows = published_band()
    gap, ctx = selection_gap(results)

    lines.append("")
    lines.append("review gap 6 — this band against the published one")
    lines.append(f"  {'published pooled range, #393':<44} "
                 f"{published:.4f}  largest range at either head budget, "
                 f"over every cell")
    for cell, head, end, rng in sorted(cell_rows, key=lambda r: -r[-1]):
        lines.append(f"  {'  same cell, ' + end + ' ' + head:<44} "
                     f"{rng:.4f}  #393, {cell}")
    if gap is not None:
        lines.append(f"  {'gap that made 1.0660 the best':<44} "
                     f"{gap:.4f}  rank {ctx.get('rank')} of "
                     f"{ctx.get('n_published')}, runner-up "
                     f"{ctx.get('runner_up')}")
    if here is None:
        lines.append("  this card's largest head-seed range: no row has two "
                     "draws yet")
        return lines
    label_range = "this card's largest head-seed range"
    label_std = "this card's pooled head-seed std"
    n_rows = sum(1 for _, _, g in table if len(g) >= 2)
    lines.append(f"  {label_range:<44} {here:.4f}  over {n_rows} "
                 f"(stop, head) rows")
    if pooled_here is not None:
        lines.append(f"  {label_std:<44} {pooled_here:.4f}")
    if gap is not None:
        if here >= gap:
            lines.append(f"  VERDICT: the band ({here:.4f}) covers the "
                         f"{gap:.4f} gap that made 1.0660 the best. This run "
                         f"cannot resolve a move of that size, and that "
                         f"sentence is the headline.")
        else:
            lines.append(f"  VERDICT: the band ({here:.4f}) is narrower than "
                         f"the {gap:.4f} gap that made 1.0660 the best, so a "
                         f"move larger than the band is readable.")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=int, action="append", dest="stops",
                    help="a stop to report on. Repeatable.")
    ap.add_argument("--results", default=full_pass.RESULTS)
    ap.add_argument("--parent", default=full_pass.PARENT_RESULTS)
    ap.add_argument("--csv", help="write the table here")
    a = ap.parse_args(argv)

    stops = a.stops or ([full_pass.RESUME_STEP] + full_pass.STOPS)
    table = rows(stops, a.results, a.parent)
    if not table:
        print("no draws on disk yet")
        return 0

    print(f"{'stop':>7}  {'head':<8} {'draws':>5}  {'mean':>7}  "
          f"{'std':>7}  {'range':>7}  seeds")
    lines = []
    for stop, head, got in table:
        values = list(got.values())
        stats = band(values)
        mean = f"{statistics.fmean(values):.4f}"
        std = f"{stats[1]:.4f}" if stats else "  -   "
        rng = f"{stats[2]:.4f}" if stats else "  -   "
        seeds = "  ".join(f"s{s}={v:.4f}" for s, v in sorted(got.items()))
        print(f"{stop:>7}  {head:<8} {len(values):>5}  {mean:>7}  "
              f"{std:>7}  {rng:>7}  {seeds}")
        lines.append((stop, head, len(values), mean, std, rng, got))

    pooled = pooled_std(table)
    if pooled is not None:
        print(f"pooled head-seed std over the rows with 2 draws or more: "
              f"{pooled:.4f}")

    again = redraw_table(stops, a.results, a.parent)
    if again:
        print("")
        print("review gap 2 — the protocol seed drawn again here")
        print(f"{'stop':>7}  {'head':<8} {'#373':>7}  {'here':>7}  "
              f"{'delta':>8}")
        for stop, head, anchor, value, delta in again:
            print(f"{stop:>7}  {head:<8} {anchor:>7.4f}  {value:>7.4f}  "
                  f"{delta:>+8.4f}")
        print("  Same head seed, same backbone. Machine and code version "
              "are the only difference.")
    else:
        print("")
        print("review gap 2 — the protocol re-draw has not landed yet")

    for line in compare(table, a.results):
        print(line)

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stop", "head", "n_draws", "mean", "std", "range",
                        "seeds", "redraw_anchor", "redraw_here",
                        "redraw_delta"])
            back = {(s, h): (anchor, value, delta)
                    for s, h, anchor, value, delta in again}
            for stop, head, n, mean, std, rng, got in lines:
                extra = back.get((stop, head), ("", "", ""))
                w.writerow([stop, head, n, mean, std, rng,
                            " ".join(f"{s}={v}" for s, v in sorted(got.items())),
                            *extra])
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
