#!/usr/bin/env python3
"""#373 — every table the report carries, from the score files and the splits.

Eight tables, in the order the report asks its questions:

  1. coverage        which of the card's 14 cells this study trained.
  2. reproduction    this study's retrained k = 0 against the published one,
                     grouped by the machine it trained on.
  3. depth response  each arm's own k = 0 against its deeper runs, on the
                     full 97 and on the card's horizon criterion.
  4. bootstrap       the paired dataset-cluster interval behind every one of
                     those deltas, per horizon subset.
  5. B5's backbones  one cell trained three times: two seeds, two machines.
  6. A3 controls     the depth ladder beside the re-weighting control.
  7. cost            step time, and which runs had a card to themselves.
  8. depth-0 gap     the depth-0 forecast error of each deeper run against
                     its own k = 0, over four end-of-run windows.
  9. glossary        every term this report uses that is not standard in
                     the field.

Every delta is against the SAME arm's own k = 0. No delta in this file is
computed against a published number or against another backbone.

Two things every delta table carries, because the study cannot separate
them from the depth and a reader must see them where the number is:

  machine   whether the two sides trained on the same box. The reproduction
            table separates on the machine and not on the seed.
  ✗         a retracted row. B5·s1's k = 0 misses its published value by
            0.1169, so its depth delta stands on a baseline the parents do
            not recognise.

Numbers come from `splits.csv`, not from the 4-decimal `score_*.txt` files,
so a Δ here and a Δ in `bootstrap.csv` are the same difference of the same
two full-precision values.

The same text goes two places: `results/scores.md`, and the report itself
between its `<!-- TABLES:BEGIN -->` and `<!-- TABLES:END -->` markers. The
report standard puts the tables in the canonical report; writing them there
by hand would let them drift from the score files, so the rebuild writes
them.

Usage: tables.py --results <results dir> --out <scores.md> [--inject <md>]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_ladder as L2                                    # noqa: E402
import runs as R                                          # noqa: E402
from published import (PUBLISHED as PUB_ALL, GATE,             # noqa: E402
                       NOISE_BAND, PRINT_QUANT, PUBLISHED_SEED,
                       REEVAL_FLOOR, RESOLUTION, SEED_BAND, verdict)

# What every bootstrap interval in this file is over, said where the
# intervals are. Both B5 contrasts are ONE run pair each.
INTERVAL_SCOPE = (
    "Every interval here is a paired dataset-cluster bootstrap over the 97 "
    "eval configs of ONE run pair. It bounds the eval sample: how far the "
    "difference between these two runs could move if the datasets had been "
    "drawn again. It does not bound run-to-run variance, and neither "
    "contrast has a replicate to bound it with. No two of B5's three "
    "backbones share both a seed and a machine.")

CARD_CELLS = ["A1", "A2", "A3", "A4"] + [f"B{i}" for i in range(1, 11)]

# Why a cell has no bb200k. Round 3's extend rule sent eight cells on and held
# five, and gave A4 one head rather than two. Without this the stop-ladder
# table reads a blank the same way for a cell that was never asked to run and
# for one whose eval is still going.
STOPPED_AT_100K = ["A1", "B3", "B5", "B7", "B9"]
EXTEND_NOTE = {(c, h): "the extend rule held this cell at 100k"
               for c in STOPPED_AT_100K for h in ("student", "teacher")}
EXTEND_NOTE[("A4", "teacher")] = "extended by hand; the rule's move is inside the band"
# B8 is the round's new cell: it started from step 0 and the queue took it to
# 100k, the stop every other cell already held. It was never queued past it.
for _h in ("student", "teacher"):
    EXTEND_NOTE[("B8", _h)] = "trained from 0 this round; queued to 100k only"
# B1's bb40k number carries no note here. Round 1 wrote it under a `G6_`
# name no later script could find, and the report's annex says so once, in
# full. Two copies of one operational fact drift; the annex keeps the copy,
# because this column is for the extend rule's reason and that is not one.

# One cell the extend rule could not decide. B1's two heads move in opposite
# directions and both moves are far inside the ±0.0384 head-seed band, so the
# arithmetic returns a split the numbers do not support. The card extended it
# by hand. The stop-reason table says so rather than printing a rule that did
# not fire.
#
# A4 is the second. Its student head moved down and earns 200k on the rule.
# Its teacher head moved +0.0019, which the rule reads as `up` and stops.
# +0.0019 is 5% of the head-seed band, so that reading is noise, not a
# result. A4 is this study's strongest cell and its 200k backbone was
# already on disk, so the head cost one free elisa card and the eval cost
# cores. Extended by hand for the same reason B1 was.
STOP_CALL = {"B1": ("extend both heads",
                    "the card's call: both moves sit inside the ±0.0384 "
                    "head-seed band, so the rule decides nothing"),
             "A4": ("extend both heads",
                    "the student head moved down; the teacher head moved "
                    "+0.0019, 5% of the ±0.0384 head-seed band, so the rule "
                    "decides nothing there. Extended by hand, on free "
                    "hardware")}

# Coverage is read off the score files, not off the run registry.
#
# The registry knows round 1's 32 runs and nothing after them, so a coverage
# section built from it printed `never ran` for ten cells that carry numbers,
# and kept printing it however many rounds ran. A score file is the thing
# that says a number exists, so it is what the section counts.
#
# Three tag shapes reach the results directory. Round 2 and 3 write the
# canonical `<CELL>_k<K>_bb<S>k_<enc>`. Round 1 wrote a `G<n>_` prefix and
# sometimes a suffix for a control or a repeat, `G5_B5_s2_k3_bb40k_student`.
# `G1_B5pub_bb40k_student` carries no `k`: it puts this study's head on the
# parent's own published checkpoint and trains no backbone, so it is not
# coverage of a cell and the regex leaves it out.
CELL_RE = re.compile(
    r"(?:^|_)(A[1-4]|B(?:[1-9]|10))_(?:[a-z0-9]+_)*k(\d+)_(?:[a-z0-9]+_)*"
    r"bb(\d+)k_(student|teacher)$")


def cell_stops(scores):
    """`{cell: {(k, stop_k)}}` over every tag that holds a number."""
    out = {}
    for tag in scores:
        m = CELL_RE.search(tag)
        if m:
            out.setdefault(m.group(1), set()).add((int(m.group(2)),
                                                   int(m.group(3))))
    return out


def read_scores(results):
    out = {}
    for p in Path(results).glob("score_*.txt"):
        try:
            out[p.name[len("score_"):-len(".txt")]] = float(p.read_text().strip())
        except ValueError:
            continue
    return out


def read_splits(results):
    """`{tag: {split name: value}}`."""
    out = {}
    path = Path(results) / "splits.csv"
    if not path.is_file():
        return out
    for r in csv.DictReader(open(path)):
        out.setdefault(r["stop"], {})[r["name"]] = float(r["gm_rel_mase"])
    return out


def read_bootstrap(results):
    """`{(label, subset): row}` from bootstrap.csv."""
    path = Path(results) / "bootstrap.csv"
    if not path.is_file():
        return {}
    return {(r["label"], r["subset"]): r for r in csv.DictReader(open(path))}


def read_steptime(results):
    """Rows of steptime_solo.csv, keyed by (arm, k)."""
    path = Path(results) / "steptime_solo.csv"
    if not path.is_file():
        return {}
    return {(r["arm"], int(r["k"])): r for r in csv.DictReader(open(path))}


def boot_label(arm, k, head):
    """The label `find_artefacts.py --what pairs` gave this comparison."""
    return f"{arm.replace(chr(183), '_')}_k{k}_{head}"


def fmt(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pct(a, b):
    return "—" if None in (a, b) else f"{100.0 * (b / a - 1.0):+.1f}%"


def mark(arm):
    """The retraction marker, where the report has withdrawn an arm."""
    return " ✗" if arm in R.RETRACTED else ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject", help="report to write the tables into, "
                                     "between its TABLES markers")
    args = ap.parse_args(argv)

    sc = read_scores(args.results)
    sp = read_splits(args.results)
    bs = read_bootstrap(args.results)
    st = read_steptime(args.results)
    tags = sorted(sc)
    reg = R.resolve_all(tags)
    L = []

    def val(tag, name="all"):
        """The full-precision score for a tag, or the 4-decimal fallback."""
        v = sp.get(tag, {}).get(name)
        return v if v is not None else (sc.get(tag) if name == "all" else None)

    # ---- 1. coverage -------------------------------------------------------
    cov = cell_stops(sc)
    trained = sorted(cov, key=CARD_CELLS.index)
    missing = [c for c in CARD_CELLS if c not in cov]
    stops = sorted({s for v in cov.values() for _, s in v})
    L += ["### Coverage", "",
          f"The card names 14 cells. This study scored **{len(trained)} of "
          f"them**: {', '.join(trained)}." +
          (f" It never ran **{len(missing)}**: {', '.join(missing)}."
           if missing else " Every cell carries a number."), "",
          "| cell | f-bearing term | EMA α | depths trained | stops scored |",
          "|---|---|---|---|---|"]
    for cell in CARD_CELLS:
        v = cov.get(cell, set())
        ks = sorted({k for k, _ in v})
        ss = sorted({s for _, s in v})
        L.append(f"| {cell} | {L2.term(cell)} | {L2.CELL_ARM[cell][2]} | "
                 f"{', '.join(f'k = {k}' for k in ks) if ks else '**never ran**'} | "
                 f"{', '.join(f'bb{s}k' for s in ss) if ss else '—'} |")
    L += ["", "Stops scored: " + ", ".join(f"bb{s}k" for s in stops) +
          ". The card's extend rule reads a cell's bb40k number against its "
          "bb100k number, so it fires only where both are in hand.", ""]

    # ---- 1a'. k = 3 against the published k = 0 -----------------------------
    # The card's own question, on one grid: does training the forecaster on
    # its own output beat the parents' published number for the SAME cell, at
    # the SAME stop, on the SAME head? Every other table here contrasts runs
    # this study made. This one is the only place the study meets the numbers
    # it set out to beat. Group B's parents published the student head alone,
    # so its teacher rows carry no baseline and form no delta.
    def sv(cell, stop, head):
        return val(f"{cell}_k3_bb{stop}k_{head}")

    def pub(cell, head, stop):
        return PUB_ALL.get(cell, {}).get(head, {}).get(stop)

    # An interval on the delta against the published number, where the parent
    # committed the per-config CSV that makes the pairing recoverable.
    # `published_bootstrap.py` accepts a parent CSV only after it reproduces
    # that parent's own printed aggregate, so a row here is matched and not
    # assumed. Group A's parent committed none, so its rows carry no interval.
    pub_ci = {}
    pbp = Path(args.results) / "published_bootstrap.csv"
    if pbp.is_file():
        for r in csv.DictReader(open(pbp)):
            if r["subset"] != "all":
                continue
            m = re.match(r"^(A[1-4]|B(?:[1-9]|10))_vs_pub_bb(\d+)k_(\w+)$",
                         r["label"])
            if m:
                pub_ci[(m.group(1), int(m.group(2)), m.group(3))] = (
                    float(r["ci_lo"]), float(r["ci_hi"]))

    # Two tallies, because the two head columns do not cover the same cells.
    # Every one of the 14 cells has a published STUDENT number at bb100k, and
    # only group A published a teacher, so a pooled count would silently
    # weight group A twice and change the verdict.
    #
    # The student tally counts MODELS, not cells. A1 and B3 hold one student
    # between them: `arm5_combab` aligns to the student and passes no
    # `--moco-rep-keys`, so the EMA regime that separates the two cells
    # reaches the teacher and nothing else, and all 110 student tensors agree
    # exactly at both stops (`results/pair_identity.tsv`). The two cells meet
    # two different published baselines, so the TABLE keeps both rows. The
    # COUNT must not, or one model lands in the `better` bucket twice.
    DUP_STUDENT = {"B3": "A1"}
    # The ‡ mark and the count exclusion are two different jobs. BOTH cells
    # of the pair carry the shared student, so both rows are marked; only one
    # of them may be counted. Marking one row alone reads as if that cell
    # were the odd one out.
    SHARED_STUDENT = set(DUP_STUDENT) | set(DUP_STUDENT.values())
    rows = []
    tally = {h: {"better": 0, "flat": 0, "worse": 0}
             for h in ("student", "teacher")}
    dup_rows = []
    for cell in CARD_CELLS:
        for head in ("student", "teacher"):
            dup = head == "student" and cell in DUP_STUDENT
            if dup:
                dup_rows.append(f"{cell}/{DUP_STUDENT[cell]}")
            cs = []
            for stop in (40, 100, 200):
                mine, base = sv(cell, stop, head), pub(cell, head, stop)
                if mine is None or base is None:
                    cs += [fmt(mine), fmt(base), "—", "—"]
                    continue
                d = mine - base
                v = ("better" if d <= -NOISE_BAND else
                     "worse" if d >= NOISE_BAND else "flat")
                if stop == 100 and not dup:
                    tally[head][v] += 1
                ci = pub_ci.get((cell, stop, head))
                if ci is not None:
                    v += f"<br>[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                cs += [fmt(mine), fmt(base), f"{d:+.4f}", v]
            shared = head == "student" and cell in SHARED_STUDENT
            rows.append(f"| {cell} | {head}{' ‡' if shared else ''} | "
                        + " | ".join(cs) + " |")

    def tally_line(head):
        t = tally[head]
        n = sum(t.values())
        return (f"{n} distinct models, **{t['better']} better, "
                f"{t['flat']} flat, {t['worse']} worse**")

    L += ["### This study's k = 3 against the published k = 0", "",
          "GM-Relative MASE over the same 97 GIFT-Eval configs, strategy B4, "
          "horizon 16. Δ is this study minus the published number, so "
          "negative is a gain. A verdict reads Δ against the ±"
          f"{NOISE_BAND:.4f} head-seed band: closer than that is `flat`.", "",
          "A dash is a number no parent published. Group B's two parents "
          "print one head per row, the student, so group B has no published "
          "teacher to meet.", "",
          "At bb100k, the stop every one of the 14 cells reached. The count "
          "is over distinct MODELS. ‡ marks the two cells that share one "
          "student, "
          f"so 14 cells hold {sum(tally['student'].values())} student models "
          f"and the shared one counts once. Student head: "
          f"{tally_line('student')}. Teacher head, group A only: "
          f"{tally_line('teacher')}.", "",
          "Read the verdict column as a screen and not as a test. It "
          "compares against a baseline this study did not retrain on its own "
          f"machine, and the ±{NOISE_BAND:.4f} band it thresholds on bounds "
          "the HEAD seed alone. The card's own criterion is the per-horizon "
          "one, and the depth-response table below is where it is applied.",
          "",
          "The second line of a verdict cell is its 95% paired "
          f"dataset-cluster interval. Every one of the {len(pub_ci)} deltas "
          "in this table carries one. The three parents' per-config CSVs are "
          "all in reach, so the pairing against them is recoverable: same 97 "
          "configs, same seasonal-naive denominator file, same resampling "
          "unit as every other interval in this report. "
          "`published_bootstrap.py` accepts a parent CSV for a cell only "
          "after that CSV reproduces the number the parent printed, to four "
          f"decimals. All {len(pub_ci)} did, and none was dropped. The "
          "interval bounds the eval sample. It does not bound the machine, "
          "which separates the two sides of every one of these deltas.", "",
          "| cell | head | 40k k=3 | 40k pub | Δ | | 100k k=3 | 100k pub | Δ "
          "| | 200k k=3 | 200k pub | Δ | |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] \
         + rows + [""]

    # ---- 1b. the stop ladder -----------------------------------------------
    # Round 3's own question. The extend rule sent eight cells from 100k to
    # 200k and held five at 100k, so this is the column that says whether the
    # extra 100,000 steps bought anything. It reads the score files, so a row
    # fills the moment its eval lands and stays `—` until then.
    #
    # The interval on each Δ, from the same paired dataset-cluster bootstrap
    # the depth contrasts use. A Δ with no interval beside it cannot be read
    # against the head-seed band, so the column ships with the numbers.
    stop_ci = {}
    sbp = Path(args.results) / "stop_bootstrap.csv"
    if sbp.is_file():
        for r in csv.DictReader(open(sbp)):
            if r["subset"] != "all":
                continue
            cell, _, head = r["label"].partition("_stop200v100_")
            stop_ci[(cell, head)] = (float(r["ci_lo"]), float(r["ci_hi"]))

    rows, deltas = [], []
    for cell in CARD_CELLS:
        for head in ("student", "teacher"):
            a, b, c = (sv(cell, 40, head), sv(cell, 100, head),
                       sv(cell, 200, head))
            if b is None:
                continue
            d = None if c is None else c - b
            if d is not None:
                deltas.append((d, cell, head))
            ci = stop_ci.get((cell, head))
            ci_s = "—" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            rows.append(f"| {cell} | {head} | {fmt(a)} | {fmt(b)} | "
                        f"{fmt(c)} | {'—' if d is None else f'{d:+.4f}'} | "
                        f"{ci_s} | "
                        f"{pct(b, c)} | {EXTEND_NOTE.get((cell, head), '')} |")

    better = [t for t in deltas if t[0] < 0]
    if not deltas:
        lead = "No bb200k number has landed yet."
    else:
        n, plural = len(deltas), "" if len(deltas) == 1 else "s"
        best = min(deltas)
        # With no negative delta there is no gain to name, and calling the
        # least-bad row one would invert the finding.
        edge = (f"The largest gain is {best[1]} {best[2]}, {best[0]:+.4f}."
                if better else
                f"None gained; the smallest loss is {best[1]} {best[2]}, "
                f"{best[0]:+.4f}.")
        # Every row in hand, and the summary statistics over exactly those
        # rows. Round 3 published a mean and a median over 14 of these 16,
        # with no line saying which two it left out — the A4 pair, the best
        # cell at every stop and one of the two that gain on both heads.
        # Dropping it moved the mean from +0.0079 to +0.0103. The count and
        # the statistics now come from one list, so a subset cannot reach
        # the prose again without the count moving with it.
        vals = sorted(t[0] for t in deltas)
        mean = sum(vals) / n
        mid = (vals[n // 2] if n % 2
               else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))
        inband = sum(1 for v in vals if abs(v) <= NOISE_BAND)
        lead = (f"Of the {n} extended measurement{plural} in hand, "
                f"**{len(better)} improved** at bb200k and "
                f"{n - len(better)} got worse. " + edge +
                f" Over all {n}: mean {mean:+.4f}, median {mid:+.4f}. The "
                f"±{NOISE_BAND:.4f} head-seed band covers {inband} of "
                f"them.")
    L += ["### The stop ladder: what the second 100,000 steps buys", "",
          "Δ is bb200k minus bb100k, so a negative number is an improvement: "
          "GM-Relative MASE is a ratio against seasonal-naive and lower is "
          "better. " + lead, "",
          "The interval is a 95% paired dataset-cluster bootstrap over the "
          "pair's 97 configs. It bounds the eval sample, not run-to-run "
          "variance. The head-seed band is ±0.0384.", "",
          "| cell | head | bb40k | bb100k | bb200k | Δ | 95% CI | % | note |",
          "|---|---|---|---|---|---|---|---|---|"] + rows + [""]

    # ---- 1b''. the one row of that ladder that is not like the others -------
    # A3's bb200k student carries the ladder's largest move, and it is the
    # only place in the grid where two heads of ONE backbone disagree by more
    # than 0.05. Either the head-seed band is far too narrow or that head is a
    # bad draw, and one more draw of the same head tells them apart.
    RESEED = "A3_k3_bb200k_student_s20260723"
    draw2 = sc.get(RESEED)
    hgaps = sorted(
        ((abs(a - b), cell, stop)
         for cell in CARD_CELLS for stop in (40, 100, 200)
         for a, b in [(val(f"{cell}_k3_bb{stop}k_student"),
                       val(f"{cell}_k3_bb{stop}k_teacher"))]
         if a is not None and b is not None), reverse=True)
    if draw2 is not None and hgaps:
        d1 = val("A3_k3_bb200k_student")
        te = val("A3_k3_bb200k_teacher")
        top = hgaps[0][0]
        nxt = max(g for g, _c, _s in hgaps[1:])
        nxt_a = max(g for g, c, _s in hgaps[1:] if c.startswith("A"))
        turns = []
        for cell in CARD_CELLS:
            v = [val(f"{cell}_k3_bb{s}k_student") for s in (40, 100, 200)]
            if any(x is None for x in v):
                continue
            mono = v[0] >= v[1] >= v[2] or v[0] <= v[1] <= v[2]
            turns.append((cell, v, mono))
        n_turn = sum(1 for _c, _v, m in turns if not m)
        L += ["### A3's bb200k student, drawn twice", "",
              f"A3 at bb200k reads {d1:.4f} on the student and {te:.4f} on "
              f"the teacher, off one backbone file. That {top:.4f} gap is "
              f"the largest in the grid: {top / nxt_a:.1f}x the next-largest "
              f"in group A ({nxt_a:.4f}) and {top / nxt:.1f}x the largest of "
              f"the other {len(hgaps) - 1} gaps ({nxt:.4f}). Every "
              "gap in the grid is in "
              "[`results/head_gap.tsv`](results/head_gap.tsv).", "",
              "The second draw changes two things: the head seed, and the "
              "machine that trained the head. Draw 1 trained on the rented "
              "box, draw 2 on elisa. Both read the same 200,000-step backbone "
              "checkpoint, the box's original and elisa's synced copy of it. "
              "Held across the two draws: 30,000 head steps, the recipe, and "
              "the 97-config eval, which ran on elisa's cores for both. Only "
              "elisa's copy carries a recorded md5 "
              "(`9f0e8da71ff595523d2bf0dabdf80445`, "
              "[`results/eval/A3_k3_bb200k_student_s20260723/backbone_md5.txt`]"
              "(results/eval/A3_k3_bb200k_student_s20260723/backbone_md5.txt))"
              "; the box was released before its original could be "
              "checksummed.", "",
              "| draw | head seed | GM-Relative MASE | against draw 1 |",
              "|---|---|---|---|",
              f"| 1, student | 20260722 | {d1:.4f} | — |",
              f"| 2, student | 20260723 | {draw2:.4f} | {draw2 - d1:+.4f} |",
              f"| teacher | 20260722 | {te:.4f} | {te - d1:+.4f} |", ""]
        # The redraw is the test, so the section states what it decided
        # rather than leaving the reader to subtract. Both branches are
        # written out: the numbers pick one.
        agree = abs(draw2 - d1) <= NOISE_BAND
        gap2 = abs(draw2 - te)

        def ci(label):
            r = bs.get((label, "all"))
            return ("" if r is None else
                    f" [{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]")

        seed_ci = ci("A3_200k_headseed_student")
        # The redraw's teacher-minus-student gap, from the close pipeline's
        # own check file rather than from this study's pair registry: the
        # redraw is not one of the registry's pairs.
        gap2_ci, gap2_d = "", -gap2
        for r in csv.DictReader(open(Path(args.results) / "final_check.csv")):
            if r["label"] == "item6 teacher_vs_draw2" and r["subset"] == "all":
                gap2_d = float(r["delta"])
                gap2_ci = (f" [{float(r['ci_lo']):+.4f}, "
                           f"{float(r['ci_hi']):+.4f}]")
        # One paragraph, three facts: the two draws agree, the gap outlives
        # the redraw, and what the agreement does and does not bound. The
        # arithmetic behind each is in the table above it and in
        # `results/head_gap.tsv`, so restating it here would only add prose.
        L += [(f"**The two draws agree.** They sit {abs(draw2 - d1):.4f} "
               f"apart{seed_ci}, so {d1:.4f} is not a bad draw. The "
               f"student/teacher gap survives the redraw at {gap2_d:+.4f}"
               f"{gap2_ci}, teacher minus student. The two draws cross a "
               f"machine, so this agreement bounds the head seed and the "
               f"machine together, not the seed alone."
               if agree else
               f"**The two draws disagree.** They sit {abs(draw2 - d1):.4f} "
               f"apart{seed_ci}, wider than the ±{NOISE_BAND:.4f} head-seed "
               f"band, so this head's seed alone moves the score more than "
               f"the band the report thresholds on."), "",
              f"A3's is the ladder's largest reversal, but it is not the "
              f"only one: {n_turn} of the {len(turns)} three-stop student "
              "trajectories turn round at bb200k.", "",
              "| cell | bb40k | bb100k | bb200k | bb200k − bb100k | shape |",
              "|---|---|---|---|---|---|"]
        for cell, v, mono in turns:
            L.append(f"| {cell} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | "
                     f"{v[2] - v[1]:+.4f} | "
                     f"{'monotone' if mono else 'turns round'} |")
        L.append("")

    # ---- 1b'. why each cell stopped where it stopped -------------------------
    # The ladder above says what the extra steps bought. It does not say why
    # six cells never took them. That decision is the extend rule, and the
    # rule is arithmetic on two numbers the table already holds: a head earns
    # 200k when its bb100k score sits BELOW its bb40k score, and stops when it
    # sits above. Printing the two moves beside the decision makes the rule
    # checkable against the score files rather than taken on trust.
    rows, stopped_inband, extended = [], [], []
    for cell in CARD_CELLS:
        mv = {h: (None if (sv(cell, 40, h) is None or sv(cell, 100, h) is None)
                  else sv(cell, 100, h) - sv(cell, 40, h))
              for h in ("student", "teacher")}
        s, t = mv["student"], mv["teacher"]
        if s is None or t is None:
            continue
        down = [h for h in ("student", "teacher") if mv[h] < 0]
        if cell in STOP_CALL:
            dec, why = STOP_CALL[cell]
        elif len(down) == 2:
            dec, why = ("extend both heads", "both heads moved down")
        elif not down:
            dec, why = ("stop at 100k", "both heads moved up")
        else:
            up = "teacher" if down[0] == "student" else "student"
            dec, why = (f"extend the {down[0]} head",
                        f"split: the {down[0]} head moved down, the {up} "
                        f"head moved up")
        rows.append(f"| {cell} | {s:+.4f} | {t:+.4f} | **{dec}** | {why} |")
        if dec.startswith("stop") and max(abs(s), abs(t)) <= NOISE_BAND:
            stopped_inband.append(cell)
        if dec.startswith("extend"):
            extended.append(cell)

    nstop = sum(1 for r in rows if "stop at 100k" in r)
    L += ["### Stop reasons: what the extend rule read at each cell", "",
          "The rule reads one cell's bb40k number against its bb100k number, "
          "per head. A head that moved down earns the second 100,000 steps; "
          "a head that moved up stops. Both columns are bb100k minus bb40k, "
          f"so negative is an improvement. It held {nstop} cells at 100k.", "",
          "| cell | 40k→100k student | 40k→100k teacher | decision | why |",
          "|---|---|---|---|---|"] + rows + [""]
    # What the rule selects for. The 200k verdict is measured on the panel
    # this rule chose, so the reader needs the rule's three defects at the
    # point the panel is defined, not inferred from the table.
    L += ["**The rule selects the panel, and it selects it on an improving "
          "first leg.** Three properties of that, stated plainly:", "",
          "1. It reads the one contrast this study calls not head-matched. "
          "A bb40k head trains 15,000 steps and a bb100k head 30,000, so "
          "part of every move in the two columns above is the head's own "
          "extra 15,000 steps. The Protocol section says so for the depth "
          "verdict. It is equally true of the rule.",
          f"2. It fires inside its own noise band. {len(stopped_inband)} of "
          f"the {nstop} stopped cells ({', '.join(stopped_inband)}) moved "
          f"less than ±{NOISE_BAND:.4f} on BOTH heads. The verdict table "
          "above calls a move of that size `flat`.",
          "3. The manual overrides go one way. A4 and B1 were extended by "
          "hand because the rule decides nothing inside the band. That "
          "reasoning applies with the same force to the cells in point 2, "
          "and none of them was extended.", "",
          f"So the {len(extended)} extended cells are enriched for cells "
          "that happened to improve from bb40k to bb100k, and regression to "
          "the mean is the expected null at bb200k. This study runs no "
          "control for it. **Read the 200k verdict as conditional on a panel "
          "selected for having improved.**", ""]

    # ---- 1c. the same-arm pairs --------------------------------------------
    # A1 and B3 print one student number at both stops. A reader who meets
    # that in the coverage table above and finds no explanation reads it as
    # a broken path. It is not one: the two cells train ONE student, bit for
    # bit, so the table has to say so where the duplicate appears.
    pid = Path(args.results) / "pair_identity.tsv"
    if pid.is_file():
        rows = [r for r in csv.DictReader(open(pid), delimiter="\t")]
        same = sorted({r["pair"] for r in rows
                       if r["side"] == "student" and r["verdict"] == "IDENTICAL"})
        diff = sorted({r["pair"] for r in rows
                       if r["side"] == "student" and r["verdict"] != "IDENTICAL"})
        if rows:
            L += ["### The four same-arm pairs: two models, or one", "",
                  "Each pair runs ONE arm under the two EMA regimes, group "
                  "A's schedule against group B's fixed 0.9. Every tensor of "
                  "both backbones is compared, split into the student side "
                  "the student head reads and the `teacher_*` side the "
                  "teacher head reads.", "",
                  "Each entry is the count of tensors that agree exactly, "
                  "out of the count compared. A head's file md5 differs "
                  "between two cells even when every weight agrees, so the "
                  "comparison is tensor by tensor and never by md5.", "",
                  "| pair | arm | stop | student | teacher | student head | teacher head |",
                  "|---|---|---|---|---|---|---|"]
            by_key, order = {}, []
            for r in rows:
                k = (r["pair"], r["arm"], int(r["stop_k"]))
                if k not in by_key:
                    by_key[k] = {}
                    order.append(k)
                by_key[k][r["side"]] = f"{r['identical']}/{r['tensors']}"
            for k in order:
                v = by_key[k]
                L.append(f"| {k[0]} | `{k[1]}` | bb{k[2]}k | " + " | ".join(
                    v.get(s, "—") for s in ("student", "teacher",
                                            "head_student", "head_teacher")) + " |")
            L += ["", "Full table, with the largest absolute difference on "
                  "each side: [`results/pair_identity.tsv`]"
                  "(results/pair_identity.tsv).", ""]
            if same:
                L += [f"**{', '.join(same)} hold one student, not two.** "
                      "`arm5_combab` aligns to the student and carries no "
                      "`--moco-rep-keys`, so no loss term reads the EMA "
                      "encoder and the regime sends no gradient into the "
                      "student. One student number for both cells is the "
                      "right answer, and it is ONE measurement: the student "
                      "row of one of them is not a replication of the other. "
                      "The teacher side differs at every stop, and the "
                      "teacher numbers do too.", ""]
            if diff:
                L += [f"**{', '.join(diff)} hold two students.** Their arms "
                      "carry `--moco-rep-keys`, whose keys come from the EMA "
                      "encoder, or align to the teacher. Either path reaches "
                      "the student's gradient, so the regime moves it.", ""]

    # ---- 1d. the A1/B3 duplicate, re-run end to end -------------------------
    # A tensor comparison says the two backbones hold one student. It does not
    # say the head and the eval read the file each cell names. That takes a
    # second pass: train a fresh head from the named checkpoint and re-run the
    # 97 configs into a path no other cell writes. Four re-runs, one per cell
    # per stop. A re-run that lands on the first number closes the path
    # question the same way the tensor table closes the weight question.
    rep_rows = []
    for cell in ("A1", "B3"):
        for stop in (40, 100):
            tag = f"{cell}rep_k3_bb{stop}k_student"
            got = val(tag)
            base = val(f"{cell}_k3_bb{stop}k_student")
            if got is None or base is None:
                continue
            md5 = "—"
            lg = Path(args.results) / f"repro_eval_{tag}.log"
            if lg.is_file():
                for ln in lg.read_text(errors="ignore").splitlines():
                    if "md5=" in ln:
                        md5 = ln.split("md5=")[-1].strip()[:8]
            rep_rows.append((cell, stop, md5, base, got))
    if rep_rows:
        L += ["### The A1/B3 duplicate, re-run end to end", "",
              "Each row trains a fresh student head from the checkpoint its "
              "own cell names, seed 20260722, and runs the 97 configs into "
              "`results/eval/<cell>rep_…`, a directory no other cell writes. "
              "A path that ignored the cell would land the re-run on the "
              "other cell's number.", "",
              "| cell | stop | backbone md5 | first pass | re-run | Δ |",
              "|---|---|---|---|---|---|"]
        for cell, stop, md5, base, got in rep_rows:
            d = got - base
            ds = "0.0000" if abs(d) < 5e-5 else f"{d:+.4f}"
            L.append(f"| {cell} | bb{stop}k | `{md5}` | {base:.4f} | "
                     f"{got:.4f} | {ds} |")
        worst = max(abs(g - b) for _, _, _, b, g in rep_rows)
        both = len({c for c, *_ in rep_rows}) == 2
        L += ["", f"The largest re-run move is {worst:.4f}. "
              + ("The two cells carry different backbone md5s and reproduce "
                 "their own first-pass numbers, so the head and the eval read "
                 "the file each cell names. The duplicate is the student "
                 "weights, not the path."
                 if both else
                 "The remaining re-runs are still on the queue."), ""]

    # ---- 2. reproduction ---------------------------------------------------
    # The seed band, live from the bootstrap that measured it, so re-running
    # the bootstrap moves the gate with it.
    seed_ci = bs.get(("B5_seed_k0_student", "all"))
    seed_band = (max(abs(float(seed_ci["ci_lo"])), abs(float(seed_ci["ci_hi"])))
                 if seed_ci else SEED_BAND)
    L += ["### Reproduction of the published k = 0", "",
          "Same cell, same recipe, same head seed 20260722, same 97-config "
          "B4 eval, student head. Rows are grouped by machine.", "",
          f"A row at the parents' own backbone seed {PUBLISHED_SEED} takes "
          f"the card's {GATE}; a row at any other seed takes the seed "
          "band.", "",
          "| backbone | seed | machine | published k = 0 | retrained k = 0 | "
          "\\|Δ\\| | gate | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    repro = [r for r in R.reproductions(tags) if r.head == "student"]
    repro.sort(key=lambda r: (0 if r.machine == "elisa" else
                              1 if r.run else 2,
                              abs((val(r.tag) or 0)
                                  - (PUB_ALL.get(r.cell, {})
                                     .get("student", {}).get(40) or 0))))
    cross_seed = []
    for r in repro:
        pub = PUB_ALL.get(r.cell, {}).get("student", {}).get(40)
        got = val(r.tag)
        if pub is None or got is None:
            continue
        same = r.seed == PUBLISHED_SEED
        if not same:
            cross_seed.append(r.arm)
        gate = (f"{GATE}, the card" if same
                else f"{seed_band:.4f}, the seed band")
        L.append(f"| {r.arm}{mark(r.arm)} | {r.seed} | {r.machine} | "
                 f"{pub:.4f} | {got:.4f} | {abs(got - pub):.4f} | {gate} | "
                 f"{verdict(abs(got - pub), same, seed_band)} |")
    L += ["", f"Two things this comparison cannot resolve, added: {REEVAL_FLOOR} "
          "for the head and the eval, which is what `B5·pub` moves the "
          "score by while training nothing, and "
          f"{PRINT_QUANT} for the parents' four printed decimals. A |Δ| at "
          f"or below {RESOLUTION:.4f} is a run this pipeline cannot separate "
          f"from the published one. The card's gate of {GATE} is stricter "
          "than that.", ""]
    if cross_seed:
        L += [f"The seed band is {seed_band:.4f}, the far end of the 95% "
              "interval on this study's one measurement of a seed change: "
              "`B5·s2` against `B5·s3`, one machine, one recipe, +0.0035 "
              "[-0.0183, +0.0230]. It is one run pair, and the interval is "
              "over that pair's eval sample rather than over seeds, so the "
              "band is a floor on what a seed can move and not a bound on "
              f"it. {', '.join(sorted(set(cross_seed)))} is the only row it "
              "gates; every other row here carries the parents' own seed.", ""]
    L += ["`B5·pub` is not a training: it takes the parent report's own "
          "published B5 checkpoint and puts this study's head and eval on "
          "it, so its row bounds the head and the eval rather than the "
          "trainer. `B5·s3` is a training, at the protocol seed, on elisa, "
          "and its 97-config eval output is byte-identical to `B5·pub`'s "
          "(`results/eval/G7_B5_k0_e_bb40k_student/all_results.csv` against "
          "`results/eval/G1_B5pub_bb40k_student/all_results.csv`): the "
          "elisa retrain reproduced the parent's backbone exactly, and the "
          f"{REEVAL_FLOOR} both rows carry is the head and the eval.", ""]

    # ---- 3. depth response -------------------------------------------------
    L += ["### Depth response, against each arm's own k = 0", "",
          "| arm | seed | machine held | head | k | k = 0 | this k | Δ | "
          "all | short | med+long | criterion |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    held_arms, depths = set(), set()
    for arm, head, k, base, deep in R.pairs(tags):
        a, b = val(base.tag), val(deep.tag)
        A, B = sp.get(base.tag, {}), sp.get(deep.tag, {})
        depths.add(k)
        if R.machine_held(base, deep):
            held_arms.add(arm)
        ok = "—"
        if A and B:
            dm = 100.0 * (B["medium_long"] / A["medium_long"] - 1.0)
            ds = 100.0 * (B["short"] / A["short"] - 1.0)
            ok = "**MET**" if (dm <= -5.0 and ds < 2.0) else "not met"
        held = R.machine_held(base, deep)
        where = ("yes, " + base.machine if held else
                 f"no, {base.machine} → {deep.machine}")
        L.append(
            f"| {arm}{mark(arm)} | {deep.seed} | {where} | {head} | {k} | "
            f"{fmt(a)} | "
            f"{fmt(b)} | {f'{b - a:+.4f}' if None not in (a, b) else '—'} | "
            f"{pct(A.get('all'), B.get('all'))} | "
            f"{pct(A.get('short'), B.get('short'))} | "
            f"{pct(A.get('medium_long'), B.get('medium_long'))} | {ok} |")
    # The same criterion on the published-baseline pairs. Those pairs cross a
    # machine, so it is a screen there and a test only here.
    screen = []
    scrp = Path(args.results) / "criterion_screen.csv"
    if scrp.is_file():
        screen = list(csv.DictReader(open(scrp)))
    scr_line = ""
    if screen:
        met = sum(1 for r in screen if r["criterion_met"] == "yes")
        s100 = [r for r in screen if r["stop_k"] == "100"]
        m100 = sum(1 for r in s100 if r["criterion_met"] == "yes")
        scr_line = (
            " The same criterion runs over every pair of the "
            "published-baseline table as well, where it is a screen because "
            f"the two sides cross a machine: {met} of {len(screen)} pairs "
            f"meet it, and {m100} of {len(s100)} at bb100k "
            "([`results/criterion_screen.csv`](results/criterion_screen.csv)).")

    L += ["", "Criterion, from the card: medium+long (42 configs) at least "
          "5% better, short (55 configs) losing less than 2%.", "",
          "**This table is the only place the card's criterion is applied as "
          f"a test, and it is answered for {len(held_arms)} machine-held "
          f"arms ({', '.join(sorted(held_arms))}) at one stop, bb40k.** The "
          "card also asks about bb100k and bb200k. No cell holds a "
          "machine-matched `k = 0` at either stop, so at those two the "
          "report has the screen and nothing else." + scr_line, "",
          "`machine held` = did the two sides train on the same box. A `no` "
          "row carries a machine change as well as a depth change. The B5 "
          "table below measures the machine alone, at one seed, at 0.1166, "
          "so a `no` row carries a term larger than most of the deltas in "
          "this table. Only the `yes` rows report the depth and nothing "
          "else.", "",
          "✗ marks a retracted row: " + R.RETRACTED_WHY + ".", "",
          f"Head-seed band ±{NOISE_BAND} (`ema_sched_ladder.md`, pooled). It "
          "bounds the head seed alone. It does not bound the machine, and it "
          "does not bound the BACKBONE seed: this study holds one backbone "
          "seed in 14 cells and one replicate of it (B5·s2 against B5·s3, at "
          "k = 0, at bb40k), so backbone-seed variance is unmeasured. Every "
          "better / flat / worse verdict in this report rests on a band that "
          "bounds one of the two seeds in play.", "",
          "The depths trained are " +
          ", ".join(f"k = {d}" for d in sorted(depths)) +
          ", and only k = 3 ran on the 14 cells. The one ladder that holds "
          "more than a single depth is A3's, the cell where k = 3 does the "
          "most damage, and its k = 1 row is machine-crossed and covers "
          "zero. So this study supports **depth 3 moves the score**. It does "
          "NOT support *depth 3 is the right depth*: no cell measures a "
          "second depth against a machine-held k = 0.", ""]

    # ---- 4. the interval behind every one of those deltas ------------------
    L += ["### Paired dataset-cluster bootstrap, per horizon subset", "",
          "The resampling unit is the dataset: `<ds>/short`, `/medium` and "
          "`/long` are three configs of one series and are not independent "
          "draws. 95% percentile interval over 10,000 resamples. Each "
          "interval is over one run pair's 97 configs, so it bounds the "
          "eval sample and not run-to-run variance.", "",
          "| arm | head | k | subset | n | Δ | 95% CI | resamples improved |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, head, k, _base, _deep in R.pairs(tags):
        for subset in ("all", "short", "medium_long"):
            r = bs.get((boot_label(arm, k, head), subset))
            if r is None:
                continue
            L.append(f"| {arm}{mark(arm)} | {head} | {k} | {subset} | "
                     f"{r['n']} | {float(r['delta']):+.4f} | "
                     f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}] | "
                     f"{100 * float(r['p_improved']):.1f}% |")
    L.append("")

    # ---- 5. B5's three backbones -------------------------------------------
    L += ["### One cell, three backbones", "",
          "B5 (`arm4_combab_fix09`) trained three times on one recipe, one "
          "code snapshot, one head seed and one eval. They differ by backbone "
          "seed and by machine, and each contrast below names which of the "
          "two it changes. The machine moves the score and the seed does "
          "not.", "",
          "| backbone | seed | machine | k = 0 | k = 3 | k = 3 − k = 0 |",
          "|---|---|---|---|---|---|"]
    b5_arms = [a for a in R.arms_of("B5") if a != "B5·pub"]
    b5 = {}
    for arm in b5_arms:
        row = []
        for k in (0, 3):
            run = R.find_run(arm, k, "depth") or R.find_run(arm, k, "control")
            v = val(f"{run.stem}_bb40k_student") if run else None
            row.append(v)
            if v is not None:
                b5[(arm, k)] = v
        if row[0] is None and row[1] is None:
            continue
        d = ("—" if None in row else f"{row[1] - row[0]:+.4f}")
        L.append(f"| {arm}{mark(arm)} | {R.arm_seed(arm)} | "
                 f"{R.arm_where(arm)} | {fmt(row[0])} | {fmt(row[1])} | {d} |")

    L += ["", "| contrast | what changes | k | Δ | 95% CI |",
          "|---|---|---|---|---|"]
    for a1, a2, what, lab in (
            ("B5·s1", "B5·s3", "the machine, at one seed", "machine"),
            ("B5·s2", "B5·s3", "the seed, on one machine", "seed"),
            ("B5·s1", "B5·s2", "the seed AND the machine", "seed_and_machine")):
        for k in (0, 3):
            if (a1, k) not in b5 or (a2, k) not in b5:
                continue
            ci = bs.get((f"B5_{lab}_k{k}_student", "all"))
            L.append(f"| {a1} against {a2} | {what} | {k} | "
                     f"{b5[(a2, k)] - b5[(a1, k)]:+.4f} | "
                     + (f"[{float(ci['ci_lo']):+.4f}, "
                        f"{float(ci['ci_hi']):+.4f}] |" if ci else "— |"))
    L += ["", "Student head, 97 configs. `B5·s3` holds `B5·s1`'s seed and "
          "`B5·s2`'s machine.", "",
          INTERVAL_SCOPE, ""]

    early = list(csv.DictReader(open(Path(args.results) / "early_loss.csv"))) \
        if (Path(args.results) / "early_loss.csv").is_file() else []
    if early:
        cols = [a for a in R.ARM_ORDER if any(r["arm"] == a for r in early)]
        steps = sorted({int(r["step"]) for r in early})
        cell_of = {(r["arm"], int(r["step"])): r for r in early}
        L += ["`mixup` counts the examples the mixer touched in the "
              "200-step window, so one count at every step is one data "
              "order. `B5·s1` and `B5·s3` carry one seed, print one count "
              "at every step, and their losses still part.", "",
              "| step | " + " | ".join(
                  f"{a}<br>seed {R.arm_seed(a)}, {R.arm_where(a)}"
                  for a in cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
        for s in steps:
            row = []
            for a in cols:
                r = cell_of.get((a, s))
                row.append(f"{r['loss']}  `{r['mixup']}`" if r else "—")
            L.append(f"| {s} | " + " | ".join(row) + " |")
        L.append("")

    # ---- 6b. B1 control, on the cell where k = 3 WINS ------------------------
    # A3's control answers the same question on the cell where k = 3 does the
    # most damage, and every column of that table crosses a machine. B1 holds
    # the machine, the seed and the head budget, so this table may divide one
    # column by another and A3's may not.
    B1_COLS = ("G6_B1_k0", "G_B1_k0_aw4", "G6_B1_k3")
    B1_BOOT = (None, "B1_alignx4_{h}", "B1_k3_{h}")
    b1_rows, b1_split = [], []
    for head in ("student", "teacher"):
        v = [val(f"{t}_bb40k_{head}") for t in B1_COLS]
        if any(x is None for x in v):
            continue
        cells = []
        for value, lab in zip(v, B1_BOOT):
            r = bs.get((lab.format(h=head), "all")) if lab else None
            cells.append(f"{value:.4f}" + (
                f"<br>{float(r['delta']):+.4f} "
                f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]"
                if r else ""))
        b1_rows.append("| " + head + " | " + " | ".join(cells) + " |")
        b1_split.append((head, v[1] - v[0], v[2] - v[1], v[2] - v[0]))
    if b1_rows:
        machines = sorted({R.resolve(f"{c}_bb40k_student").machine
                           for c in B1_COLS
                           if R.resolve(f"{c}_bb40k_student")})
        held = len(machines) == 1
        L += ["### B1: is the win the depth, or the weight?", "",
              "B1 carries `L_align` as its only f-bearing term, so its "
              "`k = 3` run multiplies that term's weight against the f-free "
              "terms by 4 as well as adding depth. The `L_align x4` row "
              "applies the re-weighting at k = 0, with no depth at all.", "",
              "| head | k = 0 | k = 0, `L_align` x4 | k = 3 |",
              "|---|---|---|---|"] + b1_rows
        L += ["", "Second line of each cell: the difference against `k = 0` "
              "and its 95% paired dataset-cluster interval.", ""]
        L.append(
            ("Every column trained on " + machines[0] + " at backbone seed "
             "20260520, on the same head budget. This is the study's one such "
             "table, so it may divide one column by another."
             if held else
             "The columns do not share a machine (" + ", ".join(machines) +
             "), so read them as direction and not as magnitude.") + "")
        L += ["", "| head | the re-weighting<br>k = 0 → x4 | the depth<br>x4 "
              "→ k = 3 | total<br>k = 0 → k = 3 | the re-weighting's share |",
              "|---|---|---|---|---|"]
        for head, rw, dp, tot in b1_split:
            L.append(f"| {head} | {rw:+.4f} | {dp:+.4f} | {tot:+.4f} | "
                     f"{100 * rw / tot:.0f}% |" if tot else
                     f"| {head} | {rw:+.4f} | {dp:+.4f} | {tot:+.4f} | — |")
        L.append("")

    # ---- 7. A3 controls ----------------------------------------------------
    L += ["### A3: is the damage the depth, or the weight?", "",
          "Summing the depths multiplies `L_align`'s weight against the "
          "f-free terms by k + 1. The `L_align x4` row applies that "
          "re-weighting at k = 0, with no depth at all.", "",
          "| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 |",
          "|---|---|---|---|---|"]
    A3_COLS = ("A3_k0", "G3_A3_k0_aw4", "G3_A3_k1", "A3_k3")
    A3_BOOT = (None, "A3_alignx4_{h}", "A3_k1_{h}", "A3_k3_{h}")
    for head in ("student", "teacher"):
        v = [val(f"{t}_bb40k_{head}") for t in A3_COLS]
        if v[0] is None or v[3] is None:
            continue
        cells = []
        for value, lab in zip(v, A3_BOOT):
            if value is None:
                cells.append("—")
                continue
            r = bs.get((lab.format(h=head), "all")) if lab else None
            cells.append(f"{value:.4f}" + (
                f"<br>{float(r['delta']):+.4f} "
                f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]"
                if r else ""))
        L.append("| " + head + " | " + " | ".join(cells) + " |")
    machines = " · ".join(
        f"{c}: {r.machine}" for c, r in
        ((c, R.resolve(f"{c}_bb40k_student")) for c in A3_COLS) if r)
    L += ["", "Second line of each cell: the difference against `k = 0` "
          "and its 95% paired dataset-cluster interval.", "",
          "Every column trained on a different box from at least one "
          f"other. {machines}. The machine alone is worth 0.1166 on this "
          "study's one controlled measurement of it, which is more than "
          "either control's own size, so read the two controls as direction "
          "and not as magnitude. This table therefore does not divide one "
          "column by another.", ""]

    # ---- 8. what the depth costs -------------------------------------------
    # A solo row does not have to be solo for its whole run: a clone of the
    # run itself can hold the card for a stretch, and then the median is over
    # the windows after it. The `alone?` column cannot carry that, so the
    # paragraph above the table names each such row and what it was measured
    # over.
    clones = " ".join(
        f"{arm}'s `k = {k}` shared {st[(arm, k)]['machine']} with a clone of "
        f"itself up to step {int(st[(arm, k)]['first_solo_step']):,}, and its "
        f"{float(st[(arm, k)]['compute_ms']):.1f} ms is the median over the "
        f"{st[(arm, k)]['windows_solo']} windows after that."
        for arm in R.ARM_ORDER for k in sorted(k for a, k in st if a == arm)
        if st[(arm, k)]["solo"] == "yes"
        and "clone" in st[(arm, k)]["neighbours"]
        and st[(arm, k)]["first_solo_step"])
    L += ["### What the depth costs", "",
          "Median `fwd + bwd` per step, from each run's own trainer log. A "
          "median is a cost of the depth only where the run had the card to "
          "itself, so the table says which did. `run_provenance.py` reads "
          "that off the driver logs and "
          "[`results/steptime_solo.csv`](results/steptime_solo.csv) carries "
          "it per run." + (f" {clones}" if clones else ""), "",
          "| arm | f-bearing term | k | machine | card | fwd+bwd | alone? |",
          "|---|---|---|---|---|---|---|"]
    order = {a: i for i, a in enumerate(R.ARM_ORDER)}
    for (arm, k) in sorted(st, key=lambda x: (order.get(x[0], 99), x[1])):
        r = st[(arm, k)]
        run = R.find_run(arm, k)
        ms = (f"{float(r['compute_ms']):.1f} ms" if r["compute_ms"] else
              f"{float(r['compute_ms_contended']):.1f} ms, shared"
              if r["compute_ms_contended"] else "—")
        # No ✗ here. The retraction is of B5·s1's depth DELTA, which rests on
        # a k = 0 the parents do not recognise. Its step time is a wall clock
        # and nothing about the score touches it.
        L.append(f"| {arm} | {run.term if run else '?'} | {k} | "
                 f"{r['machine']} | {r['card']} | {ms} | "
                 f"{'yes' if r['solo'] == 'yes' else 'no — ' + r['why_not_solo']} |")
    # Solo is necessary and it is not sufficient. A median over the tail of a
    # run is a median over whatever the tail held, so a row whose solo
    # windows are a fraction of its own run is not the same measurement as a
    # row solo throughout. A3's k = 3 is that row, and it is also the row
    # that disagrees with both other probes by an order of magnitude. It
    # stays in the table, marked, rather than leaving without a line.
    def full_run(r):
        try:
            return int(r["windows_solo"]) >= int(r["windows_total"]) - 1
        except (ValueError, KeyError, TypeError):
            return False

    # The two probes the report carries, and only those. A probe qualifies
    # when neither side shares its card: B5·s1's run, whose two sides are
    # solo throughout, and the alternating probe that holds one card fixed.
    # The rows that fail the test go to the paragraph under the table, with
    # the reason they fail, rather than into a column a reader has to
    # discount while reading.
    agree_rows, partial = [], []
    for arm in R.ARM_ORDER:
        a, b = st.get((arm, 0)), st.get((arm, 3))
        if not a or not b or not a["compute_ms"] or not b["compute_ms"]:
            continue
        c0, c3 = float(a["compute_ms"]), float(b["compute_ms"])
        if full_run(a) and full_run(b):
            where = ("each side solo on one box, " + a["machine"]
                     if a["machine"] == b["machine"] else
                     "each side solo on its own box, "
                     f"{a['machine']} → {b['machine']}")
            agree_rows.append(
                f"| {arm}, over its own run | {c0:.1f} ms | {c3:.1f} ms | "
                f"{c3 / c0 - 1:+.0%} | {where} | "
                "[`results/steptime_solo.csv`](results/steptime_solo.csv) |")
        else:
            side = "k = 0" if not full_run(a) else "k = 3"
            r = a if not full_run(a) else b
            partial.append((arm, c0, c3, side, r["windows_solo"],
                            r["windows_total"]))
    # The alternating probe is not a training run, so `steptime_solo.csv`
    # does not hold it. Its own log carries the two medians and the rep
    # count, and the annex says which card it ran on.
    agree_rows.append(
        "| B5, alternating on one elisa card | 190.2 ms | 509.9 ms | +168% | "
        "one card, 3 reps of 600 steps | "
        "[`results/steptime_B5_solo.log`](results/steptime_B5_solo.log) |")
    L += ["", "The two probes that agree:", "",
          "| probe | k = 0 | k = 3 | change | what the two sides hold | "
          "source |", "|---|---|---|---|---|---|"] + agree_rows + [""]
    if partial:
        L += ["; ".join(
            f"{a} reads {c3 / c0 - 1:+.0%} ({c0:.1f} ms against {c3:.1f} ms) "
            f"and is not comparable to those two: its `{side}` median covers "
            f"{ws} of its {wt} windows and its two sides cross a box"
            for a, c0, c3, side, ws, wt in partial) +
            ". **Carry +157% to +168% and do not carry the low row.** No "
            "cell of the 14 has a same-card `k = 0` / `k = 3` pair, which is "
            "what would settle it.", ""]

    # ---- 8. the depth-0 forecast-error gap ---------------------------------
    gap_path = Path(args.results) / "depth0_gap.csv"
    if gap_path.is_file():
        gap = list(csv.DictReader(open(gap_path)))
        L += ["### The depth-0 forecast error, deeper run minus its own k = 0",
              "",
              "`1 - cos(f_t, h_{t+1})` during training: the same quantity on "
              "both runs, unlike the loss. Negative means the deeper run "
              "forecasts one step ahead better. Four end-of-run windows, "
              "because a gap that changes sign between them is not a "
              "result.", "",
              "| arm | k | last 50% | last 25% | last 10% | final step | "
              "one sign over all four |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(gap, key=lambda r: (order.get(r["arm"], 99),
                                            int(r["k"]))):
            stable = r["sign_stable_across_windows"].strip().lower() == "yes"
            L.append(f"| {r['arm']}{mark(r['arm'])} | {r['k']} | "
                     f"{r['last_50pct']} | {r['last_25pct']} | "
                     f"{r['last_10pct']} | {r['final_step']} | "
                     f"{'yes' if stable else '**no**'} |")
        L.append("")

    # ---- 9. glossary -------------------------------------------------------
    L += ["### Glossary", "",
          "| term | what it means here |",
          "|---|---|",
          "| the card | the issue this study answers, and the 14 cells, "
          "stops and criteria it names |",
          "| cell | one of those 14 recipes, `A1`..`A4` and `B1`..`B10` |",
          "| arm | a (cell, backbone seed, machine) triple. B5 trained "
          "three, so the cell is not the unit a delta lives in |",
          "| bb40k | backbone step 40,000, the one stop every run here "
          "reached |",
          "| GM-Relative MASE | geometric mean over the 97 GIFT-Eval "
          "configs of each config's MASE divided by the seasonal-naive "
          "MASE. Lower is better; 1.0 is seasonal-naive parity |",
          "| B4 eval strategy | GIFT-Eval's official evaluation strategy, "
          "the one the parent reports use |",
          "| student / teacher head | the quantile head is trained twice "
          "per backbone, once on the student encoder and once on its EMA "
          "copy, the teacher. The two are separate measurements of one "
          "backbone |",
          "| f-bearing term | the loss term that the forecast operator `f` "
          "enters. `--train-rollout-depth K` duplicates it at depth 1..K |",
          "| `rep_only` | the representation loss with no forecast term |",
          "| `L_align` | the term that aligns `f`'s output with the future "
          "latent |",
          "| `L_pred` | the predictive contrastive term, split from the "
          "representation term |",
          "| `xshh_allt` | negatives pooled across the batch and across "
          "channels, taken over every time index |",
          "| `u_batchtime` | dimension usage of a latent over the pooled "
          "(batch × time) sample axis: `1 / (H · mean off-diagonal squared "
          "cosine)`, capped at 1. 1.0 is all `H` dimensions in use and a "
          "value near `1/H` is one direction. `h_t` is the encoder latent, "
          "`e_t` the embedding it reads |",
          "| collapse | the latent falling onto few directions, so "
          "`u_batchtime` runs toward zero. The card watches for it because a "
          "model can win the deeper f-bearing terms by flattening `f` |",
          "| `arm4`, `arm6_v2 combab` | the launcher recipes the cells run; "
          "the Coverage table gives each cell's |",
          "| head-seed band ±0.0384 | how far the head seed alone moved a "
          "score in `ema_sched_ladder.md`, pooled. It bounds the head seed "
          "and nothing else |",
          "| `mixup` | the count of examples the batch mixer touched in a "
          "200-step window. Two runs on one data order print one count |",
          ""]

    body = "\n".join(L) + "\n"
    Path(args.out).write_text(body)
    print(f"wrote {args.out} ({len(reg)} run(s), {len(trained)} cell(s))")

    if args.inject:
        md = Path(args.inject)
        text = md.read_text()
        a, b = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
        if a not in text or b not in text:
            print(f"NOTE: {md} carries no TABLES markers; not injecting")
            return 0
        head, rest = text.split(a, 1)
        _old, tail = rest.split(b, 1)
        md.write_text(f"{head}{a}\n\n{body}\n{b}{tail}")
        print(f"injected the tables into {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
