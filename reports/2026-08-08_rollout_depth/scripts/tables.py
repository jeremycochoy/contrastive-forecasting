#!/usr/bin/env python3
"""#373 — every table the report carries, from the score files and the splits.

Ten tables, in the order the report asks its questions:

  0. limits          what the study cannot support, one row per claim, with
                     every number read from the same variable the table that
                     prints it reads.
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
                       REEVAL_FLOOR, RESOLUTION, SEED_BAND,
                       best_published, verdict)

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
    EXTEND_NOTE[("B8", _h)] = "trained from step 0; scored at bb100k only"
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


def fidelity_lines(results):
    """The card's flat branch, from `results/rollout_fidelity.csv`.

    The card asks the study to name the part that failed. This reads the
    diagnostic that answers it: for every arm that trained both depths, the
    cosine between the rolled latent and the true `h` at each depth, k = 3
    against that SAME arm's k = 0. It is a within-arm reading, repeated once
    per arm, so it does not rank one arm against another.
    """
    cos = {}
    p = Path(results) / "rollout_fidelity.csv"
    if not p.exists():
        return ["*(no `rollout_fidelity.csv` in the results directory)*"]
    for r in csv.DictReader(p.open()):
        arm, _, k = r["run"].rpartition(":")
        cos.setdefault((arm, k), {})[int(r["d"])] = float(r["cos"])
    arms = sorted({a for a, k in cos if k == "3" and (a, "0") in cos})
    every, depths = [], set()
    for a in arms:
        d0, d3 = cos[(a, "0")], cos[(a, "3")]
        ds = sorted(set(d0) & set(d3))
        depths.add(len(ds))
        if ds and all(d3[d] > d0[d] for d in ds):
            every.append(a)
    n, nd = len(every), (depths.pop() if len(depths) == 1 else 0)
    if n != len(arms) or not nd:                  # the sentence below is false
        return [f"Of the {len(arms)} arms that trained both depths, {n} roll "
                f"out more faithfully than their own `k = 0` at every depth."]
    return [f"Every one of the {n} arms that trained `k = 3` rolls out more "
            f"faithfully than its own `k = 0` at all {nd} depths, and the "
            "scores do not follow. The fixed-point approximation does what "
            "it was built to do, so where a score did not improve, the "
            "approximation is not the part that failed."]


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

    # ---- 1a''. the matched-stop comparison, NOT published -------------------
    # The parent report prints a `Matched-stop comparison` table. This study
    # does not rebuild it. Every one of its rows is a Δ column of the wide
    # table below, and `k3_minus_k0.png` already draws those Δ values ranked
    # inside their own stop, with an interval on each. A third copy of one
    # set of numbers is only a place for them to drift.

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
    # The bb200k column of this table, which the limits table reads back.
    pub200 = []
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
                if stop == 200 and head == "student":
                    pub200.append((cell, d))
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
          f"{NOISE_BAND:.4f} head-seed band: closer than that is `flat`. "
          "A dash is a number no parent published, ‡ marks the two cells "
          "that share one student model, and the second line of a verdict "
          "cell is its 95% paired dataset-cluster interval.", "",
          "At bb100k, the stop every one of the 14 cells reached, counted "
          f"over distinct models. Student head: {tally_line('student')}. "
          f"Teacher head, group A only: {tally_line('teacher')}.", "",
          "| cell | head | 40k k=3 | 40k pub | Δ | | 100k k=3 | 100k pub | Δ "
          "| | 200k k=3 | 200k pub | Δ | |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] \
         + rows + [""]

    # ---- 0a. the card's two success criteria, per cell ----------------------
    # The card sets a PRIMARY criterion on the horizon split and a SECONDARY
    # one on the full 97. Both are per-cell, both take a pass or a fail, and
    # both are answered here rather than left for the reader to derive from
    # the delta tables.
    #
    #   primary    medium+long (42 configs) at least 5% better, AND short
    #              (55 configs) losing less than 2%. `published_bootstrap.py`
    #              computes both percentages per cell, per stop, per head
    #              into `criterion_screen.csv`.
    #   secondary  full-97 GM-Relative MASE lower than the k = 0 cell by
    #              more than the head-seed band, so Δ <= -NOISE_BAND. That
    #              is the same threshold the delta table's `better` verdict
    #              uses, applied as the card's own test.
    #
    # Both sides of every row cross a machine, which this study measures at
    # 0.1166. That is larger than the criterion's own threshold, so the
    # table is a SCREEN. The depth-response table holds the machine and is
    # where the primary criterion runs as a test.
    CRIT = []
    scr = {}
    scrp = Path(args.results) / "criterion_screen.csv"
    if scrp.is_file():
        for r in csv.DictReader(open(scrp)):
            scr[(r["cell"], int(r["stop_k"]), r["head"])] = r

    def crit_counts(stop, head):
        """`(primary met, secondary met, rows)` for one stop and head."""
        p = s = n = 0
        for cell in CARD_CELLS:
            row = scr.get((cell, stop, head))
            mine, base = sv(cell, stop, head), pub(cell, head, stop)
            if row is None or mine is None or base is None:
                continue
            n += 1
            p += row["criterion_met"] == "yes"
            s += (mine - base) <= -NOISE_BAND
        return p, s, n

    if scr:
        crows = []
        for cell in CARD_CELLS:
            row = scr.get((cell, 100, "student"))
            mine, base = sv(cell, 100, "student"), pub(cell, "student", 100)
            if row is None or mine is None or base is None:
                continue
            ml, sh = float(row["pct_medium_long"]), float(row["pct_short"])
            ok1 = row["criterion_met"] == "yes"
            d = mine - base
            ok2 = d <= -NOISE_BAND
            crows.append(
                f"| {cell} | {ml:+.1f}% | {sh:+.1f}% | "
                f"{'**PASS**' if ok1 else 'fail'} | {d:+.4f} | "
                f"{'**PASS**' if ok2 else 'fail'} |")
        p100, s100, n100 = crit_counts(100, "student")
        p40, s40, n40 = crit_counts(40, "student")
        p200, s200, n200 = crit_counts(200, "student")
        pT, sT, nT = crit_counts(100, "teacher")
        CRIT += [
            "| cell | med+long, 42 configs | short, 55 configs | PRIMARY | "
            "full-97 Δ | SECONDARY |", "|---|---|---|---|---|---|"] + crows + [
            "",
            f"**{p100} of {n100} cells meet the primary criterion at bb100k, "
            f"and {s100} of {n100} meet the secondary one.** At bb40k it is "
            f"{p40} and {s40} of {n40}; at bb200k, {p200} and {s200} of "
            f"{n200}; on the teacher head at bb100k, where only group A "
            f"publishes a baseline, {pT} and {sT} of {nT}.", "",
            "Primary: medium+long at least 5% better AND short losing less "
            f"than 2%. Secondary: full-97 Δ at or below −{NOISE_BAND:.4f}, "
            "the head-seed band. Δ is `k = 3` minus the cell's published "
            "`k = 0`, so negative is a gain. Student head at bb100k, the "
            "stop every one of the 14 cells reached.", "",
            "The count is over CELLS. A1 and B3 hold one student model "
            "between them, so the same 14 cells hold 13 student models and "
            "the model count of the secondary criterion is one lower than "
            "the cell count.", "",
            "**Both sides of every row trained on a different machine.** "
            "This study's one controlled measurement of the machine is worth "
            "0.1166, which is larger than either threshold, so this table is "
            "a screen. The two rows that hold the machine are in the "
            "depth-response table in the annex. Every cell here ran once, on "
            "one backbone seed, so the spread over the rows does not rank the "
            "recipes.", ""]

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
        # The parent report's two columns: where the cell ended, and what
        # ended it. 200k is the card's ceiling, so a cell that reached it was
        # stopped by the ceiling and not by the rule.
        last = max([st for st in (40, 100, 200)
                    for h in ("student", "teacher")
                    if sv(cell, st, h) is not None] or [0])
        ended = ("ladder ceiling" if last == 200 else
                 "the card's call" if cell in STOP_CALL else
                 "extend rule")
        rows.append(f"| {cell} | {s:+.4f} | {t:+.4f} | **{dec}** | "
                    f"bb{last}k | {ended} | {why} |")
        if dec.startswith("stop") and max(abs(s), abs(t)) <= NOISE_BAND:
            stopped_inband.append(cell)
        if dec.startswith("extend"):
            extended.append(cell)

    nstop = sum(1 for r in rows if "stop at 100k" in r)
    L += ["### Stop reasons: what the extend rule read at each cell", "",
          "The rule reads one cell's bb40k number against its bb100k number, "
          "per head. A head that moved down earns the second 100,000 steps; "
          "a head that moved up stops. Both columns are bb100k minus bb40k, "
          f"so negative is an improvement. It held {nstop} cells at 100k. "
          "`last stop` and `ended by` are the parent report's two columns: "
          "where each cell finished, and what finished it.", "",
          "| cell | 40k→100k student | 40k→100k teacher | decision | "
          "last stop | ended by | why |",
          "|---|---|---|---|---|---|---|"] + rows + [""]
    # What the rule selects for, once. The limits table carries the same
    # point as a row, and the annex figure carried it a third time. One
    # sentence here, at the point the panel is defined.
    # No "on an improving first leg": two of the eight extended on a move the
    # rule's own `why` column says decides nothing.
    L += [f"**The rule selects the panel.** It sent {len(extended)} cells to "
          f"bb200k, fired inside its own "
          f"±{NOISE_BAND:.4f} band on {len(stopped_inband)} of the {nstop} "
          f"cells it stopped ({', '.join(stopped_inband)}), and both manual "
          "overrides extended.", ""]

    # ---- 1b. the stop ladder -----------------------------------------------
    # Round 3's own question. The extend rule sent eight cells from 100k to
    # 200k and held five at 100k, so this is the column that says whether the
    # extra 100,000 steps bought anything. It reads the score files, so a row
    # fills the moment its eval lands and stays `—` until then.
    #
    # The interval on each Δ, from the same paired dataset-cluster bootstrap
    # the depth contrasts use. A Δ with no interval beside it cannot be read
    # against the head-seed band, so the column ships with the numbers.
    # The point estimate comes from the same file as the interval beside it.
    # Subtracting the two score files again lands on the other side of a
    # rounding boundary on B4 student (1.318241 - 1.280391 = 0.037850), so the
    # report printed +0.0378 next to an interval computed around +0.0379.
    stop_ci, stop_d = {}, {}
    sbp = Path(args.results) / "stop_bootstrap.csv"
    if sbp.is_file():
        for r in csv.DictReader(open(sbp)):
            if r["subset"] != "all":
                continue
            cell, _, head = r["label"].partition("_stop200v100_")
            stop_ci[(cell, head)] = (float(r["ci_lo"]), float(r["ci_hi"]))
            stop_d[(cell, head)] = float(r["delta"])

    rows, deltas = [], []
    for cell in CARD_CELLS:
        for head in ("student", "teacher"):
            a, b, c = (sv(cell, 40, head), sv(cell, 100, head),
                       sv(cell, 200, head))
            if b is None:
                continue
            d = None if c is None else stop_d.get((cell, head), c - b)
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
    # Everything from here on is the machine, the seed and the control
    # material a review asked for. The body carries the card's own tables;
    # these go to the annex, under the TABLES_ANNEX markers. Nothing is
    # dropped, only moved.
    BODY_TABLES = len(L)

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
              # No second table of the same eight rows. The stop-ladder
              # table above already prints every cell, every stop and the
              # same Δ, so this sentence points at it.
              f"A3's is the ladder's largest reversal, but it is not the "
              f"only one: {n_turn} of the {len(turns)} three-stop student "
              "trajectories turn round at bb200k, in the stop-ladder table "
              "above.", ""]

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
    # The card's gate is per GROUP: retrain one cell of the group at k = 0
    # and meet its published number. Group A and group B each get a verdict
    # below, because the card's instruction on failure is per group too.
    gate_by_group = {}
    for r in repro:
        # Not `pub`: that name is the published-baseline lookup this
        # function's later tables call.
        base = PUB_ALL.get(r.cell, {}).get("student", {}).get(40)
        got = val(r.tag)
        if base is None or got is None:
            continue
        same = r.seed == PUBLISHED_SEED
        if not same:
            cross_seed.append(r.arm)
        gate = (f"{GATE}, the card" if same
                else f"{seed_band:.4f}, the seed band")
        if same and r.run:
            gate_by_group.setdefault(r.cell[0], []).append(
                (abs(got - base), r.arm, r.machine))
        L.append(f"| {r.arm}{mark(r.arm)} | {r.seed} | {r.machine} | "
                 f"{base:.4f} | {got:.4f} | {abs(got - base):.4f} | {gate} | "
                 f"{verdict(abs(got - base), same, seed_band)} |")
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

    # ---- 2b. the card's gate, per group ------------------------------------
    # The card runs this gate once per group and gives an instruction for a
    # failure. Group B passes and group A does not, and the report has to
    # say both, and say what the card asked for on the failure.
    if gate_by_group:
        verdicts = []
        for g in sorted(gate_by_group):
            d, arm, machine = min(gate_by_group[g])
            verdicts.append(
                f"Group {g}: {arm} at `k = 0`, on {machine}, misses its "
                f"published number by {d:.4f}"
                + (" — **PASS**" if d <= GATE else " — **FAIL**"))
        L += ["**The card's baseline validity gate, group by group.** It "
              "retrains one cell of the group at `k = 0` on this study's "
              f"code and asks for the published number to within {GATE}. "
              + " ".join(v + "." for v in verdicts), ""]
        failed = [g for g in sorted(gate_by_group)
                  if min(gate_by_group[g])[0] > GATE]
        mrow = bs.get(("B5_machine_k0_student", "all"))
        if failed:
            gs = ", ".join(failed)
            L += [f"The card's instruction on a failure is to retrain the "
                  f"`k = 0` side of every cell of that group rather than "
                  f"read it from the parent report. This study did not do "
                  f"that for group {gs}. So every group-{gs} delta against a "
                  f"published `k = 0` is a screen and not a test, on top of "
                  f"the machine it already crosses. The gate's own row "
                  f"crosses that machine as well: it is this study's only "
                  f"group-{gs} retrain and it trained on a rented box"
                  + (f", and the machine is worth "
                     f"{abs(float(mrow['delta'])):.4f}." if mrow else "."),
                  ""]

    # ---- 3. depth response -------------------------------------------------
    L += ["### Depth response, against each arm's own k = 0", "",
          "| arm | seed | machine held | head | k | k = 0 | this k | Δ | "
          "all | short | med+long | criterion |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    held_arms, depths = set(), set()
    # The machine-held k = 0 / k = 3 pairs on the student head: the two the
    # report leads on, and the row the limits table reads them back into.
    held_student = []
    for arm, head, k, base, deep in R.pairs(tags):
        a, b = val(base.tag), val(deep.tag)
        A, B = sp.get(base.tag, {}), sp.get(deep.tag, {})
        depths.add(k)
        if R.machine_held(base, deep):
            held_arms.add(arm)
            r = bs.get((boot_label(arm, k, head), "all"))
            if head == "student" and k == 3 and r is not None:
                held_student.append((arm, float(r["delta"]),
                                     float(r["ci_lo"]), float(r["ci_hi"])))
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
          "two it changes. The machine contrast is the larger of the two, and "
          "each contrast is one run pair.", "",
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
    agree_rows, partial, agree_pct = [], [], []
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
            agree_pct.append(c3 / c0 - 1)
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
    agree_pct.append(509.9 / 190.2 - 1)
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

    CW = []
    # ---- 8b. the collapse watch the card names ------------------------------
    # `collapse_watch.py` looks for every quantity the card lists, in every
    # run, and reports the ones no run logged. A watch the report never
    # prints reads as a watch that passed.
    WATCHED = ["ff", "cos_err_d0", "cos_err_d1", "cos_err_d2", "cos_err_d3",
               "u_batchtime", "u_batchtime_e", "qk_logit_maxabs"]
    cwp = Path(args.results) / "collapse_watch.csv"
    if cwp.is_file():
        cw = list(csv.DictReader(open(cwp)))
        seen = {r["metric"] for r in cw}
        missing = [m for m in WATCHED if m not in seen]
        by = {}
        for r in cw:
            by.setdefault((r["arm"], int(r["k"])), {})[r["metric"]] = r
        crows = []
        for (arm, k) in sorted(by, key=lambda a: (order.get(a[0], 99), a[1])):
            g = by[(arm, k)]

            def cell_of(metric):
                r = g.get(metric)
                return ("—" if r is None else
                        f"{r['end_of_run']}<br>{r['min_last_half']}")
            crows.append(
                f"| {arm}{mark(arm)} | {k} | "
                + " | ".join(cell_of(m) for m in
                             ("ff", "cos_err_d0", "cos_err_d1", "cos_err_d2",
                              "cos_err_d3", "u_batchtime", "u_batchtime_e"))
                + " |")
        us = [r for r in cw if r["metric"].startswith("u_batchtime")]
        lowest = min(us, key=lambda r: float(r["min_last_half"]))
        umin = float(lowest["min_last_half"])
        INV_H = 1.0 / 64                 # d_model = 64, one direction
        # Does the deeper run use fewer directions than its own k = 0? One
        # ratio per arm that trained both, on the encoder latent the card
        # names first.
        pair = {}
        for r in cw:
            if r["metric"] == "u_batchtime":
                pair.setdefault(r["arm"], {})[int(r["k"])] = \
                    float(r["end_of_run"])
        drops = sorted(f"{a} {v[0]:.4f} → {v[max(v)]:.4f}"
                       for a, v in pair.items()
                       if 0 in v and max(v) > 0 and v[max(v)] < 0.5 * v[0])
        CW += [
              "First line of a cell is the mean over the last 10% of the "
              "run; second line is the lowest value over the run's second "
              "half.", "",
              "`ff` is `cos(f_t, h_{t+1})` and `cos_err_dj` is "
              "`1 − cos(f^(j)_t, h_{t+1+j})`, so `cos_err_d0` is `1 − ff` "
              "and `cos_err_dj` is the card's per-depth `ff`. A collapsed "
              "latent points one way, so `u_batchtime` runs toward zero "
              "WHILE `ff` runs toward 1. It is that pair, not `ff` alone, "
              "that separates collapse from a good forecast.", ""]
        if missing:
            CW += [f"**Not logged: `{'`, `'.join(missing)}`.** No run in this "
                  "study writes that column at any depth, so this study "
                  "does not watch it.", ""]
        CW += ["| arm | k | `ff` | `cos_err_d0` | `cos_err_d1` | "
              "`cos_err_d2` | `cos_err_d3` | `u_batchtime` on `h_t` | "
              "`u_batchtime` on `e_t` |",
              "|---|---|---|---|---|---|---|---|---|"] + crows + \
             ["",
              f"The lowest `u_batchtime` any arm reaches over its second "
              f"half is {umin:.4f}, on `{lowest['metric']}`, "
              f"{lowest['arm']} at k = {lowest['k']}. One direction would "
              f"give `1/H` = {INV_H:.4f} at `d_model = 64`, so that arm sits "
              f"{umin / INV_H:.1f}× above it. No arm reaches zero at any "
              "depth.", ""] + \
             ([f"On `h_t`, {len(drops)} of the {len(pair)} arms that trained "
               f"both depths ends the deeper run below half its own `k = 0` "
               f"usage: {'; '.join(drops)}. That is a reading and not a "
               "verdict. No arm reaches zero, and this study runs no control "
               "that separates a lower usage from a worse score.", ""]
              if drops else [])

    # ---- 9. glossary -------------------------------------------------------
    # The report's ONE glossary. It used to have a second, hand-written
    # `Definitions` section, and `u_batchtime` and the six launcher recipes
    # were defined in both. Two definitions of one term drift; this is the
    # only place either is defined now.
    GLOSS = ["### Glossary", "",
          "| term | what it means here |",
          "|---|---|",
          "| the card | the issue this study answers, and the 14 cells, "
          "stops and criteria it names |",
          "| cell | one of those 14 recipes, `A1`..`A4` and `B1`..`B10` |",
          "| arm | a (cell, backbone seed, machine) triple. B5 trained "
          "three, so the cell is not the unit a delta lives in |",
          "| `k`, rollout depth | the value of `--train-rollout-depth`. It "
          "copies every loss term the forecast operator `f` enters at "
          "depths 1..`k` and sums the copies. `k = 0` is today's training |",
          "| the fixed-point approximation | how training rolls the forecast "
          "out: the depth-`j` input is the model's own depth-`j-1` "
          "predictions, not the true prefix. It buys one parallel pass over "
          "every `t`, and it is the card's alternative suspect to the "
          "objective |",
          "| bb40k, bb100k, bb200k | backbone step 40,000 / 100,000 / "
          "200,000. bb40k is the one stop every run here reached |",
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
          "| `arm4 combab`, `arm5 combab`, `arm6_v2 combab`, "
          "`arm6_v2 ncpc`, `arm6_v2 nse`, `arm1 nse` | the six launcher "
          "recipes the 14 cells run. `combab` pools negatives across the "
          "batch and the channels; `ncpc` drops the CPC auxiliary; `nse` "
          "keeps it. The Coverage table gives each cell's |",
          "| head-seed band ±0.0384 | how far the head seed alone moved a "
          "score in `ema_sched_ladder.md`, pooled. It bounds the head seed "
          "and nothing else |",
          "| dataset-cluster | the resampling unit of every interval here. "
          "`<ds>/short`, `/medium` and `/long` are three configs of one "
          "series, so the bootstrap resamples the dataset, not the config |",
          "| machine-held | both sides of a comparison trained on the same "
          "physical box. A pair that is not machine-held carries a machine "
          "change as well as a depth change |",
          "| `mixup` | the count of examples the batch mixer touched in a "
          "200-step window. Two runs on one data order print one count |",
          "| ✗ | a retracted arm: its `k = 0` baseline is a rented-box "
          "artefact, so its depth delta is withdrawn |",
          ""]

    # ---- 0. what the study cannot support ----------------------------------
    # The report's limits, one row per claim. This table used to be written by
    # hand above the TABLES markers, and it re-printed four headline numbers
    # the generated tables already carried. Four copies of one value drift, so
    # every number here now comes from the same variable the table that prints
    # it comes from, and a row with no number of its own points at the table
    # that holds it.
    def boot_ci(label):
        r = bs.get((label, "all"))
        return None if r is None else (float(r["delta"]), float(r["ci_lo"]),
                                       float(r["ci_hi"]))

    ns_rows = []
    # The headline's own limit, first. The frontier drop is the report's
    # lead number, and its two ends do not hold one thing constant.
    fbase, fcell, fhead, fstop = best_published()
    fv, fc, fh, fs = min((v, c, h, s) for c in L2.CELLS for s in L2.STOPS
                         for h in ("student", "teacher")
                         for v in [sv(c, s, h)] if v is not None)
    mach0 = boot_ci("B5_machine_k0_student")
    if mach0:
        mine100, pub100 = sv(fc, 100, fh), pub(fc, fh, 100)
        ns_rows.append(
            f"| That the frontier drop of {fbase - fv:.4f} measures the "
            f"depth | Its two ends cross a head, a stop and a machine: "
            f"{fcell} on the {fhead} head at bb{fstop}k against {fc} on the "
            f"{fh} head at bb{fs}k. "
            + (f"{fc}'s own matched-stop delta is "
               f"{mine100 - pub100:+.4f} at bb100k."
               if mine100 is not None and pub100 is not None else "") + " |")
    for g in sorted(gate_by_group):
        d, arm, _machine = min(gate_by_group[g])
        if d > GATE:
            ns_rows.append(
                f"| Any group-{g} delta against a published `k = 0` | The "
                f"card's baseline validity gate fails on group {g}: {arm} "
                f"misses its published number by {d:.4f} against a gate of "
                f"{GATE}. The card then asks for the `k = 0` side of every "
                f"group-{g} cell to be retrained, and this study reads those "
                "baselines from the parent report. |")
    if len(held_student) == 2:
        (a1, d1, lo1, hi1), (a2, d2, lo2, hi2) = sorted(
            held_student, key=lambda t: t[1])
        ns_rows.append(
            "| That `k = 3` helps, or that it hurts | The two machine-held "
            f"`k = 0` / `k = 3` pairs read {a1} {d1:+.4f} and {a2} {d2:+.4f}, "
            "both 95% intervals excluding zero (`depth_response.png`). Each "
            "is one draw in the backbone seed, so this study reads a "
            "direction and not a per-recipe ranking. |")
    ns_rows += [
        "| That the gain is the depth alone | B1 is the one cell that carries "
        "the `L_align` ×4 re-weighting control on one machine, and the "
        "re-weighting moves the score on its own. The annex's B1 table and "
        "its figure print the share of the move, per head. |",
        "| That one of the two pays more than the other | The re-weighting's "
        "move and the depth's move sit inside each other's 95% intervals, in "
        "the same B1 table in the annex. That cell measures both and ranks neither. |",
        "| Any per-cell verdict | Every cell is n = 1 in the backbone seed. "
        f"The ±{NOISE_BAND:.4f} band bounds the HEAD seed alone, and "
        "backbone-seed variance is unmeasured. |"]
    k1 = boot_ci("A3_k1_student")
    if k1:
        ns_rows.append(
            "| That depth 3 is the right depth | Only `k = 3` ran on the 14 "
            "cells. One ladder holds a second depth, on A3, and its `k = 1` "
            f"delta covers zero: {k1[0]:+.4f} [{k1[1]:+.4f}, {k1[2]:+.4f}] on "
            "the student. |")
    mach = boot_ci("B5_machine_k0_student")
    if mach and held_arms:
        ns_rows.append(
            "| The per-horizon criterion of the card, the issue this study "
            "answers, at scale | Only "
            f"{len(held_arms)} arms hold the machine, and only at bb40k. "
            "Every other pair crosses a machine, and the machine is worth "
            "more than most of the deltas in this report. |")
    if pub200:
        won = sorted((d, c) for c, d in pub200 if d < 0)
        lost = sorted(((d, c) for c, d in pub200 if d > 0), reverse=True)
        ns_rows.append(
            f"| That `k = 3` leads at 200k | {len(pub200)} cells hold a "
            "published `k = 0` at 200k. "
            + (", ".join(f"{c} by {d:+.4f}" for d, c in won)
               + (" lead it. " if len(won) != 1 else " leads it. ")
               if won else "")
            + (", ".join(f"{c} by {d:+.4f}" for d, c in lost)
               + (" lose it" if len(lost) != 1 else " loses it")
               + f", against a largest gain of {won[0][0]:+.4f}, so the "
                 f"{len(pub200)} cells do not point one way. |"
               if lost and won else "|"))
    if agree_pct:
        cost = "Two solo probes agree; the annex step-time tables carry them."
        for arm, _c0, _c3, _side, ws, wt in partial:
            cost += (f" {arm}'s reading covers {ws} of its {wt} timing "
                     "windows and crosses a box, so it is not comparable to "
                     "them.")
        ns_rows.append(f"| The cost of the depth | {cost} |")
    ns_rows.append(
        "| That the 200k reading is unconditional | The extend rule reads the "
        "bb40k-to-bb100k contrast, which the Protocol calls not head-matched. "
        f"It fired inside its own ±{NOISE_BAND:.4f} band on "
        f"{len(stopped_inband)} stopped cells, and both manual overrides "
        "extended. |")
    NS = ["| the claim | what stops it |", "|---|---|"] + ns_rows

    FID = fidelity_lines(args.results)

    # Three blocks, three places in the report. The card's success criteria
    # answer its own question, so they lead; the limits qualify every number
    # above them, so they close the body; the tables sit between.
    BODY, ANNEX = L[:BODY_TABLES] + GLOSS, L[BODY_TABLES:]
    blocks = {"CRITERIA": CRIT, "COLLAPSE": CW, "TABLES": BODY,
              "TABLES_ANNEX": ANNEX, "LIMITS": NS, "FIDELITY": FID}
    Path(args.out).write_text(
        "\n".join(["## Did the card's criteria pass?", ""] + CRIT +
                  ["## Collapse watch", ""] + CW +
                  ["## What this study cannot support", ""] + NS +
                  ["", "## Tables", ""] + BODY +
                  ["", "## Annex tables", ""] + ANNEX) + "\n")
    print(f"wrote {args.out} ({len(reg)} run(s), {len(trained)} cell(s))")

    if args.inject:
        md = Path(args.inject)
        text = md.read_text()
        for name, lines in blocks.items():
            a, b = f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"
            if a not in text or b not in text:
                print(f"NOTE: {md} carries no {name} markers; not injecting")
                continue
            head, rest = text.split(a, 1)
            _old, tail = rest.split(b, 1)
            text = f"{head}{a}\n\n" + "\n".join(lines) + f"\n\n{b}{tail}"
            print(f"injected {name} into {md}")
        md.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
