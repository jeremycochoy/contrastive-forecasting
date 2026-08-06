#!/usr/bin/env python3
"""#393 — the bb40k-to-bb100k delta with an uncertainty on BOTH of its ends.

`results/seed_spread.csv` replicated the bb100k end at three head seeds and
reported its sd. Dividing the delta by that sd states a significance the
data does not contain, for two reasons.

  The bb40k end had one head seed, 20260722, and no replicate. Nothing in
  the study bounded its spread, and the #390 parent measured 15,000-step
  bb40k heads with sd up to 0.0380 — 2x to 20x the in-study bb100k sds that
  were serving as the denominator.

  The two ends do not share a head budget. Every bb40k row is 15,000 head
  steps and every bb100k row is 30,000 (`results/ladder_all.csv`), so a
  spread measured on 30k-step heads says nothing about 15k-step heads.

So the bb40k heads were retrained at the same two extra seeds, 20260723 and
20260724, at the bb40k budget of 15,000 steps
(`scripts/seed_replicates_bb40k.sh`). This file recomputes the delta from
both ends.

THE STATISTIC. The two ends share a head seed, so the deltas are PAIRED:
one delta per seed, `bb100k(s) - bb40k(s)`, and the spread of those three
deltas is the standard error of the mean delta. That is the right
denominator — it keeps whatever the seed does to both ends in common, which
an unpaired two-sample SE throws away. Both are reported; the paired one is
the number the verdict uses.

  delta_s  = bb100k(s) - bb40k(s)      for s in 20260722/3/4
  mean_d   = mean of the three
  se_pair  = sd(delta_s) / sqrt(3)     df = 2, t_crit(0.05, two-sided) = 4.303
  se_unp   = sqrt(sd40^2/3 + sd100^2/3)

AND WHAT THE RULE ACTUALLY READS. `ladder_decision` is a sign test: it asks
whether each head's bb100k value is below its bb40k value, and extends only
when at least one is. A t on the mean delta is the wrong shape for that
question on its own, so every row also carries `n_down`, the number of
seeds at which this head went down with both ends at the same seed. A head
that is down at 3/3 or 0/3 has a reproducible sign whatever the t says; one
at 1/3 or 2/3 does not, and the branch it produced was decided by the seed.

GPU MODELS ARE RECORDED, NOT CORRECTED. Three of the six cells trained
every head of theirs on rented RTX 5090s — arm5_combab_alignS,
arm5_combab_alignT, arm6_v2_nse_alignT (`results/machines.txt`,
`results/seed_boxes.txt`, and the `_broker/<box>/<cell>/` trees are the
receipt). Vast was empty when the bb40k replicates were run, so their new
bb40k heads are on elisa's RTX 4090s and their bb100k heads are not.

An earlier draft measured that term with a retrained control head
(`HEAD_TAG=hw4090`) and shifted the 4090 values by it before pairing. That
tier was cancelled before any of its jobs ran. A 4090 and a 5090 are both
consumer NVIDIA cards running the same PyTorch with a seeded head, so the
term is negligible and does not earn six head trainings or the eval slots
they would hold. No score here is adjusted for the card that produced it,
and this file reports no hardware term.

The `gpu_split` column marks the rows where a cell's bb40k and bb100k ends
sit on different card models. It is provenance for one line in the results
README, not a measurement and not a correction.

Writes:

  results/paired_delta.csv     one row per (cell, head): both ends' seeds,
                               the three paired deltas, mean, sd, se, t,
                               n_down, and the verdict
  results/paired_branches.csv  per cell: the branch re-derived at each seed
                               with BOTH ends at that seed, and whether it
                               is the branch the ladder recorded

Usage:  python3 scripts/paired_delta.py [--runs DIR] [--results DIR] [--check]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ladder import (  # noqa: E402
    CELLS as LADDER_CELLS,
    LADDER_COLUMNS,
    RUNS_DEFAULT,
    alpha_at,
    experiment_step_cap,
    head_steps_for,
    ladder_decision,
)

CELL_BY_SLUG = {c["slug"]: c for c in LADDER_CELLS}

PROTOCOL_SEED = 20260722
REPLICATE_SEEDS = [20260723, 20260724]
SEEDS = [PROTOCOL_SEED] + REPLICATE_SEEDS
HEADS = ["student", "teacher"]

# Two-sided 0.05 critical values of Student's t. Only these two df occur:
# three paired deltas give df=2, and a cell with one replicate gives df=1.
T_CRIT_05 = {1: 12.706, 2: 4.303}

# The six cells with bb100k replicates, and the two the review flags as
# marginal-and-unreplicated. `stops` is the pair the delta spans.
CELLS = [
    "arm6_v2_combab_alignS", "arm6_v2_combab_alignT",
    "arm5_combab_alignS", "arm5_combab_alignT",
    "arm6_v2_nse_alignT", "arm6_v2_nse_alignS",
    "arm4_combab", "arm6_v2_ncpc_alignT",
    "arm1_nse", "arm6_v2_ncpc_alignS",
]
# The six whose bb100k replicates seed_spread.py already audits.
SPREAD_CELLS = set(CELLS[:6])
PREV_STOP, STOP = 40000, 100000

# WHICH CARD TRAINED WHICH HEAD. Kept per (cell, stop, seed) so the report
# can name exactly which cells have a bb40k end and a bb100k end on
# different card models. Nothing downstream corrects for it.
#
#   seed 20260722, both stops — the cell's own box (results/machines.txt):
#   the five cells below on rented RTX 5090s, the rest on elisa's 4090s.
#
#   seeds 20260723/24 at bb100k — the boxes in results/seed_boxes.txt, which
#   were chosen to match each cell's first seed. Only the six cells that had
#   bb100k replicates appear there; arm4_combab and arm6_v2_ncpc_alignT get
#   theirs in this pass, on elisa.
#
#   seeds 20260723/24 at bb40k — every one of them in this pass, on elisa.
#
# So arm5_combab_alignS/alignT and arm6_v2_nse_alignT are the only cells
# with a mismatched pair: 5090 bb100k against 4090 bb40k. arm4_combab and
# arm6_v2_ncpc_alignT are 5090 cells too, but both of their replicate ends
# are on elisa, so each of their paired deltas is internally matched.
CELL_GPU_SEED22 = {
    "arm5_combab_alignS": "5090",
    "arm5_combab_alignT": "5090",
    "arm6_v2_nse_alignT": "5090",
    "arm4_combab": "5090",
    "arm6_v2_ncpc_alignT": "5090",
    "arm1_nse": "5090",
    "arm6_v2_ncpc_alignS": "5090",
}
BB100K_REPLICATE_GPU = {
    "arm6_v2_combab_alignS": "4090",   # elisa
    "arm6_v2_combab_alignT": "4090",   # vast seed-h2, RTX 4090
    "arm6_v2_nse_alignS": "4090",      # vast seed-h1, RTX 4090
    "arm5_combab_alignS": "5090",      # vast seed-s1
    "arm5_combab_alignT": "5090",      # vast seed-s2
    "arm6_v2_nse_alignT": "5090",      # vast seed-s3 / seed-s4
}
NEW_GPU = "4090"          # elisa, where every head in this pass ran


def gpu_of(cell: str, stop: int, seed: int) -> str:
    if seed == PROTOCOL_SEED:
        return CELL_GPU_SEED22.get(cell, NEW_GPU)
    if stop == STOP:
        return BB100K_REPLICATE_GPU.get(cell, NEW_GPU)
    return NEW_GPU


def gpu_split(cell: str) -> bool:
    """True when some replicate seed pairs a bb40k head with a bb100k head
    trained on the other card model. Recorded, never corrected for."""
    return any(gpu_of(cell, PREV_STOP, s) != gpu_of(cell, STOP, s)
               for s in REPLICATE_SEEDS)

DELTA_COLUMNS = [
    "cell", "head", "n_seeds",
    "bb40k_20260722", "bb40k_20260723", "bb40k_20260724",
    "bb100k_20260722", "bb100k_20260723", "bb100k_20260724",
    "bb40k_sd", "bb100k_sd",
    "delta_20260722", "delta_20260723", "delta_20260724",
    "delta_mean", "delta_sd", "se_paired", "se_unpaired", "t_paired", "df",
    "t_crit_05", "significant", "n_down", "sign_stable",
    "gpu_split", "verdict",
]
BRANCH_COLUMNS = [
    "cell", "recorded_branch", "branch_20260722", "branch_20260723",
    "branch_20260724", "n_distinct", "survives_paired",
    "student_down_seeds", "teacher_down_seeds", "verdict",
]
LF = "\n"


def score_path(runs: str, cell: str, stop: int, head: str, seed: int) -> str:
    sfx = "" if seed == PROTOCOL_SEED else f"_s{seed}"
    return os.path.join(runs, cell, "eval",
                        f"score_bb{stop // 1000}k_{head}{sfx}.txt")


def read_score(path: str) -> float | None:
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def collect(runs: str) -> dict:
    """{(cell, stop, head, seed): score} — every score as it was measured."""
    scores = {}
    for cell in CELLS:
        for head in HEADS:
            for stop in (PREV_STOP, STOP):
                for seed in SEEDS:
                    v = read_score(score_path(runs, cell, stop, head, seed))
                    if v is not None:
                        scores[(cell, stop, head, seed)] = v
    return scores


def fmt(v, spec="{:.6f}"):
    return "" if v is None else spec.format(v)


def delta_rows(scores: dict) -> list[dict]:
    rows = []
    for cell in CELLS:
        split = "yes" if gpu_split(cell) else "no"
        for head in HEADS:
            prev, cur, deltas = {}, {}, {}
            for s in SEEDS:
                p = scores.get((cell, PREV_STOP, head, s))
                c = scores.get((cell, STOP, head, s))
                prev[s], cur[s] = p, c
                if p is not None and c is not None:
                    deltas[s] = c - p
            if not deltas:
                continue

            d = list(deltas.values())
            n = len(d)
            mean_d = statistics.fmean(d)
            sd_d = statistics.stdev(d) if n > 1 else None
            se_p = sd_d / (n ** 0.5) if sd_d is not None else None
            have40 = [v for v in prev.values() if v is not None]
            have100 = [v for v in cur.values() if v is not None]
            sd40 = statistics.stdev(have40) if len(have40) > 1 else None
            sd100 = statistics.stdev(have100) if len(have100) > 1 else None
            se_u = (None if sd40 is None or sd100 is None else
                    (sd40 ** 2 / len(have40) + sd100 ** 2 / len(have100)) ** 0.5)
            df = n - 1
            tcrit = T_CRIT_05.get(df)
            t = (None if not se_p else mean_d / se_p)
            sig = ("" if t is None or tcrit is None
                   else "yes" if abs(t) > tcrit else "no")

            n_down = sum(1 for s in deltas if deltas[s] < 0)
            stable = "yes" if n_down in (0, n) else "no"

            if n < len(SEEDS):
                verdict = f"measured at {n}/{len(SEEDS)} seeds — no claim yet"
            elif t is None:
                # Three paired deltas equal to the last printed digit. Real
                # data does not do this, but a degenerate SE must not become
                # an infinite t or a crash.
                verdict = (f"{'down' if mean_d < 0 else 'up'} at {n}/{n} seeds; "
                           f"the three paired deltas are identical, so the SE "
                           f"is zero and no t is defined")
            elif stable == "no":
                verdict = (f"sign flips: down at {n_down}/{n} seeds — the head "
                           f"seed decides this head's direction")
            elif sig == "yes":
                verdict = (f"{'down' if mean_d < 0 else 'up'} at {n}/{n} seeds, "
                           f"|t|={abs(t):.1f} > {tcrit} — clears the noise of "
                           f"both ends")
            else:
                verdict = (f"{'down' if mean_d < 0 else 'up'} at {n}/{n} seeds "
                           f"but |t|={abs(t):.1f} <= {tcrit} — direction is "
                           f"reproducible, magnitude is not resolved")

            row = {
                "cell": cell, "head": head, "n_seeds": n,
                "bb40k_sd": fmt(sd40), "bb100k_sd": fmt(sd100),
                "delta_mean": fmt(mean_d, "{:+.6f}"),
                "delta_sd": fmt(sd_d), "se_paired": fmt(se_p),
                "se_unpaired": fmt(se_u),
                "t_paired": fmt(t, "{:+.3f}"), "df": df,
                "t_crit_05": fmt(tcrit, "{:.3f}"), "significant": sig,
                "n_down": f"{n_down}/{n}", "sign_stable": stable,
                "gpu_split": split,
                "verdict": verdict,
            }
            for s in SEEDS:
                row[f"bb40k_{s}"] = fmt(prev[s])
                row[f"bb100k_{s}"] = fmt(cur[s])
                row[f"delta_{s}"] = fmt(deltas.get(s), "{:+.6f}")
            rows.append(row)
    return rows


def branch_rows(scores: dict, recorded: dict) -> list[dict]:
    """The branch re-derived per seed with BOTH ends at that seed.

    `results/seed_branches.csv` held bb40k fixed at seed 20260722 because
    that was the only bb40k number there was. With the bb40k end replicated
    the comparison the rule actually makes can be replayed seed by seed.
    """
    cap = experiment_step_cap()
    out = []
    for cell in CELLS:
        got, down = {}, {h: [] for h in HEADS}
        for s in SEEDS:
            p = {h: scores.get((cell, PREV_STOP, h, s)) for h in HEADS}
            c = {h: scores.get((cell, STOP, h, s)) for h in HEADS}
            if any(v is None for v in p.values()) or \
               any(v is None for v in c.values()):
                got[s] = None
                continue
            got[s] = ladder_decision(STOP, p, c, step_cap=cap)["branch"]
            for h in HEADS:
                if c[h] < p[h]:
                    down[h].append(s)

        have = [b for b in got.values() if b is not None]
        if not have:
            continue
        n_distinct = len(set(have))
        complete = len(have) == len(SEEDS)
        survives = "" if not complete else ("yes" if n_distinct == 1 else "no")
        rec = recorded.get(cell, "")

        if not complete:
            verdict = f"{len(have)}/{len(SEEDS)} seeds paired — no claim yet"
        elif survives == "yes" and got[PROTOCOL_SEED] == rec:
            verdict = ("holds — the same branch at all three seeds with both "
                       "ends seed-matched, and it is the recorded one")
        elif survives == "yes":
            verdict = (f"stable at {got[PROTOCOL_SEED]} on all three seeds, but "
                       f"the ladder recorded {rec}")
        else:
            others = sorted({b for b in have if b != got[PROTOCOL_SEED]})
            moved = [h for h in HEADS if 0 < len(down[h]) < len(SEEDS)]
            verdict = (f"FLIPS to {', '.join(others)} — the head seed alone "
                       f"changes the sign on the "
                       f"{' and '.join(moved) if moved else 'heads'}")
        out.append({
            "cell": cell, "recorded_branch": rec,
            "branch_20260722": got[PROTOCOL_SEED] or "",
            "branch_20260723": got[REPLICATE_SEEDS[0]] or "",
            "branch_20260724": got[REPLICATE_SEEDS[1]] or "",
            "n_distinct": n_distinct, "survives_paired": survives,
            "student_down_seeds": " ".join(str(s) for s in down["student"]),
            "teacher_down_seeds": " ".join(str(s) for s in down["teacher"]),
            "verdict": verdict,
        })
    return out


def hw_rows(scores: dict, hw: dict) -> list[dict]:
    out = []
    for cell in CELLS:
        if not needs_hw_control(cell):
            continue
        for head in HEADS:
            ref = scores.get((cell, PREV_STOP, head, PROTOCOL_SEED))
            off = hw.get((cell, head))
            if off is None:
                note = "control not scored yet"
                ctl = None
            else:
                ctl = ref + off
                note = ("4090 minus 5090 at one fixed seed, same backbone "
                        "checkpoint and same 15,000-step head")
            out.append({
                "cell": cell, "head": head, "stop": PREV_STOP,
                "seed": PROTOCOL_SEED,
                "gpu_5090": fmt(ref), "gpu_4090": fmt(ctl),
                "hw_offset": fmt(off, "{:+.6f}"), "note": note,
            })
    return out


def audit_rows(runs: str) -> list[list]:
    """Every head this pass trained, in ladder_all.csv's schema.

    audit_scores.py walks a ladder-shaped CSV and traces each row to the
    GIFT-Eval summary that produced it. The bb100k replicates of the six
    cells already have such a table in seed_spread_rows.csv, so this one
    holds exactly what that file does not: every bb40k replicate, the GPU
    control, and the two late cells' bb100k replicates. Emitting the
    overlap as well would put the same row under two audits and make the
    check count meaningless.
    """
    rows = []
    for cell in CELLS:
        spec = CELL_BY_SLUG.get(cell, {})
        for head in HEADS:
            for stop in (PREV_STOP, STOP):
                if stop == STOP and cell in SPREAD_CELLS:
                    continue
                for seed in REPLICATE_SEEDS:
                    v = read_score(score_path(runs, cell, stop, head, seed))
                    if v is not None:
                        rows.append([cell, spec.get("arm", ""),
                                     spec.get("align") or "", stop, head,
                                     head_steps_for(stop), f"{alpha_at(stop):.6f}",
                                     seed, "", f"{v:.6f}"])
            if not needs_hw_control(cell):
                continue
            v = read_score(score_path(runs, cell, PREV_STOP, head,
                                      PROTOCOL_SEED, HW_TAG))
            if v is not None:
                rows.append([cell, spec.get("arm", ""), spec.get("align") or "",
                             PREV_STOP, head, head_steps_for(PREV_STOP),
                             f"{alpha_at(PREV_STOP):.6f}", PROTOCOL_SEED,
                             HW_TAG, f"{v:.6f}"])
    rows.sort(key=lambda r: (r[0], r[3], r[4], r[7], r[8]))
    return rows


def recorded_branches(results: str) -> dict:
    path = os.path.join(results, "decisions_all.csv")
    out = {}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("stop") == str(STOP) and r.get("rule_branch_now"):
                    out[r["cell"]] = r["rule_branch_now"]
    except OSError:
        pass
    return out


def write(path: str, columns: list[str], rows) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        if rows and isinstance(rows[0], dict):
            w = csv.DictWriter(fh, columns, lineterminator=LF)
            w.writeheader()
            w.writerows(rows)
        else:
            w = csv.writer(fh, lineterminator=LF)
            w.writerow(columns)
            w.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    res_default = os.path.join(os.path.dirname(SCRIPTS_DIR), "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--runs", default=os.environ.get("CF393_RUNS", RUNS_DEFAULT))
    p.add_argument("--results", default=res_default)
    p.add_argument("--check", action="store_true", help="print, write nothing")
    a = p.parse_args()

    scores, hw = collect(a.runs)
    rows = delta_rows(scores, hw)
    br = branch_rows(scores, hw, recorded_branches(a.results))
    hwr = hw_rows(scores, hw)

    n40 = sum(1 for k in scores if k[1] == PREV_STOP and k[3] != PROTOCOL_SEED)
    n_hw = len(hw)
    print(f"[paired] {n40} bb40k replicate score(s) and {n_hw} GPU-control "
          f"pair(s) on disk under {a.runs}")
    width = max((len(r["cell"]) for r in rows), default=10)
    for r in rows:
        print(f"  {r['cell']:<{width}} {r['head']:<8} "
              f"d=[{r['delta_20260722'] or '-':>9} {r['delta_20260723'] or '-':>9}"
              f" {r['delta_20260724'] or '-':>9}]  mean {r['delta_mean']:>9}"
              f"  se_pair {r['se_paired'] or '-':>8}  t {r['t_paired'] or '-':>7}"
              f"  down {r['n_down']}  {r['verdict']}")
    print()
    for r in hwr:
        print(f"  [hw] {r['cell']:<{width}} {r['head']:<8} "
              f"5090 {r['gpu_5090'] or '-':>8}  4090 {r['gpu_4090'] or '-':>8}"
              f"  offset {r['hw_offset'] or '-':>9}")
    print()
    for r in br:
        print(f"  {r['cell']:<{width}} recorded {r['recorded_branch']:<14}"
              f" paired [{r['branch_20260722'] or '-'},"
              f" {r['branch_20260723'] or '-'}, {r['branch_20260724'] or '-'}]"
              f"  {r['verdict']}")

    if a.check:
        return 0
    write(os.path.join(a.results, "paired_delta.csv"), DELTA_COLUMNS, rows)
    write(os.path.join(a.results, "paired_branches.csv"), BRANCH_COLUMNS, br)
    write(os.path.join(a.results, "hw_control.csv"), HW_COLUMNS, hwr)
    # ladder_all.csv's schema with head_seed and head_tag before the score,
    # so audit_scores.py reads it with the same DictReader and no special case.
    write(os.path.join(a.results, "paired_rows.csv"),
          LADDER_COLUMNS[:-1] + ["head_seed", "head_tag", LADDER_COLUMNS[-1]],
          audit_rows(a.runs))
    print("\n  -> paired_delta.csv, paired_branches.csv, hw_control.csv, "
          "paired_rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
