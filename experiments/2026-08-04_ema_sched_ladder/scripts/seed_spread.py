#!/usr/bin/env python3
"""#393 — the in-study head-seed spread, and which extend-rule branches survive it.

The extend rule reads one number per (cell, stop, head) and compares bb100k
against bb40k with a strict `<`. It has no idea how big the noise on that
difference is. The #390 parent measured head-seed ranges up to 0.0908 on the
same protocol (results/parent_seed_spread.csv), and six of this card's ten
cells moved less than that on BOTH heads — including the two best cells, one
of them terminated on a student move of +0.0026.

So the bb100k heads were retrained at two more seeds, 20260723 and 20260724,
the same two the parent used. Backbones were not retrained: the replicate
varies the head seed and nothing else. This reads the three seeds off the
eval directories and answers two questions.

  1. How wide is the head-seed spread INSIDE this study? Mean and sample sd
     per cell per head, which is what the report plots as error bars.
  2. Does each cell's recorded branch survive it? The branch is re-derived
     from each seed's own bb100k pair against the same bb40k pair, with
     `ladder.ladder_decision` — the same pure function the drivers call. A
     branch that is not the same on all three seeds was decided by noise,
     and the report has to say so rather than presenting it as a finding.

The branch is a joint test on two heads, so what actually decides it is one
bit per head: is this head below its bb40k value, or not. Those two bits are
reported per seed (`student_down_seeds`, `teacher_down_seeds`), because they
say WHERE a flip comes from — a branch can flip on the student's sign, on
the teacher's, or on both, and the three are different findings.

Pairing each seed's student with another seed's teacher adds nothing, and
the guard test is what showed it: the matched branches agree exactly when
each head's sign is constant across seeds, and if both signs are constant
then every one of the nine pairings gives the same branch. So the matched
check is already the complete one.

Writes three files:

  results/seed_spread_rows.csv   one row per (cell, stop, head, seed), in
                                 ladder_all.csv's schema plus `head_seed`,
                                 so audit_scores.py can trace every one of
                                 them to the summary that produced it
  results/seed_spread.csv        per (cell, head): the three values, mean,
                                 sd, range, the bb40k reference and the
                                 change it implies
  results/seed_branches.csv      per cell: the branch at each seed, whether
                                 it survives, and the verdict in words

Usage:  python3 scripts/seed_spread.py [--runs DIR] [--results DIR] [--check]
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
    CELLS,
    LADDER_COLUMNS,
    RUNS_DEFAULT,
    alpha_at,
    experiment_step_cap,
    head_steps_for,
    ladder_decision,
)

PROTOCOL_SEED = 20260722
REPLICATE_SEEDS = [20260723, 20260724]
SEEDS = [PROTOCOL_SEED] + REPLICATE_SEEDS
HEADS = ["student", "teacher"]
STOP = 100000
PREV_STOP = 40000

# The six cells whose branch turned on a difference smaller than the
# parent's 0.0908 on BOTH heads, in the order the card names them.
REPLICATE_CELLS = ["arm6_v2_combab_alignS", "arm6_v2_combab_alignT",
                   "arm5_combab_alignS", "arm5_combab_alignT",
                   "arm6_v2_nse_alignT", "arm6_v2_nse_alignS"]
CELL_BY_SLUG = {c["slug"]: c for c in CELLS}

SPREAD_COLUMNS = ["cell", "head", "bb40k", "seed_20260722", "seed_20260723",
                  "seed_20260724", "n_seeds", "mean", "sd", "range",
                  "delta_mean", "delta_seed_20260722", "resolved"]
BRANCH_COLUMNS = ["cell", "recorded_branch", "branch_20260723",
                  "branch_20260724", "n_matched_distinct", "survives_matched",
                  "student_down_seeds", "teacher_down_seeds", "verdict"]
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
    """{(cell, stop, head, seed): score} for everything on disk."""
    out = {}
    for cell in REPLICATE_CELLS:
        for head in HEADS:
            v = read_score(score_path(runs, cell, PREV_STOP, head, PROTOCOL_SEED))
            if v is not None:
                out[(cell, PREV_STOP, head, PROTOCOL_SEED)] = v
            for seed in SEEDS:
                v = read_score(score_path(runs, cell, STOP, head, seed))
                if v is not None:
                    out[(cell, STOP, head, seed)] = v
    return out


def ladder_rows(scores: dict) -> list[list]:
    """The replicate scores in ladder_all.csv's schema, plus head_seed."""
    rows = []
    for (cell, stop, head, seed), v in scores.items():
        if seed == PROTOCOL_SEED:
            continue          # already in ladder_all.csv, and audited there
        spec = CELL_BY_SLUG[cell]
        rows.append([cell, spec["arm"], spec["align"] or "", stop, head,
                     head_steps_for(stop), f"{alpha_at(stop):.6f}",
                     seed, f"{v:.6f}"])
    rows.sort(key=lambda r: (r[0], r[3], r[4], r[7]))
    return rows


def spread(scores: dict) -> list[dict]:
    rows = []
    for cell in REPLICATE_CELLS:
        for head in HEADS:
            vals = [scores.get((cell, STOP, head, s)) for s in SEEDS]
            have = [v for v in vals if v is not None]
            if not have:
                continue
            bb40k = scores.get((cell, PREV_STOP, head, PROTOCOL_SEED))
            mean = statistics.fmean(have)
            sd = statistics.stdev(have) if len(have) > 1 else 0.0
            rng = max(have) - min(have)
            row = {
                "cell": cell, "head": head,
                "bb40k": "" if bb40k is None else f"{bb40k:.6f}",
                "n_seeds": len(have),
                "mean": f"{mean:.6f}", "sd": f"{sd:.6f}", "range": f"{rng:.6f}",
            }
            for s, v in zip(SEEDS, vals):
                row[f"seed_{s}"] = "" if v is None else f"{v:.6f}"
            if bb40k is None:
                row["delta_mean"] = row["delta_seed_20260722"] = ""
                row["resolved"] = ""
            else:
                d = mean - bb40k
                row["delta_mean"] = f"{d:+.6f}"
                d22 = vals[0]
                row["delta_seed_20260722"] = (
                    "" if d22 is None else f"{d22 - bb40k:+.6f}")
                # A change is resolved only if it is bigger than the whole
                # spread of the thing it is measured on. Range, not sd:
                # three points, and the rule is a sign test, so what matters
                # is whether the seeds could have put the sign the other way.
                # Blank until all three seeds are in — one seed has a range
                # of zero and would call every change resolved.
                row["resolved"] = ("" if len(have) < len(SEEDS)
                                   else "yes" if abs(d) > rng else "no")
            rows.append(row)
    return rows


def branches(scores: dict) -> list[dict]:
    cap = experiment_step_cap()
    out = []
    for cell in REPLICATE_CELLS:
        prev = {h: scores.get((cell, PREV_STOP, h, PROTOCOL_SEED))
                for h in HEADS}
        if any(v is None for v in prev.values()):
            continue

        matched: dict[int, str | None] = {}
        down: dict[str, list[int]] = {h: [] for h in HEADS}
        for s in SEEDS:
            cur = {h: scores.get((cell, STOP, h, s)) for h in HEADS}
            if any(v is None for v in cur.values()):
                matched[s] = None
                continue
            matched[s] = ladder_decision(STOP, prev, cur, step_cap=cap)["branch"]
            for h in HEADS:
                if cur[h] < prev[h]:
                    down[h].append(s)

        recorded = matched[PROTOCOL_SEED]
        got = [b for b in matched.values() if b is not None]
        n_matched = len(set(got))

        # Incomplete is not the same as flipped, and reading it as flipped
        # would report noise wherever a box is simply still training.
        complete = recorded is not None and len(got) == len(SEEDS)
        survives = "" if not complete else ("yes" if n_matched == 1 else "no")

        if not complete:
            verdict = (f"not enough seeds scored yet "
                       f"({len(got)}/{len(SEEDS)} matched)")
        elif survives == "no":
            flipped = sorted({b for b in got if b != recorded})
            moved = [h for h in HEADS
                     if 0 < len(down[h]) < len(SEEDS)]
            verdict = (f"FLIPS to {', '.join(flipped)} — the head seed alone "
                       f"changes the sign on the "
                       f"{' and '.join(moved) if moved else 'heads'}; "
                       f"decided by noise")
        else:
            verdict = ("holds on all three seeds — every head keeps its sign "
                       "against bb40k")

        out.append({
            "cell": cell,
            "recorded_branch": recorded or "",
            "branch_20260723": matched[REPLICATE_SEEDS[0]] or "",
            "branch_20260724": matched[REPLICATE_SEEDS[1]] or "",
            "n_matched_distinct": n_matched,
            "survives_matched": survives,
            "student_down_seeds": " ".join(str(s) for s in down["student"]),
            "teacher_down_seeds": " ".join(str(s) for s in down["teacher"]),
            "verdict": verdict,
        })
    return out


def write(path: str, columns: list[str], rows: list[dict] | list[list]) -> None:
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

    scores = collect(a.runs)
    n_repl = sum(1 for k in scores if k[3] != PROTOCOL_SEED)
    sp, br = spread(scores), branches(scores)

    width = max((len(r["cell"]) for r in sp), default=10)
    print(f"[seeds] {n_repl}/24 replicate score(s) on disk under {a.runs}")
    for r in sp:
        vals = " ".join(f"{r[f'seed_{s}'] or '     -   ':>9}" for s in SEEDS)
        print(f"  {r['cell']:<{width}} {r['head']:<8} {vals}  "
              f"mean {r['mean']} sd {r['sd']} range {r['range']}"
              f"  bb40k->bb100k {r['delta_mean'] or '-':>10}"
              f"  resolved={r['resolved'] or '-'}")
    print()
    for r in br:
        print(f"  {r['cell']:<{width}} recorded {r['recorded_branch']:<14}"
              f" seeds [{r['recorded_branch']}, {r['branch_20260723'] or '-'},"
              f" {r['branch_20260724'] or '-'}]  {r['verdict']}")

    if a.check:
        return 0
    # ladder_all.csv's schema with head_seed inserted before the score, so
    # audit_scores.py reads it with the same DictReader and no special case.
    write(os.path.join(a.results, "seed_spread_rows.csv"),
          LADDER_COLUMNS[:-1] + ["head_seed", LADDER_COLUMNS[-1]],
          ladder_rows(scores))
    write(os.path.join(a.results, "seed_spread.csv"), SPREAD_COLUMNS, sp)
    write(os.path.join(a.results, "seed_branches.csv"), BRANCH_COLUMNS, br)
    print(f"\n  -> seed_spread_rows.csv, seed_spread.csv, seed_branches.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
