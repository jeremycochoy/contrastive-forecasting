#!/usr/bin/env python3
"""#373 review gap 2 — an interval on the deltas against the published k = 0.

The verdict table reads this study's `k = 3` against the number its parent
report published, and it thresholds the difference on a band. Every other
delta table in this study carries a paired dataset-cluster bootstrap. That
one did not, because the parents were read as four printed decimals.

Every parent's per-config CSV is in reach, so the pairing is recoverable:
the same 97 configs, the same seasonal-naive denominator file, the same
resampling unit as every other interval in this study. This computes it.

THE MAPPING IS PROVEN, NOT ASSUMED. A parent CSV is accepted for a cell only
when its own aggregate reproduces the number that parent published, to the
four decimals the parent printed. A mismatch means this script picked the
wrong run, and the row is dropped with a line saying so rather than carrying
an interval from another recipe.

  group B, `L_align` on the student  reports/2026-07-21_split_pred_rep_small
  group B, `L_align` on the teacher  reports/2026-08-04_lalign_teacher
  group A                            reports/2026-08-04_ema_sched_ladder
                                     committed none. Its run tree on elisa
                                     holds them, so this copies the ones it
                                     uses into `results/parent_eval/` and
                                     reads the committed copy. `--no-import`
                                     turns that off.

Usage: published_bootstrap.py [--results <dir>] [--out <csv>] [--iters N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from published import PUBLISHED                              # noqa: E402
import paired_bootstrap as PB                                # noqa: E402

REPO = HERE.parent.parent.parent
SPLIT = REPO / "reports" / "2026-07-21_split_pred_rep_small" / "results" / "eval_gm_mase"
LALIGN = REPO / "reports" / "2026-08-04_lalign_teacher" / "results" / "eval_gm_mase"

# cell -> (parent eval root, the parent's own run name for that cell).
# The arm is the cell's own arm from cells.tsv; the parent is chosen by which
# encoder the cell's `L_align` targets, which is what splits the two parent
# reports.
PARENT = {
    "B1":  (SPLIT,  "arm6_v2_combab"),
    "B3":  (SPLIT,  "arm5_combab"),
    "B5":  (SPLIT,  "arm4_combab"),
    "B6":  (SPLIT,  "arm6_v2_ncpc"),
    "B9":  (SPLIT,  "arm1_nse"),
    "B10": (SPLIT,  "arm6_v2_nse"),
    "B2":  (LALIGN, "arm6_v2_combab"),
    "B4":  (LALIGN, "arm5_combab"),
    "B7":  (LALIGN, "arm6_v2_ncpc"),
    "B8":  (LALIGN, "arm6_v2_nse"),
}

# Group A's parent is #393's ladder. It published two heads per stop and its
# eval tree is on elisa, one directory per (stop, encoder). The cell -> arm
# map is cells.tsv's own `arg` column for the four group-A rows.
LADDER_ROOT = Path("/home/jupyter/checkpoints_backup/cf-393")
LADDER = {"A1": "arm5_combab_alignS",   "A2": "arm6_v2_nse_alignT",
          "A3": "arm6_v2_combab_alignT", "A4": "arm6_v2_combab_alignS"}

# The parents print four decimals, so a reproduced aggregate can sit half a
# unit of the last place away and still be the same run.
PRINT_HALF = 0.00005


def parent_csv(cell, stop_k, head, res, do_import=True):
    """The parent's own 97-config CSV for this (cell, stop, head), inside the
    repo. Group A's is imported from #393's run tree on first use and read
    from the committed copy after that, so a rebuild needs the repo alone."""
    if cell in PARENT:
        if head != "student":
            return None                 # group B's parents published one head
        root, name = PARENT[cell]
        hd = 15000 if stop_k <= 40 else 30000
        p = root / f"{name}_bb{stop_k}k_hd{hd}s" / "all_results.csv"
        return p if p.is_file() else None

    if cell not in LADDER:
        return None
    local = res / "parent_eval" / f"{cell}_bb{stop_k}k_{head}" / "all_results.csv"
    if local.is_file():
        return local
    if not do_import:
        return None
    src = (LADDER_ROOT / LADDER[cell] / "eval" / f"bb{stop_k}k_{head}"
           / "gift" / "all_results.csv")
    if not src.is_file():
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(src.read_bytes())
    (local.parent / "source.txt").write_text(
        f"{src}\nmd5 {hashlib.md5(src.read_bytes()).hexdigest()}\n"
        f"#393 ladder, cell {cell} = {LADDER[cell]}, bb{stop_k}k, {head} head\n")
    print(f"     imported {src}")
    return local


def aggregate(csv_path, sn, subset="all"):
    """That CSV's own GM-Relative MASE over every config it shares with the
    seasonal-naive reference, on one horizon subset."""
    import math
    m = PB.read_mase(csv_path)
    common = sorted(set(m) & set(sn))
    if subset != "all":
        common = PB.subsets(common)[subset]
    if not common:
        return None, 0
    return PB.gm([math.log(m[d] / sn[d]) for d in common]), len(common)


def criterion(pcsv, mine, sn):
    """The card's own per-horizon criterion on one pair: medium+long at least
    5% better and short losing less than 2%. Returns the two percentages and
    the verdict, or None if a subset is empty."""
    out = {}
    for sub in ("short", "medium_long"):
        a, _ = aggregate(pcsv, sn, sub)
        b, _ = aggregate(mine, sn, sub)
        if a is None or b is None:
            return None
        out[sub] = 100.0 * (b / a - 1.0)
    out["met"] = out["medium_long"] <= -5.0 and out["short"] < 2.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--no-import", action="store_true",
                    help="read only CSVs already in the repo")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "published_bootstrap.csv"
    out.unlink(missing_ok=True)

    sn = PB.read_mase(PB.SN_REF)
    ok = dropped = 0
    screen = []
    for cell in sorted(set(PARENT) | set(LADDER)):
      for head in ("student", "teacher"):
        for stop_k in (40, 100, 200):
            pub = PUBLISHED.get(cell, {}).get(head, {}).get(stop_k)
            if pub is None:
                continue
            mine = res / "eval" / f"{cell}_k3_bb{stop_k}k_{head}" / "all_results.csv"
            if not mine.is_file():
                print(f"skip {cell} bb{stop_k}k {head}: this study's CSV is not in the checkout")
                continue
            pcsv = parent_csv(cell, stop_k, head, res, not a.no_import)
            if pcsv is None:
                print(f"skip {cell} bb{stop_k}k {head}: no per-config CSV from the parent")
                dropped += 1
                continue
            got, n = aggregate(pcsv, sn)
            if got is None or abs(got - pub) > PRINT_HALF:
                print(f"DROP {cell} bb{stop_k}k {head}: {pcsv.parent.name} aggregates to "
                      f"{got:.4f}, the parent published {pub:.4f} — not the same run")
                dropped += 1
                continue
            print(f"ok   {cell} bb{stop_k}k {head}: {pcsv.parent.name} reproduces "
                  f"{pub:.4f} over {n} configs")
            rc = subprocess.run(
                [sys.executable, str(HERE / "paired_bootstrap.py"),
                 "--k0", str(pcsv), "--k3", str(mine),
                 "--label", f"{cell}_vs_pub_bb{stop_k}k_{head}",
                 "--iters", str(a.iters), "--out", str(out)],
                capture_output=True, text=True)
            if rc.returncode != 0:
                print(f"     bootstrap failed: {rc.stderr.strip().splitlines()[-1:]}")
                dropped += 1
                continue
            ok += 1
            c = criterion(pcsv, mine, sn)
            if c is not None:
                screen.append((cell, stop_k, head, c["short"],
                               c["medium_long"], c["met"]))
    print(f"\n{ok} interval(s) -> {out}; {dropped} row(s) dropped")

    # The card's per-horizon criterion, on every one of these pairs. It is a
    # SCREEN here and not a test: the two sides of each pair trained on
    # different machines. The depth-response table applies the same criterion
    # to the pairs that hold the machine.
    scr = out.parent / "criterion_screen.csv"
    with open(scr, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "stop_k", "head", "pct_short", "pct_medium_long",
                    "criterion_met"])
        for row in screen:
            w.writerow([row[0], row[1], row[2], f"{row[3]:.2f}",
                        f"{row[4]:.2f}", "yes" if row[5] else "no"])
    met = sum(1 for r in screen if r[5])
    at100 = [r for r in screen if r[1] == 100]
    print(f"criterion screen -> {scr}: {met} of {len(screen)} pairs meet it; "
          f"at bb100k, {sum(1 for r in at100 if r[5])} of {len(at100)}")


if __name__ == "__main__":
    main()
