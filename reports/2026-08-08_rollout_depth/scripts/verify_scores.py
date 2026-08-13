#!/usr/bin/env python3
"""#373 — recompute every score file from its own raw eval.

The score files are the study's currency. Each was cut from the `Aggregate
GM-Relative MASE` line of a GIFT-Eval summary. This re-derives that line two
ways from the artefacts beside it, so a score that drifted from its own eval
cannot pass:

  leg 1  the geometric mean of the per-config `Relative` column in summary.txt
  leg 2  the per-config `MASE` column in summary.txt against
         `eval_metrics/MASE[0.5]` in all_results.csv, which the harness wrote

summary.txt prints to 4 decimals, so leg 1 carries a rounding error the check
must allow rather than paper over. A fixed tolerance would either hide a real
drift or fail on arithmetic that is correct, so the allowance is derived:

    log GM = mean_i log r_i,   |d log GM| <= mean_i (5e-5 / r_i)
    bound  = GM * mean_i (5e-5 / r_i)  +  5e-5 for the printed aggregate

Usage:
  verify_scores.py --results results
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

ROUND = 5e-5          # half of the last printed decimal
CONFIGS = 97          # every eval in this study runs the same panel

AGG_RE = re.compile(r"Aggregate GM-Relative MASE \((\d+) configs\): ([0-9.]+)")
ROW_RE = re.compile(r"^(\S+/\S+/\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
                    re.M)


def check(res: Path, tag: str, printed: float):
    """Return (ok, note, detail) for one score file."""
    d = res / "eval" / tag
    summ = d / "summary.txt"
    csvp = d / "all_results.csv"
    if not summ.exists():
        return False, "no summary.txt", {}
    if not csvp.exists():
        return False, "no all_results.csv", {}

    txt = summ.read_text(errors="replace")
    agg = AGG_RE.findall(txt)
    if len(agg) != 1:
        return False, f"{len(agg)} aggregate lines, want 1", {}
    n_agg, agg_val = int(agg[0][0]), float(agg[0][1])

    per = ROW_RE.findall(txt)
    if len(per) != n_agg:
        return False, f"{len(per)} table rows vs {n_agg} in the aggregate", {}

    rel = [float(r[3]) for r in per]
    gm = math.exp(sum(map(math.log, rel)) / len(rel))
    bound = gm * sum(ROUND / r for r in rel) / len(rel) + ROUND

    with open(csvp) as fh:
        raw = {r["dataset"]: float(r["eval_metrics/MASE[0.5]"])
               for r in csv.DictReader(fh)}
    if len(raw) != n_agg:
        return False, f"csv holds {len(raw)} configs vs {n_agg}", {}
    if any(r[0] not in raw for r in per):
        return False, "summary names a config the csv does not", {}
    worst = max(abs(raw[r[0]] - float(r[1])) for r in per)

    detail = {"agg": agg_val, "gm": gm, "d_gm": abs(gm - agg_val),
              "bound": bound, "d_mase": worst, "n": n_agg}
    if abs(printed - agg_val) >= ROUND:
        return False, f"score file {printed} vs summary {agg_val}", detail
    if abs(gm - agg_val) > bound:
        return False, f"GM {gm:.6f} vs {agg_val}, outside {bound:.2e}", detail
    if worst > ROUND:
        return False, f"summary MASE off csv by {worst:.2e}", detail
    if n_agg != CONFIGS:
        return False, f"{n_agg} configs, want {CONFIGS}", detail
    return True, "", detail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    res = Path(a.results)

    ok, bad = [], []
    for sf in sorted(res.glob("score_*.txt")):
        tag = sf.stem[len("score_"):]
        try:
            printed = float(sf.read_text().strip())
        except ValueError:
            bad.append((tag, "score file is not a number", {}))
            continue
        good, note, detail = check(res, tag, printed)
        (ok if good else bad).append((tag, note, detail))

    print(f"score files on the branch : {len(ok) + len(bad)}")
    print(f"reproduced from raw evals : {len(ok)}")
    print(f"failed                    : {len(bad)}")
    for tag, note, _d in bad:
        print(f"  FAIL {tag}: {note}")
    seen = [d for _t, _n, d in ok + bad if d]
    if seen:
        print(f"worst |GM recomputed - printed| : "
              f"{max(d['d_gm'] for d in seen):.2e}   "
              f"(worst allowance {max(d['bound'] for d in seen):.2e})")
        print(f"worst |summary MASE - csv MASE| : "
              f"{max(d['d_mase'] for d in seen):.2e}   "
              f"(4-decimal print, allowance {ROUND:.1e})")
        print(f"config counts seen              : "
              f"{sorted({d['n'] for d in seen})}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
