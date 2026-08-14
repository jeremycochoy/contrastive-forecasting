#!/usr/bin/env python3
"""#373 — the PUBLISHED k = 0 numbers, split by horizon term and by domain.

`split_scores.py` splits this study's own evals. The per-domain and the
per-horizon questions also need the k = 0 side, and the three parent reports
print only the 97-config aggregate. Their per-config CSVs are in reach, so
this script splits them the same way and writes `results/splits_k0.csv` with
the same columns `splits.csv` carries.

A parent CSV is accepted for a (cell, stop, head) only when its own
aggregate reproduces the number that parent PRINTED, to four decimals. That
is `published_bootstrap.py`'s own gate, reused here, so a CSV that is not
the published run cannot enter a figure.

Row key `stop` is `<cell>_k0_bb<stop>k_<head>`, which no eval directory of
this study writes, so the two files never collide.

Usage: parent_splits.py [--results DIR] [--out results/splits_k0.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_bootstrap as PB                             # noqa: E402
import published                                          # noqa: E402
import published_bootstrap as PBOOT                       # noqa: E402
import split_scores as SS                                 # noqa: E402
import r2_ladder as L                                     # noqa: E402


def gm(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "splits_k0.csv"

    sn = PB.read_mase(PB.SN_REF)
    rows, kept, dropped = [], 0, 0
    for cell in L.CELLS:
        for head in L.HEADS:
            for stop_k in L.STOPS:
                pub = published.at(cell, head, stop_k)
                if pub is None:
                    continue
                p = PBOOT.parent_csv(cell, stop_k, head, res)
                if p is None:
                    continue
                mine = PB.read_mase(p)
                shared = [d for d in mine if d in sn]
                got = gm([mine[d] / sn[d] for d in shared])
                if abs(got - pub) > 0.00005 + 1e-9:
                    print(f"DROP {cell} bb{stop_k}k {head}: CSV aggregates to "
                          f"{got:.4f}, the parent printed {pub:.4f}")
                    dropped += 1
                    continue
                kept += 1
                doms = SS.read_domains(p)
                label = f"{cell}_k0_bb{stop_k}k_{head}"
                by_term, by_dom = {}, {}
                for ds in shared:
                    r = mine[ds] / sn[ds]
                    by_term.setdefault(SS.term_of(ds), []).append(r)
                    d = doms.get(ds)
                    if d:
                        by_dom.setdefault(d, []).append(r)
                rows.append((label, "all", "all", len(shared), gm(
                    [mine[d] / sn[d] for d in shared])))
                for t in sorted(by_term):
                    rows.append((label, "term", t, len(by_term[t]),
                                 gm(by_term[t])))
                for d in sorted(by_dom):
                    rows.append((label, "domain", d, len(by_dom[d]),
                                 gm(by_dom[d])))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stop", "split", "name", "n", "gm_rel_mase"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}"])
    print(f"wrote {out}  ({kept} published (cell, stop, head) reproduced, "
          f"{dropped} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
