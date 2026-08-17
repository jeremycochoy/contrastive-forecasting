#!/usr/bin/env python3
"""#401 — when does an arm enter or leave the collapsed state?

`diag_collapse.py` measures saved checkpoints, which exist every 20,000
steps. That grid is too coarse to name a step. The trainer wrote a row for
every step, so this script reads the curves instead.

The trainer has no `eff_rank` column. It has two columns that move with the
same thing and are written every step:

    auc       the in-batch retrieval AUC of f_t against h_{t+1}. 0.50 is
              chance: the forecast picks its own future no better than it
              picks another series' future.
    u_batch   the batch uniformity of the embedding. It falls when the
              batch concentrates on one direction.

This script reports, per arm and per leg:

  * the first step at which a 500-step median of `auc` drops below 0.55,
  * every later crossing back above 0.55 and down again,
  * `auc` and `u_batch` at each saved checkpoint step, so the curve rows
    and the checkpoint rows of `collapse_all.csv` line up.

Usage:
    python3 diag_curve_state.py --out results/diag/curve_state.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

CF401 = "/home/jupyter/checkpoints_backup/cf-401"
CF393 = "/home/jupyter/checkpoints_backup/cf-393/arm6_v2_combab_alignS"

ARMS = [
    (0, CF393, "cf393_arm6_v2_combab_alignS"),
    (8, f"{CF401}/k8/arm6_v2_combab_alignS",
     "cf393_arm6_v2_combab_alignS_cf373k8"),
    (16, f"{CF401}/k16/arm6_v2_combab_alignS",
     "cf393_arm6_v2_combab_alignS_cf373k16"),
    (32, f"{CF401}/k32/arm6_v2_combab_alignS",
     "cf393_arm6_v2_combab_alignS_cf373k32"),
]
LEGS = ["leg_40k", "leg_100k", "leg_200k"]

AUC_CHANCE = 0.55      # above this the retrieval is better than chance
WIN = 500              # median window, to ignore single-step noise


def read_curve(path):
    """step, auc, u_batch, top1 from a trainer losses CSV."""
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = {c: [] for c in ("step", "auc", "u_batch", "top1")}
        for row in rd:
            for c in cols:
                try:
                    cols[c].append(float(row[c]))
                except (TypeError, ValueError):
                    cols[c].append(np.nan)
    return {c: np.asarray(v) for c, v in cols.items()}


def rolling_median(v, w):
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = v[lo:i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def crossings(step, med):
    """(step, direction) for every crossing of AUC_CHANCE, first state first."""
    ok = np.isfinite(med)
    s, m = step[ok], med[ok]
    if s.size == 0:
        return [], None
    state = m[0] >= AUC_CHANCE
    first = "above" if state else "below"
    out = []
    for i in range(1, s.size):
        now = m[i] >= AUC_CHANCE
        if now != state:
            out.append((int(s[i]), "up" if now else "down"))
            state = now
    return out, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diag/curve_state.csv")
    a = ap.parse_args()

    rows = []
    for k, root, stem in ARMS:
        for leg in LEGS:
            p = Path(root) / leg / f"{stem}_losses.csv"
            if not p.is_file():
                continue
            c = read_curve(p)
            med = rolling_median(c["auc"], WIN)
            cr, first = crossings(c["step"], med)
            lo, hi = int(c["step"][0]), int(c["step"][-1])
            print(f"\nk={k:<3} {leg:<9} steps {lo}..{hi}  "
                  f"start {first}  crossings={len(cr)}")
            for s, d in cr[:12]:
                print(f"    step {s:>7}  AUC goes {d}")
            fin = med[np.isfinite(med)]
            print(f"    AUC median first500={fin[min(499, fin.size - 1)]:.4f} "
                  f"last={fin[-1]:.4f}   "
                  f"u_batch last={np.nanmedian(c['u_batch'][-500:]):.4f}")

            # checkpoint steps, so these rows join collapse_all.csv. A leg
            # holds its own step range only, so 20k belongs to leg_40k and
            # not to the first rows of leg_100k.
            for st in range(20000, hi + 1, 20000):
                if st < lo:
                    continue
                j = int(np.searchsorted(c["step"], st))
                if j >= c["step"].size:
                    continue
                w = slice(max(0, j - WIN), j + 1)
                rows.append(dict(
                    k=k, leg=leg, step=st,
                    auc=float(np.nanmedian(c["auc"][w])),
                    u_batch=float(np.nanmedian(c["u_batch"][w])),
                    top1=float(np.nanmedian(c["top1"][w])),
                    n_crossings=len(cr), start_state=first,
                    first_down=cr[0][0] if cr and cr[0][1] == "down" else "",
                ))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["k", "leg", "step", "auc", "u_batch", "top1", "n_crossings",
            "start_state", "first_down"]
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
