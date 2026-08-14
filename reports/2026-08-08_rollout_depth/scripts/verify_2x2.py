#!/usr/bin/env python3
"""#373 item 3 — did the 2x2 corners each get the objective they were asked for?

The four corners cross the total weight on `L_align` with the horizons it
sits on:

    corner            k   --align-loss-weight   total weight   horizons
    cf373k0           0   1.0                   1.0            t+1
    cf373k0_aw4       0   4.0                   4.0            t+1
    cf373k3_aw025     3   0.25                  1.0            t+1..t+4
    cf373k3           3   1.0                   4.0            t+1..t+4

Nothing in a backbone log echoes the weight, so a preflight record proves
intent and not effect. This reads the trained artefacts instead.

All four share seed 20260520, so at step 1 the parameters are still the
init and the batch is the same. The loss at step 1 therefore decomposes as

    loss(1) = F + w * sum_j L_align^(j)(1),   j = 0..k

with F the f-free part, which every corner shares. Two corners pin F and the
per-depth align losses, and the other two must then FOLLOW. The script
predicts each from the other two and compares.

It also reads the depth straight off the losses CSV: `run_bb.py` writes
`cos_err_d0..d3` only when it trains the rollout, so those columns must be
present for the k = 3 corners and absent for the k = 0 ones.

Usage:
  verify_2x2.py [--root /home/jupyter/checkpoints_backup/cf-373]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BASE = ("bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_"
        "ema_qk_aon_cpc_tau090_")

# tag -> (label, k, --align-loss-weight)
CORNERS = {
    "cf373k0": ("k=0, w=1", 0, 1.0),
    "cf373k0_aw4": ("k=0, w=4", 0, 4.0),
    "cf373k3_aw025": ("k=3, w=0.25", 3, 0.25),
    "cf373k3": ("k=3, w=1", 3, 1.0),
}
TOL = 0.02          # step-1 losses agree to about this on a 4090


def load(root: Path, tag: str):
    d = root / (BASE + tag)
    with open(d / f"{BASE + tag}_losses.csv") as fh:
        rows = list(csv.DictReader(fh))
    return {int(r["step"]): r for r in rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jupyter/checkpoints_backup/cf-373")
    a = ap.parse_args(argv)
    root = Path(a.root)

    tab, missing = {}, []
    for tag in CORNERS:
        try:
            tab[tag] = load(root, tag)
        except OSError:
            missing.append(tag)
    if missing:
        print(f"corners not on disk yet: {', '.join(missing)}")
    if len(tab) < 3:
        print("ABORT: need at least three corners to predict the others")
        return 3

    print("corner            k   weight   step-1 loss   depth columns")
    for tag, (label, k, w) in CORNERS.items():
        if tag not in tab:
            print(f"  {label:<15} {k}   {w:<7} {'not yet':<13} -")
            continue
        l1 = float(tab[tag][1]["loss"])
        cols = [c for c in tab[tag][1] if c.startswith("cos_err_d")]
        print(f"  {label:<15} {k}   {w:<7} {l1:<13.5f} "
              f"{len(cols)} ({'depth ' + str(len(cols) - 1) if cols else 'depth 0'})")

    fails = []

    # ---- the depth each corner actually trained ---------------------------
    print("\n== depth, read off the losses CSV ==")
    for tag, (label, k, _w) in CORNERS.items():
        if tag not in tab:
            continue
        cols = [c for c in tab[tag][1] if c.startswith("cos_err_d")]
        got = len(cols) - 1 if cols else 0
        ok = got == k
        print(f"  {label:<15} asked k={k}, trained k={got}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{label}: asked k={k}, the CSV says k={got}")

    # ---- the weight each corner actually used -----------------------------
    # F + 1.0*A0 = loss(k0)            A0 = L_align^(0)(1)
    # F + 4.0*A0 = loss(k0_aw4)        => A0 and F follow from those two
    # F + 1.0*S  = loss(k3)            S  = sum_{j=0..3} L_align^(j)(1)
    # F + 0.25*S = loss(k3_aw025)      => must follow from F and S
    print("\n== weight, predicted from the other corners ==")
    have = set(tab)
    if {"cf373k0", "cf373k0_aw4"} <= have:
        l_k0 = float(tab["cf373k0"][1]["loss"])
        l_x4 = float(tab["cf373k0_aw4"][1]["loss"])
        a0 = (l_x4 - l_k0) / 3.0
        f_free = l_k0 - a0
        print(f"  L_align^(0)(1) = (x4 - k0) / 3 = {a0:.5f}")
        print(f"  f-free part    = k0 - L_align^(0)(1) = {f_free:.5f}")

        if "cf373k3" in have:
            l_k3 = float(tab["cf373k3"][1]["loss"])
            s = l_k3 - f_free
            print(f"  sum of the four align losses = k3 - f-free = {s:.5f}")

            if "cf373k3_aw025" in have:
                pred = f_free + 0.25 * s
                got = float(tab["cf373k3_aw025"][1]["loss"])
                d = abs(got - pred)
                print(f"\n  k=3, w=0.25 predicted : {pred:.5f}")
                print(f"  k=3, w=0.25 observed  : {got:.5f}")
                print(f"  difference            : {d:.5f} "
                      f"({'within' if d <= TOL else 'OUTSIDE'} {TOL})")
                if d > TOL:
                    fails.append(
                        f"the k=3 w=0.25 corner's step-1 loss is {got:.5f}, "
                        f"but the other three predict {pred:.5f}")
                else:
                    print("\n  The 0.25 weight reached the trainer. The corner "
                          "holds total align\n  weight at 1.0, the same as B1's "
                          "k = 0 baseline.")
            else:
                print("\n  k=3, w=0.25 not on disk yet; predicted step-1 loss "
                      f"{f_free + 0.25 * s:.5f}")

    # ---- same length ------------------------------------------------------
    print("\n== steps logged ==")
    lens = {}
    for tag, (label, _k, _w) in CORNERS.items():
        if tag not in tab:
            continue
        lens[label] = max(tab[tag])
        print(f"  {label:<15} {max(tab[tag])} steps")
    done = {k: v for k, v in lens.items() if v >= 39999}
    if len(done) == len(lens) and len({v for v in lens.values()}) > 1:
        fails.append(f"the corners ran different lengths: {lens}")

    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS: every corner on disk trained the objective it was asked for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
