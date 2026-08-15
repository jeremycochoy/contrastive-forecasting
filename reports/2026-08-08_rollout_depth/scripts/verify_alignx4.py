#!/usr/bin/env python3
"""#373 item 3 — did `--align-loss-weight 4.0` reach the trainer?

`results/gap3_preflight.txt` records the flags the launcher meant to pass.
The backbone log does not echo them, so the preflight alone proves intent and
not effect. This reads the trained artefact instead.

The control shares seed 20260520, batch order and init with the k = 0
baseline, so the two runs differ in exactly one term's weight. At the first
logged steps the parameters are still near-identical, so

    loss(k = 0, x4) - loss(k = 0)  ~=  (4 - 1) * L_align

must be positive and sizeable. Had the flag been dropped the two curves would
start on top of each other.

The depth is read the same way. `run_bb.py` writes `cos_err_d0..d3` only when
it trains the rollout, so those columns appear for k = 3 and must be absent
for both k = 0 columns.

Usage:
  verify_alignx4.py [--root /home/jupyter/checkpoints_backup/cf-373]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

BASE = ("bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_"
        "ema_qk_aon_cpc_tau090_")
COLS = ("cf373k0", "cf373k0_aw4", "cf373k3")
NAMES = {"cf373k0": "k = 0", "cf373k0_aw4": "k = 0, L_align x4",
         "cf373k3": "k = 3"}
PROBE = (1, 2, 5, 10, 100, 1000, 10000, 39999)


def load(root: Path, tag: str):
    d = root / (BASE + tag)
    with open(d / f"{BASE + tag}_losses.csv") as fh:
        return {int(r["step"]): r for r in csv.DictReader(fh)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jupyter/checkpoints_backup/cf-373")
    a = ap.parse_args(argv)
    root = Path(a.root)

    tab = {}
    for t in COLS:
        try:
            tab[t] = load(root, t)
        except OSError as e:
            print(f"ABORT: cannot read {t}: {e}")
            return 3

    print("step       k = 0       k = 0 x4     diff        k = 3")
    for s in PROBE:
        if s not in tab["cf373k0"] or s not in tab["cf373k0_aw4"]:
            continue
        x = float(tab["cf373k0"][s]["loss"])
        y = float(tab["cf373k0_aw4"][s]["loss"])
        z = (float(tab["cf373k3"][s]["loss"])
             if s in tab["cf373k3"] else float("nan"))
        print(f"{s:<10} {x:<11.5f} {y:<12.5f} {y - x:<+11.5f} {z:.5f}")

    d1 = float(tab["cf373k0_aw4"][1]["loss"]) - float(tab["cf373k0"][1]["loss"])
    print()
    print(f"step 1 gap, x4 over k = 0 : {d1:+.5f}"
          f"   -> L_align(1) ~= {d1 / 3:.5f}")
    ok_w = d1 > 0.5
    print(f"the x4 weight reached the trainer : {'YES' if ok_w else 'NO'}")

    print()
    print("rollout columns (written only when the rollout is trained):")
    ok_d = True
    for t in COLS:
        cols = [c for c in next(iter(tab[t].values())) if c.startswith("cos_err_d")]
        print(f"  {NAMES[t]:<18} {cols if cols else 'none — depth 0'}")
        if t == "cf373k3" and not cols:
            ok_d = False
        if t != "cf373k3" and cols:
            ok_d = False
    print(f"the control carries no depth      : {'YES' if ok_d else 'NO'}")

    print()
    for t in COLS:
        print(f"  {NAMES[t]:<18} {len(tab[t])} steps logged")
    ok_n = len({len(tab[t]) for t in COLS}) == 1
    print(f"all three columns ran the same length : {'YES' if ok_n else 'NO'}")
    return 0 if (ok_w and ok_d and ok_n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
