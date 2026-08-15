#!/usr/bin/env python3
"""#373 — did the depth reach this cell's loss?

Reads the first row of a k = 0 losses CSV and a k = 3 losses CSV of the SAME
cell, trained from the same seed. Both runs start at the same weights and
draw the same first batch, so the first row is the discriminating one.

Four checks. Every one of them has a failure the review rounds actually saw.

  1. The k = 3 CSV carries cos_err_d0..d3. A k = 0 run writes no cos_err_d*
     column, so their presence is the proof the flag reached the trainer.
  2. The k = 0 CSV carries none. If it does, the k = 0 arm is not a k = 0
     arm and the pair is not a comparison.
  3. `loss_tau_ref` matches across the two. It is pinned to depth 0, so a
     mismatch means the two runs saw different batches and nothing below
     can be read.
  4. `loss` does NOT match. This is the check that catches the silent
     failure: twelve of the fourteen cells carry L_align as their only
     f-bearing term, and an unwired depth on that arm completes the run,
     writes three plausible cos_err_dj curves, and reproduces the k = 0
     loss to the last digit.

Check 4 is the one that matters. 1 and 2 are cheap and would pass on a run
that trained at k = 0.

Usage: check_depth_reached.py <cell id> <k0 losses.csv> <k3 losses.csv> [k]
Prints one TSV line and exits non-zero on any failed check.
"""
import csv
import sys


def first_row(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"ABORT: {path} holds a header and no rows")
    return rows[0]


def main(argv):
    if len(argv) not in (4, 5):
        raise SystemExit(__doc__)
    cell, k0_path, k3_path = argv[1:4]
    k = int(argv[4]) if len(argv) == 5 else 3

    k0, k3 = first_row(k0_path), first_row(k3_path)
    want = [f"cos_err_d{j}" for j in range(k + 1)]
    fails = []

    missing = [c for c in want if c not in k3]
    if missing:
        fails.append(f"k={k} CSV lacks {','.join(missing)}")

    stray = sorted(c for c in k0 if c.startswith("cos_err_d"))
    if stray:
        fails.append(f"k=0 CSV carries {','.join(stray)} — it is not a k=0 run")

    # The depth-0 reference. Equal means same weights and same batch.
    ref0, ref3 = float(k0["loss_tau_ref"]), float(k3["loss_tau_ref"])
    if abs(ref0 - ref3) > 1e-6:
        fails.append(f"loss_tau_ref differs: {ref0!r} vs {ref3!r} — "
                     "the two runs did not see the same first batch")

    loss0, loss3 = float(k0["loss"]), float(k3["loss"])
    if abs(loss0 - loss3) <= 1e-6:
        fails.append(f"loss is unchanged at {loss0!r} — this cell trained at "
                     f"k=0 with --train-rollout-depth {k} on its command line")

    # The per-depth curves must not be one number written k+1 times: the
    # deeper entries read their own shifted window.
    depths = []
    if not missing:
        depths = [float(k3[c]) for c in want]
        if len(set(depths)) == 1:
            fails.append(f"cos_err_d0..d{k} are all {depths[0]!r} — the "
                         "deeper depths are not reading their own window")

    verdict = "FAIL" if fails else "PASS"
    print("\t".join([
        cell, verdict, f"step={k3['step']}",
        f"loss_k0={loss0:.10f}", f"loss_k{k}={loss3:.10f}",
        f"d_loss={loss3 - loss0:+.6f}",
        f"loss_tau_ref={ref0:.10f}",
        "cos_err=" + ",".join(f"{d:.4f}" for d in depths),
    ]))
    for f in fails:
        print(f"  {cell}: {f}", file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
