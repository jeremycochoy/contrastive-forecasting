#!/usr/bin/env python3
"""#373 — the first logged steps of each run, to show what the seed controls.

The study's B5 backbones differ by seed and by machine, and one of them is a
retrain at a fixed seed on a different machine. That run is only a machine
test if the seed really does pin the data order — otherwise it changes two
things as well.

train.py prints `mixup=<n>/<window>` beside every logged step. `n` counts the
examples the mixer touched in that window, so two runs that see the same
batches in the same order print the same count. Two runs that do not, do not.
The loss beside it then says how far apart the same batches took them.

Usage:
  early_loss.py --out results/early_loss.csv --steps 3 \\
      --run <arm>:<k>=<trainer log> [--run ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                            # noqa: E402

LINE = re.compile(
    r"^\[\s*(\d+)\]\s+loss=([\d.\-]+)\s+ema_loss=([\d.\-]+).*?"
    r"mixup=(\d+)/(\d+)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="ARM:K=LOG")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args(argv)

    rows = []
    for spec in args.run:
        head, path = spec.split("=", 1)
        arm, ktxt = head.split(":")
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        seen = 0
        for line in open(path, errors="replace"):
            m = LINE.match(line)
            if not m:
                continue
            run = R.find_run(arm, int(ktxt), "depth") or \
                R.find_run(arm, int(ktxt), "control")
            rows.append({
                "arm": arm, "k": int(ktxt),
                "seed": run.seed if run else "",
                "machine": run.machine if run else "",
                "step": int(m.group(1)), "loss": m.group(2),
                "ema_loss": m.group(3),
                "mixup": f"{m.group(4)}/{m.group(5)}",
            })
            seen += 1
            if seen >= args.steps:
                break
        if not seen:
            print(f"  skip {spec}: no logged step", file=sys.stderr)

    if not rows:
        raise SystemExit("ABORT: no run produced a logged step")

    order = {a: i for i, a in enumerate(R.ARM_ORDER)}
    rows.sort(key=lambda r: (order.get(r["arm"], 99), r["k"], r["step"]))
    for r in rows:
        print(f"{r['arm']:<7} k={r['k']} seed={r['seed']} "
              f"{r['machine']:<11} step {r['step']:>5}  loss={r['loss']:<8} "
              f"mixup={r['mixup']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
