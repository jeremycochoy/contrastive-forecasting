#!/usr/bin/env python3
"""#407 round-3 gap 4 — pool the teacher points, and say what they measure.

Round 3 of the review asked for this pool, and it gave a reason: the teacher
is frozen from step 100,000 on, so the teacher points at 100k, 200k, 300k,
450k and 665k read one encoder. Five draws on one encoder would be a free
null with n = 5, at no GPU cost.

`teacher_head_inputs.py` tests that reason and it fails. The teacher head
does not read teacher tensors only. `prepare_backbone_state_dict` promotes
`teacher_input_to_latent.*` and `teacher_encoder_layers.*` over the
student's slots, and it leaves every other tensor the student's: the
frequency table, the seasonality table and the three forecaster layers.
Those keep training. Between 100k and 200k, 32 of 36 student-owned tensors
move, and the latents the head reads move with them.

So this script gives the pool the review asked for, and it labels the pool
correctly. The teacher points share ONE ENCODER STACK. They do not share one
head input. The spread over them is not a null. It is how far the teacher
head's score travels while its encoder stack stands still and the rest of
its input trains.

That number is still worth having. It bounds what the frozen-encoder half of
the card can contribute, and the card reads the student curve against it.

Usage:
  teacher_pool.py [--results DIR] [--parent DIR] [--csv OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402
import head_band  # noqa: E402

# The step from which the EMA momentum is exactly 1.0, so the teacher
# tensors stop moving. `--ema-tau-ramp-steps 100000`.
FROZEN_FROM = 100_000
HEAD = "teacher"


def frozen_stops():
    """Every stop whose teacher tensors are the frozen ones."""
    stops = full_pass.PARENT_STOPS + [full_pass.RESUME_STEP] + full_pass.STOPS
    return sorted({s for s in stops if s >= FROZEN_FROM})


def points(results, parent):
    """`{stop: score}` for the teacher head, over the frozen stops."""
    out = {}
    for stop in frozen_stops():
        got = head_band.draws(stop, HEAD, results, parent,
                              seeds=[head_band.PROTOCOL_SEED])
        if got:
            out[stop] = got[head_band.PROTOCOL_SEED]
    return out


def input_moves(results):
    """What `teacher_head_inputs.py` found, for every pair it answered."""
    out = []
    pattern = os.path.join(str(results), "teacher_head_inputs_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fh:
                got = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({
            "file": os.path.basename(path),
            "moved_from_teacher": got.get("moved_from_teacher"),
            "moved_from_student": got.get("moved_from_student"),
            "from_student": got.get("from_student"),
            "forward": got.get("forward", {}),
            "verdict": got.get("verdict", ""),
        })
    return out


def one_encoder(moves):
    """True when every frozen pair leaves the teacher tensors untouched."""
    frozen = [m for m in moves if "40k" not in m["file"]]
    return bool(frozen) and all(m["moved_from_teacher"] == 0 for m in frozen)


def head_input_constant(moves):
    """True when no frozen pair moves a tensor the head reads."""
    frozen = [m for m in moves if "40k" not in m["file"]]
    return bool(frozen) and all(m["moved_from_student"] == 0 for m in frozen)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=full_pass.RESULTS)
    ap.add_argument("--parent", default=full_pass.PARENT_RESULTS)
    ap.add_argument("--csv")
    a = ap.parse_args(argv)

    got = points(a.results, a.parent)
    moves = input_moves(a.results)

    print(f"teacher points from step {FROZEN_FROM:,} on, head seed "
          f"{head_band.PROTOCOL_SEED}")
    for stop, value in sorted(got.items()):
        print(f"  bb{stop // 1000:>3}k  {value:.4f}")
    if len(got) < 2:
        print("fewer than two points on disk, so there is no spread yet")
        return 0

    values = list(got.values())
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    span = max(values) - min(values)
    print(f"  n {len(values)}   mean {mean:.4f}   std {std:.4f}   "
          f"range {span:.4f}")

    print("")
    print("what these points share, from teacher_head_inputs_*.json")
    for m in moves:
        print(f"  {m['file']}: teacher tensors moved "
              f"{m['moved_from_teacher']}, student-owned moved "
              f"{m['moved_from_student']} of {m['from_student']}")
        for tag, r in (m["forward"] or {}).items():
            print(f"     {tag:<20} identical={r['identical']}  "
                  f"rel L2 {r['rel_l2']:.3e}")

    print("")
    if not moves:
        print("VERDICT: run teacher_head_inputs.py first. Without it, this "
              "pool carries no label.")
    elif head_input_constant(moves):
        print(f"VERDICT: NULL with n = {len(values)}. Every tensor the "
              f"teacher head reads is equal across these stops, so the "
              f"{span:.4f} range is run-to-run noise at one head seed.")
    elif one_encoder(moves):
        print(f"VERDICT: NOT a null. The teacher ENCODER STACK is common to "
              f"these {len(values)} points, but the teacher head also reads "
              f"student-owned tensors that keep training, so the two inputs "
              f"differ. The {span:.4f} range measures how far the teacher "
              f"head travels while its encoder stack stands still. It is an "
              f"upper bound on the frozen-encoder contribution, not a "
              f"repeatability band.")
    else:
        print("VERDICT: the teacher tensors are not equal across these "
              "stops, so the points do not share one encoder.")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stop", "head", "seed", "score"])
            for stop, value in sorted(got.items()):
                w.writerow([stop, HEAD, head_band.PROTOCOL_SEED, value])
            w.writerow([])
            w.writerow(["n", "mean", "std", "range", "one_encoder",
                        "head_input_constant"])
            w.writerow([len(values), f"{mean:.4f}", f"{std:.4f}",
                        f"{span:.4f}", one_encoder(moves),
                        head_input_constant(moves)])
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
