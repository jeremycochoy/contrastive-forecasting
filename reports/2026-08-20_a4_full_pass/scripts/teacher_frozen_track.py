#!/usr/bin/env python3
"""#407 — the teacher points are models, not draws. Track them, do not pool.

This script replaces `teacher_pool.py`, which was wrong.

Round 3 of the review asked for a pool of the teacher points, and it gave a
reason: the teacher is frozen from step 100,000 on, so the teacher points at
100k, 200k, 300k, 450k and 665k read one encoder. Five draws on one encoder
would be a free null with n = 5, at no GPU cost.

`teacher_head_inputs.py` tests that reason and it FAILS. The teacher head
does not read teacher tensors only. It loads 110 tensors: 74 come from a
`teacher_*` key and 36 stay the student's. In every pair on disk, 32 of
those 36 move.

  src/checkpoint.py:266    out = dict(state_dict)

That line is the reason. `prepare_backbone_state_dict` starts from the FULL
student state dict. The `encoder_source='teacher'` branch then overwrites
two prefixes only, `teacher_input_to_latent.*` and `teacher_encoder_layers.*`
(`_TEACHER_PROMOTIONS`, src/checkpoint.py:230). Every key those two do not
name keeps the student's tensor: the frequency table, the seasonality table,
the channel-mixing module and the three forecaster layers. Those keep
training after step 100,000.

So the teacher points at 100k, 200k, 300k, 450k and 665k are FIVE MODELS.
They share one encoder stack. They do not share one head input. This script
does not pool them, it does not average them, and it reports no standard
deviation over them. A difference between two of them is a measurement.

THE NOISE BAND OF THIS CARD IS NOT HERE. The card measures it directly, with
three head seeds on one backbone: `head_band.py` and `results/head_band.csv`.
Student range 0.0018, teacher range 0.0064, at 200,000 steps.

What this script gives instead is the teacher track: each stop's score, and
the change from one stop to the next, with the encoder stack held still. It
answers "how far does the score move when only the student-owned half of the
teacher head's input trains".

Usage:
  teacher_frozen_track.py [--results DIR] [--parent DIR] [--csv OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402
import head_band  # noqa: E402

# The step from which the EMA momentum is exactly 1.0, so the teacher
# tensors stop moving. `--ema-tau-ramp-steps 100000`.
FROZEN_FROM = 100_000
HEAD = "teacher"

# The line that makes the teacher head load student tensors. Quoted so the
# artefact carries the reason, not a summary of it.
PROMOTION_SITE = "src/checkpoint.py:266"
PROMOTION_LINE = "out = dict(state_dict)"


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
            "loaded_tensors": got.get("loaded_tensors"),
            "from_teacher": got.get("from_teacher"),
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
    """True when no frozen pair moves a tensor the head reads.

    Only this makes the teacher points draws of ONE model. Every pair on
    disk says False.
    """
    frozen = [m for m in moves if "40k" not in m["file"]]
    return bool(frozen) and all(m["moved_from_student"] == 0 for m in frozen)


def steps(got):
    """`[(from, to, delta), ...]` between neighbouring stops."""
    order = sorted(got)
    return [(a, b, got[b] - got[a]) for a, b in zip(order, order[1:])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=full_pass.RESULTS)
    ap.add_argument("--parent", default=full_pass.PARENT_RESULTS)
    ap.add_argument("--csv")
    a = ap.parse_args(argv)

    got = points(a.results, a.parent)
    moves = input_moves(a.results)
    constant = head_input_constant(moves)

    print(f"teacher track from step {FROZEN_FROM:,} on, head seed "
          f"{head_band.PROTOCOL_SEED}")
    print("Each row is a DIFFERENT model. These are not draws of one model.")
    for stop, value in sorted(got.items()):
        print(f"  bb{stop // 1000:>3}k  {value:.4f}")

    walk = steps(got)
    if walk:
        print("")
        print("change from one stop to the next")
        for lo, hi, delta in walk:
            print(f"  bb{lo // 1000}k -> bb{hi // 1000}k   {delta:+.4f}")
    else:
        print("fewer than two points on disk, so there is no track yet")

    print("")
    print("why these points are models and not draws")
    print(f"  {PROMOTION_SITE}    {PROMOTION_LINE}")
    print("  prepare_backbone_state_dict starts from the full STUDENT state "
          "dict.")
    print("  The teacher branch overwrites two prefixes only, so every other "
          "tensor")
    print("  stays the student's and keeps training after step "
          f"{FROZEN_FROM:,}.")

    print("")
    print("measured, from teacher_head_inputs_*.json")
    for m in moves:
        print(f"  {m['file']}: loaded {m['loaded_tensors']}, teacher "
              f"{m['from_teacher']}, student {m['from_student']}. Moved: "
              f"teacher {m['moved_from_teacher']}, student "
              f"{m['moved_from_student']}")
        for tag, r in (m["forward"] or {}).items():
            print(f"     {tag:<20} identical={r['identical']}  "
                  f"rel L2 {r['rel_l2']:.3e}")

    print("")
    if not moves:
        print("VERDICT: run teacher_head_inputs.py first. Without it, this "
              "track carries no label.")
    elif constant:
        print("VERDICT: every tensor the teacher head reads is equal across "
              "these stops. Only in this case would the points be draws of "
              "one model. Re-read the JSON before you trust this line.")
    elif one_encoder(moves):
        print(f"VERDICT: NOT a null, and NOT pooled. The teacher ENCODER "
              f"STACK is common to these {len(got)} points, but the head "
              f"also reads student-owned tensors that keep training, so the "
              f"head inputs differ. Each delta above is a change between two "
              f"models. The card's noise band comes from head_band.csv "
              f"instead: three head seeds on one backbone, student range "
              f"0.0018, teacher range 0.0064.")
    else:
        print("VERDICT: the teacher tensors are not equal across these "
              "stops, so the points do not even share one encoder.")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stop", "head", "seed", "score", "is_distinct_model"])
            for stop, value in sorted(got.items()):
                w.writerow([stop, HEAD, head_band.PROTOCOL_SEED, value, True])
            w.writerow([])
            w.writerow(["from_stop", "to_stop", "delta", "kind"])
            for lo, hi, delta in walk:
                w.writerow([lo, hi, f"{delta:+.4f}", "model_to_model_change"])
            w.writerow([])
            w.writerow(["n_models", "one_encoder", "head_input_constant",
                        "pooled", "noise_band_source"])
            w.writerow([len(got), one_encoder(moves), constant, False,
                        "head_band.csv"])
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
