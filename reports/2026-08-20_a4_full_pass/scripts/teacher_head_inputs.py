#!/usr/bin/env python3
"""#407 round-3 gap 3 — which tensors does the TEACHER head read?

`teacher_move.py` shows that every `teacher_*` tensor is equal bit for bit
between step 100,000 and step 200,000. A null of 0.0046 between the two
teacher scores follows from that only under one assumption: the teacher head
reads teacher tensors ONLY. This script tests that assumption.

It does not reason about the assumption. It builds the exact state dict the
head trainer loads, for both checkpoints, and it compares them. Then it runs
the two backbones over one fixed batch and compares the latents the head
consumes.

`src.checkpoint.prepare_backbone_state_dict(sd, "teacher")` is the one
function in the path. It copies `teacher_input_to_latent.*` over `encoder.*`
and `transformer.input_to_latent.*`, and `teacher_encoder_layers.*` over
`transformer.encoder_layers.*`. It then drops every `teacher_*` and
`cpc_w1*` key. Every OTHER key stays the student's.

Reported:

  loaded        how many tensors the head's backbone loads.
  from teacher  how many of them a `teacher_*` key wrote.
  from student  the rest. These are the tensors the assumption denies.
  moved         how many of each group differ between the two checkpoints.
  latents       max |d| between the two backbones' encoder latents and
                forecaster latents, over one fixed batch. The head reads
                both under `--head-train-input e_then_f`.

Usage:
  teacher_head_inputs.py <ckpt a> <ckpt b> [--json OUT] [--no-forward]
  teacher_head_inputs.py --root <durable root> --pair 100000 200000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full_pass  # noqa: E402

REPO = full_pass.REPO_ROOT
sys.path.insert(0, REPO)

from src.checkpoint import (                              # noqa: E402
    _TEACHER_PROMOTIONS, prepare_backbone_state_dict,
)

# The card's backbone, as `head_eval_bb.sh`'s ARCH_HEAD list gives it.
ARCH = dict(C=1, H=64, W=16, nhead=8, num_layers=3, encoder_type="gru",
            rev_norm_kind="ewma", rev_norm_span=128)
T_RAW = 4096
BATCH = 8
FORWARD_SEED = 20260820


def raw_state(path):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


def teacher_written_keys(state):
    """Every loaded key that a `teacher_*` key writes under promotion."""
    out = set()
    for src, dests in _TEACHER_PROMOTIONS.items():
        for key in state:
            if key.startswith(src):
                tail = key[len(src):]
                out.update(dest + tail for dest in dests)
    return out


def split(state):
    """`(teacher-written keys, student-owned keys)` of the loaded set."""
    loaded = prepare_backbone_state_dict(state, "teacher")
    written = teacher_written_keys(state) & set(loaded)
    return loaded, written, set(loaded) - written


def moved_keys(a, b, keys):
    """The keys of `keys` whose tensors differ, with each max |d|."""
    out = []
    for k in sorted(keys):
        if k not in a or k not in b:
            continue
        if torch.equal(a[k], b[k]):
            continue
        d = float((b[k].to(torch.float64) - a[k].to(torch.float64))
                  .abs().max())
        out.append([k, d])
    return sorted(out, key=lambda kv: -kv[1])


def latent_gap(path_a, path_b):
    """max |d| of the two latent tensors the head reads, over one batch."""
    from src.checkpoint import load_backbone_from_checkpoint
    from src.forecasting_head import (extract_encoder_latents,
                                      extract_forecaster_latents)
    torch.manual_seed(FORWARD_SEED)
    x = torch.randn(BATCH, T_RAW, ARCH["C"], dtype=torch.float32)
    out = {}
    got = {}
    for name, path in (("a", path_a), ("b", path_b)):
        backbone, _ = load_backbone_from_checkpoint(
            str(path), torch.device("cpu"), encoder_source="teacher", **ARCH)
        e, _ = extract_encoder_latents(backbone, x)
        f, _ = extract_forecaster_latents(backbone, x)
        got[name] = (e, f)
    for i, tag in enumerate(("encoder_latents", "forecaster_latents")):
        p, q = got["a"][i], got["b"][i]
        out[tag] = {
            "identical": bool(torch.equal(p, q)),
            "max_abs_diff": float((q - p).abs().max()),
            "rel_l2": float((q - p).norm() / p.norm()),
        }
    return out


def report(path_a, path_b, forward=True):
    a_raw, b_raw = raw_state(path_a), raw_state(path_b)
    a_load, a_teach, a_stud = split(a_raw)
    b_load, _, _ = split(b_raw)

    moved_teacher = moved_keys(a_load, b_load, a_teach)
    moved_student = moved_keys(a_load, b_load, a_stud)
    out = {
        "a": str(path_a), "b": str(path_b),
        "loaded_tensors": len(a_load),
        "from_teacher": len(a_teach),
        "from_student": len(a_stud),
        "moved_from_teacher": len(moved_teacher),
        "moved_from_student": len(moved_student),
        "teacher_written_keys": sorted(a_teach),
        "student_owned_keys": sorted(a_stud),
        "largest_student_moves": moved_student[:8],
    }
    if forward:
        out["forward"] = latent_gap(path_a, path_b)
    out["verdict"] = verdict(out)
    return out


def verdict(out):
    """What the two counts mean. Three cases, and only one is a null."""
    tail = ""
    fwd = out.get("forward")
    if fwd:
        worst = max(v["max_abs_diff"] for v in fwd.values())
        same = all(v["identical"] for v in fwd.values())
        tail = (". The latents the head reads are identical"
                if same else
                f". The latents the head reads differ by up to {worst:.3e}")
    if out["moved_from_teacher"]:
        # Both ends of the pair sit inside the EMA ramp, so the teacher
        # encoder is still tracking the student. This is the control.
        return (f"TEACHER MOVED: {out['moved_from_teacher']} of "
                f"{out['from_teacher']} teacher-written tensors differ, and "
                f"{out['moved_from_student']} of {out['from_student']} "
                f"student-owned ones do{tail}")
    if out["moved_from_student"] == 0:
        return ("NULL: the teacher head reads the same tensors at both "
                "steps. Every loaded tensor is equal bit for bit")
    return (f"NOT A NULL: the teacher head does not read teacher tensors "
            f"only. The teacher encoder stack is frozen, but "
            f"{out['moved_from_student']} of {out['from_student']} "
            f"student-owned tensors move between these two steps{tail}")


def show(out):
    print(f"a  {out['a']}")
    print(f"b  {out['b']}")
    print(f"loaded {out['loaded_tensors']}   from teacher "
          f"{out['from_teacher']}   from student {out['from_student']}")
    print(f"moved: from teacher {out['moved_from_teacher']}, "
          f"from student {out['moved_from_student']}")
    for k, d in out["largest_student_moves"]:
        print(f"   student-owned move  {k}  max |d| {d:.3e}")
    for tag, r in (out.get("forward") or {}).items():
        print(f"{tag:<20} identical={r['identical']}  "
              f"max |d| {r['max_abs_diff']:.3e}  rel L2 {r['rel_l2']:.3e}")
    print(out["verdict"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--pair", type=int, nargs=2, metavar=("A", "B"))
    ap.add_argument("--json")
    ap.add_argument("--no-forward", action="store_true",
                    help="skip the two forward passes")
    a = ap.parse_args(argv)

    if a.pair:
        if not a.root:
            print("ABORT: --pair needs --root", file=sys.stderr)
            return 2
        paths = []
        for stop in a.pair:
            found = full_pass.ckpt_path(a.root, stop)
            if not found or not os.path.isfile(found):
                print(f"ABORT: no checkpoint at step {stop} under {a.root}",
                      file=sys.stderr)
                return 3
            paths.append(found)
    elif len(a.paths) == 2:
        paths = a.paths
    else:
        print("ABORT: give two checkpoints, or --root with --pair",
              file=sys.stderr)
        return 2

    out = report(paths[0], paths[1], forward=not a.no_forward)
    show(out)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
