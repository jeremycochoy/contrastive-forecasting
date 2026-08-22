#!/usr/bin/env python3
"""#407 review gap 4 — does A4's teacher still move after step 100k?

A4's EMA schedule is `--ema-tau 0.9 --ema-tau-end 1.0
--ema-tau-ramp-steps 100000`. `models.ema_tau_at_step` clamps the ramp
fraction at 1, so from step 100,000 on the momentum is exactly 1.0 and
`_ema_update` runs `t.mul_(1.0).add_(s, alpha=0.0)`, which changes no
parameter. The card's three stops all sit past that point.

The arithmetic says the teacher is frozen. This checks the tensors, because
the arithmetic does not cover a buffer: `_ema_update` hard-copies buffers at
every step, whatever the momentum is.

Reported per checkpoint pair, over the `teacher_*` tensors and again over
the student ones as a control:

  identical   how many tensors are equal bit for bit.
  max |d|     the largest absolute difference over all elements.
  rel L2      ||b - a|| / ||a||, over the whole tensor set.

The student is the control. It must move a great deal over the same steps.
If both sides read zero, the two files are the same checkpoint and the
answer is about the files, not about the teacher.

Usage:
  teacher_move.py <checkpoint a> <checkpoint b> [--json OUT]
  teacher_move.py --root <durable root> --pair 200000 300000
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

TEACHER_PREFIX = "teacher_"


def load(path):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and \
            isinstance(state["model"], dict):
        state = state["model"]
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def side(state, teacher: bool):
    """The teacher half of a state dict, or everything else."""
    return {k: v for k, v in state.items()
            if k.startswith(TEACHER_PREFIX) is teacher}


def compare(a: dict, b: dict) -> dict:
    """How far one tensor set moved to another."""
    shared = sorted(set(a) & set(b))
    identical = 0
    max_abs = 0.0
    num = 0.0
    den = 0.0
    moved = []
    for k in shared:
        x = a[k].to(torch.float64)
        y = b[k].to(torch.float64)
        if x.shape != y.shape:
            moved.append((k, float("nan")))
            continue
        d = (y - x)
        if bool(torch.equal(a[k], b[k])):
            identical += 1
        else:
            moved.append((k, float(d.abs().max())))
        max_abs = max(max_abs, float(d.abs().max()))
        num += float((d * d).sum())
        den += float((x * x).sum())
    return {
        "tensors": len(shared),
        "identical": identical,
        "moved": len(shared) - identical,
        "max_abs_diff": max_abs,
        "rel_l2": (num ** 0.5) / (den ** 0.5) if den > 0 else float("nan"),
        "only_in_a": sorted(set(a) - set(b)),
        "only_in_b": sorted(set(b) - set(a)),
        "largest_moves": sorted(moved, key=lambda kv: -kv[1])[:5],
    }


def report(path_a, path_b) -> dict:
    a = load(path_a)
    b = load(path_b)
    out = {"a": str(path_a), "b": str(path_b)}
    for name, is_teacher in (("teacher", True), ("student", False)):
        out[name] = compare(side(a, is_teacher), side(b, is_teacher))
    out["verdict"] = verdict(out)
    return out


def verdict(out: dict) -> str:
    t, s = out["teacher"], out["student"]
    if t["tensors"] == 0:
        return "no teacher tensors in these checkpoints"
    if s["moved"] == 0:
        return ("the STUDENT did not move either, so these two files hold "
                "the same weights and this says nothing about the teacher")
    if t["moved"] == 0:
        return ("the teacher is FROZEN: every teacher tensor is equal bit "
                "for bit, while the student moved")
    return (f"the teacher MOVED: {t['moved']} of {t['tensors']} tensors "
            f"differ, max |d| {t['max_abs_diff']:.3e}, rel L2 "
            f"{t['rel_l2']:.3e}")


def show(out: dict) -> None:
    print(f"a  {out['a']}")
    print(f"b  {out['b']}")
    print(f"{'side':<9} {'tensors':>7} {'identical':>9} {'moved':>6} "
          f"{'max |d|':>11} {'rel L2':>11}")
    for name in ("teacher", "student"):
        r = out[name]
        print(f"{name:<9} {r['tensors']:>7} {r['identical']:>9} "
              f"{r['moved']:>6} {r['max_abs_diff']:>11.3e} "
              f"{r['rel_l2']:>11.3e}")
        for k, d in r["largest_moves"]:
            print(f"            {k}  max |d| {d:.3e}")
    print(out["verdict"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="two checkpoint paths")
    ap.add_argument("--root", help="the durable root that holds the legs")
    ap.add_argument("--pair", type=int, nargs=2, metavar=("A", "B"),
                    help="two stops, resolved under --root")
    ap.add_argument("--json", help="write the whole comparison here")
    a = ap.parse_args(argv)

    if a.pair:
        if not a.root:
            print("ABORT: --pair needs --root", file=sys.stderr)
            return 2
        paths = []
        for stop in a.pair:
            if stop == full_pass.RESUME_STEP:
                found = full_pass.resume_source(a.root)
                found = found if found and \
                    full_pass.ckpt_step(found) == stop else None
                if found is None:
                    found = os.path.join(
                        full_pass.leg_dir(a.root, stop),
                        f"{full_pass.RESUME_NAME}.pth")
            else:
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

    out = report(*paths)
    show(out)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
