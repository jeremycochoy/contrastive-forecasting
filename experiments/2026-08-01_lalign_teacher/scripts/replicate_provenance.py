#!/usr/bin/env python3
"""Which backbone replicate did #379's published 40k row actually use?

`snapshot_reproduction.py` finds nine of the ten arms reproducing #379's
student number to 1e-4 and arm5 missing it by 0.0977, with zero of 97 configs
identical. Two readings fit that:

  provenance   #379's published arm5 row was evaluated on a different
               backbone replicate than the one this branch re-ran.
  irreproducible
               arm5's training path does not repeat, so the cell the headline
               names carries a same-flag run-to-run spread the size of its
               own effect.

Both are testable from artefacts already on disk, at no GPU cost.

#379's launcher resumes a crashed arm, and `train.py`'s `safe_run_name`
(docs/restart_protocol.md) gives the resumed process a fresh `_r2`, `_r3`, …
run name. So one arm can leave several `_40k.pth` snapshots, each written by a
process that saw a different slice of the HF stream. Both evals then pick

    BB=$(ls -t "$RUNS/${NAME}"_${K}k.pth "$RUNS/${NAME}"_r*_${K}k.pth | head -1)

(#379 `scripts/eval_2L_gm_mase.sh:37`, this branch `scripts/eval_arm.sh:72`)
— newest by mtime, base or resumed, whichever landed last. Two facts settle
which reading holds:

  1. The backbone filename #379's eval consumed. Its `eval.log` opens with
     `start on GPU N (backbone=<file>)`, so the replicate is recorded, not
     inferred.
  2. Whether this branch's re-run repeats a #379 replicate step for step.
     The loss CSVs carry every step. A branch run that matches one replicate
     exactly is a reproduction; a branch run that matches none is not.

For each arm this reports the replicate the published row used, the step span
of every replicate, and the per-step agreement between the branch re-run and
each replicate over the steps they share below 40 000.

Usage:
    python3 replicate_provenance.py \
        --runs-379 experiments/2026-07-21_split_pred_rep_small/runs \
        --eval-379 experiments/2026-07-21_split_pred_rep_small/eval_gm_mase \
        --runs-390 experiments/2026-08-01_lalign_teacher/runs \
        --snapshot <out>/snapshot_reproduction_40k.csv \
        --out <out>/replicate_provenance_40k.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

# The ten arms, and the family token in their run names. Same rule as
# `arm_names.sh`: `lrepmoco` for the arm6_v2 cells, `lrep` for the arm5 cells.
ARMS = ["arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
        "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
        "arm6_v2_combab"]

BB_STEPS = 40000
# The columns compared step by step. `loss` is the training objective, `ff`
# and `gap` are the two diagnostics the report plots, `hf_rows_consumed` is
# the data counter — it advances with the step regardless of which rows the
# stream served, so it agreeing while `loss` does not is itself the signature
# of a restart that re-entered the stream elsewhere.
CMP_COLS = ("loss", "ff", "gap", "hf_rows_consumed")

BACKBONE_RE = re.compile(r"backbone=([^)\s]+)")
REPLICATE_RE = re.compile(r"_(r\d+)_\d+k\.pth$")


def base_name(arm: str, suffix: str = "") -> str:
    family = "lrepmoco" if arm.startswith("arm6_v2") else "lrep"
    tail = f"_{suffix}" if suffix else ""
    return (f"bb_small_{arm}_lalign_{family}"
            f"_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090{tail}")


def read_losses(path: str) -> dict[int, dict[str, str]]:
    with open(path, newline="") as fh:
        return {int(r["step"]): r for r in csv.DictReader(fh)}


def replicate_tag(filename: str) -> str:
    """`..._r3_40k.pth` -> `r3`; the un-suffixed base name -> `r1`."""
    m = REPLICATE_RE.search(filename)
    return m.group(1) if m else "r1"


def published_backbone(eval_dir: str, arm: str) -> str | None:
    log = os.path.join(eval_dir, f"{arm}_bb40k_hd15000s", "eval.log")
    if not os.path.isfile(log):
        return None
    with open(log, errors="replace") as fh:
        for line in fh:
            m = BACKBONE_RE.search(line)
            if m:
                return os.path.basename(m.group(1))
    return None


def replicates_379(runs: str, arm: str) -> dict[str, str]:
    """replicate tag -> its losses CSV, for every replicate #379 left."""
    name = base_name(arm)
    out = {}
    if os.path.isfile(p := os.path.join(runs, f"{name}_losses.csv")):
        out["r1"] = p
    for p in sorted(glob.glob(os.path.join(runs, f"{name}_r*_losses.csv"))):
        m = re.search(r"_(r\d+)_losses\.csv$", p)
        if m:
            out[m.group(1)] = p
    return out


def agreement(branch: dict, other: dict) -> tuple[int, int, float]:
    """(identical steps, shared steps, max |Δloss|) below the 40k snapshot."""
    shared = sorted(s for s in (set(branch) & set(other)) if s <= BB_STEPS)
    if not shared:
        return 0, 0, float("nan")
    ident, worst = 0, 0.0
    for s in shared:
        same = True
        for c in CMP_COLS:
            if c not in branch[s] or c not in other[s]:
                continue
            d = abs(float(branch[s][c]) - float(other[s][c]))
            if c == "loss":
                worst = max(worst, d)
            if d != 0.0:
                same = False
        ident += same
    return ident, len(shared), worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-379", required=True)
    ap.add_argument("--eval-379", required=True)
    ap.add_argument("--runs-390", required=True)
    ap.add_argument("--branch-suffix", default="alignstudent",
                    help="the same-branch student control's name token")
    ap.add_argument("--snapshot", default=None,
                    help="snapshot_reproduction_40k.csv, to carry its numbers")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = {}
    if args.snapshot:
        with open(args.snapshot, newline="") as fh:
            snap = {r["arm_slug"]: r for r in csv.DictReader(fh)}

    rows = []
    for arm in ARMS:
        bb = published_backbone(args.eval_379, arm)
        if bb is None:
            print(f"MISSING: {arm} — no #379 eval.log at bb40k")
            return 2
        used = replicate_tag(bb)

        reps = replicates_379(args.runs_379, arm)
        branch_csv = os.path.join(
            args.runs_390, f"{base_name(arm, args.branch_suffix)}_losses.csv")
        if not os.path.isfile(branch_csv):
            print(f"MISSING: {arm} — no branch re-run losses at {branch_csv}")
            return 2
        branch = read_losses(branch_csv)

        spans, matches, best = [], [], None
        for tag, path in sorted(reps.items()):
            other = read_losses(path)
            steps = sorted(other)
            reaches_40k = steps[0] <= BB_STEPS <= steps[-1]
            spans.append(f"{tag}:{steps[0]}-{steps[-1]}"
                         + ("*" if reaches_40k else ""))
            ident, shared, worst = agreement(branch, other)
            if shared:
                matches.append(f"{tag}:{ident}/{shared}")
                if ident == shared and (best is None or tag == used):
                    best = tag
        rows.append({
            "arm_slug": arm,
            "published_backbone_379": bb,
            "replicate_used_379": used,
            "replicates_present": " ".join(spans),
            "branch_matches_replicate": best or "none",
            "per_replicate_identical_steps": " ".join(matches),
            "gm_student_379": snap.get(arm, {}).get("gm_student_379", ""),
            "gm_student_390": snap.get(arm, {}).get("gm_student_390", ""),
            "snapshot_diff": snap.get(arm, {}).get("snapshot_diff", ""),
            "reproduces": snap.get(arm, {}).get("reproduces", ""),
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'arm':16s} {'used':>5s} {'branch repeats':>15s} "
          f"{'diff':>11s}  repro   replicate spans (* = wrote a 40k snapshot)")
    for r in rows:
        print(f"{r['arm_slug']:16s} {r['replicate_used_379']:>5s} "
              f"{r['branch_matches_replicate']:>15s} "
              f"{r['snapshot_diff']:>11s}  {r['reproduces']:<6s}  "
              f"{r['replicates_present']}")

    off = [r for r in rows if r["replicate_used_379"] != "r1"]
    exact = [r for r in rows if r["branch_matches_replicate"] != "none"]
    print(f"\n{len(exact)}/{len(rows)} arms: the branch re-run repeats a #379 "
          f"replicate step for step on every compared column.")
    print(f"{len(off)}/{len(rows)} arms: #379's published 40k row was "
          f"evaluated on a resumed replicate rather than the base run.")
    for r in off:
        print(f"  {r['arm_slug']} used {r['replicate_used_379']} "
              f"({r['published_backbone_379']}); the branch repeats "
              f"{r['branch_matches_replicate']}; snapshot diff "
              f"{r['snapshot_diff']}")

    bad = [r for r in rows
           if r["reproduces"] == "no" and r["replicate_used_379"] == "r1"]
    print("\nVERDICT: "
          + ("PROVENANCE — every arm whose published row used the base run "
             "reproduces; the one that does not is the one whose published "
             "row used a resumed replicate."
             if not bad and off else
             "NOT SETTLED — "
             + ", ".join(r["arm_slug"] for r in bad)
             + " misses its published number while using the base run, so "
               "the training path itself is in question."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
