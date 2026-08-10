#!/usr/bin/env python3
"""#373 — locate each backbone run's losses CSV, trainer log and checkpoints.

The study's backbones were trained in two places, so their artefacts sit in
two trees:

  ~/cf373_sync/<box>/sync   the five runs that ran on rented vast.ai boxes,
                            pulled down by the sync loops.
  $CF373_ROOT/...           the seven runs the review asked for, which ran on
                            elisa and wrote straight to the durable root.

A figure should not care which. This walks both and prints the `--run`
arguments the plot scripts take, so the rebuild script holds no paths.

Usage:
  find_artefacts.py --what curves      # --run <arm>:<k>=<losses.csv>
  find_artefacts.py --what logs        # --log <arm>:<k>=<run.log>
  find_artefacts.py --what ckpt        # --run <arm>:<k>=<bb40k .pth>
  find_artefacts.py --what ckptdir     # --run <arm>:<k>:<name>=<dir>
  find_artefacts.py --what pairs       # <label>\t<baseline tag>\t<compared tag>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                       # noqa: E402

ROOTS = [Path(os.environ.get("CF373_SYNC_BASE",
                             Path.home() / "cf373_sync")),
         Path(os.environ.get("CF373_ROOT",
                             "/home/jupyter/checkpoints_backup/cf-373")),
         # The runs that trained on elisa wrote their trainer log straight
         # into the study's results directory, beside the score files.
         HERE.parent / "results"]
# The durable root also holds one directory per HEAD, each with its own
# `qhead_*_losses.csv`. A head is not a backbone; skip that whole subtree.
SKIP = {"eval", "verify", "gap_claims", "mem", "steptime"}


def walk(roots):
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP]
            yield Path(dirpath), filenames


def find(name_to_runs, suffix):
    """`{run name: path}` for the first file matching `<run name><suffix>`."""
    want = {f"{n}{suffix}": n for n in name_to_runs}
    out = {}
    for dirpath, filenames in walk(ROOTS):
        for f in filenames:
            n = want.get(f)
            if n is not None and n not in out:
                out[n] = dirpath / f
    return out


def emit_pairs(eval_dir):
    """The comparisons the paired bootstrap runs, one per line.

    Every depth pair comes from the registry, so a new depth cannot be left
    out by hand. The rows after them are not depth pairs and are named here
    on purpose: two hold the depth fixed and vary the backbone seed, one is
    the re-weighting control.
    """
    tags = sorted(p.name for p in eval_dir.iterdir()) if eval_dir.is_dir() else []
    have = set(tags)
    out = []
    for arm, head, k, base, deep in R.pairs(tags):
        out.append((f"{arm.replace(chr(183), '_')}_k{k}_{head}",
                    base.tag, deep.tag))
    for head in ("student", "teacher"):
        for k in (0, 3):
            out.append((f"B5_backboneseed_k{k}_{head}",
                        f"B5_k{k}_bb40k_{head}",
                        f"G5_B5_s2_k{k}_bb40k_{head}"))
        out.append((f"A3_alignx4_{head}", f"A3_k0_bb40k_{head}",
                    f"G3_A3_k0_aw4_bb40k_{head}"))
    for label, a, b in out:
        if a in have and b in have:
            print(f"{label}\t{a}\t{b}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--what", required=True,
                   choices=("curves", "logs", "ckpt", "ckptdir",
                            "pairs"))
    p.add_argument("--results", default=str(HERE.parent / "results"))
    p.add_argument("--min-rows", type=int, default=10)
    args = p.parse_args(argv)

    if args.what == "pairs":
        return emit_pairs(Path(args.results) / "eval")

    bb = R.backbones()
    names = {run: (arm, k, role) for arm, k, role, run in bb}
    # A control is not a point on a depth ladder, so it stays out of the
    # curve and checkpoint figures, which are all depth comparisons.
    depth = {n: v for n, v in names.items() if v[2] == "depth"}

    if args.what == "curves":
        got = find(depth, "_losses.csv")
        for run, (arm, k, _r) in depth.items():
            path = got.get(run)
            if path and sum(1 for _ in open(path)) > args.min_rows:
                print(f"--run\n{arm}:{k}={path}")
        return 0

    if args.what == "logs":
        # No card in the spec: `runs.py` carries the machine and the card of
        # every run, and a path that happens to sit under a box directory is
        # a weaker source than the registry.
        got = find(depth, ".log")
        for run, (arm, k, _r) in depth.items():
            path = got.get(f"run_{run}") or got.get(run)
            if path is None:
                for dirpath, filenames in walk(ROOTS):
                    if f"run_{run}.log" in filenames:
                        path = dirpath / f"run_{run}.log"
                        break
            if path is None:
                continue
            print(f"--log\n{arm}:{k}={path}")
        return 0

    # Both checkpoint modes want this run's own periodic checkpoints.
    # R.ckpt_step does the matching, because a prefix test would fold group
    # A's re-weighting control into the cell it controls for.
    for run, (arm, k, _r) in depth.items():
        hits = []
        for dirpath, filenames in walk(ROOTS):
            hits += [(R.ckpt_step(run, f), dirpath / f) for f in filenames
                     if R.ckpt_step(run, f) is not None]
        if not hits:
            continue
        if args.what == "ckpt":
            forty = [h for st, h in hits if st == 40000]
            if forty:
                # `<arm>:<k>`, the same shape every other mode emits, so a
                # consumer resolves the label through the registry instead
                # of splitting a string on `_k`.
                print(f"--run\n{arm}:{k}={forty[0]}")
        else:
            by_dir = {}
            for _st, h in hits:
                by_dir.setdefault(h.parent, []).append(h)
            for d, hs in by_dir.items():
                if len(hs) >= 2:
                    print(f"--run\n{arm}:{k}:{run}={d}")
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
