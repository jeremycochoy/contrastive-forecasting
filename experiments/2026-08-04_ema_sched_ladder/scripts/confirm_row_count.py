#!/usr/bin/env python3
"""Confirm `small_v1`'s row count and derive the ladder's step cap (#393).

A run may extend only while every sample it has seen was shown once, so
the ladder needs the number of steps in one pass over the dataset. The
2026-05-03 run reported 167,000 steps for one epoch, but at batch size
256; this experiment runs batch size 64. The row count is read from the
dataset's own manifest rather than carried over.

Writes `results/dataset_rows.json` and prints the same numbers.

Usage: python3 confirm_row_count.py [--out <path>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

HF_REPO = "jeremycochoy/gift-pretrain-full-4096"
HF_PATH = "small_v1"

# Why hf_rows_per_step is the full batch size even though --mix-ratio is
# non-zero. A reader who multiplies 64 by 1/128 gets a different cap, so
# the record says which of the two is true and why.
BASIS = (
    "step_cap = total_rows // hf_rows_per_step, where hf_rows_per_step is "
    "the number of REAL dataset rows one training step consumes. It is the "
    "full batch of 64 here, not 64*(1-mix_ratio): train.py computes "
    "synth_bs = int(round(batch_size * mix_ratio)), and 64 * 0.0078125 is "
    "exactly 0.5, which Python rounds half-to-even to 0. The nominal 1/128 "
    "synthetic fraction rounds away at this batch size, so no part of the "
    "batch is synthetic and hf_bs = 64. The 3 crossfade-triplet rows "
    "(--crossfade-triplets 1) are blended from those same 64 real rows and "
    "appended on top, so the model sees 67 rows per step while the dataset "
    "gives up 64. A larger mix_ratio, or a batch size whose product with it "
    "does not round to 0, would put synth_bs > 0, cut hf_bs, and RAISE the "
    "cap; re-run this script if either changes."
)

# The issue's cross-check. Same dataset, same all-real batches, batch 256.
PRIOR_STEPS = 167_000
PRIOR_BATCH_SIZE = 256
PRIOR_NOTE = (
    "experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL called "
    "167,000 steps at batch size 256 one full epoch of small_v1, mix-ratio "
    "0.0. The same row count gives 42,571,692 // 256 = 166,295, so that "
    "figure was the rounded-up form of this one and the two agree to 0.4%. "
    "It does not transfer to this experiment directly: batch 64 here means "
    "4x the steps per epoch."
)


def load_ladder():
    spec = importlib.util.spec_from_file_location(
        "ladder_393", os.path.join(SCRIPTS_DIR, "ladder.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(repo_id: str, path_in_repo: str) -> dict:
    """The dataset's own row count, straight from its manifest."""
    token_path = os.path.join(REPO_ROOT, "experiments", "hf_token.txt")
    if os.path.exists(token_path):
        with open(token_path) as fh:
            token = fh.read().strip()
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id, f"{path_in_repo}/manifest.json",
                            repo_type="dataset")
    with open(local) as fh:
        return json.load(fh)


def cross_check(rows: int, ladder) -> dict:
    """The issue's cross-check against the 2026-05-03 epoch figure."""
    derived = ladder.step_cap(rows, PRIOR_BATCH_SIZE)
    gap = (PRIOR_STEPS - derived) / PRIOR_STEPS
    return {
        "reported_steps": PRIOR_STEPS,
        "batch_size": PRIOR_BATCH_SIZE,
        "derived_steps": derived,
        "relative_gap": round(gap, 6),
        "agrees": abs(gap) < 0.01,
        "note": PRIOR_NOTE,
    }


def build_record(manifest: dict, ladder) -> dict:
    """Everything the report quotes, with the basis it rests on."""
    rows = int(manifest["total_rows"])
    comp = ladder.experiment_batch_composition()
    return {
        "dataset": f"{HF_REPO} / {HF_PATH}",
        "total_rows": rows,
        "num_shards": manifest.get("num_shards"),
        "batch_size": ladder.BATCH_SIZE,
        "mix_ratio": ladder.MIX_RATIO,
        "crossfade_ratio": ladder.CROSSFADE_RATIO,
        "crossfade_triplets": ladder.CROSS_TRIPLETS,
        "n_channels": ladder.N_CHANNELS,
        "batch_composition": comp,
        "hf_rows_per_step": comp["hf_rows_per_step"],
        "step_cap": ladder.step_cap(rows, comp["hf_rows_per_step"]),
        "basis": BASIS,
        "cross_check_2026_05_03": cross_check(rows, ladder),
    }


def write_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--out", default=os.path.join(EXP_DIR, "results",
                                                 "dataset_rows.json"))
    args = p.parse_args()

    ladder = load_ladder()
    record = build_record(read_manifest(HF_REPO, HF_PATH), ladder)
    write_record(args.out, record)
    print(json.dumps(record, indent=2))

    if record["total_rows"] != ladder.SMALL_V1_ROWS:
        print(f"\nWARNING: manifest says {record['total_rows']} rows, "
              f"ladder.py carries {ladder.SMALL_V1_ROWS}. Update the "
              f"constant and the report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
