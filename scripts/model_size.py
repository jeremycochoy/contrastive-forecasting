#!/usr/bin/env python3
"""Print the parameter count of this project's backbone, at one shape or many.

Three modes.

    # one shape
    python3 scripts/model_size.py --d-model 384 --num-layers 3 \
        --num-encoder-layers 3

    # a table of widths, with a mark on the row nearest to a target
    python3 scripts/model_size.py --d-model-list 256,320,384,416 \
        --num-layers 3 --num-encoder-layers 3 --target 11400000

    # a checkpoint file, which counts the patch encoder two times
    python3 scripts/model_size.py --checkpoint model.pth

Add `--tsv` for a tab-separated table that a report can read.

The three numbers of one model are different. `src/model_size.py` says why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_size import (  # noqa: E402
    backbone_counts, nearest_to_target, state_dict_counts,
)

COLUMNS = ("H", "num_layers", "num_encoder_layers", "trainable", "frozen",
           "total", "state_dict")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=64,
                   help="The width of the two transformer stacks.")
    p.add_argument("--d-model-list", default=None,
                   help="Widths, separated by commas. Replaces --d-model.")
    p.add_argument("--num-layers", type=int, default=3,
                   help="Layers of the forecaster stack.")
    p.add_argument("--num-encoder-layers", type=int, default=3,
                   help="Layers of the encoder stack.")
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--target", type=int, default=None,
                   help="Mark the row whose trainable count is nearest to this.")
    p.add_argument("--checkpoint", default=None,
                   help="Count this checkpoint file instead of a built model.")
    p.add_argument("--tsv", action="store_true",
                   help="Print tab-separated columns, with no thousands mark.")
    return p.parse_args()


def widths(args) -> list[int]:
    """The widths to count, from either flag."""
    if args.d_model_list:
        return [int(v) for v in args.d_model_list.split(",") if v.strip()]
    return [args.d_model]


def print_checkpoint(path: str, tsv: bool) -> None:
    """Print the three numbers a checkpoint file gives."""
    import torch

    counts = state_dict_counts(torch.load(path, map_location="cpu",
                                          weights_only=False))
    order = ("total", "student", "teacher")
    if tsv:
        print("\t".join(order))
        print("\t".join(str(counts[k]) for k in order))
        return
    for key in order:
        print(f"{key:>10}  {counts[key]:>12,d}")
    print("The student number counts the patch encoder two times. "
          "See src/model_size.py.")


def print_table(rows, target: int | None, tsv: bool) -> None:
    """Print one row for each shape, and mark the row nearest to the target."""
    best = nearest_to_target(rows, target) if target is not None else None
    if tsv:
        print("\t".join(COLUMNS))
        for row in rows:
            print("\t".join(str(row[c]) for c in COLUMNS))
        return
    width = {c: max(len(c), 13) + 2 for c in COLUMNS}
    print("".join(f"{c:>{width[c]}}" for c in COLUMNS))
    for row in rows:
        line = "".join(f"{row[c]:>{width[c]},d}" for c in COLUMNS)
        print(f"{line} <-" if row is best else line)


def main() -> int:
    args = parse_args()
    if args.checkpoint:
        print_checkpoint(args.checkpoint, args.tsv)
        return 0
    rows = [backbone_counts(H=h, nhead=args.n_heads,
                            num_layers=args.num_layers,
                            num_encoder_layers=args.num_encoder_layers)
            for h in widths(args)]
    print_table(rows, args.target, args.tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
