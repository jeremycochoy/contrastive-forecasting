#!/usr/bin/env python3
"""
Stitch the per-step losses CSVs for the square-loss runs into a single
trajectory each, covering 0 → 100k.

Sources:
  τ=0.10:
    - loss_ext_square_tau_0_10_losses.csv      (initial 15k run, steps 1–15100)
    - loss_ext_square_tau_0_10_100k_losses.csv (resume 15k → 100k, steps 15001–100000)
  τ=0.20:
    - loss_ext_square_tau_0_20_losses.csv      (initial 15k run, steps 1–15000)
    - loss_ext_square_tau_0_20_50k_losses.csv  (aborted 50k extension, steps 15001–29100)
    - loss_ext_square_tau_0_20_100k_losses.csv (resume 25k → 100k, steps 25001–100000)

Where overlap exists (e.g. τ=0.20 has both the aborted 50k attempt and the
resumed 100k attempt covering steps 25001–29100), we dedupe by step and
keep the LATER source (the 100k resume) — that is the run that actually
generated the model state we kept.

Output (gitignored data dir):
  loss_ext_square_tau_0_10_100k_full_losses.csv
  loss_ext_square_tau_0_20_100k_full_losses.csv
"""
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT, "experiments", "2026-05-09_exp_loss_extensions", "data")
# Sync directories live in the main checkout, never in an auxiliary worktree
# (CLAUDE.md). When this script runs from a worktree the sync_loss_ext_square
# sibling won't exist locally, so fall back to the canonical main-checkout path
# unless overridden via env.
SYNC_DIR = os.environ.get(
    "LOSS_EXT_SYNC_DIR",
    os.path.join(ROOT, "sync_loss_ext_square", "checkpoints"),
)
if not os.path.isdir(SYNC_DIR):
    SYNC_DIR = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_loss_ext_square/checkpoints"


def stitch(sources: list[str], out: str) -> None:
    """Concat in priority order (later wins on duplicate step)."""
    parts = []
    for src in sources:
        if not os.path.exists(src):
            print(f"  ! missing: {src}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(src)
        print(f"  + {os.path.basename(src):<55} rows={len(df):>6} step={df.step.min()}–{df.step.max()}")
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    # keep="last" so a step that appears in a later source overwrites the earlier one
    full = full.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
    full.to_csv(out, index=False)
    print(f"  → {os.path.basename(out):<55} rows={len(full):>6} step={full.step.min()}–{full.step.max()}")


def main() -> None:
    print("τ=0.10 stitch:")
    stitch(
        [
            os.path.join(DATA_DIR, "loss_ext_square_tau_0_10_losses.csv"),
            os.path.join(SYNC_DIR, "loss_ext_square_tau_0_10_100k_losses.csv"),
        ],
        os.path.join(DATA_DIR, "loss_ext_square_tau_0_10_100k_full_losses.csv"),
    )
    print("\nτ=0.20 stitch:")
    stitch(
        [
            os.path.join(DATA_DIR, "loss_ext_square_tau_0_20_losses.csv"),
            os.path.join(SYNC_DIR, "loss_ext_square_tau_0_20_50k_losses.csv"),
            os.path.join(SYNC_DIR, "loss_ext_square_tau_0_20_100k_losses.csv"),
        ],
        os.path.join(DATA_DIR, "loss_ext_square_tau_0_20_100k_full_losses.csv"),
    )


if __name__ == "__main__":
    main()
