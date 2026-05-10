#!/usr/bin/env python3
"""
Stitch the per-step losses CSVs for the τ=0.10 / τ=0.20 baseline runs into
single trajectories covering 0 → 150k.

Sources:
  τ=0.10:
    - tau_sweep_0_10_50k_losses.csv  (initial + first resume, steps 1–50000)
    - tau_sweep_0_10_150k_losses.csv (50k → 150k continuation on vast.ai,
                                      steps 48001–150000)
  τ=0.20: analogous (steps 49801–150000 for the second segment).

Where overlap exists (47/49k–50k), dedupe by step keeping the LATER source.

Output (gitignored data dir):
  tau_sweep_0_10_150k_full_losses.csv
  tau_sweep_0_20_150k_full_losses.csv
"""
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT, "experiments", "2026-05-09_exp_loss_extensions", "data")
# The 150k continuation runs sync into separate sync_tau_sweep_0_*0_150k dirs
# in the main checkout (CLAUDE.md: sync dirs live in main, not worktrees).
def sync_dir_for(tau_str: str) -> str:
    candidate = os.path.join(ROOT, f"sync_tau_sweep_0_{tau_str}_150k", "checkpoints")
    if os.path.isdir(candidate):
        return candidate
    return f"/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_tau_sweep_0_{tau_str}_150k/checkpoints"


def stitch(sources: list[str], out: str) -> None:
    parts = []
    for src in sources:
        if not os.path.exists(src):
            print(f"  ! missing: {src}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(src)
        print(f"  + {os.path.basename(src):<55} rows={len(df):>6} step={df.step.min()}–{df.step.max()}")
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
    full.to_csv(out, index=False)
    print(f"  → {os.path.basename(out):<55} rows={len(full):>6} step={full.step.min()}–{full.step.max()}")


def main() -> None:
    for tau in ("10", "20"):
        print(f"τ=0.{tau} stitch:")
        stitch(
            [
                os.path.join(DATA_DIR, f"tau_sweep_0_{tau}_50k_losses.csv"),
                os.path.join(sync_dir_for(tau), f"tau_sweep_0_{tau}_150k_losses.csv"),
            ],
            os.path.join(DATA_DIR, f"tau_sweep_0_{tau}_150k_full_losses.csv"),
        )
        print()


if __name__ == "__main__":
    main()
