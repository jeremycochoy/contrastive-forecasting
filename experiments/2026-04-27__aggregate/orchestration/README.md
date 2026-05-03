# Multi-experiment orchestration scripts

Historical record of session-2 (late April 2026) bash scripts that
ran multiple experiments in sequence on a single vast.ai instance.
Kept here as a reference template for "how to run a multi-experiment
pipeline end-to-end on one provisioned GPU".

For new single experiments, prefer the per-experiment `run.sh` under
`experiments/exp_*/`. These multi-experiment scripts are not
maintained going forward.

## What's here

- **`run_all_experiments.sh`** — the original session-2 driver. Ran
  EXP1 (RevIN repro on mix=0.5), EXP2 (synth-only baseline + pstats),
  EXP3 (real-data span sweep), EXP4 (patch-stats on mix=0.5 +
  GIFT-Eval) sequentially. Killed mid-EXP2 STAGE 1 when the user
  reordered priorities; superseded by `run_remaining_experiments.sh`.

- **`run_remaining_experiments.sh`** — the reordered tail used after
  the kill: EXP4 (patch-stats) first, then EXP3 (span sweep with
  cheap screens), then EXP2 (synth-only). Includes the
  `--save-every` reductions (heads 1000, backbones 2000) added
  mid-session for crash safety. Superseded by the per-experiment
  `run.sh` files now under `experiments/exp_*/`.

## Why kept

The orchestration logic — multi-stage `set -e` driver with stage
markers, copying best-loss FINAL pointers, sync-loop integration —
is reusable. Future multi-experiment pipelines can copy these as a
template.
