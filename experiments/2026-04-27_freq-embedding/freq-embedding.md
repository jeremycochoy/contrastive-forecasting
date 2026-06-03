# experiments/2026-04-27_freq-embedding

This directory used to hold the entire freq-embedding experiment
sequence (6 reports + plots + results all crammed in one folder). It
was split per experiment in late April 2026 to make each experiment
self-contained and reviewable.

## What's here now

- `notes/DESIGN.md` — the freq-embedding design doc (forward-looking; covers
  the `FrequencyEmbedding` module, mixup augmentation, and the
  ablation arms).
- `notes/FOLLOWUP.md` — proposed but not-yet-run follow-ups (within-time
  contrastive negative — note: now in flight as `2026-04-27_exp_csb_synth`).
- `scripts/` — the **shared script library** for the sequence. Every
  per-experiment `run.sh` references these by absolute path (e.g.,
  `experiments/2026-04-27_freq-embedding/scripts/train.py`):
    - `train.py` — backbone trainer (`--freq-emb-dim`, `--mixup-p`,
      `--rev-norm-kind`, `--rev-norm-span`, `--patch-stats`,
      `--loss-shape`).
    - `synth_eval.py` — held-out 1024-sample synth eval; appends rows
      to a CSV.
    - `synth_compare_grid.py` — 6-arm × 12-sample comparison grid.
    - `plot_synth_qhead.py` — 12-panel synth grid for one model.
    - `plot_qhead.py` — 4-curve plot (truth + SN + MSE + qhead median +
      band) for one config.
    - `plot_multi_model.py` — multi-model prediction plot for one
      config.
    - `skill_scores.py` — aggregate skill-score computation.

## Where to find the actual experiments

| Experiment | Location |
|---|---|
| EXP1 RevIN reproduction (mix=0.5) | `../2026-04-27_exp_revin_repro/` |
| EXP4 Patch-stats on mix=0.5 + GIFT-Eval | `../2026-04-27_exp_patch_stats_mix05/` |
| Synth-only redo (4 arms × {30k, 60k}) | `../2026-04-27_exp_synth_only_redo/` |
| Real-data span sweep (mix=0.0) | `../2026-04-27_exp_span_sweep_real/` |
| Synth-only span sweep (mix=1.0) | `../2026-04-27_exp_span_sweep_synth/` |
| RevIN on mix=1.0 synth-only | `../2026-04-27_exp_revin_synth/` |
| **In-flight** cosine_similarity_batch | `../2026-04-27_exp_csb_synth/` |
| Aggregate / umbrella REPORT + cross-cutting plots | `../2026-04-27__aggregate/` |

## Why scripts live here, not duplicated per-experiment

Decision: leave `scripts/` here as a shared library, referenced from
each experiment's `run.sh` by absolute path. Reasoning:

- Every `run.sh` already invokes `python3 -u
  experiments/2026-04-27_freq-embedding/scripts/train.py …` from
  `/workspace/app`, so the path is already external to the experiment
  dir. Copying scripts into each `exp_*/scripts/` would mean updating
  every `run.sh` post-move and would fork the codebase: a bug fix in
  `train.py` would have to be applied N times.
- The user's instruction allowed "simply a copy if it eases some code
  from the previous experiments". Here, copying actively makes future
  edits harder rather than easier.
- Per-experiment READMEs document which scripts they reference, so
  reproducing one experiment in isolation only requires checking out
  the matching `experiments/2026-04-27_freq-embedding/scripts/` revision (which
  git makes trivial via `git log --follow` on the script).
