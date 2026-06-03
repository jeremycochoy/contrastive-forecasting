# Freq-embedding sequence — shared scripts and navigation

This directory began as a single folder holding the whole freq-embedding experiment
sequence (multiple arms + plots + results), and was split per experiment in late April 2026
so each is self-contained. The directory keeps the **shared script library** for the
sequence; the experiment itself is written up in [../freq-embedding.md](../freq-embedding.md).

## Shared scripts (`../scripts/`)

Every per-experiment `run.sh` in the sequence references these by absolute path
(`experiments/2026-04-27_freq-embedding/scripts/<x>.py`):

- `train.py` — backbone trainer (`--freq-emb-dim`, `--mixup-p`, `--rev-norm-kind`, `--rev-norm-span`, `--patch-stats`, `--loss-shape`).
- `synth_eval.py` — held-out 1024-sample synth eval; appends rows to a CSV.
- `synth_compare_grid.py` — 6-arm × 12-sample comparison grid.
- `plot_synth_qhead.py` — 12-panel synth grid for one model.
- `plot_qhead.py` — 4-curve plot (truth + SN + MSE + qhead median + band) for one config.
- `plot_multi_model.py` — multi-model prediction plot for one config.
- `skill_scores.py` — aggregate skill-score computation.
- `plot_gm_mase.py` — per-arm GM-MASE bar chart used in the report.

## The rest of the sequence

The other experiments split out of this folder (also indexed in [`../../INDEX.md`](../../INDEX.md)):

| Experiment | Location |
|---|---|
| RevIN reproduction (mix=0.5) | `../2026-04-27_exp_revin_repro/` |
| Patch-stats on mix=0.5 + GIFT-Eval | `../2026-04-27_exp_patch_stats_mix05/` |
| Synth-only redo (4 arms × {30k, 60k}) | `../2026-04-27_exp_synth_only_redo/` |
| Real-data span sweep (mix=0.0) | `../2026-04-27_exp_span_sweep_real/` |
| Synth-only span sweep (mix=1.0) | `../2026-04-27_exp_span_sweep_synth/` |
| RevIN on mix=1.0 synth-only | `../2026-04-27_exp_revin_synth/` |
| cosine_similarity_batch | `../2026-04-27_exp_csb_synth/` |
| Aggregate / umbrella report + cross-cutting plots | `../2026-04-27__aggregate/` |

## Why scripts stay here rather than copied per-experiment

`scripts/` is kept as one shared library, referenced from each experiment's `run.sh` by
absolute path, because every `run.sh` already invokes
`python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py …` from `/workspace/app`.
Copying the scripts into each `exp_*/scripts/` would fork the codebase — a fix in `train.py`
would have to be applied N times — for no benefit. Per-experiment reports record which scripts
they reference, so reproducing one in isolation only needs the matching script revision (`git
log --follow`).
