# experiments/_aggregate

Cross-cutting artefacts for the freq-embedding experiment sequence
(EXP1 + EXP4 + 5 follow-ups). The individual experiments live in
sibling `experiments/exp_*` directories.

## What's here

- `REPORT.md` — umbrella report covering all 4-6 ablations
  (#23 freq embedding, #24 quantile head, #26 long head, #28 RevIN,
  plus mix=0.5 patch-stats and the 30k baseline arms). Aggregates
  skill scores across arms.
- `results/synth_eval.csv` — held-out 1024-sample synth-eval rows for
  every backbone+qhead arm in the sequence (one row per arm). Spans
  `exp_synth_only_redo`, `exp_span_sweep_synth`, `exp_revin_synth`.
- `results/comparison_with_sn.csv` — per-config MASE+WQL+SN-relative
  for each arm on the 43-config univariate slice with a local SN
  baseline.
- `results/skill_scores.txt` — plain-text aggregate skill-score table.
- `plots/synth_qhead_grid.png` — 12-panel diagnostic on the head's
  own training distribution. The single plot the umbrella REPORT uses
  to argue the periodic-tracking issue is upstream of the head.
- `plots/synth_compare_grid.png` — 6-arm × 12-sample comparison grid
  (visual cross-arm comparison).
- `plots/predictions/` — 6 periodic-focus configs, multi-model
  prediction plots (`plot_multi_model.py` output).
- `plots/predictions_qhead/` — same 6 configs with the focused 4-curve
  qhead plot (truth + SN + fe+mu MSE + qhead median + uncertainty).

## Why split this from the per-experiment dirs

- `synth_eval.csv` rows for spans 32 / 64 / 128 / 256 / 512 / 1024 +
  RevIN-synth + the synth-only redo arms span 3 separate experiments
  (`exp_synth_only_redo`, `exp_span_sweep_synth`, `exp_revin_synth`).
  Splitting per-experiment would lose the cross-arm sort that makes
  the table useful.
- The umbrella REPORT.md and the 4-6 panels under `predictions*/` argue
  about all arms together; living next to the aggregate CSV keeps the
  argument self-consistent.
