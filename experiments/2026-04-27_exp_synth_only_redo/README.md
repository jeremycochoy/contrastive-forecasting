# 2026-04-27_exp_synth_only_redo

**What**: 4 backbones (fe+mu and fe+mu+pstats × {30k, 60k}) on
mix=1.0 synth-only, then a quantile head per backbone (also 30k synth-
only), then held-out synth eval (1024 samples, seed=99999999) via
`synth_eval.py`. Replaces the mix=0.5 + GIFT-Eval setup of
`2026-04-27_exp_patch_stats_mix05` to (a) isolate architectural change from OOD
transfer and (b) iterate fast.

**When**: Late April 2026.

**Status**: Success. fe+mu @ 60k was the marginal best (GM-MASE
2.366); patch-stats was 1-3% worse than baseline at both step counts.
Single seed. The real lever (EWMA span) was found in the follow-on
`2026-04-27_exp_span_sweep_synth`.

**Run script**: `run.sh` (formerly `run_synth_only.sh` at repo root).
Re-run on a fresh remote 4090 by syncing the repo and invoking
`bash run.sh` from the workspace root with `experiments/hf_token.txt`
populated.

**Code referenced**:
- `../freq-embedding/scripts/train.py`
- `../gift-eval/scripts/train_forecasting_head.py`
- `../freq-embedding/scripts/synth_eval.py`

**Bugs caught and fixed during this run** (also in REPORT):
- `_FINAL.pth = best_gap.pth` made 30k and 60k synth backbones
  byte-identical because gap saturates at step 1600 in deterministic
  synth-only training. Fixed by repointing to end-of-training
  `_30k.pth` / `_60k.pth` snapshots.
- `synth_eval.py` C=4 spp shape bug; switched to C=1 single-channel
  to match `plot_synth_qhead.py`.
- Disk full on remote (~60GB filled with periodic snapshot + optimizer
  files).

**Note on `plots/synth_compare_partial.png`**: this is a deprecated
5-row version of the multi-arm comparison plot. Kept for provenance;
the canonical multi-arm grid is in `../2026-04-27__aggregate/plots/synth_compare_grid.png`.
