# 2026-04-27_exp_patch_stats_mix05

**What**: First patch-stats run (`compute_patch_stats(kind='diff')` —
per-patch dmean, dlogstd concatenated to encoder input) on the same
mix=0.5 + GIFT-Eval setup as the previous-session arms, so it would be
directly comparable.

**When**: Late April 2026.

**Status**: Completed but **superseded** by `2026-04-27_exp_synth_only_redo`.
Backbone gap improved +33% over the fe+mu/RevIN baselines, but
downstream GIFT-Eval was 1-3% worse than fe+mu+qh / RevIN+qh on the
23-config available SN slice. The 5h GIFT-Eval per run also ate
iteration time, motivating the synth-only redo.

**Run script**: not preserved as a `run.sh` — launched inline. Setup
parameters in `REPORT.md`.

**Code referenced**:
- `src/norm.py::compute_patch_stats(...)`
- `src/models.py::ConfigurableModel(patch_stats_kind=...)` and
  `prepare_encoder_input`
- `../freq-embedding/scripts/train.py` with `--patch-stats {none,diff,raw}`
- `../gift-eval/scripts/{train_forecasting_head,eval_gift_eval_official}.py`
  with `--patch-stats auto`

**Bug caught and fixed during this run**: `train.py::forward_step`
silently dropped the patch-stats concat by reimplementing patching
manually. Routed through `model.prepare_encoder_input` and added
regression test in `tests/test_norm.py`.
