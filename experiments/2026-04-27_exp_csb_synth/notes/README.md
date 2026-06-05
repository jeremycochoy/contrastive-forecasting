# 2026-04-27_exp_csb_synth — cosine_similarity_batch on the span=512 best arm

**What**: cosine_similarity_batch loss (paper-matching, includes
within-time and cross-time negatives) on the best arm from the span
sweep: fe+mu mix=1.0 synth-only, 30k bb + 30k qhead, ewma span=512.
Single-axis change vs the established `cosine_similarity_batch_no_time_neg`.

**When**: Late April 2026.

**Status**: Complete (single seed). Result: GM-MASE 0.886 / WQL 0.434 —
**~4.5% worse on MASE, ~5.1% worse on WQL** vs the same arm without
cross-time negatives (GM-MASE 0.848 / WQL 0.413). See `../exp_csb_synth.md`.

**Run script**: `run.sh` — copy of `/tmp/run_wtn_v2.sh` from the
remote vast.ai instance (snapshot at launch time). `run_v1.sh` is the
earlier variant with the deprecated `cosine_similarity_batch_with_within_time_neg`
loss-shape — superseded by v2.

**Code referenced**:
- `../2026-04-27_freq-embedding/scripts/train.py`
- `../2026-04-13_gift-eval/scripts/train_forecasting_head.py`
- `../2026-04-27_freq-embedding/scripts/synth_eval.py`

**Artefacts**:
- `plots/synth_qhead_grid_csb.png` — 12-panel synth grid for this arm.
- `results/synth_eval.csv` — local copy of the eval result for this arm.
- Eval row also appended to `../2026-04-27__aggregate/results/synth_eval.csv`.
- Backbone + qhead checkpoints (~80 MB + 2.5 MB) not tracked in git;
  pulled to `sync_csb/checkpoints/` locally and available on the remote.

**Run timeline (operational, not science)**: backbone training was
interrupted by remote-instance failures **three times** during this
experiment (the vast.ai instance kept stopping mid-run). Each time the
run resumed cleanly via `--resume` from the latest periodic snapshot.
The final 30k-step backbone is the result of:
`8k (fresh start) → resume → ~24k → resume → 30k`. The result is
therefore from a multi-resume run, not a clean single-shot 30k.

**Resume-vs-result caveats dropped from the report body** (single-seed
speculation, not load-bearing for the negative result):
- Whether resume RNG re-seeding + CUDA non-determinism nudged the final
  weights slightly off a hypothetical single-shot 30k run — hard to
  disentangle from the loss change. A clean single-shot rerun would
  settle "loss change hurt" vs "resume corrupted the run".
- A possible model-saturation reading (span=512 was already near SN on
  WQL, so single-knob gains get harder) — speculative, not tested here.
- Note the loss *shape* values aren't directly comparable across the two
  arms (different negatives); only the downstream synth-eval metrics are.
