# Execution log — #328 (ops / journey; NOT the report)

Operational notes kept out of RESULTS.md per the report standard.

## Environment
- On elisa. Code worktree `WT=/tmp/cf-328` (branch `experiment/2026-06-03-crossfade-triplet`,
  off `origin/experiments` 13661f4). Outputs `OUT=~/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet`.
- Two RTX 4090 (24 GB). Shared with other users/agents — must not preempt.

## Key decisions
- **Triplet count = 1 per step** (3 rows: A_norm, B_norm, C), ADDITIVE on top of the
  natural allt·0.8% batch → total batch 1027. Reading of "only these 3 samples for the
  scheme, on top of the natural samples". Configurable via `--crossfade-triplets`.
- **No bottleneck** = drop BOTH `--forecaster-d-model 128` AND `--forecaster-n-heads 4`
  → forecaster inherits encoder width 384 / 6 heads (the literal "forecaster = encoder").
- **3-layer encoder** = `--num-encoder-layers 3` (forecaster depth `--num-layers 6` kept).
- **Primary baseline = allt·0.8% (#322)**, tag `xshh_allt_forked2_qk_aon_b1024`, the stated
  base; paired bootstrap over the 97 shared GIFT-Eval configs. 3 changes at once → the Δ is a
  JOINT effect, not attributable to any single change (report must say so).
- Downstream/eval scripts drop the bottleneck flags; head/eval auto-detect num_encoder_layers
  (=3) and full-width forecaster from the checkpoint.

## Reference numbers (full-97 GM-Relative MASE, from on-disk summaries)
- allt·0.8% base (#322): 2L 1.2127, 6L 1.1982.
- allt·10% (#322):       2L 1.2225, 6L 1.1912.
- crossfade·10% (#326):  2L 1.2082, 6L 1.1784  (prior best cell = 6L 1.178).
- Backbone params: base 12.7M; this arm 16.7M (full-width forecaster outweighs −3 enc layers).

## Status / timeline
- Code + 16 new tests green (29 crossfade total); CPU smoke at batch 16 validated triplet
  wiring (total_bs grew by 3). Full suite: 567 pass; 21 pre-existing test_forecasting_head
  failures (MockTransformer lacks fcst_down_proj) are unrelated fixture rot on the base branch.
- Launchers + analysis/plot scripts written & syntax-checked; schematic rendered; analyze.py
  dry-run reproduces baseline + reference GMs exactly.
- GPU WAIT: at first check both GPUs busy (GPU0 ~13.6 GB free but shared by 7 live procs;
  GPU1 100% util, 18.9 GB held). Waiting on the hourly loop for a genuinely free GPU
  (>=20 GB free) before launching the batch-1024 backbone (~17 h at 0.2 sps, per #326).

## Run order once a GPU frees
1. `smoke.sh <gpu>` (batch 1024, ~20 steps) — validates + measures peak memory.
2. `train_backbone_triplet.sh <gpu>` (background, ~17 h; resumes from latest *_Nk.pth on crash).
3. `run_downstream.sh <gpu> [gpu2]` — 2L + 6L q-heads (30k each) + GIFT-Eval B4 full+triage.
4. `analyze.py`, `perdomain_stats.py`, `plot_schematic.py`, `plot_training_metrics.py`.

## Final status (complete)
- Backbone: trained on GPU 1 (freed after a multi-hour wait for a genuinely-free card), 14.2 h,
  ~22 GB peak. PATCH_ENC_CKPT/CHUNK added to fit batch 1024 on one card (byte-identical; a smoke
  caught the ~19 GB GRU alloc before committing the run). Best-loss = step 6400 (loss 0.937).
- Downstream: 2L + 6L q-heads trained in parallel on GPU 1 (30k steps, b256) + GIFT-Eval B4
  (triage + full-97). Full evals slow (~3 h, large datasets); ran to completion, no retries needed.
- Result: NEUTRAL-to-slightly-worse. 2L 1.213→1.220 (+0.007), 6L 1.198→1.211 (+0.013); both 90%
  paired CIs straddle 0; arm trails the prior best (crossfade·10% 1.208/1.178). Per-domain:
  Web/CloudOps reliably worse on both heads (only both-head reliable effect). Single seed; 3 changes
  at once (joint effect).
