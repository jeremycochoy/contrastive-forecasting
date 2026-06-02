# Overnight plan — 150k extension to test if more training changes the dropkey-sweep ordering

Authored 2026-05-15 (user away until ~2026-05-16). Autonomous execution.
This doc is the authoritative spec; the controller and future context follow it.

## User intent (verbatim constraints)
- After the missing 50k numbers land (v20R full-eval + v27c full-eval), train **two
  backbones to 150k total steps, in parallel (one per GPU)**:
  - **v20** (dk0.9, the fp32-warmup→fp16 v11c-reproduction — the headline reference).
  - **the best of {v27c, v16, v17}** by full GM-MASE @50k.
- Then **re-evaluate with a proper (standard 2L causal-transformer) q-head at BOTH the
  150k and the 50k checkpoint of each extended backbone** → triage + full GM-MASE.
  Goal: see if more training changes the sweep ordering.
- If spare compute remains before the user returns: extend **2 other sweep dropkey
  params** to 150k the same way.
- **HARD CONSTRAINT (user, emphatic): a cold/reset optimizer is UNACCEPTABLE — it
  destroys the purpose of the experiment.** Every 150k run's optimizer must be a
  single continuous trajectory (never re-initialised mid-training).

## Optimizer-availability reality (disk-full incident purged optimizers)
| Arm | dk | 50k optimizer present? | Valid 150k path |
|-----|----|------------------------|-----------------|
| v27c | 0.80 | **YES** (`_50k_optimizer.pth`) | **warm-resume** 50k→150k (continuous) |
| v20  | 0.90 | NO (Phase-A & B opt purged) | **from-scratch**: Phase A fp32 0→5k (saves opt) → warm Phase B fp16 5k→150k (opt continuous A→B) |
| v16  | 0.70 | NO | from-scratch pure-fp32 0→150k (only if chosen) |
| v17  | 0.95 | NO | from-scratch pure-fp32 0→150k (only if chosen) |

A from-scratch run has a continuous optimizer from step 0 → valid. A warm-resume from
an intact optimizer is valid. A resume from weights WITHOUT the optimizer is the
forbidden case — never do it for a compared arm.

## Decision rule: "best of {v27c, v16, v17}"
best = min full GM-MASE @50k. Known: v16=1.335, v17=1.409. v27c = pending (chain
running). v17 is dominated. Effective choice: **v27c if its full GM < 1.335, else v16**.

## 50k-vs-150k comparison must be on the SAME optimizer trajectory
- v20 from-scratch 150k → its OWN periodic `_50k.pth` is the 50k point (NOT the old
  optimizer-purged v20R 50k). Eval this run's _50k and _150k.
- v27c warm-continue → the existing v27c 50k IS on-trajectory (warm continuation), so
  reuse the v27c-50k q-head+full-eval already being produced by the running chain;
  only the new 150k point needs a fresh q-head+eval.
- v16/v17 from-scratch (if chosen) → like v20: eval that run's _50k and _150k.

## Recipes (exact — must match original arch or resume corrupts silently)
Common: `--batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
--save-every 5000 --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1
--num-encoder-layers 6 --encoder-dropkey-share-heads --encoder-dropkey-share-layers
--depthwise-conv 3 --deprecated-depthwise-conv 0 --mix-ratio 0.0 --freq-emb-dim 3
--seasonality-emb-dim 3 --mixup-p 0.3 --rev-norm-kind ewma --rev-norm-span 128
--tau 0.10 --loss-shape cosine_similarity_batch --encoder-type gru
--hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1`
- v20: `--encoder-dropkey 0.9`; Phase A all-fp32; Phase B `--patch-emb-dtype fp32
  --residual-dtype fp16 --attn-dtype fp16 --ffn-dtype fp16`, `--resume <PhaseA _5k.pth>`.
- v27c: `--encoder-dropkey 0.8 --patch-emb-dtype fp32 --residual-dtype fp32
  --attn-dtype fp32 --ffn-dtype fp16 --log-attn-amplitude --log-attn-amplitude-every 1000`,
  `--resume enc_fcst_v27c_dk08_ffnfp16_resume25k_50k_50k.pth` (warm; restores step=50000).
- v16: `--encoder-dropkey 0.7` all-fp32.  v17: `--encoder-dropkey 0.95` all-fp32.
- q-head (proper): the standard `run_qhead_*` recipe — 2L causal transformer head,
  `--head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1
  --head-train-input e_then_f --total-steps 30000 --schedule cosine --warmup-steps 2000
  --amp-dtype bf16 --num-layers 1 --encoder-type gru`. Triage filter + full eval =
  existing `run_full_eval_2L.sh`.

## Run names (never reuse a --save-path on resume)
- v20: Phase A `enc_fcst_v20_phaseA_fp32warmup_5k_v150`; Phase B / backbone
  `enc_fcst_v20_freshwarmup_fp16_150k`.
- v27c: `enc_fcst_v27c_dk08_ffnfp16_150k`.
- v16/v17: `enc_fcst_v16_..._150k` / `enc_fcst_v17_..._150k`.
- q-heads: `<backbone>_qhead_xfmr2L_quant_30k`. Eval dirs `gift_eval_{triage,full}_<tag>`.

## Phases (controller state machine)
1. WAIT: both `gift_eval_full_v20R/summary.txt` and `gift_eval_full_v27c/summary.txt`
   exist. (v27c chain currently produces triage then the monitor fires its full-eval.)
2. RECORD v20R-full + v27c-full GM. Pick best-of-3.
3. LAUNCH parallel: GPU0 = v20 150k (Phase A then B); GPU1 = best-of-3 150k
   (v27c warm-resume, or v16/v17 from-0).
4. On each backbone reaching 150k (final/best_loss saved): q-head+triage+full-eval at
   150k; and at that trajectory's 50k (v20/v16/v17: own _50k; v27c: reuse existing 50k).
5. REPORT the 50k-vs-150k table per arm (does ordering change?).
6. SPARE: if a GPU is idle with hours to spare, extend 2 more sweep params (from-0
   150k for the purged ones) and eval @150k.

## Safety
- Disk: cron `ed060d08` checks every 30 min; controller also `df` gates before each
  launch (abort-and-flag if <40G free; never touch non-jupyter files).
- Crash-resilience: every step idempotent (skip if FINAL/summary exists); controller
  re-entrant — re-running resumes at the first unfinished step. NO optimizer reset ever.
- Monitors: one controller-progress monitor; do not duplicate per-arm watchers.

## CORRECTION 2026-05-16 03:16 — v20-fp16 diverged; dk0.9 ref pivoted to pure-fp32
- **v20 `enc_fcst_v20_freshwarmup_fp16_150k` (Phase B) DIVERGED at ~step 45k**
  (loss 2.9→5.7→10.8 over 44k→46k). Confirms warmup→fp16 is stable only to
  ~40-50k and does NOT survive 150k (residual amplitude outgrows fp16). The
  original v20 "worked" only because it stopped at 50k. Diverged python killed;
  good periodics ≤`_40k.pth` exist but unused.
- Controller's v20 arm is therefore dead; its GPU1 subshell fell through to v17
  then that was killed too (controller's `v20x*` scorecard cells stay empty —
  expected, not a bug).
- **dk0.9@150k reference is now PURE-FP32** (v11c's own recipe extended), run by
  the standalone `gpu1_reference_pipeline.sh` on GPU1: dk0.9-fp32 0→150k FIRST
  (run-name `enc_fcst_dk09_fp32_150k`, tags `dk09fp32x{50k,150k,100k}`), THEN
  v17 dk0.95-fp32 (`enc_fcst_v17_dk095_150k`, tags `v17x*`). seed 20260516.
- Bet scorecard: dk0.9@150k = `gift_eval_full_dk09fp32x150k` (NOT v20x150k).
  v27c@150k still GPU0 via controller. The controller's auto-scorecard is now
  partial — assemble the final bet table from `gift_eval_full_{dk09fp32,v17,v27c}x*`.
- GPU0 v27c-150k + controller-main were verified UNTOUCHED by this surgery.

### Reporting convention (user, 2026-05-17): @50k table error bars
Any dropkey value with **≥2 runs at the 50k horizon** is reported as **mean ± error**
(± stderr = std/√n; for n=2 this is ± half-range), and the individual run values
are still listed. Single-run dk points reported bare. Currently multi-run @50k:
- dk0.7: v16 (1.335, seed A) + dk0.7-fp32 (seed 20260516, pending qhead/eval).
- dk0.9: v11c (1.292, pure-fp32 seed A) + dk0.9-fp32@50k periodic (1.6586 — flagged
  BAD transient periodic; same run @150k=1.3328). Report v11c as the clean point;
  show the 1.66 separately/footnoted, not silently averaged in.
All other dk @50k (0.5, 0.85, 0.92, 0.95, 0.3) are n=1 → no error bar.

### Standing rule (user, 2026-05-16): if a precision recipe DIVERGES → go fp32
Any backbone arm whose fp16/partial-fp16 recipe diverges is re-run in **pure
fp32** (not grad-clipped, not bf16-patched, not abandoned). Applied: v20 fp16
→ dk0.9 pure-fp32. v27c (dk0.8 ffn-fp16) did NOT diverge (stable to 150k,
loss ~1.94) so it stays as-is — the rule triggers on divergence only. v17 is
already pure-fp32. Same rule applies to any remaining/future arm.
