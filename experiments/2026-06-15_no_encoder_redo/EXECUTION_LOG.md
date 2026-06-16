# #348 no-encoder redo — execution log (journey notes, NOT the report)

Resume state for the agent across the hourly wake-up loop. The science goes in
`no_encoder_redo.md`. Goal: PR resolving #348 open + ready to merge, all
comments addressed, PR + report checklists pass.

## What #348 asks
Redo #344's two competing setups — **normal contrastive loss** and **+ CPC
term** — but with the **encoder stack removed** (`--num-encoder-layers 0`: only
the GRU patch-embedding + the 6-layer forecaster). Compare against the encoder'd
#344 counterparts (enc3, enc6). Verdict on: (1) does removing the encoder
improve transfer, (2) does the CPC term still help without the encoder, (3) does
CPC still stabilise late training without the encoder.

## Environment
- Running directly on **elisa** (hostname `elisa`), 2× RTX 4090. Train locally, no remote sync.
- GPU 1 is the *other* (rnd) project's run — DO NOT touch. Use GPU 0 (and GPU 1 only if it frees and is confirmed idle).
- Worktree (code + report, PR branch): `/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-no-encoder-348`
  branch `experiment/2026-06-15-no-encoder-redo`, based on the #344 branch
  `origin/experiment/2026-06-13-cpc-infonce-aux` (stacked on PR #346 — needs its CPC code).
- Outputs (gitignored, OFF the worktree so a worktree-remove can't delete them):
  `/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-15_no_encoder_redo/{runs,results}`
- GIFT-Eval data: `/home/jupyter/workspaces/gift-eval-data`. HF token in worktree `experiments/hf_token.txt`.

## Arms (two backbones)
- `base` = #339/#341/#344 recipe + `--num-encoder-layers 0`, NO `--cpc-infonce-weight`.
- `cpc`  = same + `--cpc-infonce-weight 1.0`.
NAME = `bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_{base,cpc}`.
Recipe: bs 1024, 12 500 steps, seed 20260520, d_model 384 / 6 heads, 6-layer
forecaster, xshh_allt loss, --pos-in-denominator --subtract-contrastive-floor
--stopgrad-positive-h, tau 0.10, rev-norm ewma span 128, encoder-type gru,
forked-arma mix 0.0078125 + crossfade-triplets 1, mixup-p 0.3, freq/seas emb 3.

## Memory knobs (smoke-tested 2026-06-15)
- num_encoder_layers=0 trains healthily (loss falls, AUC rises).
- The GRU **patch-embedding** is the memory hog (OOM 17.9 GB without its chunking),
  independent of the encoder ⇒ keep `PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4`.
- All knobs ON (FCST_GRAD_CKPT=1, XSHH_ALLT_CHUNK=1): ~13.5 GB, ~3.12 s/step ⇒ ~10.8 h/backbone.
- Config under test (B): FCST_GRAD_CKPT=0, XSHH_ALLT_CHUNK=64 — see smoke_B.log for mem/speed.
- All these knobs are byte-identical to the loss (memory <-> kernel launches only).

## Pipeline (autonomous, mirrors #344)
1. Backbones: `supervise_noenc.sh {base|cpc} <gpu>` -> `train_backbone_noenc.sh`. Auto-resume (8 attempts).
2. Downstream (auto): `watch_and_downstream_noenc.sh {base|cpc} <gpu>` polls for `bb_..._FINAL.pth`,
   then `chain_noenc.sh` -> `downstream_noenc.sh` trains 2L/6L × best/last q-heads + full-97 GIFT-Eval.
   Touches `results/chain_{base,cpc}.done`.
3. Analyze: `analyze_noenc.py` -> `results/{gm_table,pairwise_table}.csv`. Baselines enc3/enc6 from
   #339/#341/#344 result dirs; reuses #341/#344 GM + paired-bootstrap logic verbatim.
4. Plots: adapt #344 plot scripts (gm summary as a depth ladder; training dynamics).
5. Report: `no_encoder_redo.md` per REPORT_STANDARD; sub-agent review; PR; checklists.

## Comparison baselines (validated — analyze_noenc.py reproduces #344 to 4 dp)
| arm | 2L best / last | 6L best / last |
|---|--|--|
| base enc3 (#339) | 1.1768 / 1.1801 | 1.1587 / 1.1629 |
| base enc6 (#341) | 1.1801 / 1.2134 | 1.1606 / 1.1933 |
| +CPC enc3 (#344) | 1.1846 / 1.1531 | 1.1584 / 1.1436 |
| +CPC enc6 (#344) | 1.1786 / 1.1803 | 1.1575 / 1.1623 |

## Status log
- 2026-06-15: worktree + scripts + output dirs set up; smoke confirmed noenc trains (13.5 GB, ~3.1 s/step).
- 2026-06-15: memory — keep all safe knobs ON (FCST_GRAD_CKPT=1, PATCH_ENC_CKPT=1/CHUNK=4, XSHH_ALLT_CHUNK=1);
  relaxing forecaster ckpt or patch-enc ckpt OOMs at batch 1024. ~10.8 h/backbone.
- 2026-06-15: orchestrate_gpu0.sh launched detached on GPU 0 (base bb -> cpc bb -> base chain -> cpc chain).
  base backbone training (GPU0 100% util). Monitor log: $OUT/orchestrate.log + results/run_bb_*.log.
- 2026-06-15: DRAFT PR **#349** opened (base experiments, stacked on #346). analyze_noenc.py validated
  (baselines reproduce #344). plot_gm_ladder.py written. Hourly wake-up loop monitoring.
- 2026-06-15 ~13:17: **GPU 1 freed** (ops reclaimed a stale Jupyter kernel, Jeremy-authorized) -> parallelised.
  Killed the single-GPU orchestrator's top PID only (base kept running on GPU 0, supervise reparented to
  init). Now: **GPU 0 = base backbone (running) -> base downstream (watch_and_downstream_noenc.sh base 0)**;
  **GPU 1 = cpc backbone -> cpc downstream (orchestrate_arm.sh cpc 1)**. Both single-GPU (exact #344 recipe).
  Logs: orchestrate_cpc_g1.log, watch_base_g0.log, results/run_bb_*_{base,cpc}.log.
  NOTE: orchestrate_gpu0.sh is now the single-GPU fallback only; do NOT re-launch it while the per-arm
  pipelines run (it would double-launch on GPU 0). Re-launch per-arm via orchestrate_arm.sh / the watcher.

- 2026-06-15 ~13:12: SHARED-MACHINE CONTENTION on GPU 0 — rnd_kacper trainline runs (mahalanobis port)
  + an rnd ipykernel landed on GPU 0 (~6.4 GB combined). base stable at ~13.8 GB, GPU0 free ~3.9 GB.
  NOT mine — DO NOT kill (shared account). base won't grow (fixed model/batch) and auto-resumes on OOM via
  its supervisor. Watcher now alerts if GPU0 free < 800 MiB or a backbone crash/resume occurs. cpc on GPU 1
  has ~10 GB free (safe). base step ~4300/12500 (ETA ~8h), cpc step ~900/12500 (ETA ~11.5h, cpc_aux≈4.35).

- 2026-06-15 ~21:30: **base backbone DONE** (step 12500; _FINAL.pth=best-loss + _final.pth=last present).
  Retired the sleeping base-dl watcher and launched base downstream directly: `chain_noenc.sh base 0`
  (PID family 1126453; log chain_base_g0.log) — sequential 2L then 6L (best+last + full-97 eval) on GPU 0.
  GPU 0 shared with Kacper's ~10 GB job, so base downstream stays sequential there (can't fit 2 cells).
  cpc backbone still training on GPU 1 (~4.3 h left). 2L-base head: ~6 sps, ETA ~1.4h/30k.
  PLAN when cpc backbone finishes: parallelise cpc downstream across the freed GPUs (GPU1 free immediately,
  GPU0 free once base downstream ends). cpc orchestrate_arm.sh cpc 1 auto-starts cpc chain on GPU1 — let
  2L run there and add 6L on GPU0 in parallel (idempotent; cells skip if FINAL exists). DO NOT relaunch a
  second base/cpc downstream (double-write hazard on qhead_*.pth).

- 2026-06-16 ~02:49: **cpc backbone DONE** (GPU 1 freed, ~19 GB). Both backbones complete.
  Downstream now PARALLEL: **GPU 0 = base (sequential, chain_noenc.sh base 0)**; **GPU 1 = cpc 2-wide**
  via `/tmp/cpc_takeover.sh` -> two `downstream_noenc.sh cpc {2,6} 1` lanes (logs lane_cpc{2,6}_g1.log;
  touches results/chain_cpc.done when both lanes finish). cpc 2L head ETA ~1.4h, 6L ~3h (2-wide sharing).
  base 2L-best eval ~task 80/97. GPU0 still shares Kacper's ~10 GB (base stays 1-head-at-a-time there).
  Targets: 8 head FINALs (qhead_{2,6}L_..._{base,cpc}[_last]_FINAL.pth) + 8 eval summaries.
  When both chain_{base,cpc}.done -> run analyze_noenc.py + plot_gm_ladder.py + plot_training_dynamics_noenc.py,
  fill report Result/stability, sub-agent review, finalise PR #349, checklists.
  NOTE: cpc_takeover.sh is ONE-SHOT (its pkill would kill the running lanes) — do NOT re-run it.
- 2026-06-16 ~04:27: base downstream ALSO taken over for parallelism (`/tmp/base_finish.sh`, PID family
  1206894): stopped the base chain (in-flight 2L-last eval kept running, orphaned), now pretraining base 6L
  heads on GPU0 overlapping that eval, then runs base 6L evals concurrently; touches chain_base.done when
  all 4 base summaries exist. First result: **base no-enc 2L-best GM=1.4253** (vs enc3/enc6 ~1.18 — removing
  the encoder HURTS, opposite of the hypothesis; preliminary, 1/8 cells). GPU0 ~3 GB free (Kacper 10 + eval
  + 6L head) — fits. base_finish.sh + cpc_takeover.sh pkill cpc/base patterns — NEVER put those patterns in
  an interactive command (it self-kills, exit 144); split the string or use a file.

## CPC_All arm added (2026-06-16, user request)
User asked to add ONE arm: no-encoder + **CPC_All** = paper-exact CPC InfoNCE
(van den Oord Eq. 4) with the full marginal candidate set. Implemented
`cpc_infonce_all_loss` (src/loss.py) + flag `--cpc-infonce-negs {matched,cross,all}`
+ 7 tests (80/80 pass). Decision after MI discussion: arm uses **cross** =
{positive} ∪ every OTHER sequence b'≠b at all l — context-INDEPENDENT (marginal)
negatives, so Theorem 1 holds EXACTLY (unique optimum = density ratio, I>=logN−L
bound). `all` (full grid incl. same-sequence) kept as a flag — those negs are
correlated with c_t ⇒ approximate bound. NO masking: denominator sums over X
incl. the positive (Eq. 4). Score keeps unit-norm z + bilinear W1 (same as the
+CPC arm). cpcall arm: `orchestrate_arm.sh cpcall 1` on GPU1, seed 20260520, 12.5k
steps, CPC_ALL_CHUNK=32 (~5.3 s/step, ~18.5 h). Smoke OK (cpc_aux≈11.3, no OOM).
TODO when done: add noenc_cpcall to analyze_noenc.py + gm-ladder + report + PR.
**SINGLE-GPU constraint (user, 2026-06-16):** keep ALL cpcall work (backbone +
the 4 downstream cells) on GPU1 ONLY — leave GPU0 for Kacper. Do NOT parallelise
cpcall downstream across both GPUs (as was done for base/cpc). orchestrate_arm.sh
cpcall 1 already runs backbone+chain on GPU1 sequentially; just don't add GPU0 work.

## DONE (2026-06-16) — original 2 no-encoder arms
All 8 cells evaluated. Verdict: **removing the encoder reliably HURTS the plain
contrastive arm** (base best +0.19–0.25, last +0.05–0.08 GM, all CIs>0); **with
the CPC term it is neutral** (best 4/4 ns, last 3/4 ns) — the **CPC auxiliary
substitutes for the encoder**. Adding CPC to the no-encoder backbone: best
−0.258(2L)/−0.199(6L), last −0.099/−0.079 (all reliable; vs ~0.003 best-loss with
the encoder). Mechanism: CPC term stays ~3 without the encoder vs collapsing to
~0 with it. Report final + sub-agent reviewed (4 minor fixes applied) + checklists
pass. **PR #349 ready for review** (MERGEABLE/CLEAN, base `experiments`, stacked
on #346). 0 comments on PR/issue. Both backbones + downstream complete; GPUs idle.
Remaining: respond to any maintainer comments (hourly comment-watcher).

## Wake-up checklist (legacy, training phase)
1. Is orchestrator alive? `pgrep -af orchestrate_gpu0` — if dead, re-launch `orchestrate_gpu0.sh 0` (idempotent).
2. Backbone progress: tail `results/run_bb_*_{base,cpc}.log` (step / sps / loss healthy, no NaN/Traceback).
3. If GPU 1 became free AND confirmed idle (not the rnd run), opportunistically parallelise the next
   un-started piece on it (e.g. cpc backbone, or a downstream cell).
4. When `bb_..._FINAL.pth` lands -> watcher/chain auto-runs downstream. When `results/chain_{base,cpc}.done`
   for both -> run analyze_noenc.py + plot_gm_ladder.py + training-dynamics, fill report, sub-agent review.
5. Keep PR #349 / issue #348 comment threads answered.
