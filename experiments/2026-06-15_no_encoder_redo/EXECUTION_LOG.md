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

## Status log
- 2026-06-15: worktree + scripts + output dirs set up; smoke confirmed noenc trains; tuning mem knobs.
