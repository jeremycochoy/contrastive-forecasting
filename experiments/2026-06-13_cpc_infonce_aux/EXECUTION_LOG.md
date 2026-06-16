# #344 CPC InfoNCE auxiliary — execution log (journey notes, NOT the report)

Resume state for the agent. The science goes in `cpc_infonce_aux.md`.

## Environment
- Running directly on **elisa** (hostname `elisa`), 2× RTX 4090. No remote sync — train locally.
- Worktree (code + report, the PR branch): `/home/jupyter/contrastive-forecasting/.claude/worktrees/exp+cpc-infonce-344`
  branch `experiment/2026-06-13-cpc-infonce-aux`, based on `origin/experiments` (PR #340 merge).
- Outputs (checkpoints, eval, results — gitignored, kept OFF the worktree so a worktree-remove can't delete them):
  `/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/{runs,results}`
- Draft PR: #346 (base `experiments`).
- GIFT-Eval data: `/home/jupyter/workspaces/gift-eval-data`. HF token copied into worktree `experiments/hf_token.txt`.

## Pipeline (autonomous)
1. **Backbones** (running): `supervise_cpc.sh {enc3 0 | enc6 1}` → `train_backbone_cpc.sh`. 12.5k steps,
   batch 1024, seed 20260520, exact #339/#341 recipe + `--cpc-infonce-weight 1.0`. Both arms use
   `FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK=64` (fit 24 GB). ETA ~14 h (launched ~19:32 UTC 06-13).
   Supervisor auto-resumes from latest periodic ckpt on crash (8 attempts).
2. **Downstream** (auto): `watch_and_downstream.sh {enc3 0 | enc6 1}` polls for `bb_..._FINAL.pth`, then
   `chain_cpc.sh` → `downstream_cpc.sh` trains 2L/6L × best/last q-heads + runs full-97 GIFT-Eval
   (`DO_EVAL=1`). Writes `results/gift_eval_full_*/summary.txt`. Touches `results/chain_{arm}.done`.
3. **Analyze** (manual): `analyze_cpc.py` → `results/{gm_table,pairwise_table}.csv`. Reuses #341 GM +
   paired-bootstrap logic verbatim; verified to reproduce baseline GMs to 3 dp.
4. **Plots** (manual): `plot_gm_summary_cpc.py`, `plot_training_dynamics_cpc.py` → `plots/`.
5. **Report**: fill `cpc_infonce_aux.md` Result/Dynamics/Verdict; sub-agent review; PR ready; checklists.

## Baselines (from the stop-grad-capacity report, reproduced by analyze_cpc.py)
| arm | 2L best/last | 6L best/last |
|---|--|--|
| enc3 (#339 arm2) | 1.1768 / 1.1801 | 1.1587 / 1.1629 |
| enc6 (#341 arm3) | 1.1801 / 1.2134 | 1.1606 / 1.1933 |

## Observations
- CPC term (`cpc_aux`) drops from ~8.6 (init) to ~0.20 by step 100 — the unbounded bilinear `W₁`
  satisfies the next-step prediction easily. Watch whether this means little pressure on the encoder
  (→ neutral transfer). Log per-step in losses.csv column `cpc_aux`; `loss_tau_ref` stays a clean
  CPC-term-free contrastive reference comparable to baselines.
- Memory: enc3 with the baseline chunk-2/no-grad-ckpt recipe + CPC hit 96% of 24 GB → restarted both
  arms with grad-ckpt + chunk 1 (byte-identical to the loss; memory↔kernel-launches only).

## Final result (2026-06-14)
All 8 cells evaluated. Verdict: **neutral at best-loss (4/4 ns), reliably better at last (4/4
CI<0)**. Last-Δ: enc3·2L −0.027, enc3·6L −0.019, enc6·2L −0.033, enc6·6L −0.031. The CPC term's
value collapses to ~0 by step ~1000 (unbounded W₁) yet improves the pretext representation and
reverses the baselines' best→last degradation ⇒ a late-training stabiliser. Eval was accelerated:
6L heads pretrained on the idle GPUs during the CPU-bound 2L evals, and the 6L cells evaluated
4-wide via orchestrate_6L_evals.sh + eval_cell.sh (byte-identical do_eval command).

## Follow-up arm (2026-06-14): enc6 + CPC + align, NO main contrastive loss
User-requested. Tests if CPC + a separate forecaster loss (BYOL align, encoder sg) beats the
xshh_allt contrastive loss. Code: standalone `align_loss` + `--no-main-contrastive-loss` (skip
contrastive_latent_loss; add align standalone). Launch: `train_backbone_cpcalign.sh 0` (supervised)
+ `watch_cpcalign.sh` (heads parallel 2L@g0/6L@g1, then 4-wide eval; TAG_OVERRIDE reuses
downstream_cpc.sh/eval_cell.sh). Backbone NAME=bb_allt08_xftrip_nobn_enc6_cpcalign_qk_aon_b1024_cpc.
analyze_cpc.py arm key `cpcalign_enc6`; pairs vs base_enc6 (main loss) and cpc_enc6 (main+cpc).
After eval: add arm to report table + a paragraph, regen plots, update PR #346.

### cpcalign arm runs 2-GPU DDP (2026-06-14)
Switched the cpcalign backbone to torchrun --nproc_per_node=2, --batch-size 512/rank ⇒ GLOBAL
batch 1024 (== single-GPU baselines). Loss on the gathered global batch (gather_latents: "global
negatives, == 1-GPU @ global B"), verified at startup ("DDP rank x/2, global bs=1024, gathered
loss"); CPC cross-batch negatives + align span both GPUs. ~2× faster (~6h). The 13-min single-GPU
partial (step ~400, only best_loss.pth, no periodic full-state ckpt) was NOT resumable across the
single→DDP topology change, so it was cleared for a clean CSV (no single+DDP mix). Crash-resume:
supervisor --resume's the latest periodic _Nk.pth; train.py appends to the same CSV (conserved).

### cpcalign final result (2026-06-15)
All 4 cells done (4-wide eval). GM: 2L best 1.993, 2L last 1.378, 6L best 1.432, 6L last 1.214 —
reliably/substantially WORSE than baseline (Δ +0.81/+0.16/+0.27 reliable; 6L-last +0.02 ns) and
worse than main+CPC on all 4. Verdict: CPC + a separate forecaster loss does NOT replace the
contrastive loss. Training was persistently unstable (CPC term oscillates ~0.01↔10; elevated
reference loss / 1−R² / 1−AUC). Added as the report's Ablation section + plots/cpcalign_gm.png;
training_dynamics.png now 5 arms (panel 1 = loss_tau_ref, comparable across all). Task #6 done.
