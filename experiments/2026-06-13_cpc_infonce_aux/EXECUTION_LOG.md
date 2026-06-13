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
