# Stop-grad × capacity — plan (#341)

Follow-up to #339 / PR #340. #339's encoder-side stop-grad on the InfoNCE positive is
reliably better than its reference at best-loss on both heads. Hypothesis under test:
stop-grad changes what the encoder learns, so capacity knobs that hurt *without* it
(#336: enc6 + no-bottleneck reliably worse than enc3 + no-bottleneck) may flip sign
*with* it.

## Arms

| # | arm | backbone | status |
|---|---|---|---|
| 1 | base+triplet (enc6 + 128-wide bottleneck forecaster) | `bb_allt08_xftrip_bn_enc6_qk_aon_b1024` | done (#336) |
| 2 | enc3 + no-bottleneck + stop-grad | `bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024` | done (#339) |
| 3 | **NEW** enc6 + no-bottleneck + stop-grad | `bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024` | this exp |
| 4 | **NEW** enc6 + bottleneck + stop-grad (= arm 1 + stop-grad) | `bb_allt08_xftrip_bn_enc6_sgpos_qk_aon_b1024` | this exp |

No-stop-grad capacity twins for context (both #336): `…xftrip_nobn_enc6…` (enc6+nobn,
reliably worse than enc3+nobn) and `…xftrip_nobn_enc3…` (the #328 best arm).

All arms: allt·0.8% + crossfade triplet, qk-norm, attn-out-norm,
`--subtract-contrastive-floor`, `--pos-in-denominator`, τ=0.10, batch 1024,
12 500 steps, seed 20260520, one RTX 4090 per arm (elisa, per issue directive).
New arms differ from #339's launcher only by `--num-encoder-layers 6` (arm 3) plus
`--forecaster-d-model 128 --forecaster-n-heads 4` (arm 4).
Param-count cross-check vs the #336 twins (stop-grad adds no params):
arm 3 = 22,063,164, arm 4 = 12,714,684 (asserted by `smoke_sgcap.sh`).

## Protocol

1. Backbones: `scripts/chain_sgcap.sh <arm> <gpu>` (one per GPU) → 12.5k steps,
   periodic checkpoints every 2.5k; `watchdog_sgcap.sh` monitors.
2. Heads per arm: 2L and 6L quantile heads on **best-loss** (30k steps) and **last**
   (resume best head, 10k re-adapt) checkpoints — identical to #339's
   `downstream_sgpos.sh` (chained automatically after each backbone).
3. GIFT-Eval full-97 (strategy B4) per cell, sharded across free GPU capacity
   (`shard_evals.py` / `mopup_evals.py` / `merge_shards.py`, raw-config-name filters,
   97-completeness gate).
4. Analysis (`analyze_sgcap.py`): GM-Relative MASE + paired bootstrap (2000 resamples,
   90% CI) for all pairwise contrasts of the 4 arms, per head × checkpoint; arm 1 and
   arm 2 per-task relatives reused from the #336 / #339 result dirs.
5. Dynamics (`plot_training_metrics_sgcap.py`): 6-panel log-log, new arms vs the two
   references; early version posted to issue #341 at a few k steps (user directive).

## Success criteria (issue #341)

- GM-Relative MASE + paired-bootstrap CIs for all pairwise contrasts of the 4 arms,
  per head × checkpoint.
- Verdict: does stop-grad flip the sign of the capacity knobs (arm 3 vs arm 2;
  arm 4 vs arms 1 and 2)?
- Report PR into `experiments` passing the report checklist.

## Artifact map

- New runs/results: `~/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity/{runs,results}` (elisa)
- Arm 1 + no-sg twins: `…/experiments/2026-06-03_crossfade_triplet/{runs,results}`
- Arm 2: `…/experiments/2026-06-10_stopgrad_positive/{runs,results}`
- Code worktree: `/tmp/cf-341`, branch `experiment/2026-06-11-stopgrad-capacity`
  (from PR #340's branch — needs `--stopgrad-positive-h` + `--crossfade-triplets`)
