# Stop-gradient on the encoder positive term — plan (#339)

Follow-up to #328 / PR #336. The #328 best arm (L3 + no-bottleneck + crossfade-triplet)
reliably beats its base on both heads at full training (last checkpoint). Test whether a
SimSiam/BYOL-style stop-gradient on the encoder side of the InfoNCE positive changes the
learning dynamics and the downstream transfer.

## Arms

| arm | backbone | change |
|---|---|---|
| reference | `bb_allt08_xftrip_nobn_enc3_qk_aon_b1024` (#328, already trained) | — |
| stop-grad | `bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024` | `--stopgrad-positive-h` |

One change only: in the positive similarity sim(h_{t+1}, f_{t+1}), detach h_{t+1}
everywhere that term appears (numerator and denominator; `--pos-in-denominator` is on).
Negatives keep gradient on h. Everything else identical to the #328 best recipe:
allt·0.8% mix, crossfade triplet, no forecaster bottleneck, 3-layer encoder, QK-norm,
attn-out-norm, `--subtract-contrastive-floor`, τ=0.10, batch 1024, 12 500 steps,
seed 20260520, single RTX 4090.

## Protocol

1. Backbone 12 500 steps (`scripts/train_backbone_sgpos.sh`).
2. Downstream per head size (2L, 6L) via `scripts/downstream_sgpos.sh` — identical to
   #328's `downstream_generic.sh`: 30k-step quantile head on the best-loss checkpoint +
   GIFT-Eval full-97 (strategy B4); then the full-training (last-checkpoint) head,
   resumed from the best head and re-adapted 10k steps, + eval.
3. Analysis (`scripts/analyze.py`): GM-Relative MASE, paired bootstrap (2000 resamples,
   90% CI) stop-grad vs reference, per head × checkpoint. Reference eval CSVs copied to
   `results/reference/` from #328's runs.
4. Training dynamics (`scripts/plot_training_metrics.py`): the 6-panel log-log layout of
   #328 (loss, ratio gap, 1−R², used dims), stop-grad vs reference.

## Success criteria (issue #339)

- GM-Relative MASE vs reference at both head sizes, both checkpoints, with CIs.
- Log-log training-dynamics curves of both models.
- Report PR into `experiments` passing the report checklist.
