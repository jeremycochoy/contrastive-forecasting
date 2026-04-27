# Real-data RevEWMNorm span sweep

## Why

Per the previous handoff: `span=32` was the default settled in
`experiments/revnorm-span-search/report.md` on ARMA-only data. Once
periodic content dominates the input we have reason to think a longer
span might help (preserve more periodic amplitude, less periodic
distortion). The user's initial intent: "swipe on real data, because
we want to know the good span for the real distribution".

## Setup

| Knob | Value |
|---|---|
| Spans tested | 32, 64, 128, 256 |
| Mix ratio | **0.0** (pure HF base-bundles, no synth confound) |
| Steps per backbone | 20000 (cut from 30k for budget at the time of launch) |
| Architecture | fe+mu (freq_emb=3, mixup=0.3) |
| Patch stats | none |
| Loss | `cosine_similarity_batch_no_time_neg` |
| Output | training loss / gap curves; backbones-only — no qhead, no eval |

## Results

Last-200 ema:

| Span | ema_loss | ema_gap |
|---|---:|---:|
| 32 | 3.004 | **0.330** |
| 64 | 2.887 | 0.321 |
| **128** | **2.827** | 0.316 |
| 256 | 2.917 | 0.300 |

(`experiments/freq-embedding/plots/span_sweep_real.png` — both metrics
across all 4 spans on the same axes.)

## Findings

1. **Loss is U-shaped** with the bottom at span=128 (5.9% lower than
   span=32, 3.0% lower than span=256).
2. **Gap is monotonically decreasing**. span=32 has the highest gap
   (0.330), span=256 the lowest (0.300).
3. The two metrics **disagree** on the optimum.

## Interpretation (speculative)

Possible reading: with longer spans, RevEWMNorm tracks the slow-moving
local mean more aggressively, leaving less periodic / trend signal in
the patch values. Lower contrastive loss because there's less
"residual structure" the model needs to fit. But the within-batch
positive (FF) goes UP a similar amount as the cross-batch negative
(FP), shrinking the gap.

## Caveats

- Single seed.
- 20k steps may not be enough to reveal the full effect — synth sweep
  (later) showed the trend continues monotonically up to span=512 on
  in-distribution data, suggesting the real-data optimum may also be
  higher than 128 with more compute.
- Backbones-only — no downstream eval. The "best span" decision needs
  to be revisited with held-out validation.

## Bug caught and fixed during this run

`create_mixed_periodic_dataloader(mix_ratio=0.0, ...)` short-circuited
to `create_hf_dataloader` which doesn't yield freq_ids, crashing
`train.py::main` with "too many values to unpack" when `freq_emb_dim>0`.
Fixed in `src/dataloader.py` to fall through to MixedPeriodicLoader
(synth_bs=0, hf_bs=batch_size) when `emit_freq_ids=True`.

## Open questions

- Real-data sweep at 30k+ steps — does the optimum move to span=256
  or beyond?
- Real-data sweep with downstream qhead + held-out HF validation —
  is the loss/gap signal a good proxy for downstream forecast quality
  on real data?

## Artefacts

- 4 backbones: `checkpoints/tiny_femu_real_span{32,64,128,256}_FINAL.pth`.
- 4 loss CSVs: `checkpoints/tiny_femu_real_span*_losses.csv`.
- Plot: `experiments/freq-embedding/plots/span_sweep_real.png`.
