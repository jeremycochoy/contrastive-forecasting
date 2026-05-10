# Encoder+forecaster (6L+6L, bf16, τ=0.10) — training in progress

Training a GRU patch embedding → 6 causal transformer-encoder layers → 6 causal transformer-forecaster layers at τ=0.10 with bf16 autocast, otherwise identical to the τ-sweep baseline (`run_encoder_forecaster.sh` on elisa GPU 1, target 50k steps, batch 256).

**log–log scale**

![progress (log–log)](plots/progress.png)

**linear scale**

![progress (linear)](plots/progress_linear.png)

## Per-batch training metrics at the baseline's last common step (~15k)

Means over the last 200 steps of each window.

| arm | step | loss | U_temporal | U_batch | AUC | top-1 | error-gap-closure |
|---|---:|---:|---:|---:|---:|---:|---:|
| τ=0.10 baseline (6L fcst) | 15 000 | 7.040 | 0.050 | 0.096 | 0.8982 | 0.7472 | 0.599 |
| encoder+forecaster (6L+6L, bf16) @ 15k | 14 800–14 999 | 1.432 | 0.283 | 0.507 | 1.0000 | 1.0000 | 0.996 |
| encoder+forecaster (6L+6L, bf16) — CURRENT | 24 601–24 800 | 1.387 | 0.300 | 0.565 | 1.0000 | 1.0000 | 0.996 |

For reference the long-run τ=0.10 baseline (separately resumed, 48k–150k segment) lands at loss=6.917, U_t=0.059, U_b=0.119, AUC=0.9011, top1=0.7562, egc=0.603 by step 150k — i.e. the baseline's per-batch metrics barely move past 15k. That long reference is omitted from the trajectory plots above because the in-progress arm currently only reaches ~25k; including it would compress the visible curves.

## What the curves show

On the per-batch training metrics plotted (with `1-metric` on the y-axis for AUC / top-1 / error-gap-closure), the encoder+forecaster arm separates from the τ=0.10 baseline within the first few hundred steps and stays separated through 24.8k. Loss settles around 1.4 vs the baseline's ~7.0 plateau, and the per-batch AUC and top-1 saturate at the numerical 1.0 floor while the baseline holds at ~0.90 / ~0.75. Error-gap-closure tracks the same picture: the new arm closes ~99.6 % of the cosine-similarity gap between the cross-batch reference (`fp`) and a perfect positive, vs ~0.60 for the baseline. Dimension usage is also much higher under the encoder arm (U_temporal 0.28 vs 0.05; U_batch 0.51 vs 0.10 at 15k) and is still rising. The new arm's curves show no sign of degradation across the 24.8k steps seen so far.

## Caveats

- Training is still running (~24.8k of 50k). Numbers will move.
- All AUC / top-1 / loss / egc values plotted are PER-BATCH on the training distribution (256 samples, in-distribution). They are not held-out and do not directly speak to GIFT-Eval metrics. Per-batch AUC=1.0 means perfect retrieval within a 256-sample minibatch, not zero error on real forecasts.
- The encoder arm's `fp` (cross-batch reference cosine) goes negative, while the baseline keeps `fp` ≈ +0.17. Some of the "egc improvement" is the denominator `(1 - fp)` widening, not the numerator `(1 - ff)` shrinking. Held-out eval is needed to interpret.
