# Encoder + Forecaster — FAILED

## Goal

Test whether adding 6 causal transformer-encoder layers between the existing GRU patch embedding and the existing 6-layer causal transformer-decoder forecaster improves the backbone, measured by GIFT-Eval MASE through a q-head trained on top of the frozen backbone. Same architectural building block as the forecaster (causal `DecoderOnlyTransformerLayer`, H=384, n_heads=6, ffn_mult=4, depthwise_conv=3, dropout=0.1). Backbone trained with bf16 autocast, batch=256, τ=0.10, 25 600 steps on `jeremycochoy/gift-pretrain-full-4096:small_v1`.

## Headline result

**Triage GIFT-Eval GM-Relative MASE = 1.596** (`results/gift_eval_triage/summary.txt`). Seasonal naive = 1.000. R9_E13 reference (same q-head recipe on `backbone-beta_167k`) = 0.990. Moirai = 0.809. The triage gate (< 1.0 → full eval) failed; full eval was not launched.

| arm                                                                   | triage GM-MASE |
| --------------------------------------------------------------------- | -------------: |
| Seasonal naive                                                        |          1.000 |
| Moirai (leaderboard)                                                  |          0.809 |
| R9_E13 reference (backbone-beta_167k + xfmr-12L head)                 |          0.990 |
| **ours (encoder+forecaster backbone + xfmr-12L head, 15.2k head steps)** |     **1.596** |

Two of 11 configs were below seasonal naive (`bizitobs_l2c/5T` 0.61, `us_births/D` 0.98); the rest worse. High-frequency configs particularly bad (`bizitobs_application/10S` 4.98, `bizitobs_service/10S` 4.59, `electricity/H` 1.94, `covid_deaths/D` 1.80).

## The held-out contrastive metrics looked great — and they were misleading

The canonical N=50 multisample eval (`results/encoder_forecaster_metrics_multisample_n50.csv`):

| metric    | τ=0.10 baseline      | encoder+forecaster      | Δ      |
| --------- | -------------------- | ----------------------- | -----: |
| AUC       | 0.8993 ± 0.0054      | **0.9778 ± 0.0028**     | +0.078 |
| top-1     | 0.7535 ± 0.0099      | **0.9409 ± 0.0062**     | +0.187 |
| r²_naive  | 0.6153 ± 0.0095      | **0.8550 ± 0.0330**     | +0.240 |
| U_temporal| 0.0512 ± 0.0012      | **0.2956 ± 0.0023**     | +0.244 |
| U_batch   | 0.1019 ± 0.0015      | **0.5575 ± 0.0064**     | +0.456 |

All deltas are many SEM apart, baseline numbers reproduce the canonical tau-sweep CSV exactly, so the pipeline is correct. The numbers themselves are not in dispute. What's in dispute is **what they measure**.

## Why the metric was misleading — the AUC/top-1 measure only temporal cheating

Look at `src/metrics.py:retrieval_auc_top1`. For each query `(b, t, c)`:

  * **Positive**: `h_full[b, t+1, c, :]` — the encoder latent one step ahead, **same sample, same channel**.
  * **Negatives**: `h_full[b, t-k, c, :]` for k ∈ {1, 2, 4, 8} — four latents from **the same sample, same channel, at recent past time positions**.

So the model only has to rank "the next time step" above "very recent past time steps" within a single window. **There are no cross-batch negatives, no cross-channel negatives, and no far-future negatives.** Any signal that distinguishes time positions inside a window is sufficient — including a pure position counter that ignores all content.

Six causal transformer-encoder layers can learn that counter trivially: depth × causal attention is enough to encode "I'm at position *t*" along a few dimensions of `h[t]`. The L2-normalized encoder output then makes the counter a unit-norm signal that nails the 4-negative retrieval task. The contrastive loss (with cross-batch negatives at training time) further rewards "position-encoded" features because cross-batch retrieval is also easier when the positional channel is sharp and content channels are similar between random samples.

**Predicted consequence**: the encoder learns position, not content. Latents look great on `retrieval_auc_top1` (which only tests temporal ordering), but contain little information a downstream q-head can use to actually forecast. We observed exactly this:

  * **Q-head training loss** (R9_E13 recipe, identical to the reference, only the backbone differs): plateaued at 0.28 vs the reference's 0.20 asymptote. The head can't extract enough forecasting content to drop further.
  * **GIFT-Eval triage MASE**: 1.596 — well above seasonal naive, well above the reference q-head on the older backbone.

The q-head failure is the test that adjudicated. The metric we relied on early couldn't see it.

## Why the per-batch metrics saturated at AUC=1.0 by step ~1k

Per-batch retrieval (same shape as the held-out N=50 eval but on the training batch) reached AUC = top-1 = 1.0 by ~1 k steps and stayed there for the rest of training. We interpreted this as "training has saturated" and stopped at step 25 600 (PR #262). Under the positional-shortcut hypothesis this was **early stopping on the shortcut convergence**, not on real-feature convergence: the model had already mastered the counting trick and was no longer getting much gradient pressure from cross-batch negatives (which the trick also solves) to learn content. We can't tell from the data we have whether much longer training would have eventually transitioned to content features.

## What we will fix in the follow-up

Three changes, intended to be applied together:

1. **`--encoder-dropkey p` regularizer** (already merged on `experiments` in #268). Per-step, per-encoder-layer fresh random mask that drops a fraction *p* of below-diagonal attention entries. Above-diagonal stays −∞ (causality preserved); diagonal stays 0 (self always allowed). The position-counting trick becomes lossy: at p=0.5 the count at position *t* becomes ~Binomial(*t*, 0.5) per layer, noise compounds across layers, and the marginal value of allocating capacity to a position counter drops. Forces content features to carry more of the contrastive signal.

2. **Augment `retrieval_auc_top1` to use time AND batch negatives.** Add cross-batch and cross-channel negatives so a positional counter alone can't ace the metric. A first cut: at each query `(b, t, c)`, draw negatives from `{h[b', t+1, c, :] : b' ≠ b}` (cross-batch positive-time) and `{h[b, t+1, c', :] : c' ≠ c}` (cross-channel) alongside the existing temporal ones. The metric should then be sensitive to whether the encoder learned content that distinguishes between samples and channels, not just time positions.

3. **Train longer.** Per-batch saturation is no longer a reliable early-stop signal under this analysis. With the dropkey regularizer and the augmented metric, we likely want 50 k+ steps to give the model time to transition past the shortcut budget into content features. Re-evaluating at 50 k will tell us whether the encoder block, *given a metric that catches the shortcut*, actually helps.

The dropkey flag and the auto-detect plumbing for `num_encoder_layers` are landed (PRs #262, #268). The augmented metric is the next infra change before re-running the experiment.

## Files in this directory

  * `PROGRESS.md` — in-progress notes from the run, including per-batch training trajectories and the q-head-vs-reference comparison plot. Kept verbatim as the live record; this REPORT supersedes its conclusions.
  * `QHEAD_PLAN.md` — recipe + step-budget rationale for the q-head training (R9_E13 mirror, halved to 30 k; we stopped early at 15.2 k per the "head ≤ backbone steps" principle).
  * `results/encoder_forecaster_metrics_multisample_n50.csv` — held-out contrastive metrics (see caveat above on what they measure).
  * `results/encoder_forecaster_metrics_persample_n50.csv` — per-window breakdown.
  * `results/gift_eval_triage/{all_results.csv, summary.txt}` — the triage that closed the experiment.
  * `plots/{progress, progress_linear, qhead_compare, qhead_compare_loglog}.png` — trajectory plots.
  * `scripts/` — `run_encoder_forecaster.sh`, `eval_one.py`, `plot_progress.py`, `plot_qhead_compare.py`, `run_qhead_training.sh`, `run_gift_eval_triage.sh`.

## Cost recap

  * Backbone training: 25 600 steps × ~165 steps/min ≈ 2.6 h on elisa GPU 1 (RTX 4090, bf16 autocast).
  * Held-out N=50 eval: ~7 min on the same GPU.
  * Q-head training: 15 200 steps × ~165 steps/min ≈ 1.5 h on the same GPU.
  * GIFT-Eval triage (11 configs): ~5 min.
  * Total: ~4.2 GPU-hours. No full GIFT-Eval (saved ~6 GPU-hours by the gate).
