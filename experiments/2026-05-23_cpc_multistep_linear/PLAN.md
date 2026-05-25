# #316 — CPC-style multi-step linear forecast on β (k=12)

## Question
β (the (B) recipe + AdamW β2=0.98) reaches full-97 GM-MASE **1.3272**, still
+2.7% over the encoder-forecaster champion **v11c (1.292)**. β's forecaster is
CPC specialised to **k=1** (predict only the next latent), and on the (B) line
more contrastive training does not translate into better GM-MASE — the latent
can satisfy "predict the next latent + separate" while drifting from
forecasting.

Does replacing β's transformer forecaster with a CPC-style **multi-step linear
forecast** (predict the next **k=12** latents with linear heads — van den Oord
et al. 2018) close the gap to v11c and remove the more-training-doesn't-help
decoupling?

## Change (one axis vs β)
- Forecaster: the 1L transformer (d=128 bottleneck) is **replaced** by K=12
  linear heads W_k: H→H, each predicting the encoder latent k steps ahead
  (h_{t+k}) from the causal 6L-encoder output h_t. No attention, no bottleneck.
- Loss: `cpc_multistep` — for each k, InfoNCE positive cos(W_k h_t, h_{t+k})
  with negatives = encoder latents at other batch rows (matched offset) and
  other in-sequence times; averaged over k. Normalized InfoNCE
  (--pos-in-denominator), τ=0.10. Cosine on L2-normalised latents.
- Everything else = β: 6L causal encoder, dropkey 0.70 (shared heads+layers),
  β2=0.98, lr1e-3, 50k, global batch 256, RevEWMNorm span 128, freq+seasonality
  emb, mixup 0.3, seed-controlled.
- Precision: **fp32** (fp16 at β's lr1e-3 diverges for the multi-step objective;
  fp32 is stable and matches v11c — see EXECUTION_LOG.md).

The k=1 head doubles as the single forecaster latent for downstream, so the
q-head/GIFT-Eval protocol is identical to β / #315 and the numbers line up.

## Arms
- Two seeds (20260520 = β's seed; 20260523) for a variance estimate — the
  single-seed ±0.02 caveat is the standing limitation of the β/v11c line.

## Evaluation (downstream kept comparable to #315)
Each backbone, with **both** a small (2L causal transformer) q-head and a **6L**
q-head: q-head 30k, forecast-len 16, reconstruction=forecaster, e_then_f input;
report **triage(11)** and **full(97)** GM-Relative MASE.

## Targets
- Beat β (1.3272); ideally reach/beat v11c (1.292).
- Beyond the number: does GM-MASE keep improving with training steps (vs flat /
  regressing on β)? — steps-curve of GM-MASE on periodic checkpoints, plus the
  training curves (gap / R²_naive / dim-usage).

## Deliverables (per REPORT_STANDARD)
`RESULTS.md`: GM-MASE summary bars (CPC vs β / v11c), per-domain radar vs v11c,
training curves, GM-MASE-vs-step. Sub-agent review before merge.
