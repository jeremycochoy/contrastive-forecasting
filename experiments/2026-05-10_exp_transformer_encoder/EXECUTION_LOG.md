# Execution log — transformer-encoder

## 2026-05-10 — kickoff

**Goal.** Replace the GRU+linear-skip patch encoder with a small (4-layer)
decoder-only causal transformer; otherwise mirror the τ=0.10 baseline at
[experiments/2026-05-08_exp_tau_sweep/RESULTS.md](../2026-05-08_exp_tau_sweep/RESULTS.md)
exactly so the comparison isolates the encoder change.

**Architecture diff vs τ=0.10 baseline.**
- old: per-patch `GRUEncoder` = bidir-GRU(input=1, hidden=128, 2 layers) on
  W=16 timesteps + `Linear(W', H)` skip + LayerNorm.
- new: per-patch `Linear(W', H=384)` upscale, then 4 decoder-only causal
  transformer layers attending across **T patches** (not within a patch),
  same `DecoderOnlyTransformerLayer` recipe as the backbone (norm-first,
  depthwise causal conv kernel=3, GeLU, no bias, dropout=0).
- "Highway" at init: with norm-first residual chain, encoder output ≈
  `Linear(patch)` when attention/FFN weights are small at init — same
  shape of inductive bias the GRU's `linear_skipping` provided.

**Baseline reference.** τ=0.10 from-scratch 50k from
[results/tau_sweep_metrics_multisample.csv](../2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_multisample.csv):

| metric        | τ=0.10 baseline (GRU+skip) |
|---------------|----------------------------|
| AUC           | 0.8993 ± 0.0053            |
| Top-1         | 0.7535 ± 0.0098            |
| R²_random     | 0.6683 ± 0.0074            |
| R²_naive      | 0.6153 ± 0.0094            |
| U_temporal    | 0.0512 ± 0.0012            |
| U_batch       | 0.1019 ± 0.0015            |

**New CSV column.** `gap_ratio = (1 - ff) / (1 - fp)` — added right after
the existing `gap = ff - fp`. Lower-is-better, complements the standard
gap by being scale-aware: a forecast that nearly matches the future
(ff → 1) drives the numerator to 0 even if the past is also slightly
correlated, capturing alignment quality rather than absolute margin.

**Run config (matches τ=0.10 baseline arm exactly except encoder):**
- backbone: 6 layers, d=384, 6 heads (`Tiny`)
- encoder: 4 layers, d=384, 6 heads, ffn_mult=3.0, depthwise_conv=3
- 50,000 steps, B=256, T_raw=4096, C=1, mix_ratio=0.0 (HF-only)
- AdamW lr=1e-3, weight_decay=0.1, betas=(0.9, 0.98)
- RevEWMNorm span=128, freq_emb_dim=3, seasonality_emb_dim=3, mixup_p=0.3
- τ=0.10 fixed, loss=cosine_similarity_batch
- HF dataset jeremycochoy/gift-pretrain-full-4096 / small_v1

**Compute.** elisa GPU 1 (RTX 4090, full memory free at launch). Encoder
adds ~5.9M params on top of the ~9M backbone for ~15M total — about 1.7×
the GRU baseline params, so per-step compute roughly scales the same.

**Sync target.** Local on elisa. Save dir
`/home/jupyter/contrastive-forecasting/sync_transformer_encoder/checkpoints/`
(MAIN checkout, per CLAUDE.md rule 4 — never under a worktree). Code
changes live on branch `transformer-encoder-experiment` (worktree
`/home/jupyter/cf-transformer-encoder/`).
