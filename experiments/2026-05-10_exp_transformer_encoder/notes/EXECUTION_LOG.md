# Execution log — transformer-encoder

## 2026-05-10 — kickoff

**Goal.** Replace the GRU+linear-skip patch encoder with a small
decoder-only causal transformer; otherwise mirror the τ=0.10 baseline at
[experiments/2026-05-08_exp_tau_sweep/exp_tau_sweep.md](../../2026-05-08_exp_tau_sweep/exp_tau_sweep.md)
so the comparison isolates the encoder change.

**Architecture diff vs τ=0.10 baseline (as shipped).**
- old: per-patch `GRUEncoder` = bidir-GRU(input=1, hidden=128, 2 layers)
  consumes the W'=22 scalars of one patch as a sequence + `Linear(W', H)`
  skip + LayerNorm.
- new: per-patch transformer — each of the W'=22 scalars in one patch is
  upscaled by a single shared `Linear(1 → H=384)` to a token, then **2**
  decoder-only causal transformer layers attend OVER THOSE 22 TOKENS
  (within the patch). Patch summary = last token. Same
  `DecoderOnlyTransformerLayer` recipe as the backbone (norm-first,
  depthwise causal conv kernel=3, GeLU, no bias, dropout=0, **ffn_mult=4**).
  Patches are still encoded independently in parallel along (B, T, C)
  exactly like the GRU.
- "Highway" at init: with norm-first residual chain, attention/FFN are
  near-zero at init, so the layer stack is approximately identity over
  `Linear(1, H)` applied to the last scalar — same shape of inductive
  bias the GRU's `linear_skipping` provided through its skip path.

**Note on Run #1 (stopped at step ~1400).** The first launch had the
encoder attending across T (one token per patch, attending over 256
patches) — wrong axis. Killed and restarted with the within-patch
design. Old checkpoints discarded (architecture incompatible).

**Memory adaptations to fit elisa's 24 GB RTX 4090** (the τ-sweep ran on
vast.ai with bigger GPUs):
- Cross-batch broadcast in `cosine_similarity_batch` allocated
  [B, B, T-1, C, H] ~ 25 GB at B=256 fp32. Replaced with an equivalent
  batched matmul in `src/loss.py` (numerically identical, max diff 6e-8;
  peak drops to ~67 MB). Even the GRU baseline at B=256 OOMs without
  this change on elisa.
- Encoder reduced from 4 → 2 within-patch transformer layers (per user
  feedback) — keeps the same ffn_mult=4 backbone-matched recipe.
- Encoder chunked along the (B*T*C) axis at chunk_size=16384 — N=65k
  length-22 sequences in one go also tripped the SDPA kernel's
  invalid-config error, so chunking is a correctness fix as well as a
  memory fix.
- Activation checkpointing on encoder layers — per-chunk peak ~10 GB
  without it; with it ~3 GB.
- Forward + loss in bf16 autocast — ~2× tensor-core speedup; safe for
  τ=0.10 (`cos/τ ∈ [-10, 10]`, bf16 keeps fp32's exponent range).
- Result: peak ~10 GiB at B=256, ~1.1 s/step (≈ 0.9 sps). 50k steps in
  ~15 h.

**Baseline reference.** τ=0.10 from-scratch 50k from
[results/tau_sweep_metrics_multisample.csv](../../2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_multisample.csv):

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

**Run config (as shipped — matches τ=0.10 baseline arm except encoder):**
- backbone: 6 layers, d=384, 6 heads (`Tiny`)
- encoder: 2 layers, d=384, 6 heads, ffn_mult=4.0, depthwise_conv=3,
  chunk_size=16384, activation checkpointing on
- 50,000 steps, B=256, T_raw=4096, C=1, mix_ratio=0.0 (HF-only),
  bf16 autocast on the forward + loss
- AdamW lr=1e-3, weight_decay=0.1, betas=(0.9, 0.98)
- RevEWMNorm span=128, freq_emb_dim=3, seasonality_emb_dim=3, mixup_p=0.3
- τ=0.10 fixed, loss=cosine_similarity_batch
- HF dataset jeremycochoy/gift-pretrain-full-4096 / small_v1

**Total params.** 14.5 M (vs ~11.4 M for the GRU baseline; encoder
adds ~3 M on top of the 6L backbone).

**150k continuation.** A resume-to-150k launch was started after 50k
completed, then **killed at step ~51,700** — the held-out N=50 eval at
50k FINAL had already shown retrieval was tied with the GRU baseline,
so the long continuation no longer added information. The artifact of
record is `transformer_encoder_tau_0_10_50k_FINAL.pth`. The
`scripts/run_resume_to_150k.sh` launcher is left in-tree as a recipe
for future arms but produced no published checkpoint.

**15k-baseline eval on file.** `results/tau_sweep_0_10_baseline_metrics_persample_n50.csv`
is the held-out N=50 eval against the **15k** GRU baseline FINAL.pth
(`sync_tau_sweep/checkpoints/tau_sweep_0_10_FINAL.pth`). Used for an
early read that turned out to be confounded by training duration; the
headline RESULTS.md table uses the 50k baseline
(`sync_tau_sweep_0_10_50k/.../tau_sweep_0_10_50k_r2_FINAL.pth`)
instead. The 15k file is kept as supplementary data only.

**Sync target.** Local on elisa. Save dir
`/home/jupyter/contrastive-forecasting/sync_transformer_encoder/checkpoints/`
(MAIN checkout, per CLAUDE.md rule 4 — never under a worktree). Code
changes live on branch `transformer-encoder-experiment` (worktree
`/home/jupyter/cf-transformer-encoder/`).
