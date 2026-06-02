# ARMA × correlation — design notes

## Motivation

Combine the two existing experiments. Each batch sample now has **both**
per-channel ARMA(p,q) dynamics and cross-channel correlation. The contrastive
backbone gets a real reason to use cross-channel information (channels share
correlated innovations) on top of per-channel temporal structure (ARMA filters).

This is the first run where the `Simple_channel_mixing_module` is actually in
the loss path: the train script calls `model.forward(x)` so the channel-mixing
Kronecker matrices are gradient-active, instead of being dead weight as they
were in the contrastive-arma and contrastive-correlation V1–V5 runs.

After contrastive pretraining we freeze the backbone and train **two** small
recovery heads on top of the same frozen latent:

- **ARMA head** — same recipe as the ARMA experiment (GRU h=128, 2 layers,
  bidirectional, MSE) operating per-channel on `h`. Output: 4 AR + 4 MA
  coefficients per channel.
- **Correlation head** — same topology as the V5 GRU head, operating on
  `h_hat` (channel-mixed latent). Output: 6 unique upper-triangle off-diagonal
  pairs.

Goal: recover both kinds of structure from the same backbone with quality
comparable to the dedicated ARMA and V5 experiments.

## Data

For each batch sample b ∈ [0, B):

1. Sample 4×4 correlation matrix `C_b` with the **uniform-per-pair** sampler
   (off-diagonals ~ U[0, 1] with PSD rejection).
2. Cholesky factor `L_b = chol(C_b)`.
3. Sample iid 4-dim Gaussian innovations `z_t ~ N(0, I_4)` for `t = 1..T`.
4. Apply Cholesky: `ε_t = L_b z_t` → cov(ε) = C_b.
5. Sample 4 random ARMA(p, q) processes with p, q ∈ {1..4}, AR/MA coefficients
   shrunk to keep the filter stable.
6. Apply each channel's ARMA filter to its corresponding innovation channel:
   `y_{t,k} = lfilter(ma_k, ar_k, ε_{·,k})`.
7. z-score each (b, k) so each channel has zero mean and unit variance.

Output: `y ∈ R^{B × T × 4}`, `C ∈ R^{B × 4 × 4}`, list of `(ar_poly, ma_poly)`
per `(b, k)`.

## Backbone

`ConfigurableModel`, 12 layers, H=1024, GRU encoder — same architecture as the
ARMA and corrV4 backbones. The training loop calls `model.forward(x)` so the
channel-mixing Kronecker module is active in the contrastive loss path:

- `f_lat`, the forecaster path, is `channel_mixed(transformer_f)` →
  `[B, T, C, H]`.
- `o_lat`, the encoder/transformer original, is per-channel → `[B, T, C, H]`.
- Loss `cosine_similarity_batch_no_time_neg` with C=4 — both cross-channel
  and cross-batch negatives are now meaningful.

LR: constant 7e-5, AdamW, batch 16, ~150–200 k steps. No clip, no schedule.

## Heads

Both heads operate on the frozen backbone latents.

### ARMA head (existing recipe)

`GRURecoveryHead(H=1024, hidden_dim=128, num_arma_params=4, num_gru_layers=2,
bidirectional=True)` from `src/recovery.py`. Input `h.permute(0,2,1,3).reshape(B*C, T, H)`,
output AR/MA per channel. MSE loss + improvement-vs-zero metric.

### Correlation head (V5 recipe)

`GRUCorrelationHead(H=1024, hidden_dim=128, num_gru_layers=2)` from
`experiments/2026-05-04_contrastive-correlation/scripts/joint_channel_model.py`. Input is `h_hat`
collapsed to a single sequence per sample by flattening the C dim and
projecting back to H — i.e. we do the joint projection at the head's input
boundary (since the backbone's transformer ran per-channel). Output: 6 pairs.

## Comparison targets

- ARMA head r vs the contrastive-arma V2 result (mean Pearson r ≈ 0.94 across
  8 coefficients).
- Correlation head r vs the V5 result (mean Pearson r 0.962).

If the joint backbone matches both, that's the proof that the channel-mixing
module + the ARMA-style data really earn their keep.

## Files

- `data.py` — generator + helpers.
- `train_backbone.py` — contrastive pretraining (channel-mixing in loss).
- `train_arma_head.py` — ARMA recovery head.
- `train_correlation_head.py` — correlation recovery head.
- `evaluate.py` — figure generation, per-pair scatter, per-coefficient ARMA
  scatter, baselines.
- `scripts/run.sh` — full pipeline.
