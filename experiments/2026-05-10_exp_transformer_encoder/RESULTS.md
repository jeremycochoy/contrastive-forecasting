# Transformer encoder vs GRU+skip — RESULTS

## Goal / question

Does swapping the per-patch GRU+linear-skip encoder for a small (2-layer)
within-patch decoder-only transformer change the encoder's representation
quality, retrieval discrimination, dimension usage, and forecast-match
metrics? GRU+skip is the patch-level building block under every
backbone-beta and τ-sweep arm so far; never previously ablated.

## In-training trajectory

![training trajectory, log/log](plots/transformer_vs_baseline_loglog.png)

Smoothed rolling-mean window=200 steps. log/log axes. x ∈ [5k, 50k].

In-batch (smoothed, window=200) at step 50k: the transformer is below
the GRU on retrieval — AUC −0.005, Top-1 −0.011 — and above on
representation spread — U_temporal +0.007, U_batch +0.021. The
U_temporal gap is roughly flat across the window (+0.0068 at 25k,
+0.0066 at 50k); the U_batch gap is still widening at 50k
(+0.0165 → +0.0206).

## Per-sample distribution (held-out N=50)

![held-out per-sample variance](plots/held_out_persample_auc_top1.png)

Each dot is one of the 50 held-out batches. AUC and Top-1
distributions overlap almost completely — the means are within ~1
SEM_diff of each other (table below).

## What we learned

The values below come from evaluating, **for both arms, the
`_best_loss.pth` checkpoint within their 0..50,000-step training
window** (saved by the launcher as `_FINAL.pth` at end of training).
That is "best EMA-loss within 0..50k", not a specific timestep —
which matters because a single-step checkpoint would mix in
per-batch noise. Same EMA-loss tracking code, same step budget, same
dataset on both arms; full paths in [Details](#which-checkpoint-is-evaluated)
below.

| backbone (`*_FINAL.pth` = best EMA-loss within 0..50k) | encoder | R²_random | R²_naive | U_temporal | U_batch | AUC | Top-1 |
|---|---|---|---|---|---|---|---|
| `tau_sweep_0_10_50k_baseline` | GRU+skip | 0.6717 ± 0.0074 | 0.6225 ± 0.0093 | 0.0567 ± 0.0015 | 0.1178 ± 0.0026 | 0.9063 ± 0.0054 | 0.7645 ± 0.0099 |
| `transformer_encoder_tau_0_10_50k` | 2L within-patch transformer | 0.6606 ± 0.0073 | 0.6128 ± 0.0091 | **0.0649 ± 0.0017** | **0.1402 ± 0.0035** | 0.9053 ± 0.0054 | 0.7622 ± 0.0099 |
| Δ (transformer − GRU) |  | −0.0111 | −0.0097 | **+0.0082** | **+0.0224** | −0.0010 | −0.0023 |
| Δ / SEM_diff |  | −7.6 | −5.3 | **+25.6** | **+36.3** | −0.9 | −1.2 |

Mean ± stdev across the 50 held-out batches; SEM = stdev/√50 ≈ stdev/7.
SEM_diff = √(SEM_te² + SEM_b²).

- **Retrieval (AUC, Top-1).** Effectively tied. Both gaps are within
  ~1 SEM_diff and would easily flip on a different draw of batches.
- **Representation spread (U_temporal, U_batch).** Strong win for the
  transformer — 25-36 SEM_diff, well past noise. The transformer
  fills 14 % more of the latent space on the time axis and 19 % more
  on the batch axis.
- **Forecast-match (R²).** Significant loss for the transformer: −5
  SEM_diff on R²_naive, −7 SEM_diff on R²_random.

**Caveats.** The result confirms swapping out GRU+skip changes the
representation; it does not by itself prove the within-patch
transformer is the *best* replacement. *Hypothesis:* a 22-token
sequence at d=384 is overdimensioned, and most of the gain comes from
the per-scalar `Linear(1→H)` ingress rather than the within-patch
attention; a 0-layer ablation would test this. U_temporal and U_batch
absolute values had not plateaued at 50k for either arm.

## Details

### Architecture

Identical to `tau_sweep_0_10` except the per-patch encoder. Backbone:
6-layer × 384-D, 6 heads, ffn_mult=4, depthwise causal conv kernel=3.
Training: T_RAW=4096, B=256, AdamW lr=1e-3 (wd=0.1, betas=0.9/0.98),
RevEWMNorm span=128, freq_emb=3, seasonality_emb=3, mixup_p=0.3,
τ=0.10 fixed, loss=`cosine_similarity_batch`, 50,000 steps from scratch.

| | baseline | this run |
|---|---|---|
| per-patch encoder | bidir GRU on W'=22 scalar sequence + `Linear(W', H)` skip + LayerNorm | `Linear(1, H=384)` per scalar + 2 decoder-only causal transformer layers attending across the 22 tokens of one patch + last-token pool |

### Which checkpoint is evaluated

Both arms write `_best_loss.pth` whenever the EMA of the per-step loss
reaches a new minimum, and the launcher copies it to `_FINAL.pth` at
end of training. Eval loads `FINAL.pth` — so the numbers describe the
**best-loss checkpoint within each run's 0..50k window**, not the
last step. Same step budget, dataset, and EMA-tracking code on both
arms.

| arm | window | checkpoint |
|---|---|---|
| baseline (GRU+skip) | 0..50,000 (15k from-scratch + 35k continuation) | `sync_tau_sweep_0_10_50k/checkpoints/tau_sweep_0_10_50k_r2_FINAL.pth` |
| this run | 0..50,000 from-scratch | `sync_transformer_encoder/checkpoints/transformer_encoder_tau_0_10_50k_FINAL.pth` |

### Held-out evaluation

50 disjoint B=256 batches from
`jeremycochoy/gift-pretrain-full-4096/small_v1` at skip_rows
50,000,000 + i × (42,740,000 / 50) for i ∈ [0, 50). Run by
`scripts/eval_held_out.py` (fp32, `model.eval()`).

### Metrics

| metric | what it measures |
|---|---|
| `R²_random` | improvement over pairing forecasts with arbitrary held-out targets |
| `R²_naive` | improvement over a "no-change" baseline that copies the previous encoder output as the prediction |
| `U_temporal` | fraction of the 384-D space one series spans across its time positions (averaged over batch) |
| `U_batch` | fraction of the 384-D space different series at the same time position span (averaged over time) |
| `AUC` | per-query fraction of past-window negatives the positive ranks above |
| `Top-1` | fraction of queries where the positive beats every past-window negative simultaneously |

A new training-time CSV column `gap_ratio = (1−ff)/(1−fp)` was added on
this run (forecast-vs-future cosine gap normalised by past-vs-future
gap; lower is better) — per-step diagnostic only, not used for
held-out comparison.
