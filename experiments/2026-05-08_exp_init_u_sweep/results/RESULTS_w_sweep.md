# Init-time U on `o_lat` vs patch-encoder per-token input dim

**Setup:** `ConfigurableModel` (encoder=`gru`, H=384, C=1, rev_norm=ewma span=128, num_layers=6).
B=32. T_raw=4096 for W∈{16,32,64,128}; T_raw=4032 for W=192 (since 4096%192≠0). 3 model seeds (42,43,44). CPU. No training. U is `dim_usage` on `o_lat = transformer.encoder_output` (post-LN, pre-6L-transformer). Hypothesis: U at init scales as ≈ (W+freq_emb_dim+seas_emb_dim)/H, where H=384.

## Summary table (sorted by sub-experiment, then W or freq_emb_dim)

| experiment | W | freq | seas | gru_zero | u_b_o (mean ± std) | u_t_o (mean ± std) | predicted ≈ (W+f+s)/H |
|---|---:|---:|---:|---:|---|---|---:|
| A_vary_W | 16 | 3 | 3 | 0 | 0.0427 ± 0.0002 | 0.0426 ± 0.0001 | 0.0573 |
| A_vary_W | 32 | 3 | 3 | 0 | 0.0801 ± 0.0004 | 0.0798 ± 0.0005 | 0.0990 |
| A_vary_W | 64 | 3 | 3 | 0 | 0.1449 ± 0.0005 | 0.1452 ± 0.0006 | 0.1823 |
| A_vary_W | 128 | 3 | 3 | 0 | 0.2506 ± 0.0031 | 0.2490 ± 0.0019 | 0.3490 |
| A_vary_W | 192 | 3 | 3 | 0 | 0.3316 ± 0.0004 | 0.3336 ± 0.0069 | 0.5156 |
| B_vary_freq_emb | 16 | 3 | 0 | 0 | 0.0426 ± 0.0003 | 0.0425 ± 0.0001 | 0.0495 |
| B_vary_freq_emb | 16 | 16 | 0 | 0 | 0.0425 ± 0.0002 | 0.0424 ± 0.0003 | 0.0833 |
| B_vary_freq_emb | 16 | 64 | 0 | 0 | 0.0412 ± 0.0004 | 0.0410 ± 0.0002 | 0.2083 |
| B_vary_freq_emb | 16 | 192 | 0 | 0 | 0.0364 ± 0.0003 | 0.0358 ± 0.0002 | 0.5417 |
| C_gru_zero | 16 | 3 | 3 | 1 | 0.0428 ± 0.0003 | 0.0428 ± 0.0001 | 0.0573 |

## Findings

Sub-experiment A (vary W with freq=3, seas=3): U_b_o went from 0.0427 at W=16 to 0.3316 at W=192, increasing monotonically with W. **Hypothesis-consistent** with the predicted (W+f+s)/H ceiling at the order-of-magnitude level (predicted 0.057 → 0.516; observed 0.043 → 0.332), with measured U sitting below the predicted ceiling at every W.

Sub-experiment B (vary freq_emb_dim with W=16, seas=0): U_b_o was 0.0426 at freq=3, 0.0425 at freq=16, 0.0412 at freq=64, and 0.0364 at freq=192 — non-increasing across the sweep, with the freq=3 vs freq=16 step within ±1σ and clear drops at freq=64 and freq=192. **Hypothesis-inconsistent**: the rank-ceiling story predicts U would rise from ≈0.050 to ≈0.542 as the per-token input dim grows from 19 to 208, but observed U was flat-to-decreasing. Concatenated freq embeddings did not contribute usable rank to `o_lat` at init.

Sub-experiment C (gru_zero, W=16, freq=3, seas=3): U_b_o = 0.0428 ± 0.0003, indistinguishable (within ±2σ) from the matched A_vary_W W=16 row at 0.0427 ± 0.0002. The GRU branch contributes effectively no additional rank to `o_lat` at default init; the `skip` linear path is the dominant rank source.
