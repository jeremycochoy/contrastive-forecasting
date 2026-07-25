# Small-model long-training sweep — 29 arms of #374 at d_model=64, B=64, up to 200k steps

*v1 — Wave D (all arms at 40k) + 11-cell 2L GM-MASE evaluation complete. Wave E (extension to 100k) in progress.*

## Question

Do the training-dynamics observations from #374 ([`reports/2026-07-10_split_pred_rep/split_pred_rep.md`](../2026-07-10_split_pred_rep/split_pred_rep.md)) at 12.5–50k steps on the 17M-parameter backbone hold when the model is shrunk to `d_model=64` (~1M params) and trained longer? And can we identify a knob that reduces the observed late-training latent drift?

## Design

Small backbone (`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64, seed=20260520`), trained on the `gift-pretrain-full-4096` dataset. Six loss-shape arms carried from #374 (arm 1 · arm 3 · arm 4 · arm 5 · arm 6 v2 · bimoco), each rerun under four ablations:

- **base**: original spec (all `τ=0.10`, `sigreg_e=1.0`, `cpc=1.0`)
- **tr1**: all `τ` raised to `1.0` (originally mis-specified as `τ_rep`-only on arm 1/3/bimoco; corrected mid-experiment; arm 5/6_v2 rep-only shape unaffected by the correction; arm 4 gets a new `arm4_tr1` at pooled `τ=1.0`)
- **nse**: `sigreg_embedding_weight=0.0` (SIGReg on `e_t` disabled)
- **ncpc**: `cpc_infonce_weight=0.0` (CPC auxiliary disabled)
- **combab**: all `τ=1.0` + `cpc=0` + `nse` (only for arm 1/3/4 where nse helped per stat test)

Total 29 arm configurations. Staged rollout brings every arm to step 40k first (Wave D). Wave E extends non-completed arms to step 100k. Wave F would extend to 200k.

## Results — Wave D (all arms at 40k)

### Latent stability (paired stat tests, N=6, base vs ablation)

Wilcoxon signed-rank on end-of-40k mean `1 − cos(h_prev, h_next)` and `1 − cos(e_prev, e_next)` per arm:

| Ablation | h_t (win/6) | h_t p | e_t (win/6) | e_t p | verdict |
|----------|-------------|-------|-------------|-------|---------|
| ncpc     | 6/6         | 0.016 | 5/6         | 0.031 | **reduces shaking** |
| nse      | 3/6         | 0.219 | 4/6         | 0.109 | mixed (helps arm 1/3/4, hurts arm 5/6v2/bimoco) |
| tr1      | 1/5         | 0.906 | 2/5         | 0.688 | does not reduce shaking |

Disabling CPC is the only single-axis fix that reliably reduces late-window encoder-latent drift on the fixed held-out batch.

### 2L GM-Relative MASE at 40k (11-cell subset)

Head trained 15k steps on the frozen 40k backbone; GIFT-Eval B4 full-97 configs. Candidate arms picked from three separate criteria: (a) lowest end-of-40k `1 − ff`, (b) trajectory still improving with least rebound, (c) lowest `h_t` movement. Plus researcher-added coverage for arm 3 (combab) and arm 4 (tr1, nse).

| Rank | Arm             | GM-Rel MASE | Notes |
|------|-----------------|-------------|-------|
| 1    | arm6_v2_combab  | **1.2025**  | winner by 0.12 |
| 2    | arm5_tr1        | 1.3254      | best single-fix |
| 3    | arm3_combab     | 1.4056      | |
| 4    | arm4_tr1        | 1.4414      | pooled-τ=1.0 helps arm 4 too |
| 5    | bimoco_combab   | 1.4420      | |
| 6    | arm3_tr1        | 1.4547      | |
| 7    | arm5_nse        | 1.4682      | |
| 8    | arm6_v2_tr1     | 1.4684      | |
| 9    | arm4_nse        | 1.4852      | |
| 10   | bimoco_tr1      | 1.4892      | |
| 11   | arm5_ncpc       | 1.5079      | worst tested |

Seed-noise band ≈ ±0.01 (per 2026-05-08 τ-sweep paired reruns, referenced in [LeJEPA-SIGReg-τ report annex F](../2026-06-21_lejepa_sigreg_tau098/lejepa_sigreg_tau098.md#f-seed-noise-band)). The 0.12 gap between #1 and #2 is ~12× seed-noise → real. Positions 3-11 are within a 0.10 band and mostly separated by 1-6× seed-noise.

**combab (all-τ=1 + cpc=0 + conditional nse) dominates the top of the ranking (positions 1, 3, 5).** arm5_tr1 is the best single-axis fix, landing ahead of every base and every non-arm6_v2 ablation.

## Figures

### 1. `1 − ff` per arm (log perplexity), 2×3 grid by variant, shared y

![cos_error per arm](plots/cos_error_per_arm.png)

### 2. Dimension usage `u_batchtime` per arm

![dim usage per arm](plots/dim_usage_per_arm.png)

### 3. Latent movement between adjacent checkpoints (variant × h_t/e_t grid)

![latent movement](plots/latent_movement_per_arm.png)

### 4. GM-Relative MASE bars — 11 arms at 40k, ±0.01 seed-noise error bars

![eval bars](plots/eval_2L_gm_mase_bars.png)

## Status

- Wave D: **DONE** for all 29 arms.
- 11-cell 2L GM-MASE eval at bb=40k: **DONE**.
- Wave E (bring all non-completed arms to 100k): **IN PROGRESS** (~35h).
- Wave F (extend to 200k): pending.
- Full report (with 100k trajectories + optional 100k GM-MASE cells): pending Wave E completion.

## Preliminary answers to the questions

From Wave D + eval only; extended answers require Wave E and beyond.

1. **arm 6 v2 / bimoco h_t movement at 40k**: still elevated for base; combab drops it dramatically for arm 6 v2 (`mv_h` from 0.635 → 0.425).
2. **arm 5 alignment plateau**: `1 − ff` at 40k ≈ 0.30 in base — trending toward but not yet at the #374 50k plateau of ≈0.4. Wave E will resolve.
3. **`τ_rep=1.0` (misspec) vs base**: 4/5 arms have LOWER `1 − ff`; but 4/5 arms have HIGHER latent movement. Softening the temperature helps the loss surface, not the encoder stability.
4. **`nse` (sigreg_e=0)**: helps arm 1/3/4 latent movement, hurts arm 5/6v2/bimoco. Loss-shape-dependent.
5. **`ncpc` (cpc=0)**: the clearest single-axis fix — reduces both `h_t` and `e_t` movement across 5-6 of 6 arms.
