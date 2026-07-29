# SIGReg on e_t is the only single term whose end-of-100k h_t drift stays below 0.1; the other seven terms end between 0.29 and 0.86

Trained alone for 100k steps on the small backbone, only `sigreg_e` brings end-of-100k `h_t` drift below 0.1 (0.05213); the remaining seven terms end between 0.2884 (`rep_moco`) and 0.8605 (`pred`). `pred` climbs across the second half of training and `cpc` oscillates over a wide range (0.2675–1.0515) without settling.

## Headline figure

![Per-arm h_t drift trajectory, one value per adjacent 5k-step checkpoint pair, drift_cos_h in [0,2] on a linear y-axis, log training step on x. One panel per arm.](plots/latent_movement_per_arm.png)

At step 95k→100k the arms rank, lowest drift first: `sigreg_e` (0.05213), `rep_moco` (0.28837), `align` (0.31101), `sigreg_h` (0.37894), `cpc` (0.43988), `pred_moco` (0.45757), `rep` (0.60357), `pred` (0.86053).

## Definitions

- `ff` = mean `cos(f̂, h_{t+1})` between the forecaster's next-step prediction `f̂` and the encoder's next-step latent `h_{t+1}` (unit-normalised on the sphere). `1 − ff` is a distance in [0, 2]; smaller = better forecast.
- `h_t`, `e_t`: the encoder-output latent and the patch-embedding latent respectively, shape `[B, T, C, H]`.
- **InfoNCE**: the standard cross-entropy contrastive loss `−log(exp(pos/τ) / Σ_j exp(neg_j/τ))` used throughout, with temperature `τ`.
- **SIGReg**: the LeJEPA-style spectral regulariser that pushes each latent's Gram matrix toward a Gaussian off-diagonal; `--sigreg-embedding-weight` (`sigreg_e`) applies it to `e_t`, `--sigreg-encoding-weight` to `h_t`.
- **CPC** (`--cpc-infonce-weight`, "cpc"): the CPC-InfoNCE auxiliary of van den Oord et al. 2018 predicting `e_{t+1}` from a bilinear projection of `h_t`.
- **MoCo**: negatives drawn from an EMA teacher (momentum contrast).
- `u_batchtime` = `1/(d · off-diag Gram mean)` over `(B×T)` samples of the specified latent, clamped to `[0, 1]`. 1 = every H dim carries independent info; low = collapsed onto a subspace. Exactly what SIGReg regularises.
- **Latent drift** at checkpoint pair `(step_i, step_j)` on a fixed held-out batch (`torch.manual_seed(20260722)`, `B=8` ARMA-synthetic): `mean_{b,t,c} 1 − cos(h_t(model_j), h_t(model_i))`. Range `[0, 2]`; low = the mapping learned by the model on this fixed input hasn't rotated between the two checkpoints. Referred to below as "drift".
- **L_pred** — batch-pooled f-anchored InfoNCE from `cosine_similarity_batch_split_pred_rep`: numerator `cos(f_t, h_{t+1})/τ`, denominator LSE over the f-anchored families (adjacent `f_t↔f_{t+1}` + cross-batch `f_t↔h'_{t+1}`).
- **L_rep** — h-anchored logsumexp from the split shape's `L_rep`: pooled LSE of the three h-anchored families (cross-channel `h↔h`, within-series all-time `h_t↔h_l`, cross-series all-time `h_t↔h_{b',l}`). No positive.
- **L_align** — BYOL/SimSiam alignment: `2 − 2·cos(f_t, sg(h_{t+1}))` (stop-grad on the encoder side).

## Backbone

`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, T=1024, C=1, rev_norm=ewma(span=128), encoder_type=gru, batch_size=64, lr=1e-3, wd=0.1, adam_beta1=0.9, adam_beta2=0.98, seed=20260520`, dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`. Every arm trains for 100k steps with a checkpoint every 5k (20 checkpoints per arm). T=1024 is the HF-stream cap on the experiments branch; #379's `--t-raw 4096` flag is downgraded by the loader and its runs are effectively also T=1024, so the two studies compare like-for-like.

## Arms

Each arm activates exactly one loss term; every other term (the two SIGReg regularisers and the CPC auxiliary included) is at weight 0. All eight arms completed the full 100k steps (20 drift intervals each).

| Arm       | Only-term active                                                       |
|-----------|------------------------------------------------------------------------|
| pred      | `L_pred` from the split shape (`--pred-loss-weight 1 --rep-loss-weight 0`) |
| rep       | `L_rep` from the split shape (`--pred-loss-weight 0 --rep-loss-weight 1`)  |
| align     | `L_align` standalone (`--no-main-contrastive-loss --align-loss-weight 1`)  |
| pred_moco | `L_pred` with `--moco-negatives` (EMA-teacher cross-batch keys)         |
| rep_moco  | `L_rep` with `--moco-rep-keys` (EMA-teacher h-anchored keys)            |
| sigreg_e  | SIGReg on `e_t` only (`--sigreg-embedding-weight 1.0`)                 |
| sigreg_h  | SIGReg on `h_t` only (`--sigreg-encoding-weight 1.0`)                  |
| cpc       | CPC-InfoNCE auxiliary only (`--cpc-infonce-weight 1.0`)                 |

## Latent-drift results

End-of-100k `h_t` drift (`drift_cos_h`, step 95k→100k) per arm, ranked lowest first. Values from `results/latent_movement_per_arm.csv`.

| Rank | Arm       | End-of-100k h_t drift | Δ from #1 |
|------|-----------|-----------------------|-----------|
| 1    | sigreg_e  | 0.05213               | —         |
| 2    | rep_moco  | 0.28837               | +0.2362   |
| 3    | align     | 0.31101               | +0.2589   |
| 4    | sigreg_h  | 0.37894               | +0.3268   |
| 5    | cpc       | 0.43988               | +0.3877   |
| 6    | pred_moco | 0.45757               | +0.4054   |
| 7    | rep       | 0.60357               | +0.5514   |
| 8    | pred      | 0.86053               | +0.8084   |

`sigreg_e` reaches 0.15901 by step 10k and stays below 0.066 from step 20k onward. `pred` sits above 0.63 for every interval after step 40k. `cpc` swings between 0.26753 (60k→65k) and 1.05153 (70k→75k) with no downward trend.

## Supporting figures

![1 − ff per arm across training step, one panel per arm, `1 − ff` in [0,2] on the y-axis, log training step on x.](plots/cos_error_per_arm.png)

The four arms that carry no `L_pred`/`L_rep` positive (`sigreg_e`, `sigreg_h`, `cpc`, `align`) do not optimise `1 − ff`; among the four arms that do, `pred_moco` and `rep_moco` reach lower `1 − ff` than `pred` and `rep`.

![u_batchtime per arm, `h_t` solid and `e_t` dashed, u_batchtime in [0,1] on the y-axis, log training step on x.](plots/dim_usage_per_arm.png)

`u_batchtime_e` (the dashed `e_t` line) is logged only for `sigreg_e` and `sigreg_h` and is empty for the other six arms; `sigreg_e` is the only arm whose `h_t` `u_batchtime` stays near 1 across training.

## Annex

### A. Per-arm loss recipes with CLI flags

| Arm       | Active term | CLI flags                                                            |
|-----------|-------------|---------------------------------------------------------------------|
| pred      | `L_pred`    | `--pred-loss-weight 1 --rep-loss-weight 0`                           |
| rep       | `L_rep`     | `--pred-loss-weight 0 --rep-loss-weight 1`                           |
| align     | `L_align`   | `--no-main-contrastive-loss --align-loss-weight 1`                  |
| pred_moco | `L_pred`    | `--pred-loss-weight 1 --rep-loss-weight 0 --moco-negatives`          |
| rep_moco  | `L_rep`     | `--pred-loss-weight 0 --rep-loss-weight 1 --moco-rep-keys`           |
| sigreg_e  | SIGReg(e_t) | `--no-main-contrastive-loss --sigreg-embedding-weight 1.0`           |
| sigreg_h  | SIGReg(h_t) | `--no-main-contrastive-loss --sigreg-encoding-weight 1.0`            |
| cpc       | CPC-InfoNCE | `--no-main-contrastive-loss --cpc-infonce-weight 1.0`               |

Terms that degenerate by construction under single-term training: `align` reaches loss=0 / ff=1 (`f≡h` collapse), `sigreg_e`/`sigreg_h` loss→0, `cpc` loss→0, and `rep` sits at loss≈13.25 with ff≈0 (h-anchored logsumexp with no positive). These are the designed outcomes of isolating one term; the comparison in this report is on drift, not on loss magnitude.

### B. Code changes shipped in the scaffold PR

- Two new CLI flags on `experiments/2026-04-27_freq-embedding/scripts/train.py`: `--pred-loss-weight`, `--rep-loss-weight` (both default 1.0). Threaded into `train_configuration` and consumed inside `contrastive_latent_loss`'s `cosine_similarity_batch_split_pred_rep` branch as `loss = w_pred·L_pred + w_rep·L_rep`. Default 1.0/1.0 is byte-for-byte the historical split objective.
- Loosened `--no-main-contrastive-loss` guard: the flag is now accepted with any SIGReg term on, so the `sigreg_e` / `sigreg_h` arms can drop the contrastive forward entirely while training on the SIGReg regulariser alone.

### C. Compute and provenance

Vast.ai run under label `cf-382-loss-term-isolation`, single 1×RTX 4090 instance (vast id 46124214), aggregate wall-clock ~20h42m (aggregate provenance 4.27; per-arm timestamps not retained). Per-arm checkpoints and training logs are mirrored on elisa under `/home/jupyter/checkpoints_backup/cf-382/runs_vast/<arm>/`.
