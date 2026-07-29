# h_t drift trends down for six of eight single-term arms, flat for CPC, up for `L_pred` alone

Trained alone for 100k steps on the small backbone, `h_t` drift on a fixed held-out batch trends **down** for six of the eight single-term arms (`L_align`, `SIGReg on h_t`, `L_rep`, `SIGReg on e_t`, `L_rep_moco`, `L_pred_moco`), stays **flat** for the `CPC` arm, and trends **up** for `L_pred` alone. Among the six down-trending arms, `SIGReg on e_t` reaches the lowest late-training drift.

## Headline figure

![Per-arm h_t drift trajectory, one value per adjacent 5k-step checkpoint pair, drift_cos_h in [0,2] on a linear y-axis, log training step on x. One panel per arm; red dotted line at drift = 0.1; black dashed line at drift = 1 for scale reference.](plots/latent_movement_per_arm.png)

Drift trend per arm, grouped by direction. Early drift is the mean over steps 5k–25k, late drift the mean over steps 80k–100k, slope the log-linear fit of `drift_cos_h` against `log10(step)` over the full 5k–100k range. Within each group, arms are ordered by decreasing slope magnitude. Values from `results/latent_movement_per_arm.csv`.

| Trend | Arm            | Early drift | Late drift | Slope    |
|-------|----------------|-------------|------------|----------|
| down  | `L_align`      | 1.065       | 0.2653     | −0.7986  |
| down  | `SIGReg on h_t`| 0.5878      | 0.3415     | −0.3632  |
| down  | `L_rep`        | 0.8143      | 0.4090     | −0.3110  |
| down  | `SIGReg on e_t`| 0.1977      | 0.03670    | −0.2683  |
| down  | `L_rep_moco`   | 0.4400      | 0.2531     | −0.2588  |
| down  | `L_pred_moco`  | 0.5400      | 0.4198     | −0.2155  |
| flat  | `CPC`          | 0.7272      | 0.6393     | −0.1325  |
| up    | `L_pred`       | 0.6461      | 0.7837     | +0.07480 |

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

`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, T=1024, C=1, rev_norm=ewma(span=128), encoder_type=gru, batch_size=64, lr=1e-3, wd=0.1, adam_beta1=0.9, adam_beta2=0.98, seed=20260520`, dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`. Every arm trains for 100k steps with a checkpoint every 5k (20 checkpoints per arm).

## Arms

Each arm activates exactly one loss term; every other term (the two SIGReg regularisers and the CPC auxiliary included) is at weight 0. All eight arms completed the full 100k steps (20 drift intervals each).

| Arm       | Only-term active                                          |
|-----------|-----------------------------------------------------------|
| pred      | `L_pred` from the split shape                             |
| rep       | `L_rep` from the split shape                              |
| align     | `L_align` standalone                                      |
| pred_moco | `L_pred` with EMA-teacher cross-batch keys               |
| rep_moco  | `L_rep` with EMA-teacher h-anchored keys                 |
| sigreg_e  | SIGReg on `e_t` only                                      |
| sigreg_h  | SIGReg on `h_t` only                                      |
| cpc       | CPC-InfoNCE auxiliary only                               |

## Supporting figures

![1 − ff per arm across training step, one panel per arm, `1 − ff` in [0,2] on the y-axis, log training step on x.](plots/cos_error_per_arm.png)

The four arms that carry no `L_pred`/`L_rep` positive (`sigreg_e`, `sigreg_h`, `cpc`, `align`) do not optimise `1 − ff`; among the four arms that do, the non-MoCo variants reach lower end-of-100k `1 − ff` (`pred` 0.02572, `rep` 1.01788) than their MoCo counterparts (`pred_moco` 0.24110, `rep_moco` 1.02540).

![u_batchtime per arm, `h_t` solid and `e_t` dashed, u_batchtime in [0,1] on the y-axis, log training step on x.](plots/dim_usage_per_arm.png)

`u_batchtime_e` (the dashed `e_t` line) is logged only for `sigreg_e` and `sigreg_h` and is empty for the other six arms. On `h_t`, `rep` ends highest (0.99893, near the 1.0 ceiling) and `align` lowest (0.01563).

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
