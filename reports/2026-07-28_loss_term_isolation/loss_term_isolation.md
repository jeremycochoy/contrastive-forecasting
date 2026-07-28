# Single-term loss isolation on the small backbone

TODO: results — the 8 arms of #382 train on cf-382-loss-term-isolation, one loss term at a time. This report will summarise which terms stabilise h_t and which keep drifting, following the small_long protocol (`reports/2026-07-21_split_pred_rep_small/small_long.md`). Numbers, tables, and figures will be filled in by the report-writer sub-agent once the 8 backbones reach 100k steps.

## Definitions

- `ff` = mean `cos(f̂, h_{t+1})` between the forecaster's next-step prediction `f̂` and the encoder's next-step latent `h_{t+1}` (unit-normalised on the sphere). `1 − ff` is a distance in [0, 2]; smaller = better forecast.
- `h_t`, `e_t`: the encoder-output latent and the patch-embedding latent respectively, shape `[B, T, C, H]`.
- **InfoNCE**: the standard cross-entropy contrastive loss `−log(exp(pos/τ) / Σ_j exp(neg_j/τ))` used throughout, with temperature `τ`.
- **SIGReg**: the LeJEPA-style spectral regulariser that pushes each latent's Gram matrix toward a Gaussian off-diagonal; `--sigreg-embedding-weight` (`sigreg_e`) applies it to `e_t`, `--sigreg-encoding-weight` to `h_t`.
- **CPC** (`--cpc-infonce-weight`, "cpc"): the CPC-InfoNCE auxiliary of van den Oord et al. 2018 predicting `e_{t+1}` from a bilinear projection of `h_t`.
- **MoCo**: negatives drawn from an EMA teacher (momentum contrast).
- `u_batchtime` = `1/(d · off-diag Gram mean)` over `(B×T)` samples of the specified latent, clamped to `[0, 1]`. 1 = every H dim carries independent info; low = collapsed onto a subspace. Exactly what SIGReg regularises.
- **Latent drift** at checkpoint pair `(step_i, step_j)` on a fixed held-out batch (`torch.manual_seed(20260722)`, `B=8` ARMA-synthetic): `mean_{b,t,c} 1 − cos(h_t(model_j), h_t(model_i))` (and analogously for `e_t`). Range `[0, 2]`; low = the mapping learned by the model on this fixed input hasn't rotated between the two checkpoints. Referred to below as "drift".
- **L_pred** — batch-pooled f-anchored InfoNCE from `cosine_similarity_batch_split_pred_rep`: numerator `cos(f_t, h_{t+1})/τ`, denominator LSE over the f-anchored families (adjacent `f_t↔f_{t+1}` + cross-batch `f_t↔h'_{t+1}`).
- **L_rep** — h-anchored logsumexp from the split shape's `L_rep`: pooled LSE of the three h-anchored families (cross-channel `h↔h`, within-series all-time `h_t↔h_l`, cross-series all-time `h_t↔h_{b',l}`). No positive.
- **L_align** — BYOL/SimSiam alignment: `2 − 2·cos(f_t, sg(h_{t+1}))` (stop-grad on the encoder side).

## Backbone

`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, T=4096, C=1, rev_norm=ewma(span=128), encoder_type=gru, batch_size=64, lr=1e-3, wd=0.1, adam_beta1=0.9, adam_beta2=0.98, seed=20260520`, dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`. Every arm trains for 100k steps with a checkpoint every 5k (20 checkpoints per arm).

## Arms

Each arm activates exactly one loss term; every other term (the two SIGReg regularisers and the CPC auxiliary included) is at weight 0.

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

TODO: fill in once training reaches 100k. Per-arm drift trajectories (h_t, e_t) from the in-training probe (`--latent-drift-probe`, seed `20260722`, B=8 ARMA batch), one adjacent-pair drift value per 5k-step interval.

## Supporting figures

TODO once training completes. Placeholders (`plots/…`, produced by `experiments/2026-07-28_loss_term_isolation/scripts/make_*.py`):

- `plots/cos_error_per_arm.png` — 1 − ff per arm across training step, 3×3 grid.
- `plots/dim_usage_per_arm.png` — u_batchtime per arm, h_t solid + e_t dashed, log-x step.
- `plots/latent_movement_per_arm.png` — per-arm adjacent-checkpoint drift (h_t).

## Annex

### A. Code changes shipped in the scaffold PR

- Two new CLI flags on `experiments/2026-04-27_freq-embedding/scripts/train.py`: `--pred-loss-weight`, `--rep-loss-weight` (both default 1.0). Threaded into `train_configuration` and consumed inside `contrastive_latent_loss`'s `cosine_similarity_batch_split_pred_rep` branch as `loss = w_pred·L_pred + w_rep·L_rep`. Default 1.0/1.0 is byte-for-byte the historical split objective.
- Loosened `--no-main-contrastive-loss` guard: the flag is now accepted with any SIGReg term on (in addition to `--cpc-infonce-weight` / `--align-loss-weight`), so the `sigreg_e` / `sigreg_h` arms can drop the contrastive forward entirely while training on the SIGReg regulariser alone.

### B. Compute

Single vast.ai instance under label `cf-382-loss-term-isolation`, provisioned via `docs/vastrun_guide.md` (2×RTX 4090 preferred at ≤$1.10/h combined; single-4090 fallback). Elisa-side `vast_sync_382.sh` mirrors checkpoints and training logs every ~30 s so results survive a bad instance.
