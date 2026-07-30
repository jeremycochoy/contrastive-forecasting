# Data behind this report

| Path | Contents |
|------|----------|
| `eval_gm_mase/<cell>_summary.txt` | one line, the aggregate GM-Relative MASE and the config count for that cell |
| `eval_gm_mase/<cell>/all_results.csv` | per-config GIFT-Eval output for that cell — one row per dataset/config, with MASE, the seasonal-naive reference and the relative ratio. This is what the aggregate is the geometric mean of. |
| `training_curves/<run>_losses.csv` | per-step backbone training metrics: `loss`, `gap`, `ff`, `u_batchtime`, `u_batchtime_e`, `sigreg_e`, `sigreg_h`, `cpc_aux`, `auc`, `top1`, `top3` and more (26 columns). **Downsampled**: every step up to 500, then every 200th step. |
| `attn_amplitude/<run>_attn_amplitude.csv` | per-layer attention amplitude diagnostics (`qk_logit_maxabs`, `sa_in_maxabs`, `sa_out_maxabs`, `resid_post_sa_maxabs`, `resid_post_ffn_maxabs`), logged every 200 steps. |
| `wave_d_metrics.csv` | one row per cell at the step-40k snapshot: end-of-window `ff`, `u_batchtime`, and `h_t` / `e_t` latent drift. |

Cell naming is `<arm><variant>_bb<backbone step>k_hd<head steps>s`, e.g. `arm6_v2_combab_bb100k_hd30000s`.

A run that was resumed writes a fresh `_r<N>` file rather than appending, so a full trajectory is the concatenation of the `_losses.csv` files sharing a run name, ordered by their first `step`.

Full-resolution (per-step, ~87 MB per run) loss curves and all backbone / head checkpoints stay on elisa under `experiments/2026-07-21_split_pred_rep_small/{runs,eval_gm_mase}/`; they are too large for the repository.
