# Data behind this report

| Path | Contents |
|------|----------|
| `eval_gm_mase/<cell>_summary.txt` | one line, the aggregate GM-Relative MASE and the config count for that cell |
| `eval_gm_mase/<cell>/all_results.csv` | per-config GIFT-Eval output for that cell — one row per dataset/config, with MASE, the seasonal-naive reference and the relative ratio. This is what the aggregate is the geometric mean of. |
| `training_curves/<run>_losses.csv` | per-step backbone training metrics: `loss`, `gap`, `ff`, `u_batchtime`, `u_batchtime_e`, `sigreg_e`, `sigreg_h`, `cpc_aux`, `auc`, `top1`, `top3` and more (26 columns). **Downsampled**: every step up to 500, then every 200th step. |
| `attn_amplitude/<run>_attn_amplitude.csv` | per-layer attention amplitude diagnostics (`qk_logit_maxabs`, `sa_in_maxabs`, `sa_out_maxabs`, `resid_post_sa_maxabs`, `resid_post_ffn_maxabs`), logged every 200 steps. |
| `seasonal_naive_all_results.csv` | the seasonal-naive baseline, one row per config. This is the denominator of every GM-Relative MASE in the report — `MASE(model) / MASE(seasonal_naive)`, geometric mean over configs. |
| `latent_movement_pairs.csv` | one row per cell per adjacent-checkpoint pair: `step_later`, `drift_h`, `drift_e`. Produced by `plots/_make_latent_movement.py --dump-csv`, so the drift figures can be rebuilt without a GPU. |
| `wave_d_metrics.csv` | one row per cell at the step-40k snapshot: end-of-window `ff`, `u_batchtime`, and `h_t` / `e_t` latent drift. |
| `elisa_artefacts_sha256.txt` | sha256 of all 1144 files kept on elisa but not in the repository: 990 checkpoints (5.1 GB) and the 154 full-resolution loss curves (1.88 GB). See the note at the end of this file. |

Cell naming is `<arm><variant>_bb<backbone step>k_hd<head steps>s`, e.g. `arm6_v2_combab_bb100k_hd30000s`.

A run that was resumed writes a fresh `_r<N>` file rather than appending, so a full trajectory is the concatenation of the `_losses.csv` files sharing a run name, ordered by their first `step`.

Full-resolution (per-step, ~87 MB per run) loss curves and all backbone / head checkpoints are too large for the repository. They stay on elisa under `/home/jupyter/checkpoints_backup/cf-379/runs/`, alongside the other experiments' backups (`cf-369`, `cf-374`, `cf-382`, `cf-388`). `elisa_artefacts_sha256.txt` lists every one of them with its checksum, so the copy on disk can be verified against what this report was built from:

```bash
cd /home/jupyter/checkpoints_backup/cf-379/runs
sha256sum -c /path/to/results/elisa_artefacts_sha256.txt
```

The experiment directory keeps a `runs` symlink pointing there, so the scripts under `experiments/2026-07-21_split_pred_rep_small/scripts/` still resolve their paths. Removing the git worktree removes the symlink, not the artefacts.
