# τ sweep — committed plan

> Persisted before context compaction so future sessions can pick up cleanly.

## Goal

Determine whether the contrastive temperature τ — fixed during training rather
than learned — affects the converged AUC of the contrastive backbone. AUC was
the strongest predictor of downstream MASE in the proxy-correlation analysis
across 5 backbones (Spearman ρ = +0.70).

## Hypothesis

backbone-beta's learnable τ converged to ~0.072 over 167k steps. Five fixed-τ
arms in {0.03, 0.05, 0.07, 0.10, 0.20} probe whether nearby fixed values match
the learnable optimum, and whether sharper / softer values shift AUC
materially.

## Sweep mechanics

Five from-scratch arms, identical architecture/HP except `--tau`:

| τ    | run name           | rationale                                          |
|------|--------------------|----------------------------------------------------|
| 0.03 | `tau_sweep_0_03`   | sharp — punishes near-misses harder                |
| 0.05 | `tau_sweep_0_05`   | moderately sharp                                   |
| 0.07 | `tau_sweep_0_07`   | closest fixed value to backbone-beta's converged τ |
| 0.10 | `tau_sweep_0_10`   | moderately soft                                    |
| 0.20 | `tau_sweep_0_20`   | soft — high entropy, harder to discriminate        |

**Recipe (per arm)** — matches backbone-beta exactly, only τ varies:
- T_RAW=4096, C=1, d_model=384, num_layers=6, n_heads=6
- freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128
- AdamW lr=1e-3 wd=0.1 β1=0.9 β2=0.98
- mixup_p=0.3, mix_ratio=0.0, loss_shape=cosine_similarity_batch
- batch_size=256, total_steps=50000, save_every=5000
- **fixed τ** (no `--learnable-tau`), pass `--tau X`

Budget rationale: backbone-beta's AUC was already near-converged well before
167k. 50k = "just enough to acquire knowledge". Winner extendable later.

**Launcher**: `scripts/run_tau_sweep.sh` — loops the 5 τ values, idempotent
(skips arms whose `<run>_FINAL.pth` exists). Merged via PR #162.

## 6 metrics tracked per minibatch

Trainer change pending in a separate PR. The four functions in `src/metrics.py`
yield 6 columns appended to `<run>_losses.csv` at every step (cheap — reuses
`f`, `h`, `z` already computed for the loss; all `@torch.no_grad`):

| Column        | Source                                    |
|---------------|-------------------------------------------|
| `r2_random`   | `1 - q_random(f, h_target)`               |
| `r2_naive`    | `1 - q_naive_latent(f, h_target, h_prev)` |
| `u_temporal`  | `dim_usage(z, axis="temporal")`           |
| `u_batch`     | `dim_usage(z, axis="batch")`              |
| `auc`         | `retrieval_auc_top1(...)` 1st return      |
| `top1`        | `retrieval_auc_top1(...)` 2nd return      |

Column names match `experiments/2026-05-05_exp_qhead_improvements/results/backbone_proxy_correlation.csv` so post-hoc + per-batch metrics merge cleanly.

## Status & blockers

| Item                                            | State                              |
|-------------------------------------------------|------------------------------------|
| τ sweep launcher (PR #162)                      | merged                             |
| Per-batch metric logging (PR #164)              | merged — smoke-tested on elisa     |
| Dead code cleanup (PR #165, −1118 LOC)          | merged                             |
| Vast 5090 provisioned                           | NOT — credit $3.67 vs ~$15 needed  |
| Sync_loop on elisa                              | not yet set up (deferred)          |

## Pre-launch gate (do all before provisioning)

- [x] Per-batch metric logging PR merged (#164)
- [x] Dead code cleanup PR merged (#165)
- [x] All 6 columns visible in a smoke `_losses.csv` on elisa
- [ ] Vast credit ≥ $20 (user must top up — agent cannot)

**All code work is done and merged on `experiments`.** The only remaining
blocker is vast.ai credit. Once topped up, an agent can read this PLAN.md
and execute the launch directly (provision via vastrun-kit, push code, run
`scripts/run_tau_sweep.sh` under nohup, set up sync_loop on elisa).

## Post-sweep evaluation

For each `tau_sweep_*_FINAL.pth`:

1. **Per-arm 6-metric eval** on a fixed eval batch — comparable across arms.
2. **Per-arm trajectory** from sync_loop's 5k periodic saves — 10 points × 5 arms = 50 evals (adapt `experiments/2026-05-05_exp_qhead_improvements/scripts/eval_backbone_metrics.py`).
3. **Proxy MASE** via R3_E4 quantile head trained on each arm's backbone (recipe at `experiments/2026-05-05_exp_qhead_improvements/scripts/run_round10_proxy.sh`); evaluate on the existing triage subset.
4. **`RESULTS.md`**: per-arm final R²/U/AUC/Top-1 + proxy MASE + which τ wins on each metric.
5. **Decision**: does the AUC winner match the proxy-MASE winner? Is fixed τ better than learnable τ in either dimension?

## Cost

5090 spot ~$0.60/h × ~5h/arm × 5 arms = **$15**. Margin → $25. Eval ~$0.50.

## File layout

```
experiments/2026-05-08_exp_tau_sweep/
├── PLAN.md                          ← this file
├── README.md                        ← high-level overview
├── scripts/run_tau_sweep.sh         ← launcher (5 arms × 50k each)
├── results/                         ← per-arm metric eval (post-sweep)
└── plots/                           ← per-arm trajectory plots (post-sweep)
```

## Reference points (NOT arms in the sweep)

- **backbone-beta**: the 167k learnable-τ training. Final τ ≈ 0.072. Reference for "what the trained τ chose".
- Other backbones (`moirai_hp_FINAL_run1`, `FRESH_50k`, `moirai_hp_early`, `learnable_tau`): used for the prior 5-backbone proxy correlation — not retrained here.

## How to resume after compaction

1. `cat experiments/2026-05-08_exp_tau_sweep/PLAN.md` — this file.
2. `gh pr list --state all --search 'tau-sweep|metrics'` — what's merged.
3. `vastrun-balance` — current credit.
4. `vastrun-status` — any live instances.
5. If everything green and budget topped up: launcher entry point is `experiments/2026-05-08_exp_tau_sweep/scripts/run_tau_sweep.sh`. Provision via vastrun-kit, push code, run launcher in nohup, set up sync_loop on elisa.

## Operational notes

- vast.ai is a shared account. Verify any contract before destroying — match the label this experiment sets (`tau-sweep-2026-05-08` or similar).
- HF token must be exported in the launcher (already wired).
- sync_loop pulls 5k periodic checkpoints + best_loss/best_gap + losses.csv + run.log every 15 min (atomic-mv pattern, per-class size thresholds).
- No grad-clip — fix divergence at the data/normalization layer if it appears.

## Open questions

- Do all 5 arms converge by 50k, or do some need longer? Watch `r2_random`/AUC trajectory — if any arm is still climbing at 50k, extend it.
- Are AUC and proxy-MASE consistent across τ values? If they diverge, the AUC-as-proxy-MASE-predictor hypothesis weakens.

## Run length revision (2026-05-08)

Arm 1 (τ=0.03) plateaued on every metric by step ~3k–5k (loss 11→7,
AUC 0.51→0.88, U_b 0.003→0.009, then stable through step 22.4k). Killed
at step 22,400 and `cp`'d the 15k periodic save to `_FINAL.pth` for
fair cross-arm comparison. Edited `run_tau_sweep_elisa.sh` to
`--total-steps 15000` so arms 2–5 also stop at 15k. Saves ~22h.

## HF dataloader incident (2026-05-08)

Arm 2 launch repeatedly failed on HF API errors (ReadTimeout, then 500).
Root cause: `datasets.load_dataset` resolves per-shard metadata via a
64-thread pool calling `/api/datasets/<id>/revision/<sha>`. With 4274
shards in `gift-pretrain-full-4096/small_v1` the thundering herd 500s
intermittently. Arm 1 worked because metadata was cached in-process;
arm 2 hit a fresh process every time. Fix in PR #170: bypass
`load_dataset` for flat parquet layouts via `HfFileSystem.ls` +
`pyarrow.ParquetFile.iter_batches`. Throughput also up from 1.5 sps
(arm 1) to 2.7 sps (arm 2) — fewer Python overheads in the new path.

## Future work (post-sweep)

User-noted (2026-05-08), tracked here for later:

- **Pick best τ → retest with `residual_silu` ("patch fm") encoder.** Goal:
  see if AUC/Top-1 improve with the FM-style encoder vs. the GRU encoder
  at the winning τ.
- **Symmetric-negatives loss extension.** Currently the loss has cross-batch
  negatives between `(f_t, h_{j,t+1})` for `j` in the batch. Extend to also
  use `(h_{t-1}, f_t)` cross-batch — i.e. encoder-anchor with forecaster
  negatives, in addition to the existing forecaster-anchor with encoder
  negatives.
- **Symmetrise the loss.** Compute `0.5 · L(anchor=f, pos=h_target) + 0.5 ·
  L(anchor=h_target, pos=f)`. SimCLR (NT-Xent) and CLIP (symmetric
  InfoNCE) both use this swap-anchor trick — proven and easy to drop in.
- **Sweep range can flex.** If arm-by-arm AUC/Top-1 trends monotonically
  worse, stop the τ sweep early. If it keeps improving at the extremes,
  add τ=0.01 / τ=0.40 etc. to extend the range.
