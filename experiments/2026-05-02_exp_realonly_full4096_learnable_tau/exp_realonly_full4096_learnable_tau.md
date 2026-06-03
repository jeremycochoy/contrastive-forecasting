# 2026-05-02_exp_realonly_full4096_learnable_tau — full-4096 30k learnable τ baseline (#6)

*Date: 2026-05-02. Author: agent (jeremycochoy).*

## tl;dr

The full-data 30k-step learnable-τ baseline finished its full pipeline
(backbone + qhead + STAGE E gift_eval). Final **GM-MAPE_SN = 1.3698,
GM-CRPS_SN = 1.1000, GM-MASE = 1.8043** across 97 GIFT-Eval configs.

**Surprising finding.** The full-data run *did not* outperform the
small-data learnable-τ run (#32) at 30k steps:

| arm                          | GM-MAPE_SN | GM-CRPS_SN | GM-MASE |
|------------------------------|-----------:|-----------:|--------:|
| #32 small-4096 learnable τ   | 1.3500     | 1.0907     | 1.7770  |
| **#6 full-4096 learnable τ** | **1.3698** | **1.1000** | **1.8043** |

#6 is slightly *worse* on every GM metric. The head-loss story tells
the opposite direction (small #32 head ema_loss at 30k = 0.0682;
#6 final head EMA loss = 0.0644 — i.e. full-data fits *tighter* in
quantile space but evaluates *worse* on GIFT-Eval). The fitted model
isn't translating the additional data variety into better eval skill
within this step budget.

**Hypothesis.** 30k steps × bs=96 = 2.88M samples processed —
≈6.78% of one full epoch on the 42.5M-window dataset. The small-data
arm sees each window ~47× over the same step count and converges to
something more specialised; the full-data arm sees each window <1×
and is still in the early-data-coverage phase. The right next move
is more steps (e.g. 90k–150k = ~20–35% of one epoch on full-4096),
not richer-data-fewer-steps.

A second axis was also tested: #9 (`2026-05-02_exp_realonly_full4096_moirai_hp`)
runs the same recipe with MOIRAI-paper optimizer hyperparameters
(10× lr, 10× weight_decay). #9 has now finished STAGE E gift_eval
(2026-05-03) and **wins on every GM metric** —
GM-MAPE_SN 1.1850 (vs #6 1.3698, −13.5%), GM-CRPS_SN 1.0155 (vs #6
1.1000, −7.7%), GM-MASE 1.6391 (vs #6 1.8043, −9.2%). See
[`experiments/2026-05-02_exp_realonly_full4096_moirai_hp/exp_realonly_full4096_moirai_hp.md`](../2026-05-02_exp_realonly_full4096_moirai_hp/exp_realonly_full4096_moirai_hp.md)
for the full final eval breakdown. The MOIRAI-HP win means the next
follow-up (#10) will use that recipe over default HP.

## 1. Setup

| knob                | value                                                  |
|---------------------|--------------------------------------------------------|
| arch                | smaller (L=6, H=384, nhead=6, **11.4M** params)        |
| backbone trainer    | `train_contrastive_v2.py` (B / forecaster reconstruction) |
| head                | quantile head (R1q), `--reconstruction forecaster`, `--forecast-len 16` |
| dataset             | `jeremycochoy/gift-pretrain-full-4096` (path `small_v1`, 42.5M windows, 619 GB, 4274 zstd parquet shards) |
| t_raw               | 4096                                                   |
| n_channels          | 1                                                      |
| RevIN               | EWMA, span=128                                         |
| mix_ratio           | 0.0 (100% real, no synth)                              |
| batch_size          | 96                                                     |
| total_steps (BB)    | 30,000 → 30k × 96 = 2.88M samples ≈ **6.78% of one epoch** |
| total_steps (head)  | 30,000                                                 |
| LR (BB / head)      | 1e-4 / 3e-4 (default; not MOIRAI HP)                   |
| weight_decay        | 0.01 (AdamW default in trainer)                        |
| betas               | (0.9, 0.999)                                           |
| save-every          | 2.5k                                                   |
| grad-clip           | NONE (banned in this project)                          |
| freq-emb-dim        | 3                                                      |
| seasonality-emb-dim | 3                                                      |
| mixup-p             | 0.3                                                    |
| τ-policy            | `--tau 0.07 --learnable-tau` (CLIP-style log_inv_tau, init τ=0.07, clamp [0.01, 1.0]) |
| eval                | STAGE E `gift_eval` (97 configs, B4 forecast_len=16)   |

This mirrors #32 exactly except for the dataset (full-4096 vs
small-4096) — the optimizer/arch/τ-policy/step-budget knobs are
identical so the comparison isolates the data-coverage axis.

## 2. Results — GIFT-Eval (97 configs)

Computed as the geometric mean over each metric column of
`results/all_results.csv` (filtered to positive values). Verified
against the run's own `summary.txt` ("SN-normalized skill scores").

| metric                        | #6 (this run) | #32 small | Aksu MOIRAI-Small target |
|-------------------------------|--------------:|----------:|-------------------------:|
| GM-MASE                       | 1.8043        | 1.7770    | (n/a)                    |
| GM-MAPE_SN                    | 1.3698        | 1.3500    | 0.882                    |
| GM-CRPS_SN (= WQL_SN)         | 1.1000        | 1.0907    | 0.642                    |

`results/summary.txt` reports `GM-MAPE_SN (97 configs): 1.3698` and
`GM-CRPS_SN (97 configs): 1.1000`, matching to 4 decimal places.

Even the better-coverage arm sits ~55% above MOIRAI-Small on
GM-MAPE_SN and ~71% above on GM-CRPS_SN — within this step budget
the absolute gap to MOIRAI is dominated by step count and arch size,
not data coverage.

## 3. Backbone + head training

Computed by `pandas.DataFrame.ewm(alpha=0.01)` over the per-step
loss CSV (the run.log was overwritten by the eval-stage relaunch
on May 2 around 12:35 UTC, so the in-run progress lines are gone —
the per-step CSV is the canonical record).

| stage                          | EMA at last step | best EMA       | best step |
|--------------------------------|-----------------:|---------------:|----------:|
| backbone (`tiny_*_30k.pth`)    | 5.3229           | 5.2986         | 29,304    |
| qhead (`R1q_*_FINAL.pth`)      | 0.0644           | 0.0641         | 29,556    |

Both stages descend smoothly to the end of the budget — no plateau,
suggesting more steps would still drop loss. Backbone gap (positive
similarity − cross-batch negative similarity) climbed to 0.41 around
step 16k, then drifted back to 0.39 by step 30k (the slight late-run
drop is consistent with the model spending capacity on tighter
contrast at lower τ rather than wider gap, see §4).

## 4. Learnable τ trajectory

τ descends *monotonically* from init 0.0700 to 0.0472 over 30k steps
— same shape as in the small-data #32 run, just with a slightly less
extreme final value (#32 ended at 0.0526; #6 ends at 0.0472). The
larger pool of data lets the model commit to a tighter contrast
without overfitting.

| step   | τ       | log_inv_tau |
|-------:|--------:|------------:|
|     0  | 0.0700  | 2.6593      |
|  2,500 | 0.0696  | 2.6644      |
|  5,000 | 0.0645  | 2.7408      |
| 10,000 | 0.0562  | 2.8792      |
| 15,000 | 0.0526  | 2.9450      |
| 20,000 | 0.0502  | 2.9914      |
| 25,000 | 0.0485  | 3.0267      |
| 30,000 | 0.0472  | 3.0542      |

Source: `sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv`
(13 samples extracted post-hoc from periodic `*_2k.pth … *_30k.pth`
checkpoints, since the run.log was wiped). End-of-run value is also
auto-detected at eval load: `[eval] auto-detected learnable τ
(log_inv_tau=3.0512, τ=0.0473)`.

## 5. Discussion

This run was the natural follow-up to #32 — same recipe, more data —
under the assumption that data variety would translate into a better
eval. It didn't, within this step budget. Three plausible reasons:

1. **Step starvation.** 6.78% of one epoch on 42.5M windows is
   fundamentally a different regime from 47 epochs on 61k windows.
   The full-data arm has barely seen most of its windows; the
   small-data arm has overfit gracefully on a smaller manifold.
   At 30k steps the small-data overfit is a *better* GIFT-Eval
   distributional fit than the under-trained full-data arm.
2. **The full dataset's domain mix is broader.** Some shards are
   from harder domains than the small subset (pretrain-full
   includes the long tail). The optimizer is splitting capacity
   across a noisier window-mix and the eval metrics catch this.
3. **Optimizer / step-schedule mismatch.** Default lr=1e-4 with no
   warmup may be too conservative for the broader window mix; the
   companion run #9 tests lr=1e-3 and gets a tighter head EMA at
   the same step (#9 head EMA 0.0552 vs #6 0.0644, ~14% lower) and
   a better GM-MAPE_SN at eval (1.1850 vs 1.3698, −13.5%). See
   [the #9 final report](../2026-05-02_exp_realonly_full4096_moirai_hp/exp_realonly_full4096_moirai_hp.md).

The right next moves (in priority order):

- ~~Wait for #9 final eval to land~~ — DONE 2026-05-03. MOIRAI HP wins
  on every GM metric, so the optimizer was a real lever; step count
  is the remaining one.
- The follow-up (#10) is a 1-full-epoch retrain on full-4096 (≈443k
  steps at bs=96), resuming from the #9 30k pair under MOIRAI HP. This
  tests the step-starvation hypothesis directly.

For the final cross-experiment plot covering both arms over the full
30k (backbone loss + τ + head loss, log-step axes), see
[`plots/full4096_3panel_final.png`](../2026-05-02_exp_realonly_full4096_moirai_hp/plots/full4096_3panel_final.png)
(PR #102). The earlier 2-panel
`plots/full4096_default_vs_moirai_hp.png` (PR #98) was superseded by
this 3-panel version.

## 6. Local artifacts

All paths under
`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting`:

- **Backbone FINAL checkpoint** —
  `sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_FINAL.pth`
  (45.7 MB)
- **Periodic backbone snapshots (2k…30k)** —
  same dir, `tiny_realonly_full4096_learnable_tau_{2,5,7,10,12,15,17,20,22,25,27,30}k.pth`
  + companion `*_optimizer.pth` files
- **Backbone losses CSV** —
  `.../checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv`
  (3.98 MB, 30k rows; columns step, loss, gap, ff, fp, tp, cross_batch,
  hf_rows_consumed, synth_rows_consumed, mixup_applied)
- **Qhead FINAL checkpoint** —
  `.../checkpoints/R1q_realonly_full4096_learnable_tau_FINAL.pth`
  (2.46 MB)
- **Qhead losses CSV** —
  `.../checkpoints/R1q_realonly_full4096_learnable_tau_losses.csv`
  (1.02 MB, 30k rows; columns step, loss, hf_rows_consumed)
- **τ trajectory** —
  `sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv`
  (13 rows from periodic checkpoint extraction)
- **GIFT-Eval per-config CSV** —
  `sync_realonly_full4096_learnable_tau/learnable/results/all_results.csv`
  (97 rows, 19 columns)
- **GIFT-Eval summary** —
  `sync_realonly_full4096_learnable_tau/learnable/results/summary.txt`
  (formatted GM scores)
- **Run log (eval portion only)** —
  `sync_realonly_full4096_learnable_tau/learnable/run.log`
  (the train-stage log was overwritten during the eval relaunch;
  per-step train state lives in the CSVs)
- **Companion run script** —
  `experiments/2026-05-02_exp_realonly_full4096_learnable_tau/scripts/run.sh`

## 7. Cross-references

- **Small-data baseline (#32)**:
  [`experiments/2026-05-02_exp_realonly_4096_smaller_tau_sweep/exp_realonly_4096_smaller_tau_sweep.md`](../2026-05-02_exp_realonly_4096_smaller_tau_sweep/exp_realonly_4096_smaller_tau_sweep.md)
  (PR #95). The "Learnable τ" arm there is the direct small-data
  counterpart of this run.
- **MOIRAI-HP companion (#9), FINAL**:
  [`experiments/2026-05-02_exp_realonly_full4096_moirai_hp/exp_realonly_full4096_moirai_hp.md`](../2026-05-02_exp_realonly_full4096_moirai_hp/exp_realonly_full4096_moirai_hp.md).
- **Final 3-panel comparison plot (#6 vs #9, full 30k)**:
  [`plots/full4096_3panel_final.png`](../2026-05-02_exp_realonly_full4096_moirai_hp/plots/full4096_3panel_final.png)
  (PR #102).
