# exp_realonly_4096_smaller — τ sweep (#27) + learnable τ (#32)

*Date: 2026-05-02. Author: agent (jeremycochoy).*

## tl;dr

Within this 47-epoch memorization-regime comparison, **all four τ policies land within noise** on both head loss (9k spread ~0.0017, ≈2%) and GIFT-Eval (GM-MASE spread <1.2%, 1.7770–1.7982). τ=0.05 edges τ=0.20 on all three GIFT-Eval metrics, but only the GM-MAPE_SN gap (~5%) is clearly outside that noise; the GM-MASE/CRPS gaps (<1%) are not. Learnable τ (init 0.07, drifts to ~0.0526) takes the best GM-MASE of the three eval'd arms. Among those three, the step-9k head-loss ranking (learnable < τ=0.05 < τ=0.20) **agrees with** the GM-MASE ranking — lower head loss tracks better eval here, not the reverse — but with both spreads at the noise floor, neither ordering is established. Bottom line: **nothing about τ is settled in this memorization regime**; the full-dataset runs (#6 / #9 / #10), where each window is seen <1×, will settle it.

> *MASE = Mean Absolute Scaled Error (point error scaled by the in-sample seasonal-naive MAE); GM-MASE = its geometric mean over the 97 GIFT-Eval configs. GM-MAPE_SN / GM-CRPS_SN = geometric means of the seasonal-naive-normalised MAPE and weighted-quantile-loss ratios (model ÷ seasonal-naive). SN = seasonal-naive (repeat the value one season back). RevIN = reversible instance normalisation; "EWMA span" sets how fast its moving average adapts. τ = the contrastive (InfoNCE) temperature — smaller τ = sharper/tighter contrast.*

## 1. Setup

All four arms share the same architecture, optimizer, dataset, and hyperparameters; only the τ policy varies.

| knob                | value                                                  |
|---------------------|--------------------------------------------------------|
| arch                | smaller (L=6, H=384, nhead=6, **11,428,668** params)   |
| backbone trainer    | `experiments/2026-04-27_freq-embedding/scripts/train.py` (per `scripts/run.sh`; B / forecaster reconstruction) |
| head                | quantile head (R1q), `--reconstruction forecaster`, `--forecast-len 16` |
| dataset             | `jeremycochoy/gift-pretrain-small-4096` (61,717 rows, small_v1) |
| t_raw               | 4096                                                   |
| n_channels          | 1                                                      |
| RevIN               | EWMA, span=128                                         |
| mix_ratio           | 0.0 (100% real, no synth)                              |
| batch_size          | 96                                                     |
| total_steps (BB)    | 30,000 → 30k × 96 = 2.88M samples ≈ **47 epochs**      |
| LR (BB)             | 1e-4                                                   |
| LR (head)           | 3e-4                                                   |
| save-every          | 1k–2.5k                                                |
| grad-clip           | NONE (banned in this project)                          |
| freq-emb-dim        | 3                                                      |
| seasonality-emb-dim | 3                                                      |
| mixup-p             | 0.3                                                    |
| τ-policy variation  | fixed {0.05, 0.07, 0.20} OR learnable (init 0.07, log_inv_tau, clamp [0.01, 1.0]) |

**Memorization-regime caveat.** 47 epochs on 61k rows means each window has been seen ~47 times. Within-sweep ranking under this much repetition is almost certainly noisier than the inter-arm differences. The follow-up runs (#6, #9, #10) sit on the full ~42.5M-window pretraining set so each window is seen <1× over 30k steps and that ranking will actually mean something.

## 2. Headline — head EMA-loss comparison (rolling 100 over the R1q losses CSV)

Computed via `pandas.read_csv(...).sort_values('step').loss.rolling(100, min_periods=1).mean()` over each arm's `R1q_*_losses.csv`. Step ~9k is what's available for τ=0.07 (it was halted later but we quote step 9k for everyone for fairness), plus the final step-30k value where applicable.

| arm                  | head ema_loss @ step 9k | head ema_loss @ step 30k |
|----------------------|-------------------------:|-------------------------:|
| τ = 0.05             | 0.07818                  | 0.06923                  |
| τ = 0.07             | 0.07666                  | —                        |
| τ = 0.20             | 0.07829                  | 0.07005                  |
| learnable τ          | 0.07743                  | 0.06818                  |

Notes:
- At step 9k the four arms are within 0.0017 of each other; the spread (~2%) is comparable to the rolling-100 noise floor.
- At step 30k learnable τ has the lowest head ema_loss, then τ=0.05, then τ=0.20.
- τ=0.07 was halted mid-head at step ~11.5k and has no eval; its CSV has 11,800 rows, last ema_loss 0.07467. The crash details and the not-resumed decision are in [notes/RUN_LOG.md](notes/RUN_LOG.md).

![Backbone + head loss curves (log-log)](plots/loss_curves.png)

*Top: backbone contrastive loss (the learnable arm's curve is missing steps 0–17,100 — early samples not retained locally). Bottom: quantile head loss. 100-step rolling mean over the raw per-step CSV; raw loss faintly behind. τ=0.07 head curve stops at step ~11,800 (truncated run). Generated by `scripts/plot_loss_curves.py`.*

## 3. Headline — GIFT-Eval geometric means (97 configs)

Computed as the geometric mean (`scipy.stats.gmean`) over `eval_metrics/MASE[0.5]`, `eval_metrics/SN_MAPE_ratio`, `eval_metrics/SN_WQL_ratio` columns of each arm's `results/all_results.csv`.

| arm                  | GM-MASE | GM-MAPE_SN | GM-CRPS_SN |
|----------------------|--------:|-----------:|-----------:|
| τ = 0.05             | 1.7883  | 1.2969     | 1.0872     |
| τ = 0.07             | —       | —          | —          |
| τ = 0.20             | 1.7982  | 1.3622     | 1.0948     |
| learnable τ          | **1.7770** | 1.3500  | 1.0907     |
| Aksu MOIRAI-Small ref | (n/a)  | 0.882      | 0.642      |

(The MOIRAI-Small MASE target from the Aksu paper isn't directly comparable here — the GIFT-Eval official MASE is the model-only number, not seasonal-naive normalised — so we omit a target.)

Reading:
- **τ=0.05 wins τ=0.20 on all three metrics**, by 0.55%/4.8%/0.7% on GM-MASE/GM-MAPE_SN/GM-CRPS_SN respectively. The MAPE gap is the only one that's clearly outside noise.
- **Learnable τ has the best GM-MASE** of the three eval'd arms (1.7770 vs 1.7883 for τ=0.05) but is worse than τ=0.05 on both distributional metrics (MAPE_SN, CRPS_SN). The learned τ stabilises around 0.0526 — i.e. *looser* than the τ=0.05 fixed arm (smaller τ = tighter contrast), landing between 0.05 and the 0.07 init. So the best-MASE point is **not** "tighter than 0.05"; it sits just above it, with a small quantile-metric penalty. All three cross-arm gaps are <1% here except MAPE_SN, so read this as directional at most.
- Even the best arm sits ~47% above MOIRAI-Small on MAPE_SN and ~69% above on CRPS_SN; the small-data regime dominates the absolute scores. The cross-arm comparison is what's interesting, not the absolute.

![GIFT-Eval geometric means by arm](plots/eval_metrics_bars.png)

*Three GM metrics across the four arms; τ=0.07 has an explicit "no eval" placeholder (run halted before STAGE E). Single run per arm — point estimates with no error bars; the cross-arm gaps (<1% except the ~5% MAPE_SN) sit at the noise floor. Aksu MOIRAI-Small reference is overlaid as a dashed horizontal line on the SN-normalised metrics. Generated by `scripts/plot_eval_metrics_bars.py`.*

## 4. τ trajectory for the learnable arm

The learnable run uses CLIP-style `log_inv_tau` as a single trainable scalar (`τ = exp(-log_inv_tau).clamp(0.01, 1.0)`). Init was `τ=0.07` (`log_inv_tau ≈ 2.659`). The local `run.log` only retains `τ=…` lines from step 17,200 onward (earlier samples were on a different vast.ai instance pre-resume and aren't in the local artifacts), but every sample we have shows τ monotonically decreasing.

| step       | τ      |
|-----------:|-------:|
| 17,200     | 0.0587 |
| 19,100     | 0.0573 |
| 21,100     | 0.0560 |
| 23,100     | 0.0549 |
| 25,100     | 0.0543 |
| 27,100     | 0.0535 |
| 29,100     | 0.0528 |
| 30,000     | 0.0525 |

End of run: `log_inv_tau=2.9453, τ=0.0526` (auto-detected by the head trainer and eval from the backbone checkpoint — confirmed in `run.log`). Qualitatively: τ decreases monotonically across the full visible range, by ~0.0062 over the last ~12.8k steps (≈0.5e-3 per 1k steps). The descent is approximately log-linear in step — log_inv_tau increases roughly linearly. The model "wants" tighter contrast than the 0.07 init, ending in the same neighbourhood as the τ=0.05 fixed arm.

![Learnable τ trajectory](plots/learnable_tau_trajectory.png)

*Learnable τ over training (red solid: 129 observed samples from step 17,200 → 30,000; red dotted: unobserved early portion bridged from init τ=0.07 at step 0 — early samples not retained locally). Horizontal dashed lines are the three fixed-τ values from the #27 sweep for context. Generated by `scripts/plot_learnable_tau_trajectory.py`.*

## 5. Discussion

**Is τ=0.05 > τ=0.20 within this sweep, decisive?** The MAPE gap (1.2969 vs 1.3622, ~5%) is large enough to look real; the MASE/CRPS gaps are tighter (<1%). I'd call it a directional win for tighter τ within this 47-epoch regime, not a definitive result.

**Does step-9k head loss predict step-30k eval ranking?** Among the three eval'd arms, the two orderings **agree** — though both spreads are at the noise floor:
- 9k head ema_loss (eval'd arms): learnable **0.07743** < τ=0.05 **0.07818** < τ=0.20 **0.07829** (τ=0.20 is the highest, not the lowest).
- GM-MASE (eval'd arms): learnable **1.7770** < τ=0.05 **1.7883** < τ=0.20 **1.7982**.
- Same order: the arm with the lowest 9k head loss also has the best GM-MASE, and the highest 9k head loss the worst. (τ=0.07 has the lowest 9k loss of all four but no eval, so it can't be placed.)
- But the 9k spread (~0.0017, ≈2%) and the GM-MASE spread (<1.2%) are both noise-floor-level, so the agreement may be coincidental: it does **not** establish that head loss predicts eval, only that the two do not contradict each other here. The honest read is that nothing separates the arms in this regime.

**Memorization caveat is load-bearing.** The 47-epoch repetition makes intra-sweep gaps suspect; the next runs (#6 30k learnable τ on full ~42.5M-window data, #9 MOIRAI HP on the same, #10 1-epoch FINAL retrain) will exit the memorization regime and show whether τ choice actually matters for generalisation. Pending those, the safe operational stance is: keep learnable τ for #6 since it auto-tunes toward what looks like a sensible neighbourhood, and don't promote τ=0.05 as the new fixed default until #9/#10 confirm.

## 6. Per-arm operational detail → notes

Per-arm timelines, crash/resume history, final ema-loss per stage, and the full checkpoint inventory (absolute paths, including the external sync-dir CSVs the metrics above derive from) are pure journey and live in [notes/RUN_LOG.md](notes/RUN_LOG.md). One arm (τ=0.07) was halted mid-head at step ~11.5k and has no eval — hence the "—" rows above.

## 7. Plots

All three figures are embedded inline above; their source scripts are in `scripts/` (`plot_loss_curves.py`, `plot_eval_metrics_bars.py`, `plot_learnable_tau_trajectory.py`).

## 8. Next steps

The within-sweep ranking here is only suggestive (47-epoch memorization regime). The full-dataset runs **#6** (30k learnable-τ), **#9** (MOIRAI optimizer-HP variant), and **#10** (1-full-epoch FINAL retrain) sit on the ~42.5M-window pretraining set — each window seen <1× over 30k steps — and will settle whether τ choice actually matters for generalisation. Until they land, the safe stance is to keep learnable τ (it auto-tunes toward a sensible neighbourhood) and not promote τ=0.05 as the fixed default. Operational placement of those runs is in [notes/RUN_LOG.md](notes/RUN_LOG.md).
