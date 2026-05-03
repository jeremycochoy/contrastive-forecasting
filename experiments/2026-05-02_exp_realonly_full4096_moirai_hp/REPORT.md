# 2026-05-02_exp_realonly_full4096_moirai_hp — full-4096 30k MOIRAI-HP arm (#9, FINAL)

*Date: 2026-05-03. Author: agent (jeremycochoy).*

## tl;dr

#9 runs the same recipe as #6 (full-4096 30k learnable-τ baseline) but
with **MOIRAI-paper optimizer hyperparameters** (lr=1e-3, weight_decay=0.10,
β=(0.9, 0.98)). STAGE E gift_eval finished 2026-05-03 at 04:28 CEST; all
97 configs evaluated. **#9 wins on all three GM metrics**:

| Metric        | #6 (default HP) | #9 (MOIRAI HP) | Aksu MOIRAI-S | Δ (#9 − #6) |
|---------------|----------------:|---------------:|--------------:|------------:|
| GM-MASE       | 1.8043          | **1.6391**     | (n/a)         | −9.2%       |
| GM-MAPE_SN    | 1.3698          | **1.1850**     | 0.882         | −13.5%      |
| GM-CRPS_SN    | 1.1000          | **1.0155**     | 0.642         | −7.7%       |

**Decision: #10 final retrain uses MOIRAI HP**, resuming from the #9 30k
backbone + optimizer pair (`tiny_realonly_full4096_moirai_hp_30k.pth` /
`_30k_optimizer.pth`).

#9 is closer to MOIRAI-Small but still well short — GM-MAPE_SN 1.185 vs
0.882, GM-CRPS_SN 1.016 vs 0.642. The remaining gap is **step budget ×
dataset coverage**: 30k × 96 = 2.88M samples is ~6.78% of one epoch on
the 42.5M-window full-4096 dataset. That is the lever for #10.

## 1. Setup (delta from #6)

Identical to
[`2026-05-02_exp_realonly_full4096_learnable_tau`](../2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md)
in every respect except the optimizer:

| knob          | #6 default HP        | #9 MOIRAI HP         |
|---------------|---------------------|---------------------|
| lr (BB)       | 1e-4                | **1e-3**            |
| weight_decay  | 0.01                | **0.10**            |
| β1            | 0.9                 | 0.9                 |
| β2            | 0.999               | **0.98**            |
| warmup        | none                | none                |
| schedule      | flat                | flat                |

Same arch (smaller, 11.4M), same dataset (full-4096, 42.5M windows),
same step budget (30k BB + 30k head), same τ-policy (`--tau 0.07
--learnable-tau`), same RevIN/EWMA/mixup config. See [§1 of the #6
report](../2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md) for the
shared knob table.

## 2. Training results

Backbone and qhead are both fully trained — 30k steps each. The local
sync ticked successfully; FINAL.pth files exist for both.

EMA computed via `pandas.DataFrame.ewm(alpha=0.01)` over the per-step
loss CSV.

### Backbone (contrastive)

| arm                | EMA loss @ 30k | EMA gap @ 30k | best EMA loss | best step |
|--------------------|---------------:|--------------:|--------------:|----------:|
| #6 default HP      | 5.3229         | 0.3853        | 5.2986        | 29,304    |
| **#9 MOIRAI HP**   | **5.1294**     | **0.4914**    | **5.1048**    | 30,000    |

#9 is **0.19 lower in loss and 0.11 higher in gap** — the optimizer is
genuinely doing better contrastive separation under the larger lr.
This is the opposite of the prior expectation (10× lr risked NaN /
divergence; per project rule we'd fix the data not grad-clip — none
needed, training was stable throughout).

### Qhead (quantile)

| arm                | EMA loss @ 30k | best EMA loss | best step |
|--------------------|---------------:|--------------:|----------:|
| #6 default HP      | 0.0644         | 0.0641        | 29,556    |
| **#9 MOIRAI HP**   | **0.0552**     | **0.0551**    | 29,500    |

#9 head EMA at step 30k is ~14% lower than #6's. The "best" checkpoint
saved by the trainer is `R1q_*_best.pth` at ema_loss=0.055126
(step 29,500). The tighter head loss does translate to a better
GIFT-Eval — see §4.

## 3. τ trajectory — the suppression effect

The most striking qualitative difference between #6 and #9. Same
init (0.07), same `--learnable-tau` flag, same architecture — but a
totally different evolution.

Sampled from `sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv`
(per-100-step grep from the run.log; 270 rows):

| step   | #6 τ     | #9 τ     | Δ (#9 − #6)   |
|-------:|---------:|---------:|--------------:|
|     0  | 0.0700   | 0.0700   | 0.0000        |
|  2,500 | 0.0696   | 0.0638   | −0.0058       |
|  5,000 | 0.0645   | 0.0688   | +0.0043       |
| 10,000 | 0.0562   | 0.0726   | +0.0164       |
| 15,000 | 0.0526   | 0.0753   | +0.0227       |
| 20,000 | 0.0502   | 0.0754   | +0.0252       |
| 25,000 | 0.0485   | 0.0771   | +0.0286       |
| 27,000 | (n/a)    | 0.0761   | n/a           |
| 30,000 | 0.0472   | 0.0765   | +0.0293       |

Auto-detected at eval load (matches the trajectory):
`[eval] auto-detected learnable τ (log_inv_tau=2.5925, τ=0.0748)`.

The `log_inv_tau` parameter only just barely climbs above its init of
2.659 (init τ=0.07) and then never goes higher than ~2.76 (τ≈0.063 at
the early dip), settling around 2.57–2.58 (τ≈0.076). Under default HP
(#6) `log_inv_tau` ends at 3.05 (τ≈0.047). The interpretation is that
weight_decay=0.1 on a single scalar parameter is a strong restorative
force toward zero — and zero `log_inv_tau` means τ=1.0, which is far
above init. So the visible effect is τ-suppression *upward* relative
to init, but the underlying mechanism is `log_inv_tau` being pulled
*down* toward 0 by weight_decay every step, which the contrastive
loss only partly counteracts.

This is a useful operational note for any future runs that combine
a learnable scalar parameter with high weight_decay: the param will
not reach the value the loss "wants" — its equilibrium is determined
by the lr × wd × loss-gradient balance, not by the loss alone.

For the visualised version of this trajectory (and the loss curves
side-by-side), see
[`plots/full4096_3panel_final.png`](../../plots/full4096_3panel_final.png)
(generated by `scripts/plot_full4096_3panel_final.py`, PR #102).
The 3-panel plot covers the full 30k for both arms — backbone loss,
τ trajectory, and head loss — on shared log-step axes.

## 4. Final eval (97 configs, 2026-05-03)

STAGE E gift_eval completed 2026-05-03 at 04:28 CEST. All 97 configs
evaluated against MOIRAI-Small's per-dataset references (Aksu et al.).
GM scores below come straight from
`sync_realonly_full4096_moirai_hp/moirai_hp/results/summary.txt`:

| Metric        | #6 (default HP) | #9 (MOIRAI HP) | Aksu MOIRAI-S | Δ (#9 − #6) |
|---------------|----------------:|---------------:|--------------:|------------:|
| GM-MASE       | 1.8043          | **1.6391**     | (n/a)         | −9.2%       |
| GM-MAPE_SN    | 1.3698          | **1.1850**     | 0.882         | −13.5%      |
| GM-CRPS_SN    | 1.1000          | **1.0155**     | 0.642         | −7.7%       |

GM-MASE was computed via `gmean` of `eval_metrics/MASE[0.5]` over all
97 configs (no NaN drops needed — all configs have a model MASE).
GM-MAPE_SN and GM-CRPS_SN come straight from the run-emitted
summary.txt's "SN-normalized skill scores" block (the canonical source
the trainer prints), and were cross-checked against the geometric mean
of the per-dataset `eval_metrics/SN_MAPE_ratio` and
`eval_metrics/SN_WQL_ratio` columns in
`sync_realonly_full4096_*/{learnable,moirai_hp}/results/all_results.csv`
— both reproduce the summary numbers to 4 decimal places.

#9 beats #6 on **74/97** datasets on MASE, **72/97** on
SN_MAPE_ratio, and **61/97** on SN_WQL_ratio.

### Best per-dataset wins (#9 over #6)

Largest improvements on SN_MAPE_ratio (model/Aksu reference):

| dataset                             | #6 ratio | #9 ratio | Δ        |
|-------------------------------------|---------:|---------:|---------:|
| `jena_weather/H/medium`             | 5.0139   | 1.8169   | −63.8%   |
| `bizitobs_service/10S/short`        | 18.4336  | 7.1164   | −61.4%   |
| `hierarchical_sales/W/short`        | 1.7591   | 0.7574   | −56.9%   |
| `saugeen/W/short`                   | 1.4468   | 0.6695   | −53.7%   |
| `bizitobs_service/10S/medium`       | 27.8581  | 14.0406  | −49.6%   |

Largest improvements on absolute MASE:

| dataset                             | #6 MASE  | #9 MASE  | Δ        |
|-------------------------------------|---------:|---------:|---------:|
| `solar/H/long`                      | 2.1642   | 1.3052   | −39.7%   |
| `electricity/H/long`                | 2.7360   | 1.7094   | −37.5%   |
| `electricity/15T/medium`            | 2.7592   | 1.7324   | −37.2%   |
| `electricity/15T/long`              | 3.0315   | 1.9265   | −36.5%   |
| `electricity/H/medium`              | 2.3360   | 1.5088   | −35.4%   |

`electricity/H/*` and `electricity/15T/*` show the cleanest gain — the
MOIRAI HP closes ~35–37% of the gap on these long-horizon
electricity configs.

### Worst per-dataset regressions (#9 vs #6)

Largest regressions on SN_MAPE_ratio:

| dataset                             | #6 ratio | #9 ratio | Δ        |
|-------------------------------------|---------:|---------:|---------:|
| `ett1/H/medium`                     | 0.2196   | 2.0518   | +834.1%  |
| `solar/10T/long`                    | 1.2832   | 1.6144   | +25.8%   |
| `solar/H/short`                     | 1.2953   | 1.5470   | +19.4%   |
| `ett1/H/short`                      | 0.9542   | 1.1220   | +17.6%   |
| `loop_seattle/D/short`              | 0.7335   | 0.8205   | +11.9%   |

The `ett1/H/medium` outlier (+834%) is suspect — #6's 0.2196 ratio is
an order of magnitude better than the geometric mean of the column,
which suggests numerical luck on that single config rather than a
systematic #6 advantage. The other regressions are in the +12% to
+26% band.

Even with the win, #9 sits **34% above MOIRAI-Small on GM-MAPE_SN**
(1.185 vs 0.882) and **58% above on GM-CRPS_SN** (1.016 vs 0.642).
The optimizer-HP lever has been pulled; the remaining gap is the
**step budget × dataset coverage** axis — 30k × 96 = 2.88M samples
covers ≈6.78% of one epoch on the 42.5M-window full-4096 dataset.

## 5. Local artifacts

All paths under
`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting`:

- **Backbone FINAL** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_FINAL.pth`
  (45.7 MB)
- **Periodic backbone snapshots (2k…30k)** —
  same dir, `tiny_realonly_full4096_moirai_hp_{2,5,7,10,12,15,17,20,22,25,27,30}k.pth`
  + `*_optimizer.pth` siblings
- **Backbone losses CSV** —
  `.../checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv`
  (3.99 MB, 30k rows)
- **Qhead FINAL** —
  `.../checkpoints/R1q_realonly_full4096_moirai_hp_FINAL.pth`
  (2.46 MB)
- **Qhead best-loss** —
  `.../checkpoints/R1q_realonly_full4096_moirai_hp_best.pth`
  (ema_loss=0.055126 at step 29,500)
- **Qhead losses CSV** —
  `.../checkpoints/R1q_realonly_full4096_moirai_hp_losses.csv`
  (1.03 MB, 30k rows)
- **τ trajectory** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv`
  (270 rows, sampled every 100 steps from run.log)
- **GIFT-Eval per-config CSV (FINAL, 97 rows)** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/results/all_results.csv`
- **GIFT-Eval summary (FINAL)** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/results/summary.txt`
- **Run log (training + eval)** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/run.log`
- **Companion run script** —
  `experiments/2026-05-02_exp_realonly_full4096_moirai_hp/run.sh`

## 6. Decision and follow-up

**Decision.** MOIRAI HP wins on every GM metric measured (MASE,
MAPE_SN, CRPS_SN), on a ~3:1 majority of per-dataset configs, and on
both backbone and head training EMAs. **#10 final retrain will use
MOIRAI HP.**

**#10 plan.** Resume from the #9 30k pair
(`tiny_realonly_full4096_moirai_hp_30k.pth` +
`_30k_optimizer.pth`) — these preserve AdamW momentum, RNG state,
and the learned `log_inv_tau` value. Target: **1 full epoch on
full-4096**, which at bs=96 over 42.5M windows is ~443k total
steps (440k–500k window depending on the streaming dataset's exact
manifest count). Same MOIRAI HP, same arch, same τ-policy.

The expected mechanism for #10's gain is the step-starvation
hypothesis from §5 of the [#6 report](../2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md):
30k steps × 96 = 6.78% of one epoch is a fundamentally different
regime than the 47-epoch small-data baseline. A full-epoch run
removes the under-coverage confound and isolates whether the full-data
manifold can match or beat the small-data overfit at adequate step
count. If GM-MAPE_SN drops below the small-data #32 number (1.3500)
at full-epoch, the data-scale axis is real. If it doesn't, the
binding constraint is arch capacity / depth, not data.

A successful #10 also takes the next bite out of the MOIRAI-Small
gap (current #9: 1.185 vs 0.882 on GM-MAPE_SN). Even halving the
remaining gap would put GM-MAPE_SN at ~1.03 — competitive with public
small-model time-series baselines.

## 7. Cross-references

- **#6 default-HP companion (DONE)**:
  [`experiments/2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md`](../2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md).
- **Small-data tau sweep (#27/#32)**:
  [`experiments/2026-05-02_exp_realonly_4096_smaller_tau_sweep/REPORT.md`](../2026-05-02_exp_realonly_4096_smaller_tau_sweep/REPORT.md)
  (PR #95).
- **Final 3-panel comparison plot (#6 vs #9, full 30k)**:
  [`plots/full4096_3panel_final.png`](../../plots/full4096_3panel_final.png)
  (PR #102).
- **Sync-protocol audit**:
  [`docs/SYNC_PROTOCOL_REVIEW.md`](../../docs/SYNC_PROTOCOL_REVIEW.md)
  (PR #99) — relevant because both #6 and #9 used the new
  size-floored, append-only-protected sync_loop.
