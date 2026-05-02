# exp_realonly_full4096_moirai_hp — INTERIM (#9, eval still in flight)

*Date: 2026-05-02. Author: agent (jeremycochoy).*

---

## ⚠️ INTERIM ⚠️

**The GIFT-Eval STAGE E is still running on the remote instance.** Only
2 of 97 configs have been evaluated as of this snapshot (May 2 ~22:30
UTC). Backbone + qhead training are both **complete** (full 30k steps).
**Final eval scores are pending** — this report covers the training
phase + the partial eval signal we have. A follow-up `REPORT.md` will
land in the same directory once STAGE E finishes.

---

## tl;dr (so far)

#9 runs the same recipe as #6 (full-4096 30k learnable-τ baseline) but
with **MOIRAI-paper optimizer hyperparameters**: lr=1e-3 (10×),
weight_decay=0.10 (10×), β=(0.9, 0.98), no warmup, no cosine. Goal: an
apples-to-apples isolation of the optimizer-HP axis.

Two early findings worth recording before final eval:

1. **MOIRAI HP makes the contrastive backbone *better*, not worse**, on
   contrastive loss and gap (final EMA 5.13 vs #6's 5.32 on loss; gap
   0.49 vs #6's 0.39). The bigger lr + weight-decay aren't destabilising
   training — they're delivering a meaningfully tighter contrastive fit.
2. **The MOIRAI-HP qhead also fits tighter** (final EMA 0.0552 vs #6's
   0.0644, ~14% lower). Whether this translates to a better GIFT-Eval
   score is the open question; the partial 2-of-97 sample looks
   modestly better than #6 on `loop_seattle/5T/short` and roughly
   matched on `loop_seattle/5T/medium`, but n=2 is noise.
3. **τ trajectory is *suppressed* under MOIRAI HP.** Under default HP
   (#6) τ descends monotonically from 0.07 → 0.047 over 30k steps. Under
   MOIRAI HP the optimizer never lets τ drop: it oscillates above init,
   ending at τ=0.0765 (slightly *above* init). The 10× weight_decay is
   the most likely cause — `log_inv_tau` is a single trainable scalar
   and is regularised back toward init faster than the contrastive
   loss can pull it down.

If MOIRAI HP wins on eval, the takeaway is "default optimizer was
under-tuned for full-data" — which would be a clean lever for #6's
underperformance vs the small-data baseline. If MOIRAI HP loses, the
takeaway is "tighter head loss does not translate to better
GIFT-Eval on this short step budget" — i.e. step count is the
binding constraint.

## 1. Setup (delta from #6)

Identical to
[`exp_realonly_full4096_learnable_tau`](../exp_realonly_full4096_learnable_tau/REPORT.md)
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
report](../exp_realonly_full4096_learnable_tau/REPORT.md) for the
shared knob table.

## 2. Training results (complete)

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
(step 29,500).

For reference, the brief that triggered this report cited a
mid-training checkpoint EMA of 0.0614 at step 11k for #9 — that's the
EMA we logged at that step before training continued through to 0.0552
at 30k. The 11k value is in
`sync_realonly_full4096_moirai_hp/moirai_hp/run.log` but the FINAL
loss in the CSV supersedes it.

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
[`plots/full4096_default_vs_moirai_hp.png`](../../plots/full4096_default_vs_moirai_hp.png)
(generated by `scripts/plot_full4096_default_vs_moirai_hp.py`,
PR #98).

## 4. Eval status (partial, do not cite)

STAGE E started at 21:51:48 UTC on 2026-05-02 and is currently mid-eval
on the remote instance. The local sync has pulled the in-progress
`results/all_results.csv` with 2 rows so far:

| dataset                     | #6 MAPE_SN | #9 MAPE_SN | #6 WQL_SN | #9 WQL_SN |
|-----------------------------|-----------:|-----------:|----------:|----------:|
| `loop_seattle/5T/short`     | 1.1344     | 1.0918     | 0.9752    | 0.9675    |
| `loop_seattle/5T/medium`    | 1.4197     | 1.3333     | 1.3312    | 1.1272    |

n=2 is far from conclusive — these two configs are early in the
alphabetical eval order. The full 97-config GM result is what
matters and will land in the follow-up REPORT.md.

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
- **GIFT-Eval partial CSV (in-flight)** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/results/all_results.csv`
  (2 rows as of this snapshot; **do not** cite for final results)
- **Run log (training + partial eval)** —
  `sync_realonly_full4096_moirai_hp/moirai_hp/run.log`
- **Companion run script** —
  `experiments/exp_realonly_full4096_moirai_hp/run.sh`

## 6. What lands in the follow-up

The follow-up `REPORT.md` (replacing this `REPORT_INTERIM.md`) will
add:

- Final GM-MASE / GM-MAPE_SN / GM-CRPS_SN over all 97 configs
- A side-by-side eval table vs #6
- A go/no-go recommendation on MOIRAI HP for any larger
  step-budget follow-up

**Do not** scp from the remote until STAGE E completes and the run
emits its `=== run_full4096_moirai_hp ALL DONE ===` marker — the
local sync_loop will pull the final CSV and summary.txt automatically
on its next 15-minute tick.

## 7. Cross-references

- **#6 default-HP companion (DONE)**:
  [`experiments/exp_realonly_full4096_learnable_tau/REPORT.md`](../exp_realonly_full4096_learnable_tau/REPORT.md).
- **Small-data tau sweep (#27/#32)**:
  [`experiments/exp_realonly_4096_smaller_tau_sweep/REPORT.md`](../exp_realonly_4096_smaller_tau_sweep/REPORT.md)
  (PR #95).
- **Cross-arm loss/τ plot**:
  [`plots/full4096_default_vs_moirai_hp.png`](../../plots/full4096_default_vs_moirai_hp.png)
  (PR #98).
- **Sync-protocol audit**:
  [`docs/SYNC_PROTOCOL_REVIEW.md`](../../docs/SYNC_PROTOCOL_REVIEW.md)
  (PR #99) — relevant because both #6 and #9 used the new
  size-floored, append-only-protected sync_loop.
