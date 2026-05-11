# encoder-forecaster v2 — dropkey 0.7 instability + recovery

## Verdict

**Gate FAIL** at the q-head triage step: GM-Relative MASE = **1.505**
on 11 configs, vs gate threshold 1.0 and Sundial leaderboard 0.673.
Marginally better than yesterday's failed run (1.596) but still
seasonal-naive territory. **Full GIFT-Eval not launched** (chain
correctly stopped at the gate).

The result is *not* a real test of the encoder-forecaster architecture,
because the backbone never actually completed training. All three
dropkey p=0.7 configurations diverged in the 11k–15k step regime,
forcing an early stop at ~10k effective training steps — about 20% of
the planned 50k.

## Diagnostic — three p=0.7 attempts, three divergences

| Attempt | Mask sharing axis        | Outcome                  | Step  | Final ema_loss |
|---------|--------------------------|--------------------------|-------|----------------|
| 1       | Shared `(T,T)`           | NaN                      | 11700 | (NaN)          |
| 2       | Per-(B, head) indep.     | Diverged (loss 2 → 7+)   | 14900 | ~5.2 (climbing)|
| 3       | Per-(B, layer) indep., heads tied | Diverged (loss 2 → 4) | 14400 | ~4.0 (sustained)|

All three trained cleanly for 10–14k steps in a low-loss attractor
(`R² ≈ 0.99`, `U_b ≈ 0.4`, loss ema ≈ 1.5–2.0), then a single bad mask
draw kicked the model out into a higher-loss attractor that the
optimizer couldn't escape under p=0.7. Reducing variance by tying heads
within a batch row (attempt 3) lowered the divergence severity but did
not prevent it.

**Conclusion:** p=0.7 is the underlying problem regardless of mask
sharing axis. Lowering p (e.g. 0.3 or 0.5) is the likely fix but
departs from PLAN's prescription, so the night was stopped after
attempt-3 to await user OK in the morning.

## Recovery path used overnight

Promoted attempt-2's `_best_loss.pth` (step 10200, inherited by
attempt-3 which never improved on it) to
`enc_fcst_dropkey07_BACKBONE_step10200_FINAL.pth`. This is the lowest
ema_loss (1.32) checkpoint across all three attempts.

Q-head trained on this frozen backbone:
- Recipe: R9_E13 (xfmr 12L causal quantile, e_then_f, Moirai HP,
  cosine + 2k warmup) at 30k steps + bf16.
- Final ema_loss: 0.295 at step 30000 (1.7h on elisa GPU 1).

GIFT-Eval triage (11 configs) results:

```
Config                                 MASE   SN_MASE  Relative
bizitobs_application/10S/short         9.61   2.24     4.28
bizitobs_l2c/5T/short                  0.57   0.99     0.58
bizitobs_l2c/H/short                   1.25   1.21     1.03
bizitobs_service/10S/short             5.10   1.23     4.16
covid_deaths/D/short                  85.68  46.91     1.83
electricity/H/short                    2.31   1.36     1.70
ett1/15T/short                         1.63   0.93     1.75
ett1/H/short                           1.52   0.98     1.56
ett2/15T/short                         1.07   1.07     1.01
ett2/H/short                           1.03   0.92     1.12
us_births/D/short                      1.66   1.86     0.89
GM-Relative MASE                                       1.505
```

Reference (other systems on same triage configs):
Sundial 0.673 / TimesFM 0.680 / PatchTST 0.762 / Chronos 0.786 /
Moirai 0.809 / Naive 1.000.

The triage beats naive on `bizitobs_l2c/5T` and `us_births/D` (the
two configs where pattern is mostly periodic and the limited backbone
training was sufficient to learn it), and is comparable on
`ett2/15T` and `bizitobs_l2c/H`. Loses badly on the
`bizitobs_application/10S`, `bizitobs_service/10S`, and
`covid_deaths/D` configs that need richer learned representations.

## Held-out (not run)

Held-out contrastive-metric eval on the saved backbone is not in this
report — would re-confirm the in-training story (R² 0.99, top-1 1.0
on training-batch retrieval) but doesn't address the downstream
forecasting bottleneck, which is undertraining + p=0.7 instability.

## What did and didn't work

**Worked:**
- The architecture itself (GRU patch encoder → 6 causal encoder layers
  → 6 causal forecaster) trains cleanly and reaches `R²=0.99` on
  training-batch retrieval within 5–10k steps.
- The `--encoder-dropkey` flag's per-layer + encoder-only application
  is implemented correctly.
- The per-(B, head) mask shape (PR landed today as the attempt-2 fix
  for attempt-1's NaN at 11700) drops noise correlation across the
  batch, but doesn't fix the core p=0.7 instability.
- The chain-script approach (post_qhead_chain.sh) auto-running
  triage + gate + (full eval) reliably handled the q-head DONE event
  overnight without needing the agent to re-engage on each milestone.

**Didn't work:**
- p=0.7 in any of the three sharing axes (shared / per-(B,head) /
  per-(B,layer)). All three diverge in the 11k–15k step regime.
- 10k effective training steps is far too few to test the
  architecture against GIFT-Eval. The 50k target was never reached.

## What changed in this codebase

PR [#280](https://github.com/jeremycochoy/contrastive-forecasting/pull/280)
landed:

- `src/blocks.py`: `_dropkey_causal_mask(T, B, num_heads, ...)` now
  returns `(B*num_heads, T, T)`. Added `share_heads` arg to draw a
  `(B, T, T)` mask and replicate across heads.
- `src/models.py`: `encoder_dropkey_share_heads` plumbed to
  `TransformerBlock`.
- `experiments/2026-04-27_freq-embedding/scripts/train.py`: added
  `--encoder-dropkey-share-heads` flag.
- `experiments/2026-05-11_exp_encoder_forecaster/scripts/`: run.sh
  (50k attempt-1 → killed for attempt-2 with per-(B,head) fix),
  run_headshared_resume.sh (resume from attempt-2 best_loss with
  heads-shared), run_qhead.sh, run_gift_eval_triage.sh,
  post_qhead_chain.sh, plot_progress.py (4-arm), watchdog.sh.

## Recommended next step

Two options for the user in the morning:

1. **Lower dropkey to 0.3–0.5** with per-(B, head) (attempt-2's mask
   shape — known to fix the attempt-1 NaN), and retry the 50k
   backbone. If stable, continue to 166k for one full epoch.

2. **Drop dropkey entirely** (back to yesterday's failed-experiment
   spec but with the per-(B,head) mask architecture) and accept the
   position-counting risk. Use the new `auc_bt / top1_bt` metrics
   (PR #272) to detect counting on held-out — since downstream MASE
   is the real signal anyway.

Either choice needs explicit OK because it departs from PLAN.md's
`--encoder-dropkey 0.7` prescription.

## Artifacts retained

All three attempts archived under `checkpoints/` and copied as
`*_attempt{1,2,3}_losses.csv` + `run_..._attempt{1,2,3}.log`.
- `enc_fcst_dropkey07_50k_attempt1_*` — shared (T,T), NaN'd
- `enc_fcst_dropkey07_pb_50k_attempt2_*` — per-(B,head), diverged
- `enc_fcst_dropkey07_pb_hs_50k_attempt3_*` — heads-shared, diverged
- `enc_fcst_dropkey07_BACKBONE_step10200_FINAL.pth` — best-of-3
  backbone (= attempt-2's `_best_loss`)
- `enc_fcst_dropkey07_qhead_xfmr12L_quant_30k_FINAL.pth` — q-head
  trained on the above
- `gift_eval_triage/` — 11-config triage results (GM-MASE 1.505)
- `plots/progress.png` + `plots/progress_linear.png` — 4-arm log–log
  + linear comparison
