# exp_realonly_4096_smaller_learnable_tau — CLIP-style learnable τ (#32)

*Written: 2026-05-02.*

## Question

#27 (the τ-sweep) tests fixed τ values 0.05 / 0.07 / 0.20. This
experiment tests whether **letting τ be a trainable scalar** improves
on the best fixed τ. CLIP-style: the model carries a single
`log_inv_tau` parameter, the loss uses τ = exp(-log_inv_tau), and
after every optimizer.step we clamp `log_inv_tau` to [0, log(100)] so
τ stays in [0.01, 1.0].

Hypothesis (per user spec): "the optimal τ drifts during training;
fixing τ is leaving signal on the table — particularly because L_in
is sensitive to τ. Small τ = hard-negative focus but unstable, large
τ = no contrast."

## Setup

| knob          | value                                            |
|---------------|--------------------------------------------------|
| arch          | smaller (L=6 H=384 nhead=6, 11.43M)              |
| norm          | EWMA-128                                         |
| dataset       | jeremycochoy/gift-pretrain-small-4096, small_v1  |
| t_raw         | 4096                                             |
| n_channels    | 1                                                |
| mix_ratio     | 0.0                                              |
| batch_size    | 96                                               |
| total_steps   | 30,000                                           |
| lr            | 1e-4                                             |
| save-every    | 2,500                                            |
| grad-clip     | NONE (banned)                                    |
| freq-emb-dim  | 3, seasonality-emb-dim 3                         |
| mixup-p       | 0.3                                              |
| τ initial     | **0.07**  (init log_inv_tau = log(1/0.07) ≈ 2.66)|
| τ trainable   | **YES**, clamped to [0.01, 1.0] post-step        |

Per user (May 2): #32 is NOT gated on #27's winner — runs in parallel
on machine4. Init at the established 0.07 baseline regardless of which
fixed τ wins.

## Code path (committed in 7fd89b0)

- `src/models.py` — ConfigurableModel gains `learnable_tau` and
  `tau_init` knobs. When True, registers `log_inv_tau` as a scalar
  nn.Parameter. `tau()` getter returns `exp(-log_inv_tau)` as a 0-d
  tensor (gradient-tracking). `clamp_log_inv_tau()` is the
  post-`optimizer.step` clamp.
- `src/loss.py` — `contrastive_latent_loss(..., tau_override=None)`.
  When the trainer passes `model.tau()` as `tau_override`, the loss
  uses that tensor → gradient flows back to `log_inv_tau`.
- `experiments/freq-embedding/scripts/train.py` — adds `--learnable-tau`
  CLI flag. Trainer passes `model.tau()` to the loss when set; runs
  `model.clamp_log_inv_tau()` after `optimizer.step()`; appends
  `τ=<value>` to the per-step log line so we can see the drift.
- `experiments/gift-eval/scripts/{train_forecasting_head,
  eval_gift_eval_official}.py` — auto-detect `log_inv_tau` in the
  backbone state_dict and set `BACKBONE_CONFIG['learnable_tau']=True`
  so `load_state_dict` succeeds. Head loss doesn't use τ; this is
  just to keep the param around.

## Acceptance

- Eval GM-MASE / GM-MAPE_SN / GM-CRPS_SN ≥ #27's best fixed-τ result.
- τ-trajectory plot (parsed from run.log τ=… lines paired with steps).
- Notes on whether τ converges, drifts up, drifts down, or oscillates.

## Status

- [x] Code committed (7fd89b0)
- [x] Run launched on machine4 (35970908 ssh8.vast.ai:10908)
- [x] sync_loop alive (sync_realonly_4096_smaller_learnable_tau/learnable/)
- [x] τ logging confirmed (saw 0.0704 → 0.0706 over first 700 steps)
- [ ] Backbone 30k done
- [ ] Qhead 30k done
- [ ] GIFT-Eval B4 done
- [ ] Plot + REPORT.md (#29)
