# Phase 5+ follow-up ideas — comprehensive dump

*Written: 2026-04-30. Date-stamp added: 2026-05-02.*

Captured before pausing experimental work after phase 5. Includes both
ideas I'd act on next if budget allowed and ideas I'd dismiss with
reasoning. Keep as a thinking log — useful when we resume.

## Where the recipe stands at end of phase 4

Best per norm (synth-side knobs):
* **EWMA-128**: `--more-primitives` (v3 = sin/sq/saw/triangle/half_sin pool). GM 1.621.
* **RevIN**: `--enable-pulse` (v2pulse = sin/sq/saw/pulse pool). GM 1.782.

The diversity-vs-dilution sweep showed an **upper bound** on pool size:
3→4→5 primitives helps, 6 doesn't (v4 combined regresses).
"v2b seas-heavy" confirmed redundancy hurts.

74/97 configs are still worse than seasonal naive at v3+EWMA-128 (best).
The dominant remaining failure modes:
1. Explosive-trend extrapolation (covid 67.8, m4_yearly 8.4, saugeen 5.0)
2. Spike-driven CloudOps (bizitobs_application 16.3, bitbrains/rnd 5+)
3. M4 short-history (m4_yearly with 1 sample, m4_weekly ~52)
4. Long-horizon energy (electricity/15T/long, solar/10T/long)
5. Bulk of marginal failures (MASE 1–2)

## Synth-side ideas worth trying (highest signal first)

### A. Wider exp envelope range (phase 5, currently running)
Bumped `env_gain_range` from (0.1, 10) to (0.01, 100). Tests whether
exposing covid-scale 100× growth/decay reduces top-tail MASE. Outcome
will inform whether the env knob is dynamic-range-limited (fix:
extend further to (0.001, 1000)) or shape-limited (fix: try a
different envelope shape — see B).

### B. Tanh-shaped explosive trend
Real-world explosive trends (covid, infectious disease growth, viral
content) often follow a **logistic / saturating curve** rather than a
pure exponential. The series accelerates, then plateaus.

Concrete: add an `exp_logistic` envelope option with shape
`y(t) = scale × tanh(steep × (t - t_mid))` where:
* `steep ~ LogUniform(0.005, 0.05)` — controls the transition rate
* `t_mid ~ Uniform(0.2 * T, 0.8 * T)` — random transition midpoint
* `scale ~ LogUniform(1, 100)` — final magnitude

Currently we have `exp(λt)` (monotonic). A logistic variant adds
saturation. Add as an alternative to the existing env, fired with its
own coinflip (e.g., 30% logistic when env triggers).

### C. Poisson-burst primitive (aperiodic spikes)
Phase 2A's pulse primitive is **periodic** (a comb). Real CloudOps
bursts (bizitobs_application, bitbrains) are **aperiodic** — they
arrive at random times. Pulse helped some configs but the residual
errors on bizitobs are still in the 6–18 MASE range.

Concrete: add `_PRIM_POISSON_BURST`:
* sample N burst times from `Uniform(0, T)`, where N ~ Poisson(rate)
* each burst contributes `±1` for `width ~ Uniform(1, 4)` samples
* zero elsewhere

Different from pulse: pulse has a fixed period; Poisson-burst is
aperiodic. Each row gets a different number and timing of bursts.

### D. Regime-shift / piecewise-constant primitive
Some real series have **sudden level shifts** (regime changes:
business cycles flipping, sensor recalibration, accounting changes).
None of our current waveforms or trends model this.

Concrete: add `_PRIM_STEP`:
* sample K change-points from `Uniform(0, T)` where K ~ U{1, 4}
* between change-points the signal is constant at level `~ N(0, 1)`

### E. Damped oscillator primitive
`sin(2π·t/spp) × exp(-λt)` — periodic with decaying amplitude. Models
"event response" patterns: a perturbation that oscillates and decays
(seismic, financial shocks, epidemic waves with seasonality).

Decided NOT immediately useful: our trend × wave multiplicative
combination already produces something similar (multiplicative trend
with waves). Marginal additional value.

### F. Per-channel mixture instead of per-row coinflip
Currently each channel either has a primitive or doesn't (Bernoulli
gate). A more flexible recipe: each primitive is *always* present but
with a learned mixing weight `α_i ~ Beta(0.5, 2)` (heavy-tailed
toward 0). This way every channel sees every primitive at some
intensity, with most of them at low weight.

Predicted: incremental improvement on the diversity hypothesis but
risks dilution of every primitive's distinct character. Lower priority.

### G. Wider ARMA innovation scale
Currently `arma_target_std_range = (0.5, 3)` (where ARMA is rescaled
post-cumsum). Bumping to (0.3, 10) would let ARMA dominate more
strongly some rows. Predicted: helps trend-extrapolation configs
where ARIMA-d=1 is the right mental model.

### H. Wider trend slope_std
`slope_std=0.003` is very small (intended to give cumsum'd typical
amplitude ~1 without normalisation). Bumping to (0.001, 0.02)
log-uniform per-row would diversify trend strength. Predicted:
similar to G.

## Architecture / training-side ideas (not synth-side)

These won't be addressable purely by changing synth.

### J. Long-horizon rollout strategy
electricity/15T/long has horizon 720, ett1/H/long has 192. The
forecasting head is currently W=16-step reconstruction; rollout to
720 steps means 45 W-step iterations. Drift accumulates badly.

Ideas:
* Train head to predict longer horizons directly (multi-W reconstruction)
* Different rollout strategy: chunked vs. fully-autoregressive
* Use the encoder backbone's deeper context to ground the rollout

This was discussed in the head-rollout-comparison experiment
(`experiments/head-rollout-comparison/`) — R1 forecaster reconstruction
won. Could revisit at long-horizon-specific test.

### K. Larger model (Tiny → Small or Base)
Foundation models that beat seasonal-naive on GIFT-Eval are 100M+
params. Our Tiny is ~20M. Scale up: H=768 L=6 (Small ~42M params per
the bundle config), or H=1024 L=12 (Base ~150M).

Cost: training ~5x slower at Small, ~20x at Base. Need 24h+ runs.
Different experiment, different budget.

### L. Different reversible normaliser
We've covered RevIN, RevEWMNorm span ∈ {32, 128, 512}. Other options:
* RevTSN (time-series normalisation)
* DAIN (deep adaptive input normalisation)
* Static z-score per-config (no per-instance estimation)

Lower priority — RevEWMNorm-128 is competitive with the foundation
models we benchmark against.

### M. Longer training
30k steps on a 20M-param model is ~1× pass through bundle. Foundation
models train for 100s of GPU-days. Even doubling to 60k might help
(this was the original "scaling search" experiment, which showed
20L matches 12L gap but slower).

Not the highest-leverage knob — diminishing returns past 30k on Tiny.

## Data-side ideas

### P. Mix ratio sweep
We've fixed `mix_ratio=0.5` throughout. A 0.3 / 0.7 / 0.9 sweep
might find a sweet spot. Per-norm winners might prefer different mix
ratios.

### Q. Real-data-only baseline (next phase)
Pre-emptively scheduled: train best configs on
`jeremycochoy/gift-pretrain-small-4096` with `mix_ratio=0.0` (NO
synth). Measures: how much of our gain comes from the synth recipe
vs from the real-data pre-training? If real-data-only is competitive,
synth was a useful regulariser; if not, synth was the main lever.

### R. Curriculum: simple-to-complex synth
Start training with mix_ratio=1.0 + small env_gain, gradually shift
to mix_ratio=0.5 + full env_gain. Hypothesis: synth provides scaffold
for early training, real data refines later.

Speculative; needs design.

## Eval-side ideas

### S. SN-normalised MAPE / CRPS (queued task #18)
Add gluonts SeasonalNaivePredictor pass to emit per-config SN_MAPE
and SN_WQL columns; aggregate via per-config ratio to produce
Aksu-et-al-paper-comparable scores. Targets: 0.882 MAPE, 0.642 CRPS
(Moirai-Small-on-GiftEvalPretrain reference).

### T. Multi-seed runs
Single-seed phase 1–4 results have ~3–5% cross-seed variance per the
recovery-head search (Mar 2026). The ~4% v3 vs v2pulse gap at EWMA-128
is in that range. Multi-seed (3 seeds × 2 arms × 2 norms = 12 arms)
would tell us how much of the v3 win is real vs noise.

### U. Per-domain ablation
Some configs win at v3 but lose at v2pulse, others vice versa. A
mixture-of-recipes approach (different config classes use different
synth) would beat both. Requires meta-controller; speculative.

## Things NOT to try (with reasoning)

* **More-of-the-same redundancy** — phase 2B confirmed adding more of
  an existing modality (more seas-tied waves at the same period bucket)
  hurts. Rule of thumb: every new knob should add a *distinct* mode.

* **6+ primitives in the wave pool** — phase 4 confirmed 6 dilutes too
  much. Don't push past 5 unless we discover a way to keep individual
  primitive rates high.

* **Aggressive scale_max bumps** — current scale_max=1000 already
  produces float32 values up to ~1e4 post-env. Going to 10× or 100×
  more risks float32 overflow + makes RevIN normalisation less stable.

* **Pure-synth training** — the bundle's HF stream provides
  domain-realistic patterns (seasonality, trend, noise) the synth
  doesn't replicate well. mix_ratio < 0.3 hurt in early experiments.

## Operational lessons

* **macOS case-insensitive FS** vs lowercase `_final.pth` /
  uppercase `_FINAL.pth` is a recurring papercut. Future runs should
  use distinct names from the start (e.g. `_endoftrain.pth` and
  `_eval_canonical.pth`) to avoid case-collision rename gymnastics.
* **sync_loop is too slow** for FINAL/best_loss to land before run
  completes when GPU is fast (5090). Solution implemented per-run:
  manual scp of the 8 missing checkpoint files at end-of-run.
* **vastrun-provision SSH-attach failures** continue (one v5-revin
  retry needed). Workaround documented in CLAUDE.md.
* **HF httpx transient errors** can kill a training run mid-init
  (v5-revin first launch). Restart from scratch was clean. Consider
  adding retry logic in HFStreamingLoader.

## Cost summary so far

| phase | recipe | arms | hours | cost |
|---|---|---:|---:|---:|
| 1 (composite v1) | base composite | 2 | ~10h | ~$3.40 |
| 2A (v2pulse) | + pulse | 2 | ~5h | ~$3.40 |
| 2B (v2b) | + seas-heavy | 2 | ~5h | ~$3.40 |
| 3 (v3) | + more-primitives | 2 | ~5h | ~$3.40 |
| 4 (v4) | + both | 2 | ~5h | ~$3.40 |
| 5 (v5) | + env-bump | 2 | ~5h (in progress) | ~$3.40 |
| **total** | | **12** | | **~$20** |

## Next-up plan

After phase 5 reports + SN metric integration, the user has prioritised:
1. **Real-data-only training** on `gift-pretrain-small-4096` — answers
   the "how much of our gain is from the synth recipe vs from real-data
   pretraining?" question.
2. Architecture-search experiments later (using those checkpoints).

Then synth-side ideas A–H above would resume in priority order if budget allows.
