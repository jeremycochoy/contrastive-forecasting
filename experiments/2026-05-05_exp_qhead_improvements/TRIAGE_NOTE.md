# Triage subset bias — note for future GIFT-Eval triage proxies

## What we did this round

`run_eval_elisa.sh --triage` filters the 97 GIFT-Eval configs by
`--config-filter` to keep only configs whose **test set is small**
(≤ ~1k instances): bizitobs_*, ett{1,2}/{15T,H}, electricity/H,
covid_deaths, us_births. This made triage finish in ~5 min instead
of ~6 h.

## Why it was biased

This drops entire configs (the m4_*, the long/medium variants,
loop_seattle, bitbrains_*) instead of sub-sampling within them. Two
problems:

1. **Domain skew**: the kept set is dominated by web/cloudops
   (bizitobs) and energy (ett, electricity). M4 (econ/finance), long-
   horizon transport (loop_seattle), and large IT-ops (bitbrains) are
   never sampled.
2. **Difficulty skew**: short-test-set configs are typically also
   short-horizon and short-context. Our forecasts degrade more on
   long-horizon configs (medium/long terms in m4 and loop_seattle).
   Excluding them inflates the triage score.

Empirically: baseline triage 1.128 vs full 1.183 (+0.055 bias).
R9_E13 triage 0.990 vs full 1.029 (+0.039 bias). The triage
**unfairly suggested R9_E13 beat seasonal naive** (0.990 < 1.000)
when on the unbiased full eval it does not (1.029 > 1.000).

## Better triage shape

Pick a small **random subsample of test instances within every
config**, instead of restricting to small configs. Two approaches:

- **Per-config cap**: cap each config's test set at e.g. 32 random
  instances. Run all 97 configs, but each is fast. Total ≈ 97 × <
  0.5 s ≈ 1 min. Domain coverage is preserved.
- **Stratified random subset**: keep all configs, but for the
  expensive ones (test set > 256 instances) deterministically
  subsample to 256 with a fixed seed. Cheap configs run in full;
  expensive configs are just truncated.

Either way: don't drop configs.

GluonTS exposes the test set length per dataset — the eval script
already knows which configs are big. The cap can go in
`run_eval_elisa.sh` as e.g. `--max-test-instances 32`.
