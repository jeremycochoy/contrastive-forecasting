# Periodic Synth Mix — Design

## Motivation

v3b-120k evaluation (April 2026) showed the Tiny backbone is strongly worse
than seasonal-naive on every strongly-periodic dataset in GIFT-Eval:

| Dataset | Ours | Seasonal-Naive | Ratio |
|---|---|---|---|
| m4_hourly/H/short | 5.22 | 1.19 | 4.38× worse |
| solar/10T/medium | 3.15 | 0.93 | 3.40× worse |
| solar/10T/long | 2.08 | 0.87 | 2.39× worse |
| solar/H/short | 2.07 | 0.95 | 2.18× worse |
| ett2/W/short | 1.64 | 0.78 | 2.11× worse |
| ett1/15T/short | 1.78 | 0.93 | 1.91× worse |

Working hypothesis (HANDOFF.md): the training mix (base-bundles + TimesFM-style
synthetic composite) contains only *random-period* sinusoids inside a broader
ARMA+trend mix. The model sees "there is sometimes oscillation" but never
"this series has a clean, fixed period". It has no incentive to learn
seasonal-naive behaviour.

## Intervention

Add a **clean, single-primitive periodic synthesizer** to the training stream,
half-and-half with the existing base-bundles data.

## Experimental design

| Dimension | Value |
|---|---|
| Architecture | Tiny (C=4, H=512, W=16, GRU encoder, 6-layer transformer), identical to v3b |
| Total steps | 30 000 |
| Effective batch size | 24 (bs=24 * C=4 = 96 rows/step) |
| From-scratch | Yes |
| LR, norm, loss, augmentation | Identical to v3b |
| Seed | Fixed, paired between arms |

Two arms, paired on everything except data mix:

1. **CONTROL (v3c_ctrl)** — 24 samples/step from base-bundles HF stream.
2. **MIX (v3c_mix)** — 12 samples/step from base-bundles + 12 samples/step
   from on-the-fly periodic synthesizer.

30k steps is under-trained; the CONTROL arm is what lets us attribute any
periodic-dataset improvement to *the mix*, not to *more training*.

## Synthesizer spec

One primitive per series. No additive noise.

- **Primitive** — uniform choice over {sinusoid, square, saw}.
- **Samples-per-period (P)** — log-uniform in `[8, 256]` samples. This is the
  invariant that matters for the model; the physical sampling period (10s,
  1min, 1h, 1d, …) is informational only.
- **Phase** — uniform in `[0, 1)` (full period).
- **Sign flip** — 50% for square (random up/down), 50% for saw (random slope).
- **Envelope** — with `p = 0.3`, multiply by `exp(λ * t)` where `λ` is chosen
  so the total gain over `T=1024` samples is log-uniform in `[0.1×, 10×]`
  (i.e. `λ` can be positive or negative).
- **Scale** — log-uniform in `[0.1, 1000]`. Safe for float32 (max ≈ 10 000
  with envelope).

Per-batch: draw `batch_size * C` independent series, stack into
`[batch_size, T, C]`. All `C` channels are independent primitives with
independent parameters.

## Validation before training

- Plot 100 sample series (10×10 grid) and inspect visually.
- Dump per-series metadata + first/last/min/max/mean/std as text.
- Verify seasonal-naive MASE << naive MASE on the synth stream. If not,
  the synth is broken or misaligned with how the model sees the data.

## Hypothesis to confirm / falsify

- **H1**: MIX improves periodic-dataset MASE relative to CONTROL at matched
  compute.
- **H2**: MIX does not significantly degrade non-periodic-dataset MASE.
- **H3**: The aggregate GM-Rel MASE improves on the 6 periodic datasets
  identified in HANDOFF.

Effect size we expect to see if the hypothesis holds: factor-of-2 improvement
(or better) on the worst periodic datasets. Smaller effects at 30k steps may
be ambiguous given the noise of under-training.

## Pipeline

```
scaffold dir  ──▶  synth impl  ──▶  unit tests + visual inspect + naive sanity
                                 │
                                 ▼
                          mix dataloader  ──▶  train.py with --mix-ratio
                                                    │
                                                    ▼
                                            CPU smoke test (50 steps)
                                                    │
                                                    ▼
                                     provision vast.ai 4090 + sync code
                                                    │
                        ┌──────── backbone train (30k) ────────┐
                        │                                       │
                     v3c_ctrl                                v3c_mix
                        │                                       │
                     R1 head (30k)                          R1 head (30k)
                        │                                       │
                      GIFT-Eval B4                          GIFT-Eval B4
                        │                                       │
                        └────────▶  compare + report  ◀────────┘
```

## Budget

- Backbone: 2× 30k steps @ ~2.7 sps on 4090 ≈ 6h.
- R1 head: 2× 30k steps ≈ 1h each (frozen backbone, lighter compute).
- GIFT-Eval: 2× ~2h on 97 configs.
- Total: ~12h of single-4090 time ≈ $4–7 at on-demand prices.

## Known risks

- 30k is short; training-dynamics noise may dominate the effect.
  → We fix the seed and pair arms aggressively.
- Synthetic half might dominate early and slow base-bundles learning.
  → Covered by the CONTROL arm — if MIX loses aggregate MASE that's a
     real finding, not a bug.
- Seasonal-naive is not always well defined (needs a known period). The
  model isn't given the period explicitly. What we want is the model to
  *infer* the period from context and then copy. Synthetic coverage of
  many period values is how we force this skill.
