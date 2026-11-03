# SIGReg: the integration window

`--sigreg-embedding` and `--sigreg-encoding` add the LeJEPA SIGReg statistic
(#355) on the patch embedding `e` and on the encoder latent `h`. The
statistic projects the latents on random unit directions. For each
direction, it integrates the squared distance between the characteristic
function of the projections and the characteristic function of the target
`N(0, σ²)`, with `σ² = 1/K`:

```
EP = ∫ from -W to W of |φ̂(ω) - exp(-σ²ω²/2)|² dω
```

## The window

`W` is a frequency. The target characteristic function decays at about
`ω = 1/σ = √K`. The default bound is `W = 6/σ = 6√K`, where the target is
`exp(-18)`.

Before this fix, the default bound was `6/√K`. That value is the spread of
the projections, in the unit of the projections. At `K = 384` the window then stops where both
characteristic functions are within about `1e-4` of 1. So the statistic saw
almost nothing of the distribution, and SIGReg did not act on a collapse.

`sigreg_loss` on Gaussian features with the target's total variance,
`K = 384`, `N = 8192` samples, `M = 512` directions:

| features | bound `6/√K` | bound `6√K` |
|---|---|---|
| isotropic | 5.7e-9 | 0.025 |
| top direction with 66% of the variance | 1.4e-8 | 3.9 |
| all on 10 directions | 7.2e-9 | 1.8 |
| all on one direction | 8.6e-9 | 38 |

The isotropic value is the sampling noise of the `N` samples, about
`10 √K / N`.

## The scale of the term

The fixed statistic is about `√K` times the statistic of the standardised
projections against `N(0, 1)`. A weight that was tuned with the old bound
now acts much more strongly. Check the `sigreg_e` and `sigreg_h` columns of
`<run>_losses.csv` against the other terms.

## Reproduce an older run

`--sigreg-legacy-window` restores the bound `6/√K`. Use it only to reproduce
a run that trained before this fix. For example:

```bash
python experiments/2026-04-27_freq-embedding/scripts/train.py \
  --sigreg-embedding --sigreg-encoding --sigreg-legacy-window ...
```

The flag has no effect without `--sigreg-embedding` or `--sigreg-encoding`.
