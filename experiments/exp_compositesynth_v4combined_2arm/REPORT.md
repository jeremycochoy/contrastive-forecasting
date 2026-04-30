# Phase 4 — pulse + more-primitives combined

## TL;DR — v4 doesn't beat the individual winners

| metric | v4 EWMA-128 | v3 EWMA-128 (winner) | v4 RevIN | v2pulse RevIN (winner) |
|---|---:|---:|---:|---:|
| GM-MASE ↓ | 1.655 | **1.621** | 1.807 | **1.782** |
| median ↓ | **1.405** | 1.408 | 1.538 | 1.522 |
| max ↓ | **61.9** | 67.8 | 173.2 | **152.1** |
| configs<1.5 | 56/97 | **57/97** | 44/97 | **47/97** |
| H2H wins | **38/97** vs v3 | — | **36/97** vs v2pulse | — |

**6-way pool dilution confirmed.** With pool {sin, sq, saw, pulse, triangle, half_sin}, each primitive lands at 1/6 probability per wave slot, vs 1/4 (v2pulse) or 1/5 (v3). v4 inherits *neither* phase-2/3 winner's edge:
* At EWMA-128, the GM regression vs v3 (1.655 vs 1.621) outweighs the slight median + tail wins.
* At RevIN, v4 ties v3's GM but has the worst median + tail of the four composite arms; v2pulse keeps the RevIN crown.

The interesting positive: **v4 has the best max-MASE at EWMA-128** (61.9 vs v3 67.8 vs v2pulse 67.1). Pulse content does help tail-control under EWMA-128, just not enough to offset the dilution loss on the bulk of configs.

## Setup

Identical to phase 2/3 except both flags together: `--enable-pulse --more-primitives`. Two arms, two fresh Vast.ai instances, same recipe otherwise (mix=0.5, freq+seas emb dim=3, mixup=0.3, 30k+30k steps, B4 eval).

## Combined phase 1–4 ranking (97 configs, single seed each)

### EWMA-128

| rank | arm | flags | GM | median | max | <1.5 |
|---:|---|---|---:|---:|---:|---:|
| 1 | **v3** | `--more-primitives` | **1.621** | 1.408 | 67.8 | 57/97 |
| 2 | v4 | `--enable-pulse --more-primitives` | 1.655 | **1.405** | **61.9** | 56/97 |
| 3 | periodic baseline | (no composite synth) | 1.659 | 1.528 | 70.8 | 47/97 |
| 4 | v2pulse | `--enable-pulse` | 1.670 | 1.414 | 67.1 | 54/97 |
| 5 | composite-v1 (phase 1) | (no extra flags) | 1.697 | 1.459 | 66.3 | 51/97 |
| 6 | v2b | `--seas-heavy` | 1.704 | 1.450 | 66.1 | 53/97 |

### RevIN

| rank | arm | flags | GM | median | max | <1.5 |
|---:|---|---|---:|---:|---:|---:|
| 1 | **v2pulse** | `--enable-pulse` | **1.782** | 1.522 | **152.1** | 47/97 |
| 2 | composite-v1 (phase 1) | (no extra flags) | 1.785 | 1.514 | 194.3 | 48/97 |
| 3 | v3 | `--more-primitives` | 1.807 | **1.477** | 173.5 | 50/97 |
| 4 | v4 | `--enable-pulse --more-primitives` | 1.807 | 1.538 | 173.2 | 44/97 |
| 5 | periodic baseline | (no composite synth) | 1.859 | 1.568 | 190.4 | 43/97 |
| 6 | v2b | `--seas-heavy` | 1.866 | 1.554 | 200.9 | 44/97 |

## What the dilution-vs-diversity sweep teaches

```
pool size  →  GM-MASE EWMA-128  RevIN
3 (phase 1)        1.697         1.785
4 (v2pulse)        1.670         1.782
4 (v2b)            1.704         1.866   (different change: redundancy not diversity)
5 (v3)             1.621         1.807   ← sweet spot at EWMA-128
6 (v4)             1.655         1.807
```

* **Adding distinct primitives helps up to 5** — diversity adds capacity.
* **Adding a 6th (combined) regresses** — dilution starts dominating when each primitive's exposure drops to 1/6.
* **Adding redundancy (v2b) regresses unconditionally** — confirmed in phase 2.

The "diversity > quantity > redundancy" lesson from phase 2/3 holds, but with a ceiling: the pool can be too big.

## Decision: Best-of-breed, no combined recipe

Ship per-norm winners:
* **EWMA-128 → use `--more-primitives` (v3)**: GM 1.621.
* **RevIN → use `--enable-pulse` (v2pulse)**: GM 1.782.

Don't combine. The "natural next test" of phase 4 came back negative.

## Phase 5 candidates (synth-side knobs we haven't explored)

Per the worst-config analysis (74/97 configs still worse than seasonal naive at v3-EWMA-128), the remaining failure modes are:

1. **Explosive trends** (covid 67.8, m4_yearly 8.4, saugeen 5.0, healthcare growth 1.25 ish): we *do* have the multiplicative `exp(λt)` envelope at p=0.3 but `env_gain_range=(0.1, 10)` only spans 10× growth. Bumping to `(0.01, 100)` or `(0.001, 1000)` would expose covid-scale dynamics. Single-knob change, easy win to test.
2. **Spike-driven CloudOps** (bizitobs_application 16.3 / bizitobs_service 7.6 / bitbrains 6.3): pulse helped (max 194 → 152 on covid-RevIN) but Poisson-like burst arrivals aren't well-modelled by periodic pulse trains. A separate **shot-noise / Poisson-burst primitive** (random arrival times rather than periodic) would target this.
3. **Long-horizon drift** (electricity/15T/long, solar/10T/long): not a synth issue — rollout-strategy / forecast-head problem.

Phase 5 should pick #1 (env_gain bump) — 1-line synth change, parallel A/B vs v3 (best EWMA-128) and v2pulse (best RevIN). 4 instances total, ~3.5 h.

## Cost so far (phase 1–4, all 12 arms)

12 instance-runs × ~$0.34/h × ~5h = **~$20 total** for the full diversity sweep.

## Artefacts

* `run.sh` — single-arm driver for v4 (takes `revin`/`ewma128`)
* `results/gift_eval_{revin,ewma128}/all_results.csv` — 97 configs each
* `plots/gift_eval_v4combined_compare.png` — 6-arm comparison plot
* `scripts/plot_compare_2arm.py` — plotter (clone of v3, with v4 paths)
