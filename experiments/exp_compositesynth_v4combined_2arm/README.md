# exp_compositesynth_v4combined_2arm — pulse + more-primitives

## Question

Phase 2/3 results showed:
* `--enable-pulse` (v2pulse) wins RevIN — pulse adds spike content
  that helps tail-prone domains (Web/CloudOps, Healthcare, Econ/Fin).
* `--more-primitives` (v3) wins EWMA-128 — triangle + half-sin add
  periodic-shape diversity that helps the strong-period domains
  (Energy, Healthcare, Sales).

The two flags target different aspects (new modality vs new shapes)
and win disjoint domain sets. Combining them should plausibly give
the best of both.

## What's new

Both flags simultaneously: `--enable-pulse --more-primitives`. Wave
pool grows to 6 primitives `{sin, square, saw, pulse, triangle,
half_sin}`. Each wave slot has 1/6 chance of any primitive.

No code change beyond the flag combination — both have already been
unit-tested independently. The combination test is empirical.

## Hypothesis

Best case: v4 inherits v2pulse's RevIN tail-control + v3's EWMA-128
median-and-aggregate-win.
* Target at RevIN: GM ≤ 1.78 (match v2pulse), median ≤ 1.48 (match v3).
* Target at EWMA-128: GM ≤ 1.62 (match v3), max ≤ 67.

Worst case: 6-way pool dilutes too much and v4 sits between v2pulse
and v3 on each metric, with neither's best.

## Setup

Identical to phase 2/3 except `--enable-pulse --more-primitives`
together. Two arms, parallel on two fresh Vast.ai instances.

## Phase 2/3 reference results (97 configs, single seed)

| arm | GM-RevIN | GM-EWMA-128 |
|---|---:|---:|
| v3 (more-primitives) | 1.807 | **1.621** |
| v2pulse (pulse) | **1.782** | 1.670 |
| v2b (seas-heavy) | 1.866 | 1.704 |
| composite-v1 (phase 1) | 1.785 | 1.697 |
| periodic baseline | 1.859 | 1.659 |

## Status

- [x] Code (both flags exist; combination is just `--enable-pulse --more-primitives`)
- [ ] 2 fresh Vast.ai instances provisioned
- [ ] Both arms launched
- [ ] Plotted vs phases 2/3
