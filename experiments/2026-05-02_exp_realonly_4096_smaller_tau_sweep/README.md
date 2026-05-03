# 2026-05-02_exp_realonly_4096_smaller_tau_sweep — τ sweep on smaller-EWMA

*Written: 2026-05-01. Last updated: 2026-05-02 (added τ=0.07 anchor at bs=96).*

## Question

Phases 1–5 + #19/#20/#22 all used `contrastive_divergence_temperature =
0.07` (CLIP convention). The optimal τ for our setting (T=4096, C=1,
mix=0.0, smaller arch, EWMA-128) hasn't been verified.

Quick 2-arm sweep around 0.07 to confirm we're at (or close to) the
optimum:

| τ | role |
|---|------|
| 0.05 | tighter contrast, more hard-negative focus |
| 0.07 | reference (already covered by #20 EWMA-smaller GM-MASE 1.78, MAPE_SN 1.24) |
| 0.20 | softer contrast, less hard-negative pressure |

If 0.07 wins on all three metrics → we're set, fixed-τ optimum confirmed.
If 0.05 or 0.20 wins → that's the new fixed default and the init point
for #28 (learnable τ).

## Setup change vs #19/#20/#22 — bigger batch

These runs use **`--batch-size 96`** (vs 24 in #19/#20/#22). On the 5090
the full forward+backward+optimizer step at bs=96 peaks at 19.8 GB
(measured); bs=128 OOMs at ~32 GB. Going to bs=96 means 4× the samples
per step → effectively faster data exposure at the same step budget.

**This breaks comparability with #19/#20/#22 absolute numbers.** Within
this τ-sweep the comparison is valid (all 3 arms use bs=96). Going
forward (#28 learnable τ, #23 train-to-completion, #21 if it launches)
will also use bs=96 unless documented otherwise.

The user explicitly accepted the comparability tradeoff for upcoming
runs in exchange for better MASE/MAPE/CRPS numbers.

## Setup

| knob          | value                                            |
|---------------|--------------------------------------------------|
| arch          | smaller (L=6 H=384 nhead=6, 11.43M)              |
| norm          | EWMA-128                                         |
| dataset       | jeremycochoy/gift-pretrain-small-4096, small_v1  |
| t_raw         | 4096                                             |
| n_channels    | 1                                                |
| mix_ratio     | 0.0                                              |
| **batch_size**| **96** (changed from 24)                         |
| total_steps   | 30,000                                           |
| lr            | 1e-4                                             |
| save-every    | 2,500                                            |
| grad-clip     | NONE (banned)                                    |
| freq-emb-dim  | 3, seasonality-emb-dim 3                         |
| mixup-p       | 0.3                                              |
| τ             | 0.05 / 0.20 (per arm)                            |

## Arms

Two new arms (0.07 already covered by #20 EWMA-smaller).

| arm   | τ    | host                                  |
|-------|------|---------------------------------------|
| t005  | 0.05 | RevIN box (35927139 ssh9.vast.ai:17138) — was freed when span=64 was killed |
| t020  | 0.20 | NEW 3rd 5090 — provisioned for parallel run |

EWMA box stays busy with #22 span=256 (in flight, ~2h to ALL DONE).

## Code change

`experiments/freq-embedding/scripts/train.py` got a new `--tau` CLI flag
(line ~115) that overrides
`LOSS_SPEC.train_configuration["contrastive_divergence_temperature"]`
when set. Defaults to None (preserves the LOSS_SPEC built-in 0.07).

## Status

- [x] `--tau` CLI flag added.
- [x] bs benchmark on 5090: bs=96 fits at 19.8 GB peak.
- [x] run.sh built (parameterised on τ).
- [ ] τ=0.05 launched on RevIN box (with bs=96).
- [ ] 3rd 5090 provisioned + τ=0.20 launched there.
- [ ] Plot + REPORT.md.
