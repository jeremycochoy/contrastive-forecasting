# B=1024 retrain of the τ=0.90 last-ckpt winner

Follow-up to the τ=0.90 (λ_e, λ_h) grid landed at
`experiments/2026-06-28_sigreg_lambda_tau_cross/`.

## Question

Does doubling the contrastive batch size from B=512 to B=1024 — everything
else held fixed (seed, steps, τ, λ pair, optimiser, dataset) — reduce
GM-Rel MASE at the two spec-defined checkpoints?

- **Parent best-loss locus.** The step at which the parent B=512 arm
  reached its lowest training loss (step 533, snapped to the nearest
  trajectory-checkpoint multiple of 500 = step 500).
- **Spec-budget locus.** The parent's training budget of 12,500 steps.

**GM-Rel MASE** = geometric mean over the 97 GIFT-Eval-full tasks of
`MASE / SN_MASE`, where `SN_MASE` is the seasonal-naive baseline. Lower
is better; 1.000 = seasonal-naive.

Success criterion from the issue: if both retrained cells (2L and 6L
quantile heads) beat the parent's `last-ckpt` on GM-Rel MASE, extend to
≈ 25,000 steps and re-evaluate; otherwise stop.

## Design

One arm: the joint winner of the previous τ=0.90 A–I grid on both
`2L / last-ckpt` and `6L / last-ckpt` — **arm C** (λ_e = 1, λ_h = 1,
τ = 0.90). Backbone retrained at B=1024 for 12,500 steps with
backbone checkpoints saved every 500 steps. Two quantile heads (2L,
6L; 2-layer and 6-layer MLP over the frozen backbone) trained from
each of the step-500 and step-12,500 backbone checkpoints; every head
evaluated on the full 97-task GIFT-Eval grid. Single seed, matched to
the parent.

Vocabulary. **τ** is the EMA rate for the target encoder. **λ_e, λ_h**
are the SIGReg weights on the embedding and head branches respectively.

## Result — spec-scoped

At the issue's fixed 12,500-step budget, **B=1024 loses every cell to
parent B=512**: 2L 1.1621 vs 1.1491 (+0.0130); 6L 1.1407 vs 1.1254
(+0.0153). Under the issue's stop rule, the experiment stopped here.

![GM-Rel MASE vs backbone step](plots/gm_vs_step.png)

| head | ckpt | parent B=512 | B=1024 retrain | Δ (B=1024 − parent) |
|:---:|:---:|:---:|:---:|:---:|
| 2L | best-loss step (500) | 1.1682 | 1.1873 | +0.0191 |
| 6L | best-loss step (500) | 1.1561 | 1.1746 | +0.0185 |
| 2L | last (12,500) | **1.1491** | **1.1621** | **+0.0130** |
| 6L | last (12,500) | **1.1254** | **1.1407** | **+0.0153** |

The parent B=512 numbers come from the previous experiment's committed
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`
at commit `ba1df52`, arm `cross_C`.

## Result — owner-authorized extension

The repo owner authorized continuing past the stop rule in-session
(*"We can try to push a bit longer yes"*, after the 12,500-step numbers
were reported). Backbone training was extended to 25,000 steps, then to
37,500 steps, keeping the same B=1024 recipe and 500-step trajectory
saves. Heads were retrained from the step-15,000/20,000/25,000/30,000/35,000/37,500
backbone checkpoints and re-scored on the same 97-task grid.

- **6L crosses parent at step 30,000** (1.1206 < parent 1.1254) —
  2.4× the parent step budget, 4.8× the parent sample budget
  (batch doubling × step ratio). Best 1.1197 at step 37,500 (3.0×
  steps, 6.0× samples).
- **2L never crosses parent** (best 1.1560 at step 25,000) and degrades
  past step 30,000 (1.1789 at 35,000, 1.1731 at 37,500).

![Backbone training loss](plots/backbone_loss.png)

Training loss falls to a first minimum near step 500 (matching the
parent's best-loss step), rebounds to ~4.25 by step 2,500, and settles
to ~3.75 by step 12,500. The extension drifts down to ~3.65 by step
37,500 — a decrease not tracked by the 2L head-eval curve, which
degrades past step 30,000.

## Notes

- **Single seed, no confidence intervals.** The seed is fixed by spec
  (matched to the parent). All differences above are point-estimate GM
  aggregates over 97 tasks; the ~0.003 – 0.015 GM-Rel MASE gaps are
  quoted without a variance band. "Worse" vs "within noise" is not
  separated.
- **The step-500 locus is a near-untrained backbone** (4 % of the
  12,500-step budget). Its cells — 2L 1.1873 and 6L 1.1746, the worst
  in the table — reflect that, not model quality. The locus is included
  only because the parent's best-loss step landed there.
- **`step25000_6L` per-task CSV is a clean re-evaluation** whose
  aggregate reproduces the original exactly (GM-Rel MASE = 1.1289).
  The original per-task CSV is quarantined uncommitted.
