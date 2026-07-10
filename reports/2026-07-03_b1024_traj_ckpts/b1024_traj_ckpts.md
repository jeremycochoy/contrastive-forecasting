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
backbone checkpoints saved every 500 steps. Two causal transformer
quantile heads (2-layer and 6-layer, 6 attention heads, trained over
the frozen backbone) per backbone checkpoint; every head evaluated on
the full 97-task GIFT-Eval grid. Head-training protocol matches the
parent's: the step-500 head is trained 30,000 steps from scratch, and
every later locus resumes from that head for a 10,000-step re-adapt.
Single seed, matched to the parent.

Vocabulary. **τ** is the EMA rate for the target encoder. **λ_e, λ_h**
are the SIGReg weights on the embedding and head branches respectively.

## Result — spec-scoped

At the issue's fixed 12,500-step budget, **B=1024 loses every cell to
parent B=512**. Under the issue's stop rule, the experiment stopped
here.

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

## Result — extension to 50,000 steps

Backbone training was extended past the stop rule (out of issue
scope), in three legs (to 25,000, 37,500, then 50,000 steps), same
B=1024 recipe and 500-step trajectory saves. Heads retrained at nine
steps between 15,000 and 50,000, scored on the same 97-task grid.

- **6L traces a U-curve with its minimum at step 40,000** (1.1179 vs
  parent 1.1254, margin −0.0075): below the parent over steps
  30,000–45,000, back above at 50,000 (1.1286). The minimum sits at
  3.2× the parent step budget, 6.4× its sample budget. The margin is a
  single-seed point estimate.
- **2L never durably reaches parent.** Past step 25,000 it oscillates
  over a ~0.03 range (1.1485 – 1.1789); its single sub-parent point
  (1.1485 at step 40,000, margin −0.0006) is far smaller than that
  oscillation.
- Past step 40,000 both heads degrade.

![Backbone training loss](plots/backbone_loss.png)

Backbone training loss keeps decreasing through the extension; the
head-eval curves decouple from it, both degrading past step 40,000.

## Notes

- **Single seed, no confidence intervals.** The seed is fixed by spec
  (matched to the parent). All differences above are point-estimate GM
  aggregates over 97 tasks; the 0.006 – 0.019 GM-Rel MASE gaps are
  quoted without a variance band. "Worse" vs "within noise" is not
  separated.
- **The step-500 locus is a near-untrained backbone** (4 % of the
  12,500-step budget). It is included only because the parent's
  best-loss step landed there.
