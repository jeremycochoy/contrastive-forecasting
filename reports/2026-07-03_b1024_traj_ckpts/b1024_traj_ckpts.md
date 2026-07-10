# B=1024 retrain of the τ=0.90 last-ckpt winner

**Question: does doubling the training batch size (B=512 → B=1024,
everything else fixed) make forecasts better? Answer: not within the
issue's 12,500-step budget — the doubled batch is worse in every spec
cell. Trained ~3× longer (≈ 6.4× the samples), it overtakes B=512 on
the 6-layer head only, by a small single-seed margin, and degrades
past step 40,000.**

![GM-Rel MASE vs backbone step](plots/gm_vs_step.png)

The retrained model is **arm C** (λ_e = 1, λ_h = 1, τ = 0.90), the
winner of the grid at `experiments/2026-06-28_sigreg_lambda_tau_cross/`;
its B=512 original is the **parent**. **GM-Rel MASE** = geometric mean
over the 97 GIFT-Eval-full tasks of `MASE / SN_MASE` (seasonal-naive
baseline); lower is better.

At the issue's two checkpoints, B=1024 loses every cell and the
issue's stop rule fired:

| head | ckpt | parent B=512 | B=1024 | Δ |
|:---:|:---:|:---:|:---:|:---:|
| 2L | best-loss step (500) | 1.1682 | 1.1873 | +0.0191 |
| 6L | best-loss step (500) | 1.1561 | 1.1746 | +0.0185 |
| 2L | last (12,500) | **1.1491** | **1.1621** | **+0.0130** |
| 6L | last (12,500) | **1.1254** | **1.1407** | **+0.0153** |

Extending past the stop rule (out of issue scope), to 50,000 steps:

- **6L bottoms at step 40,000** (1.1179, −0.0075 vs parent last) and
  sits below the B=512 re-run at every scored step from 15,000 through
  37,500 (the re-run's current end).
- **2L never durably beats B=512**: it oscillates over ~0.03
  (1.1485 – 1.1789) past step 25,000.
- Both batch sizes spike at step 35,000; past 40,000 both B=1024
  heads degrade.

![Backbone training loss](plots/backbone_loss.png)

The B=1024 loss minimum (≈ 3.23 at step ~680 on the 200-step moving
average, the early dip) does not coincide with its head-eval minimum
(step 40,000); within the extension the loss bottoms near step 35,700
and rises after.

## Protocol

- Backbone retrained at B=1024, seed matched to the parent, 12,500
  steps (extended to 50,000), checkpoint every 500 steps. The
  "best-loss step (500)" locus is the parent's measured best (step
  533) snapped to that 500-step checkpoint grid.
- Two causal transformer quantile heads (2- and 6-layer) per scored
  checkpoint, over the frozen backbone; the step-500 head trains 30,000
  steps from scratch, later loci resume from it for a 10,000-step
  re-adapt (parent-matched protocol). Full 97-task GIFT-Eval per head.
- τ is the EMA rate of the target encoder; λ_e, λ_h are the SIGReg
  weights on the embedding and encoding branches.
- Parent numbers in the table: `#366`'s committed `gm_table.csv` at
  `ba1df52`, arm `cross_C`. The B=512 curve on the plots is the
  parent's best point (step 533) continued by the seed-2 re-run
  (`results/b512_seed2/`, extracted from the in-flight #371 run);
  the seed-1 parent has only the two committed checkpoints. The
  B=1024 run is one seed; margins are point estimates with no
  variance band.
- The step-500 locus is a near-untrained backbone (4 % of the budget),
  included only because the parent's best-loss step landed there.
