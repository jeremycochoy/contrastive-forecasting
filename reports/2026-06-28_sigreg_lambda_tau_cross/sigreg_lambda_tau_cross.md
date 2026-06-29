# Crossing the SIGReg λ and EMA-τ single-axis winners does not produce a cell that beats either axis alone

**Question.** SIGReg's per-term weights (`λ_e`, `λ_h`) and the EMA-teacher
temperature `τ` were each tuned in isolation in prior work — a λ sweep
chose `λ_e` at fixed `τ=0.99`, and a τ sweep chose `τ=0.90` at fixed
`λ_e=λ_h=0.1`. Does pairing each axis's winner in a single run compound
the gain, or does the new operating point sit between its single-axis
parents?

**Answer.** It does not compound. On all four GM aggregates the best cell
is a single-axis `anchor_363` cell, not a cross arm.

| GM aggregate | best cell | arm | head / ckpt |
| --- | --: | --- | --- |
| Rel-MASE         | **1.1294** | anchor_363  `λ_e=10,  τ=0.99` | 6L / best |
| MASE (raw)       | **1.5788** | anchor_363  `λ_e=10,  τ=0.99` | 6L / best |
| MAPE / SN_MAPE   | **1.0618** | anchor_363  `λ_e=10,  τ=0.99` | 6L / last |
| CRPS / SN_CRPS   | **0.8502** | anchor_363  `λ_e=1000,τ=0.99` | 2L / last |

*GM-Relative MASE: geometric mean over GIFT-Eval's 97 tasks of model MASE
divided by seasonal-naive MASE. Lower is better; 1.0 = seasonal-naive.
The other three are the analogous geometric means of raw MASE, of
MAPE / SN_MAPE, and of mean-weighted-sum-quantile-loss / SN_CRPS.*

![GM-Relative MASE per (head × checkpoint) group, for the two crosses
(`cross_A`, `cross_B`) and the three single-axis anchors. The global
minimum across all twenty bars is an anchor (anchor_363 `λ_e=10, τ=0.99`,
6L / best, 1.1294). Within the two 'best' groups every anchor bar sits
below every cross bar; within the two 'last' groups cross_B sits below
every anchor, by ≤ 0.3 % of the best anchor in that group.](plots/headline_relmase.png)

## Arms

| arm | λ_e | λ_h | τ | provenance |
| --- | --: | --: | --: | --- |
| **cross_A**       | 10   | 1   | 0.90 | λ pair = prior λ-sweep best-at-best; τ = prior τ-sweep winner |
| **cross_B**       | 1000 | 1   | 0.90 | λ pair = prior λ-sweep best-at-last; τ = prior τ-sweep winner |
| anchor_363 (low)  | 10   | 1   | 0.99 | prior λ-sweep best-at-best (its native τ) |
| anchor_363 (high) | 1000 | 1   | 0.99 | prior λ-sweep best-at-last (its native τ) |
| anchor_357        | 0.1  | 0.1 | 0.90 | prior τ-sweep winner (its native λ pair) |

Backbone, q-heads, optimiser, batch (`B=512`), step count (12,500),
dataset, and seed (`20260520`) are held constant across arms. Only
`λ_e`, `λ_h`, and `τ` change. The launch-time manifest — including the
git revisions of the source sweeps and the cell values the manifest
points at — is `results/winners.locked.txt`.

## Where the cross lands vs the best anchor in the same group

![For each (head × checkpoint) group: the two cross bars and a black
tick marking the lowest single-axis anchor in that group. cross_A trails
the best anchor in every group; cross_B sits just below the best anchor at
both 'last' cells (Δ = −0.0020 on 2L / last, Δ = −0.0033 on 6L / last)
and above it at both 'best' cells.](plots/cross_vs_best_anchor.png)

Cross-vs-best-anchor deltas at last-checkpoint cells are −0.17 % (2L) and
−0.29 % (6L) in favour of cross_B. Within a single arm the best→last gap
on this same data spans **−3.4 % to +0.9 %** (`results/notes.md`), so
the cross-vs-anchor differences sit inside the per-arm best-vs-last band.

## Four aggregates

![Same five arms scored on GM-Relative MASE (top-left), raw GM-MASE
(top-right), GM-MAPE / SN_MAPE (bottom-left), GM-CRPS / SN_CRPS
(bottom-right). The globally lowest bar on each panel is an anchor_363
cell. cross_B is the lowest within the two last-checkpoint groups on
Rel-MASE and on raw MASE, but loses both last-checkpoint groups on
MAPE_SN and on CRPS_SN.](plots/four_aggregates.png)

## Training curves

![Backbone training curves for the two arms, rolling 100-step mean,
log-x. The total `loss`, the fixed-τ=0.07 diagnostic `loss_tau_ref`,
and the two utilisation diagnostics (`U_temporal`, `U_batch`) sit on
top of each other to the eye. The two SIGReg per-term penalties
(`sigreg_e`, `sigreg_h`) separate by roughly two orders of magnitude
because `λ_e` differs by 100× between the arms.](plots/training_curves.png)

## Method

Each backbone is trained 12,500 steps on `gift-pretrain-full-4096 /
small_v1` with `B=512`, enc3 + CPC auxiliary, EMA teacher on both the
GRU patch-embedding and the 3-layer encoder, fixed `τ` for the EMA copy,
SIGReg with per-term weights `λ_e` (embedding) and `λ_h` (encoding) on
the student. Each scored cell freezes the backbone, trains a fresh
quantile head (2L or 6L) at the same step budget, and evaluates on
GIFT-Eval's 97 tasks at `best-loss` and at `last` (step 12,500). Two
ckpts × two head depths × five arms = 20 GIFT-Eval cells; the cross arms
contribute eight of them, the three anchors contribute the other twelve.
GM aggregates are computed by `scripts/_compute_gm.py` against the
seasonal-naive `all_results.csv` from `~/workspaces/gift-eval/results/`.

## Caveat — single seed

Each cell is `N=1`. The cross-vs-best-anchor deltas at the last-checkpoint
cells are 0.17 % and 0.29 %, which is inside the **−3.4 % to +0.9 %**
within-arm best→last band on the same arms. Per the issue spec — *"if an
arm wins a headline cell clear of noise, multi-seed-replicate before
claiming"* — no cell wins clear of noise and the negative headline above
does not require multi-seed replication.

If the project wants to investigate the marginal `cross_B` 6L / last
advantage (1.1340 vs `anchor_357` 1.1373 = −0.29 % on Rel-MASE),
that confirmation is its own chained follow-up issue, not part of #366.
