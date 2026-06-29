# Crossing the SIGReg λ and EMA-τ single-axis winners does not produce a cell that beats either axis alone

**Question.** SIGReg's per-term weights (`λ_e`, `λ_h`) and the EMA-teacher
temperature `τ` were each tuned in isolation in prior work — a λ sweep
chose `λ_e` at fixed `τ=0.99`, and a τ sweep chose `τ=0.90` at fixed
`λ_e=λ_h=0.1`. Does pairing each axis's winner in a single run compound
the gain, or does the new operating point sit between its single-axis
parents?

**Answer.** It does not compound. On all four GM aggregates the best cell
is a single-axis anchor, not a cross arm.

| GM aggregate | best cell | arm | head / ckpt |
| --- | --: | --- | --- |
| Rel-MASE         | **1.1294** | SIGReg sweep, λ_e=10  λ_h=1, τ=0.99   | 6L / best |
| MASE (raw)       | **1.5788** | SIGReg sweep, λ_e=10  λ_h=1, τ=0.99   | 6L / best |
| MAPE / SN_MAPE   | **1.0618** | SIGReg sweep, λ_e=10  λ_h=1, τ=0.99   | 6L / last |
| CRPS / SN_CRPS   | **0.8502** | SIGReg sweep, λ_e=1000 λ_h=1, τ=0.99 | 2L / last |

*GM-Relative MASE: geometric mean over GIFT-Eval's 97 tasks of model MASE
divided by seasonal-naive MASE. Lower is better; 1.0 = seasonal-naive.
The other three are the analogous geometric means of raw MASE, of
MAPE / SN_MAPE, and of mean-weighted-sum-quantile-loss / SN_CRPS. Each
"best cell" is the global minimum across all twenty
(2 head depths × 2 ckpts × 5 arms) configurations for that aggregate.*

Within the two `best`-checkpoint groups every single-axis anchor sits
below every cross bar; within the two `last`-checkpoint groups the
`λ_e=1000` cross sits below every anchor by ≤ 0.3 % of the best anchor
in that group.

![Grouped-bar chart: GM-Relative MASE per (head depth × checkpoint) group, five
arms per group — two cross arms (τ=0.90) and three single-axis anchors
(τ=0.99 at λ_e=10, τ=0.99 at λ_e=1000, τ=0.90 at λ_e=λ_h=0.1). Horizontal
dotted line at GM-Rel MASE = 1.0 marks the seasonal-naive
baseline.](plots/headline_relmase.png)

## Arms

| arm                                              | λ_e   | λ_h | τ    | provenance |
| ---                                              | --:   | --: | --:  | --- |
| **cross, λ_e=10 λ_h=1, τ=0.90**                  | 10    | 1   | 0.90 | λ pair = prior λ-sweep best-at-best; τ = prior τ-sweep winner |
| **cross, λ_e=1000 λ_h=1, τ=0.90**                | 1000  | 1   | 0.90 | λ pair = prior λ-sweep best-at-last; τ = prior τ-sweep winner |
| SIGReg sweep, λ_e=10 λ_h=1, τ=0.99               | 10    | 1   | 0.99 | prior λ-sweep best-at-best (its native τ) |
| SIGReg sweep, λ_e=1000 λ_h=1, τ=0.99             | 1000  | 1   | 0.99 | prior λ-sweep best-at-last (its native τ) |
| EMA-τ sweep, λ_e=λ_h=0.1, τ=0.90                 | 0.1   | 0.1 | 0.90 | prior τ-sweep winner (its native λ pair) |

Backbone, q-heads, optimiser, batch (`B=512`), backbone step count
(12,500), dataset, and backbone seed (`20260520`) are held constant
across arms. The downstream q-head protocol (head arch, total-steps
30000 best / 10000 resume-on-last, lr `1e-3`, schedule cosine, warmup
2000/1000, AMP off, full-97 GIFT-Eval at strategy B4) matches byte-for-byte
between this experiment's `scripts/downstream_sigreg.sh` and the two
anchor branches' equivalents
(`feature/contrastive-forecasting-363-v2:experiments/2026-06-24_sigreg_lambda_sweep/scripts/downstream_sigreg.sh`
and
`feature/contrastive-forecasting-357:experiments/2026-06-20_lejepa_sigreg/scripts/downstream_sigreg_tau090.sh`).
Only `λ_e`, `λ_h`, and `τ` change between arms. The launch-time manifest
— including the git revisions of the source sweeps and the cell values
the manifest points at — is `results/winners.locked.txt`.

## Where the cross lands vs the best anchor in the same group

The `λ_e=10` cross trails the best single-axis anchor in every group.
The `λ_e=1000` cross sits just below the best anchor at both
`last`-checkpoint cells (by 0.17 % at 2L / last and 0.29 % at 6L / last)
and above it at both `best`-checkpoint cells. Both deltas sit inside
the **−3.4 % to +0.9 %** within-arm best→last band measured on the
same data (`results/notes.md`).

![Per (head × checkpoint) group: the two cross bars (coloured) and a
black tick marking the lowest single-axis anchor in that
group.](plots/cross_vs_best_anchor.png)

## Four aggregates

The same five arms scored on the other three GM aggregates — raw MASE,
MAPE / SN_MAPE, CRPS / SN_CRPS — keep the same ordering at the
`best`-checkpoint groups (a single-axis anchor wins each). On the
`last`-checkpoint groups the `λ_e=1000` cross is the lowest cell on
Rel-MASE and on raw MASE, but loses both `last` groups on MAPE_SN and
on CRPS_SN.

![Four-panel grouped-bar chart: GM-Relative MASE (top-left), raw GM-MASE
(top-right), GM-MAPE / SN_MAPE (bottom-left), GM-CRPS / SN_CRPS
(bottom-right). Five arms per (head × checkpoint) group, same colour
coding as the headline figure.](plots/four_aggregates.png)

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

Each cell is `N=1`. The two `last`-checkpoint cross-vs-anchor deltas
sit inside the within-arm best→last band reported above, so a
multi-seed replicate that cleared the band would be required to claim
either as a win.
