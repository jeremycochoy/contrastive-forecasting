# Crossing the SIGReg λ and EMA-τ winners does not compound; a nine-arm (λ_e, λ_h) grid at τ=0.90 moves every GM global minimum to a λ_e=1 cell, within single-seed noise

**Question.** SIGReg's per-term weights (`λ_e`, `λ_h`) and the EMA-teacher
temperature `τ` were each tuned in isolation in prior work — a λ sweep
chose `λ_e` at fixed `τ=0.99`, and a τ sweep chose `τ=0.90` at fixed
`λ_e=λ_h=0.1`. Two questions, answered in sequence: does pairing each
axis's winner in a single run compound the gain? And, extending the two
crossed arms to a nine-arm grid, where does the τ=0.90 optimum actually
sit in the (λ_e, λ_h) plane?

**Answer.** Crossing the winners does not compound: neither crossed arm
(λ_e=10 or λ_e=1000, λ_h=1, τ=0.90) holds any of the four global
minima. The grid instead moves every global minimum to a λ_e=1 cell at
τ=0.90 — but each by a margin (0.08–0.47 %) inside the single-seed
noise band.

| GM aggregate | best cell | arm | head / ckpt | margin over best other arm |
| --- | --: | --- | --- | --- |
| Rel-MASE         | **1.1254** | grid, λ_e=1 λ_h=1, τ=0.90  | 6L / last | 0.35 % (1.1294, SIGReg sweep λ_e=10, 6L/best) |
| MASE (raw)       | **1.5732** | grid, λ_e=1 λ_h=1, τ=0.90  | 6L / last | 0.35 % (1.5788, SIGReg sweep λ_e=10, 6L/best) |
| MAPE / SN_MAPE   | **1.0568** | grid, λ_e=1 λ_h=10, τ=0.90 | 6L / best | 0.47 % (1.0618, SIGReg sweep λ_e=10, 6L/last) |
| CRPS / SN_CRPS   | **0.8495** | grid, λ_e=1 λ_h=1, τ=0.90  | 6L / last | 0.08 % (0.8502, SIGReg sweep λ_e=1000, 2L/last) |

*GM-Relative MASE: geometric mean over GIFT-Eval's 97 tasks of model MASE
divided by seasonal-naive MASE. Lower is better; 1.0 = seasonal-naive.
The other three are the analogous geometric means of raw MASE, of
MAPE / SN_MAPE, and of mean-weighted-sum-quantile-loss / SN_CRPS. Each
"best cell" is the global minimum across all forty-eight
(2 head depths × 2 ckpts × 12 arms) configurations for that aggregate.*

At the two `best`-checkpoint groups the lowest Rel-MASE cell is the
τ=0.99 SIGReg-sweep anchor; at the two `last`-checkpoint groups it is
the grid's (λ_e=1, λ_h=1) arm.

![Grouped-bar chart: GM-Relative MASE per (head depth × checkpoint)
group, seven arms per group — the two crossed arms, the two best grid
arms (λ_e=1 λ_h=1 and λ_e=1 λ_h=10, both τ=0.90) and three single-axis
anchors. Horizontal dotted line at GM-Rel MASE = 1.0 marks the
seasonal-naive baseline.](plots/headline_relmase.png)

## Arms

| arm                                              | λ_e   | λ_h | τ    | provenance |
| ---                                              | --:   | --: | --:  | --- |
| **arm A (crossed winners)**                      | 10    | 1   | 0.90 | λ pair = prior λ-sweep best-at-best; τ = prior τ-sweep winner |
| **arm B (crossed winners)**                      | 1000  | 1   | 0.90 | λ pair = prior λ-sweep best-at-last; τ = prior τ-sweep winner |
| arm C (grid)                                     | 1     | 1   | 0.90 | grid extension |
| arm D (grid)                                     | 10    | 10  | 0.90 | grid extension |
| arm E (grid)                                     | 100   | 100 | 0.90 | grid extension |
| arm F (grid)                                     | 1000  | 1000| 0.90 | grid extension |
| arm G (grid)                                     | 100   | 10  | 0.90 | grid extension |
| arm H (grid)                                     | 1     | 10  | 0.90 | grid extension |
| arm I (grid)                                     | 100   | 1   | 0.90 | grid extension |
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

The two crossed arms hold none of the sixteen (aggregate × head × ckpt)
group minima. At the `best`-checkpoint groups the τ=0.99 SIGReg-sweep
anchor holds seven of eight — the grid's (1, 10) arm takes MAPE_SN at
6L. At the `last`-checkpoint groups the grid's (1, 1) arm holds six of
eight — the τ=0.99 anchors keep MAPE_SN at 6L and CRPS_SN at 2L.

![Four-panel grouped-bar chart: GM-Relative MASE (top-left), raw GM-MASE
(top-right), GM-MAPE / SN_MAPE (bottom-left), GM-CRPS / SN_CRPS
(bottom-right). Seven arms per (head × checkpoint) group, same colour
coding as the headline figure.](plots/four_aggregates.png)

## (λ_e, λ_h) grid at τ=0.90

The same headline GM-Relative MASE plotted as a (λ_e, λ_h) heatmap
restricted to the τ=0.90 runs. Ten cells are measured — the EMA-τ-sweep
anchor at (λ_e=λ_h=0.1) plus the nine arms at (1, 1), (1, 10), (10, 1),
(10, 10), (100, 1), (100, 10), (100, 100), (1000, 1) and (1000, 1000).
Hatched cells are not run. Performance degrades from the (1, 1) corner
toward large λ on both axes; the (1000, 1000) arm holds the worst cell
of the whole experiment on every aggregate.

![2×2 grid: (λ_e, λ_h) heatmap of GM-Relative MASE, one panel per
(head × checkpoint) group. Ten filled cells; the rest hatched.
Blue = better, red = worse.](plots/lambda_grid_tau090.png)

## (last − best) drift per grid cell

Same ten cells, plotted as the checkpoint drift `GM-Rel MASE(last) −
GM-Rel MASE(best)`, one panel per q-head depth. Negative (blue) means the
`last` checkpoint improved on the `best` checkpoint after the extra
training steps; positive (red) means it regressed.

![Two-panel heatmap of (last − best) drift per (λ_e, λ_h) cell at
τ=0.90, one panel per q-head depth. Blue = last beats best; red = last
regresses.](plots/lambda_grid_last_minus_best_tau090.png)

## Method

Each backbone is trained 12,500 steps on `gift-pretrain-full-4096 /
small_v1` with `B=512`, enc3 + CPC auxiliary, EMA teacher on both the
GRU patch-embedding and the 3-layer encoder, fixed `τ` for the EMA copy,
SIGReg with per-term weights `λ_e` (embedding) and `λ_h` (encoding) on
the student. Each scored cell freezes the backbone, trains a fresh
quantile head (2L or 6L) at the same step budget, and evaluates on
GIFT-Eval's 97 tasks at `best-loss` and at `last` (step 12,500). Two
ckpts × two head depths × twelve arms = 48 GIFT-Eval cells; the two
crossed arms contribute eight, the seven grid-extension arms twenty-eight,
and the three anchors twelve.
GM aggregates are computed by `scripts/_compute_gm.py` against the
seasonal-naive `all_results.csv` from `~/workspaces/gift-eval/results/`.

## Caveat — single seed

Each cell is `N=1`. The four global-minimum margins in the headline
table (0.35 %, 0.35 %, 0.47 %, 0.08 %) and the two crossed-arm deltas
above all sit inside the **−3.4 % to +0.9 %** within-arm best→last band
measured on the same data (`results/notes.md`), so a multi-seed
replicate that cleared the band would be required to claim any of them
as a win.
