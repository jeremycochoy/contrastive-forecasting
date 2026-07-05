# Combining the SIGReg λ and EMA-τ sweep winners gives no clear gain; the best cells of a (λ_e, λ_h) grid at τ=0.90 have λ_e=1, within single-seed noise

**Question.** SIGReg's two loss weights (`λ_e` on the embedding, `λ_h`
on the encoding) and the EMA-teacher temperature `τ` were tuned in
separate sweeps: the λ sweep picked `λ_e=10` and `λ_e=1000` (at
`λ_h=1`, `τ=0.99`); the τ sweep picked `τ=0.90` (at `λ_e=λ_h=0.1`).
Does a run that uses both winners together beat them, and which
(λ_e, λ_h) works best at `τ=0.90`?

**Answer.** Combining the winners gives no clear gain: neither
combined arm holds the best score of any aggregate. The best scores
all sit at `λ_e=1`, `τ=0.90` — but by margins smaller than the
single-seed noise band (see Caveat), so no setting is a confirmed
winner.

![GM-Relative MASE bars for seven of the twelve arms, grouped by head
depth and checkpoint.](plots/headline_relmase.png)

| aggregate | best score | λ_e | λ_h | τ | head / ckpt | margin over best other arm |
| --- | --: | --: | --: | --: | --- | --: |
| Rel-MASE       | **1.1254** | 1 | 1  | 0.90 | 6L / last | 0.35 % |
| MASE (raw)     | **1.5732** | 1 | 1  | 0.90 | 6L / last | 0.35 % |
| MAPE / SN_MAPE | **1.0568** | 1 | 10 | 0.90 | 6L / best | 0.47 % |
| CRPS / SN_CRPS | **0.8495** | 1 | 1  | 0.90 | 6L / last | 0.08 % |

*GM-Relative MASE: geometric mean, over GIFT-Eval's 97 tasks, of model
MASE divided by seasonal-naive MASE; 1.0 = seasonal-naive, lower is
better. The other three aggregates are the same geometric mean of raw
MASE, of MAPE / SN_MAPE, and of CRPS / SN_CRPS. "Best score" is the
minimum over all 48 evaluations (12 arms × 2 head depths × 2
checkpoints).*

## Arms

One arm = one backbone trained with one (λ_e, λ_h, τ) setting;
everything else is identical (12,500 steps, `B=512`, seed `20260520`,
`gift-pretrain-full-4096 / small_v1`, enc3 + CPC auxiliary, EMA
teacher on the GRU patch embedding and the 3-layer encoder).

- Nine arms at `τ=0.90` cover the `λ_h=1` row, the `λ_h=10` row and
  the diagonal of the (λ_e, λ_h) plane (see heatmap below). Two of
  them, (10, 1) and (1000, 1), pair the λ-sweep winners with the
  τ-sweep winner.
- Three arms are reused from the earlier sweeps without retraining:
  (10, 1) and (1000, 1) at `τ=0.99`, and (0.1, 0.1) at `τ=0.90`.
  `results/winners.locked.txt` records their git revisions.

## Four aggregates

![Four-panel bar chart: GM-Relative MASE, raw GM-MASE, GM-MAPE /
SN_MAPE and GM-CRPS / SN_CRPS for the same seven
arms.](plots/four_aggregates.png)

All 48 evaluations are in `results/gm_table.csv`.

## (λ_e, λ_h) grid at τ=0.90

![Heatmap of GM-Relative MASE per (λ_e, λ_h) cell at τ=0.90, one
panel per head depth × checkpoint. Hatched cells were not
run.](plots/lambda_grid_tau090.png)

## Last vs best checkpoint

![Heatmap of GM-Relative MASE (last − best checkpoint) per (λ_e, λ_h)
cell at τ=0.90, one panel per head depth. Blue = the last checkpoint
is better.](plots/lambda_grid_last_minus_best_tau090.png)

## Method

Each arm freezes its backbone and trains a fresh quantile head (2 or
6 layers; 30,000 steps at the `best-loss` checkpoint, 10,000 resumed
steps at the `last` checkpoint, step 12,500), then evaluates on
GIFT-Eval's 97 tasks. The downstream protocol is byte-for-byte
identical for new and reused arms. Aggregates are computed by
`scripts/_compute_gm.py` against the seasonal-naive results in
`~/workspaces/gift-eval/results/`.

## Caveat — single seed

Every evaluation is `N=1`. All margins in this report (0.08–0.47 %)
are smaller than the −3.4 % to +0.9 % spread between `best` and `last`
checkpoints of the same arm measured on this data
(`results/notes.md`), so a multi-seed replicate would be needed to
call any ordering real.
