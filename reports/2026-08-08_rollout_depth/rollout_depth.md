# Rollout depth k = 3

Training the composed forecaster works, and it costs forecast accuracy.

`--train-rollout-depth 3` duplicates every loss term that ties `f` to `h` at
depths 1, 2 and 3, so the forecaster is trained on its own output. The
composed operator gets measurably better at exactly that. GM-Relative MASE
gets worse, by three to four times the head-seed band, and the damage lands
hardest on the short horizons the model was already winning.

## The objective did what it was designed to do

![rollout fidelity](plots/rollout_fidelity.png)

`cos(rollout_d, h_{T0+d})` on one fixed batch, for d = 1..16, using the
eval's own `rollout_latent`. No head, no metric — this is the composed
operator measured directly. `k = 3` is above `k = 0` at every depth on both
cells, and never below.

## And the metric moved the other way

![k = 3 against k = 0](plots/k3_vs_k0.png)

GM-Relative MASE over the 97 GIFT-Eval configs, this study's own `k = 0`
against its `k = 3`. The grey band is `ema_sched_ladder.md`'s pooled
head-seed range, ±0.0384. Every bar is several times wider than it.

## Where the damage lands

![horizon split](plots/horizon_split_student.png)

Short is 55 configs, medium and long 21 each. The card's criterion is drawn
on the right panel: medium+long at least 5% better, short losing less than
2%. Neither is met on any cell. But the *shape* is the one the mechanism
predicts — the loss is concentrated on short, and medium+long is hurt about
a third as much in relative terms.

![per-domain radar](plots/domain_radar_student.png)

Per-domain, the same picture: the `k = 3` polygon is outside the `k = 0`
one on every domain.

## Why: depth 0 pays for the deeper depths

![per-depth forecast error](plots/cos_err_depth.png)

`1 − cos(f^(j)_t, h_{t+1+j})` per training step. On B5 the `k = 3` run's
own depth-0 curve sits ABOVE the `k = 0` run's `1 − ff` for the whole run.
The depths are summed, so at `k = 3` the f-side carries four times its
baseline weight, and three quarters of that weight is on predictions the
head never reads. The one-step term is what the quantile head consumes, and
it is the term that got worse.

That is the whole result in one sentence: the fixed-point approximation
improved, the one-step prediction degraded, and the head reads the one-step
prediction.

## The published k = 0 numbers are not a baseline for this code

The card puts a gate before the comparison: retrain one cell per group at
`k = 0` on the new code and match the published number to within 0.0002.

| group | cell | head | published | retrained here | \|Δ\| |
|---|---|---|---|---|---|
| A | A3 | student | 1.1895 | 1.2189 | 0.0294 |
| B | B5 | student | 1.2748 | 1.3917 | **0.1169** |

Both fail the 0.0002 threshold. Group A's miss sits inside the pooled
head-seed band of 0.0384 and is the size of ordinary run-to-run noise.
Group B's does not: 0.1169 is three times that band and is **larger than
the effect this study set out to measure**.

So every number here is compared against this study's own `k = 0`, retrained
on the same commit, the same protocol and the same hardware. The card
prescribes exactly this. The consequence is that **B9 has no baseline**: its
`k = 3` run has no same-code `k = 0`, and the published 1.5579 cannot stand
in for one.

What moved between the two snapshots is not identified here. It is not the
loss — 48 frozen values and a digit-for-digit trainer reproduction pin
`k = 0` to the pre-change objective (PR #400) — so it is somewhere else in
the trainer, the head, the eval, or the hardware.

## What it costs

Cell B5, from the production runs' own timing lines, both depths on
identical RTX 5090s:

| | k = 0 | k = 3 | change |
|---|---|---|---|
| B5 forward + backward | 117.6 ms | 301.9 ms | **+157%** |
| A3 forward + backward | 115.9 ms | 137.8 ms | **+19%** |
| B5 GPU memory | 5375 MiB | 5585 MiB | +4% |

The depth is cheap when the f-bearing term has no denominator to rebuild —
A3's only such term is `L_align` — and expensive when it has one: B5's
pooled `xshh_allt` rebuilds `log_pos`, `log_neg_zy` and
`log_neg_cross_batch` at every depth.

Memory is nearly free because `FCST_GRAD_CKPT=1`, which all 14 cells already
set, checkpoints each depth's non-last forecaster layers. The depth costs
time, not VRAM.

A separate 600-step probe on elisa's shared 4090, alternating the two depths
three times, read +139% on B5's fwd+bwd. PR #400's CPU-only estimate was
+40%.

## Deviation from the card

**The h-anchored negative families shift with the depth.** The card's
default was to compute them once and reuse them unshifted, and it asks the
implementation to state which one it does. PR #400 takes the alternative, so
that a depth-`j` copy is a literal copy of the depth-0 objective under one
rule: every `h` index shifts by `j`.

It touches exactly one cell. B5 is the only cell whose f-bearing denominator
holds h-anchored families. B9's `L_pred` denominator is f-anchored only, and
the other twelve cells' f-bearing term is `L_align`, which has no
denominator at all.

## What ran, and what did not

The card lists 14 cells, each to bb40k and bb100k and conditionally bb200k,
two heads per stop, 97 configs per head — over 200 GPU-hours at the rates
measured here. The study had $7.31 of vast.ai credit and two elisa 4090s
that another session was already holding above 90% utilisation. It ran the
front of the card's own run order and stopped.

| | ran | did not run |
|---|---|---|
| cells at k = 3 | B5, B9, A3 | A1, A2, A4, B1, B2, B3, B4, B6, B7, B8, B10 |
| same-code k = 0 | B5, A3 (the two gates) | the rest |
| stops | bb40k | bb100k, bb200k |
| head-seed replicates | none | the card's annex figures |

Three of the five cells the card names as its run-early set are missing, so
rule 2 — `f` in the numerator and the denominator — is exercised by the
pooled shape and the split shape but not by the CPC auxiliary. Nine of the
ten cells whose only f-bearing term is `L_align` did not run.

Two consequences. There is no bb100k, so nothing here says whether a `k = 3`
cell that starts behind catches up — and the parents show cells that move by
0.05 between 40k and 100k. And there is no head-seed replicate in this
study, so the noise band is the parents' pooled ±0.0384 rather than one
measured here.

## The depth reached the loss, on every cell

Twelve of the fourteen cells carry `L_align` as their only f-bearing term.
Unwire the depth on that arm and the run completes, writes k+1 plausible
`cos_err_dj` curves, and reproduces the `k = 0` loss to the last digit. So
every cell was checked, not assumed — including the eleven that did not
go on to train.

Each cell ran its own launcher twice for one step, at `k = 0` and at
`k = 3`. Step 1 is the discriminating row: both runs start from the same
weights and draw the same batch, so `loss_tau_ref` — pinned to depth 0 —
must match, and `loss` must not.

All fourteen pass. `results/verify_summary.tsv` holds one row per cell, and
`results/verify_<cell>_k{0,3}_losses.csv` the CSVs each was read from.

## Collapse watch

![dimension usage](plots/dim_usage_per_arm.png)

The card names `u_batchtime` on `h_t` and `e_t` as the thing to watch: at
`k = 3` the f-side carries four times its baseline weight against the f-free
`L_rep` and SIGReg, and a model can win the deeper terms by flattening `f`.
Neither curve collapses.

![latent movement](plots/latent_movement.png)

Latent movement between the 20k and 40k checkpoints, on #379's committed
fixed batch. `k = 3` moves the encoder-output latent slightly more and the
patch-embedding latent less.

## Method

#393's protocol, unchanged.

Backbone `d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3,
batch_size=64, seed=20260520`; dataset `gift-pretrain-full-4096 /
small_v1`. Group A raises the EMA α linearly from 0.9 towards 1.0, anchored
to step 100k; group B holds α at 0.9. Every cell starts fresh at step 0.

Two heads per checkpoint, student and teacher, trained separately, each
evaluated on the encoder it was trained on. Head budget 15,000 steps at
bb40k, head seed 20260722, `--grad-clip 1.0` on the head. 97 GIFT-Eval
configs, official B4 strategy, forecast horizon 16, and #379's committed
seasonal-naive denominator.

GM-Relative MASE over a subset of the 97 is the geometric mean of
`our MASE / seasonal-naive MASE` over that subset. Over all 97 the number is
the eval's own, from `summary.txt`, not recomputed;
`scripts/split_scores.py` reproduces one of those aggregates as 1.414099
against a published 1.4141, which is what pins the subsets to the same
definition.

Head training ran on elisa's GPU, GIFT-Eval on elisa's 32 cores, backbones
on rented RTX 5090s and one 4090. That split is what made five backbone
runs affordable on $7.31: PR #394 measured the eval at 2.86 core-hours for
the 97 configs, with no VRAM at all.

**How a cell gets the flag.** Both published launchers ASSIGN `EXTRA_ARGS`
inside their per-cell `case` block and never read the environment, so an
exported value is silently overwritten. `scripts/make_launchers.sh` copies
each of the three launchers that carry the 14 cells and adds
`--train-rollout-depth "$K"` to the SHARED flag block. `diff` against the
parent is this study's whole deviation from the baseline protocol: that
flag, `OUT` moved to this directory, checkpoints moved to the durable root,
a `_cf373k<K>` run-name suffix, and `--log-every` made an env override with
its default unchanged.

## Uncertainty

Three sources, only one of them measured here.

- **Config sampling.** A paired dataset-cluster bootstrap, resampling
  datasets rather than configs because `<ds>/short`, `/medium` and `/long`
  are three configs of one series. `results/bootstrap.csv`. Every interval
  excludes zero, on the wrong side.
- **Head seed.** Not measured here. The parents' pooled range is ±0.0384.
- **Backbone training.** Not measured here, and not measured by the parents
  either. The gate failure above is the only handle on it, and it says the
  spread across code snapshots reaches 0.1169 on one cell — the same size as
  the effect. Read every delta in this report with that in mind.

## Tables

See [`results/scores.md`](results/scores.md) for the per-cell per-head table,
the gate table and the horizon split; [`results/bootstrap.csv`](results/bootstrap.csv)
for the intervals; [`results/steptime_runs.csv`](results/steptime_runs.csv)
for the per-run step times; and [`results/verify_summary.tsv`](results/verify_summary.tsv)
for the per-cell depth check.
