# The flags of the Moirai parts on the contrastive objective

Script: `experiments/2026-04-27_freq-embedding/scripts/train.py`, and the
scoring scripts in `experiments/2026-04-13_gift-eval/scripts/`.

Issue: [#412](https://github.com/jeremycochoy/contrastive-forecasting/issues/412).
The flags came with the value-space objective of #415, #417 and #421.
Since #412, three of them also reach the contrastive objective.

## The contrastive objective by row

A contrastive run (no `--value-space-objective`) trains by row when it
names one of these flags:

- `--rev-norm-kind meanstd`
- `--multi-patch-sizes`
- `--skip-nan-samples`

A contrastive term couples the rows of a batch. Its in-batch negatives, its
MoCo keys and its SIGReg statistics read every row. So a row leaves the
objective through the padding masks of #419. An inert row reads zeros, and
the masks mark all of its patches as padding.

The run needs terms that take a row mask:

| term | flag |
|---|---|
| `L_rep`, with or without MoCo keys | `--loss-shape cosine_similarity_batch_rep_only`, `--moco-rep-keys` |
| `L_pred` and `L_rep`, with or without MoCo negatives and keys | `--loss-shape cosine_similarity_batch_split_pred_rep`, `--moco-negatives`, `--moco-rep-keys` |
| `L_align`, in the loss or alone | `--align-loss-weight` |
| the CPC auxiliary | `--cpc-infonce-weight` with `--cpc-infonce-negs matched` |
| SIGReg | `--sigreg-embedding`, `--sigreg-encoding` |

The trainer refuses every other main shape, `--align-moco-loss-weight`,
`--subtract-contrastive-floor` and the other CPC negatives. It also refuses a
run on more than one rank (`WORLD_SIZE > 1`) and a forecaster other than
`--forecaster-kind transformer`.

## `--multi-patch-sizes P1,P2,...`

Default: off, one patch size (16).

Each size has its own GRU patch encoder. The body stays shared. Each sample
draws its size from its frequency's range (`src/patch_size.py`).

| objective | what a size group trains |
|---|---|
| value space (#417) | the value head of its size, on the values |
| contrastive (#412) | its patch encoder, while the batch terms read every row |

On the contrastive objective:

- Every group first runs its forward at its size P.
- L_rep, its MoCo keys and SIGReg read every row of the step. They run once
  on the time grid of the finest size G. On that grid a latent of size P
  repeats P/G times. So the rows line up in time, and each row has the same
  number of positions and the same weight. L_rep leaves every copy of an
  anchor's own patch out of the anchor's within-row negatives.
- L_align pulls each patch toward the next patch of its own row, rollout
  depths included. So it runs on the grid of each group. The group values
  add up with weights equal to each group's share of the active rows.
- L_pred of the split shape reads every row of the step. Each group first
  pairs the forecast of each patch with the key 1 + j patches ahead on its
  own grid, at rollout depth j. Then each pair fills P/G positions of the
  grid of the finest size, so the forecast and its positive move together.
  A coarse pair counts P/G times, as on the grid of L_rep. The positions
  after the last pair of a group hold no pair.
- At a grid position, an anchor takes the cross-batch keys of the other
  rows at that position. The key of a row there is the patch after the
  row's own patch at that position. With several sizes, that patch is not
  at the horizon of the anchor. The two agree when all the rows have one
  size.
- In the losses CSV, `l_pred`, `l_rep` and `sigreg_*` come from the whole
  step, and `l_align` adds up by the share of the rows.
- A step whose rows all read one size runs the whole objective on that
  group, as a single-size run does.
- Up to 10-01, every term ran on the rows of its group only. So L_rep, its
  MoCo keys and SIGReg read about one fifth of the batch. The runs #412om,
  #412oc and #412oe trained that way.
- The trainer refuses the CPC auxiliary (`--cpc-infonce-weight`): it does not
  run on the common grid.
- The trainer refuses a run on more than one rank (`WORLD_SIZE > 1`), on both
  objectives: two ranks draw different sizes.
- The EMA teacher copies the encoder of each size and updates by EMA.
- The rollout depth advances P values per depth at size P.
- The EWMA scaling starts from the mean and variance of the first 16 values
  (the first 16 real values after zero padding). No term reads a patch whose
  next patch starts in these 16 values, on either objective: its input reads
  values of its target. At size 8 this is the first real patch.
- The diagnostics read one sequence length, so they read the group of the
  base size 16. With no row at 16 they read the largest group. Only rows of
  the classes D, B, W and M, and rows with no frequency label, can draw the
  size 16. So `gap`, `auc`, `r2_*`, `cos_err_d*` and `loss_tau_ref` read
  those rows, and `<run>_best_gap.pth` selects on them. `loss_tau_ref`
  then pools fewer negatives than on a run of one group.

## `--rev-norm-kind meanstd` and `--meanstd-z-max Z`

The mean/std scaling of Moirai 1.0 (#421). Each window draws a target ratio
r ~ U[0.15, 0.5] of its whole patches. loc and scale read only the observed
values before that split, so no value after the split enters them.

| objective | positions the loss reads |
|---|---|
| value space (#421) | the values after the split |
| contrastive (#412) | every real (non-padding) position |

`--meanstd-z-max Z` (default 100) drops each window whose target part holds
a value more than Z scales from its context loc. 0 turns the filter off.
The filter can drop every window of a step. The contrastive step then reads
no row, and the trainer skips it: no weight moves and the teacher waits.

## `--skip-nan-samples` and `--skip-spike-samples K`

A step can give a non-finite loss or gradient. The trainer then finds the
rows that cause it (`src/nan_skip.py`) and drops only them. The step goes on
with the other rows. The trainer skips a step only when every row is bad.
`--skip-spike-samples K` uses the same search for a gradient norm above K
times the recent median (the clip norm before the guard holds 50 norms).

On the contrastive objective, a dropped row leaves every term: as an anchor,
a key, a positive, and a sample of the SIGReg statistics. The step after the
drop then equals the step on the batch without the row. A skipped step does
not update the teacher.

The search reads the forward of each row on the contrastive objective. One
row whose forward is not finite makes the input gradient of every row of its
size group NaN, through the coupled terms. So the rows with a non-finite
forward (student or teacher) are the outliers that the search drops first.
A row with only a non-finite input gradient ranks high, and the search
checks it on its own before it drops it. Every pass of the search runs the
forward of each group, so a NaN step costs one full step per pass.

## `--patch-stats none|diff|raw`

Default: `none`. The statistics come from the normaliser of the whole batch.
So the trainer refuses `diff` and `raw` with `--gift-pretrain` (the zero
padding moves them), `--rev-norm-kind meanstd`, `--multi-patch-sizes` and
`--skip-nan-samples` (a row reads the statistics of other rows).

## The scoring head

`train_forecasting_head.py` reads the patch sizes and the mean/std scaling
from the backbone checkpoint. For such a backbone it trains a head bank:

- One quantile head per patch size P. Head P decodes the 9 quantiles of the
  P values of the next patch. `--head-arch` picks the kind of each head.
- Each row draws its size and its split as the backbone trained. loc and
  scale read the context before the split. The pinball loss counts the real
  values after the split.
- Each row trains the head of its size. The group losses add up by their
  share of the rows.
- The checkpoint names the scaling, so `--rev-norm-kind` is not read.
  `--forecast-len` is not read either. The script refuses
  `--reconstruction forecaster`, `--mixed-rollout` and heads without
  quantiles.

`--reconstruction encoder` (#425) trains the same heads on another target.
Head P reads the encoder latent of each patch and decodes the P values of
that patch. The loss counts the same values as above. A backbone with one
patch size trains one quantile head on this target, and its zero padding
counts in no term.

`eval_gift_eval_official.py` loads the bank, checks that its sizes are the
backbone's, and gives each config the head of its frequency's inference
size. Strategy B4 reads the context at that size and rolls out one latent per
patch of P values. The forecast is unscaled with the loc and the scale of the
whole context. The script refuses a bank under a strategy other than B4 and
R: those read the context at the base size.

Strategy R (#425) is not a forecast. The encoder reads the B4 context and
the true horizon, and a reconstruction head decodes the latents of the
horizon patches. Each value is unscaled with the statistics that normalised
it. Under the mean/std scaling, these are the loc and the scale of the
context. Under the EWMA, they are the EWMA at that value. So the score
measures how much of each patch the encoder latent keeps.

`--zero-head` (R only) scores the floor of R. A head with no weights gives
the normalised value 0. So each value is the mean that normalised it. The
EWMA at a horizon value reads the true horizon until that value. Thus an
EWMA run has a lower floor than a mean/std run. Compare an R score with the
floor of its own scaling.

`head_eval_bb.sh` and `eval_local.sh` need no new argument for B4.
`CF_BB_SHAPE` gives the backbone shape, as before. For #425:

- `CF_RECONSTRUCTION=encoder` trains a reconstruction head and scores it
  under R.
- `HEAD_SAVE_EVERY` sets the head snapshot interval.
- `CF_SKIP_EVAL=1` stops after the head.
- `EVAL_DEVICE=cuda` runs the eval shards on the GPU.
- `EVAL_STRATEGY=R0` runs R with `--zero-head`.
- `CF_HEAD_ARGV_TO=<file>` writes the flags of the head trainer to the file
  and trains nothing.

`train_forecasting_heads_shared.py --jobs <file>` (#425) trains the heads of
some runs on one data stream. Each line of the file holds the flags of one
run of `train_forecasting_head.py`, as a JSON list. Each job keeps its
backbone, head, optimizer, seed and files. Each batch goes to each job in
turn. Each job uses its own random state for its step. So each job gets the
losses and the weights of its solo run, bit for bit. The script refuses jobs
that read a different data stream: source, seed, vocabulary, batch size,
start or step count.
