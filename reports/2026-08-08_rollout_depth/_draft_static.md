## What the depth costs

Cell B5 (`arm4_combab_fix09`, pooled `xshh_allt` + floor + MoCo), on one
RTX 4090 shared with another session's training. Alternating k = 0 and
k = 3 runs, three repetitions of 600 steps each, median over the
post-warm-up windows.

| | k = 0 | k = 3 | change |
|---|---|---|---|
| forward + backward | 221.8 ms | 529.2 ms | +139% |
| whole step | 229.0 ms | 536.6 ms | +134% |
| throughput | 3.50 sps | 1.70 sps | −51% |
| GPU memory | 5375 MiB | 5585 MiB | +4% |

Three added depths cost 2.3x the step time and 4% of the memory.
`FCST_GRAD_CKPT=1`, which every one of the 14 cells already sets,
checkpoints each depth's non-last forecaster layers, so the depth buys its
memory back with recomputation and pays for it in time.

PR #400 estimated +40% from a CPU-only measurement. On the GPU, with the
per-layer fp16 sub-blocks the cells actually run, it is +134%. Plan with
the GPU number.

The cost is not uniform across the cells. B5's f-bearing term is the
pooled `xshh_allt` shape, whose f-anchored negative families rebuild at
every depth. The nine cells whose only f-bearing term is `L_align` have no
denominator to rebuild: measured on the running cells, A3 (`rep_only` +
`L_align`) trains at 5.5 sps against B5's 3.0 on comparable hardware.

## Deviation from the card

**The h-anchored negative families shift with the depth.** The card's
default was to compute them once and reuse them unshifted; it asks the
implementation to state which one it does. PR #400 takes the alternative,
so that a depth-`j` copy is a literal copy of the depth-0 objective under
one rule: every `h` index shifts by `j`.

It touches exactly one cell. B5 is the only cell whose f-bearing
denominator holds h-anchored families. B9's `L_pred` denominator is
f-anchored only, and the other twelve cells' f-bearing term is `L_align`,
which has no denominator at all.

## The depth reached the loss, on every cell

Twelve of the fourteen cells carry `L_align` as their only f-bearing term.
Unwire the depth on that arm and the run completes, writes k+1 plausible
`cos_err_dj` curves, and reproduces the k = 0 loss to the last digit. So
every cell is checked, not assumed.

Each cell runs its own launcher twice for one step, at k = 0 and at k = 3.
Step 1 is the discriminating row: both runs start from the same weights and
draw the same batch, so `loss_tau_ref` — pinned to depth 0 — must match, and
`loss` must not.

All fourteen pass. `results/verify_summary.tsv` holds the row per cell, and
`results/verify_<cell>_k{0,3}_losses.csv` the two CSVs it was read from.

## How a cell gets the flag

Both published launchers ASSIGN `EXTRA_ARGS` inside their per-cell `case`
block and never read the environment, so an exported value is overwritten
and the flag never reaches the trainer. `scripts/make_launchers.sh` copies
each of the three launchers that carry the 14 cells and adds
`--train-rollout-depth "$K"` to the SHARED flag block. Running `diff`
against the parent is the study's whole deviation from the baseline
protocol:

- `OUT` moves to this study's directory,
- checkpoints move to the durable root, outside any git worktree,
- the run name gains a `_cf373k<K>` suffix,
- `--log-every` becomes an env override, default unchanged,
- `--train-rollout-depth "$K"` on the line before `"${EXTRA_ARGS[@]}"`.

## Protocol

#393's, unchanged, for every cell here.

Backbone `d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3,
batch_size=64, seed=20260520`; dataset `gift-pretrain-full-4096 /
small_v1`. Group A raises the EMA α linearly from 0.9 at step 0 towards 1.0
at step 100k, anchored to step 100k; group B holds α at 0.9. Every cell
starts fresh at step 0.

Two heads per checkpoint, student and teacher, trained separately, each
evaluated on the encoder it was trained on. Head budget 15,000 steps at
bb40k, head seed 20260722, `--grad-clip 1.0` on the head, for comparability
with the parent reports. 97 GIFT-Eval configs, official B4 strategy,
forecast horizon 16, and the same seasonal-naive denominator file #379
committed.

GM-Relative MASE over a subset of the 97 is the geometric mean of
`our MASE / seasonal-naive MASE` over that subset. Over all 97 the number
comes from the eval's own `summary.txt` and is not recomputed;
`scripts/split_scores.py` reproduces one of those aggregates to
1.414099 against a published 1.4141, which is what pins the subsets to the
same definition.

Head training runs on elisa's GPU; GIFT-Eval runs on elisa's 32 cores.
PR #394 measured the eval at 2.86 core-hours for the 97 configs, against
58.7 s on a 4090 and 97.3 s on one core for the same six configs, with no
VRAM at all. That split is why a rented card in this study only ever trains
backbones.

## What ran, and what did not

The card lists 14 cells, each to bb40k and bb100k and conditionally
bb200k, two heads per stop, 97 GIFT-Eval configs per head. Measured on this
hardware that is over 200 GPU-hours. The study had $7.31 of vast.ai credit,
which buys 16 to 22 GPU-hours depending on the card, and two elisa 4090s
that another session was already holding at over 90% utilisation.

So the study ran the front of the card's own run order and stopped.

| | ran | did not run |
|---|---|---|
| cells at k = 3 | B5, B9, A3, A4 | A1, A2, B1, B2, B3, B4, B6, B7, B8, B10 |
| same-code k = 0 | B5, A3 (the two gates) | the other twelve |
| stops | bb40k | bb100k, bb200k |
| head-seed replicates | none | the card's annex figures |

Four of the five cells the card names as its run-early set are missing:
only B5 and B9 of {B5, B9, A2, B8, B10} ran, so rule 2 — `f` in the
numerator and in the denominator — is exercised by the pooled and the split
shape but not by the CPC auxiliary. Nine of the ten cells whose only
f-bearing term is `L_align` did not run either.

Two consequences for reading the numbers. There is no bb100k, so nothing
here says whether a k = 3 cell that starts behind catches up. And there is
no head-seed replicate in this study, so the noise band is the parents'
pooled ±0.0384 rather than one measured here.
