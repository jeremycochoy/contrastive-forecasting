# Execution log

Operational events of the run. The report keeps the science, this file keeps
the events (`reports/REPORT_STANDARD.md`).

## The machine

elisa runs every arm. It holds two RTX 4090 cards. Other agents already train
on both cards, so this card shares them and stops no other run.

## The first attempt, and why it was stopped

The first implementation moved the study to #373's k = 3 cell, under the `sum`
reduction, with `--align-target student`, on the Fable opinion in
`scripts/fable_opinion.md`. It ran two control arms for about 2.5 hours.

The user stopped it. That cell answers a different question, and the card gives
the cell: k = 32 under `mean`, against the teacher, at the sweep's best EMA
momentum. See the issue comment of 2026-08-23.

Everything that attempt wrote is deleted: the checkpoints under
`/home/jupyter/checkpoints_backup/cf-409` and `cf-409-trial`, and every file
under `results/` and `plots/`. No number of that attempt enters this card.

## The latent-drift probe

The probe is a diagnostic. It draws a fixed ARMA batch once, then does one
no-grad forward of it at every save step. At the trainer's own batch of 64 that
forward allocates a 4.32 GB block, and the allocator keeps it for the run.

elisa's cards carry other agents' work, so every arm of this card runs at
`--latent-drift-probe-batch-size 16`. That is `CF409_PROBE_BS` in `study.sh`,
and every arm takes the same value.

That flag cannot move the training. `generate_arma_batch` draws the probe batch
from `np.random.default_rng(seed)`, a LOCAL generator, and `probe()` runs under
`torch.no_grad()`. It changes the drift CSV, which this card does not read.

## The checkout

`cf409_check_checkout` refused `~/workspaces/contrastive-forecasting`. That
checkout carries none of the things this card needs, and `origin/experiments`
does not carry them either: `--rep-loss-weight-end` is on this card's branch
and no merge has happened.

The study therefore runs from the branch worktree,
`/tmp/contrastive-forecasting-409`, which `study.sh` takes as `CF409_WT` by
default. `experiments/hf_token.txt` is gitignored, so a fresh worktree needs
one copy of the token before the gate passes.

## Events

| time (UTC) | event |
|---|---|
| 2026-08-22 22:44 | first attempt, at the k = 3 cell |
| 2026-08-23 01:48 | the user stopped it. Its two legs killed, its GPUs cleared |
| 2026-08-23 02:00 | its checkpoints and results deleted |
