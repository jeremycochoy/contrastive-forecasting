# Execution log

Operational events of the run. The report keeps the science, this file keeps
the events (`reports/REPORT_STANDARD.md`).

## The machine

elisa runs every arm. It holds two RTX 4090 cards. Other agents already train
on both cards, so this card shares them and stops no other run.

## What each part of the card holds

Measured on elisa, beside the other agents' runs.

| part | VRAM | rate |
|---|---|---|
| one backbone leg, probe batch 64 | 5388 MiB | 3.5 steps/s |
| one backbone leg, probe batch 16 | 1766 MiB | 3.5 steps/s |
| one 30,000-step head | 5464 MiB | 6.7 steps/s |
| one 97-config GIFT-Eval | none, CPU | about 2.9 core-hours |

The latent-drift probe, not the training, set what one leg held. It draws a
fixed batch and does one no-grad forward of it at every save step. At the
trainer's own batch of 64 that forward allocates a 4.32 GB block, and the
allocator keeps it for the run.

Card 0 held 3.9 GB free and card 1 held 9.6 GB. At 5.4 GB a leg fitted on
neither card beside a head, and a leg died in the probe on card 0 in its first
seconds. So every arm of this card runs at `--latent-drift-probe-batch-size
16`, which is `CF409_PROBE_BS` in `study.sh`.

That flag cannot move the training. `generate_arma_batch` draws the probe
batch from `np.random.default_rng(seed)`, a LOCAL generator, and `probe()`
runs under `torch.no_grad()`. It changes the drift CSV of every arm, which
this card does not read, and nothing else. Every arm takes the same value.

The heads of both lanes run on card 1. `head_eval_bb.sh` holds one head at a
time per card through a `flock`, so card 1 carries at most one backbone and
one head: 7.2 GB of its 9.6 GB.

## The checkout

`cf409_check_checkout` refused `~/workspaces/contrastive-forecasting`. That
checkout is 434 commits behind `origin/experiments`, and it carries none of
the three things this card needs.

`origin/experiments` does not carry them either: `--rep-loss-weight-end` is on
this card's branch and no merge has happened. So a checkout that is current
with `origin/experiments` would still refuse.

The study therefore runs from the branch worktree,
`/tmp/contrastive-forecasting-409`, which `study.sh` takes as `CF409_WT` by
default. The gate passed there after one copy: `experiments/hf_token.txt` is
gitignored, so a fresh worktree carries no token.

## The trial

`CF409_TRIAL=400` ran the whole pipeline before the eight arms started. It
used one GIFT-Eval config (`^us_births/M/short$`), not 97, because the wiring
this card adds sits above the eval and #373's 97-config path is unchanged.

It proved four things:

1. The backbone-to-score path. `ctrl_s20` trained 400 backbone steps, then a
   200-step head, then the eval, then `collect.sh` wrote `scores.csv`.
2. The decay reaches the trainer. `run_arm.sh` read the decay, the seed and
   the L_align target off the trainer's own command line for both shapes.
3. The decay reaches the loss. On `dec0_s20` the `rep_w` column fell 0.99 to
   0.0 over the scaled ramp, and `l_rep` went blank at weight 0.0.
4. The AUC table. `auc_watch.py` gave a verdict on the run that trained, and
   an error line on the run that did not.

The trial also showed the collapse the card warns about: on the 100-step
scaled ramp the AUC of `dec0_s20` fell from 0.86 to 0.52. The study ramp is
10,000 steps and the AUC gate watches it.

Trial artefacts stay out of git (`.gitignore`), because a trial score beside
the study's own results is one glob away from a study number. They are on
elisa at `results/trial/` and `/home/jupyter/checkpoints_backup/cf-409-trial/`.

## The lanes

The two lanes hold disjoint arms, so no arm can train twice.

| lane | card | arms |
|---|---|---|
| A | 1 | `ctrl_s20`, `dec0_s20`, `flr05_s20`, `flr02_s20` |
| B | 0 | `ctrl_s24`, `dec0_s24`, `flr05_s24`, `dec0T_s20` |

Lane A holds the control and the whole decay walk at seed 20260520, so the
card's main question has an answer even if lane B loses its card.

`lane_when_free.sh` starts a lane when its card can hold it. The first lane B
used it, because a 5.4 GB leg did not fit on card 0. The smaller probe made
the wait unnecessary, and the script stays for the next agent who must share
a full card.

## Events

| time (UTC) | event |
|---|---|
| 2026-08-22 22:44 | trial start |
| 2026-08-22 22:47 | trial score written, path proved |
| 2026-08-22 22:49 | first lane A start, probe batch 64 |
| 2026-08-22 23:05 | both lanes stopped after the VRAM measurements |
| 2026-08-22 23:11 | lane A restart on card 1, probe batch 16 |
| 2026-08-22 23:12 | lane B start on card 0 |

The first lane A ran `ctrl_s20` to about 3,000 steps at probe batch 64. Its
checkpoints were deleted, so no artefact of this card mixes the two probe
settings.
