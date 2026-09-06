# Where the artefacts live, and why no sync loop runs

This card runs on elisa, on its two RTX 4090 cards. No box on vast.ai holds any
part of it.

| artefact | path on elisa |
|---|---|
| backbones, optimizers, heads, GIFT-Eval output | `/home/jupyter/checkpoints_backup/cf-409/<arm>/` |
| losses CSVs | `<arm root>/arm6_v2_combab_alignT/leg_40k/` |
| logs, scores, AUC verdicts, `RUN_STATE.md` | `reports/2026-08-22_rep_weight_decay/results/` |

## No sync loop

`CLAUDE.md` asks for a sync loop on every REMOTE run, because a box can die and
take the only copy with it. This card writes to elisa's own disk. A copy from
elisa to elisa lands on that same disk, so it protects nothing.

The report states this in one sentence: elisa holds every artefact of this card,
and no other machine needs them.

## The two rules that replace it

The checkpoints sit outside any checkout, so no `git` command can touch them.
The `results/` tree does not: it sits in the checkout that runs the card.

1. Commit and push `results/` after each round. `git worktree remove --force`
   erases every untracked file of a worktree (`CLAUDE.md`, checkpoint safety
   rule 4), and the cards before this one ran from a worktree
   (`/home/jupyter/wt-cf-<issue>-train`).
2. Keep that worktree until the report merges.

`cf409_check_checkout` refuses a checkout that is too old for this card. It
reads the trainer flag, `GAP_ARGS`, #373's head script and the HF token.

## The reference scores are not artefacts of this card

This card runs no control. Its two references, 1.1507 at seed 20260520 and
1.1491 at seed 20260524, come from
`reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md`. Read them there. No
file under `cf-409/` holds them.
