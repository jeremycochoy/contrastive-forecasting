# #401 — why the phase-1 scores are 2x to 7x worse than #373

Diagnosis note, 2026-08-16. Supporting artefacts for PR #402.
This is not the study report. The study report comes when phase 1 ends.

## Verdict

The head and the GIFT-Eval path are correct. The three scores are correct
measurements of the backbones they were given. The backbones are collapsed.

At `k = 8` and `k = 16` the encoder becomes a constant function. It maps
every input series to one direction in latent space. A forecasting head on
a constant encoder has nothing to read, so the GIFT-Eval score is bad on
every config, at every horizon.

| cell | GM-Relative MASE | encoder effective rank | mean cos(h) between two series |
|---|---|---|---|
| #393 parent, k = 0, bb40k | 1.0862 (#373, k = 3) | 7.22 | 0.065 |
| #379 B5pub, k = 0, bb40k | 1.2751 (#373) | 10.76 | 0.079 |
| #401 k = 8, bb40k | 2.0357 | 1.00 | 1.000 |
| #401 k = 16, bb40k | 4.5297 | 1.01 | 1.000 |
| #401 k = 8, bb100k | 7.9344 | 1.00 | 1.000 |

![Collapse onset](../../plots/collapse_onset.png)

The k = 16 arm reaches chance AUC by step 200. The k = 8 arm holds until
about step 5,000, then joins it. The same cell at k = 3 climbs to AUC 0.95.

![Latent rank](../../plots/latent_rank.png)

The saved checkpoints, loaded the way the GIFT-Eval loads them, on 21 real
GIFT-Eval windows. Effective rank 1.0 means one direction.

## The cause

`docs/train_rollout_depth.md` states it, and the collapse follows it:

> At `k = 3` the f-side therefore carries four times its baseline weight
> against the h-side terms.

`--train-rollout-depth k` sums `k + 1` copies of every term that ties the
forecaster output `f` to the encoder latent `h`. This cell runs
`--loss-shape cosine_similarity_batch_rep_only`, whose depth copies return
zeros, so the only term that repeats is `L_align`:

```
L_align = weight * (2 - 2*cos(f_t, sg(h_{t+1})))          src/loss.py:686
```

`L_align` is a BYOL alignment term. It has no negatives. Its minimum is
`cos = 1`, which a constant encoder reaches for every depth at once.

The three terms that resist collapse enter ONCE at every `k`:
`L_rep` with MoCo keys, SIGReg on the embedding, SIGReg on the encoding.
This cell also sets `--cpc-infonce-weight 0.0`, so there is no InfoNCE.

So the ratio of alignment pressure to anti-collapse pressure is `k + 1`.
The step-1 loss measures it: 18.39 at k = 3, 27.37 at k = 8, 42.63 at
k = 16, a straight line of about 1.87 per added depth.

| k | L_align copies | anti-collapse copies | result |
|---|---|---|---|
| 3 | 4 | 3 | AUC 0.95, GM-Relative MASE 1.0862 |
| 8 | 9 | 3 | collapses at about step 5,000 |
| 16 | 17 | 3 | collapses at about step 200 |

The dose-response is monotone in `k`. `k = 32` will collapse faster still.

## What this rules out

**The `ARCH` list is not the cause.** `eval_gift_eval_official.py` reads the
architecture off the checkpoint, then calls `load_state_dict` STRICTLY
(`load_models`, line 442). The shard logs show the detection:
`num_encoder_layers=3`, `qk_norm=True`, `attn_out_norm=True`,
`freq_emb_dim=3`, `seasonality_emb_dim=3`. A list that did not match the
checkpoint would raise, not score badly. This diagnosis script hit exactly
that error on a first draft.

**The two checkouts run the same code.** `git diff c7e8af9d 8a9b567a`
outside `reports/` is one unrelated probe script and one test file. #373's
worktree is clean. So `WT=/tmp/contrastive-forecasting-401` and
`WT=/home/jupyter/wt-cf-373-train` give the same trainer, head and eval.

**The trainer command is #373's.** #401 adds no flag. `run_arm_k.sh` calls
#373's own `run_leg_k.sh`, which holds the whole command line. The one
input that differs is `K`.

**The damage is not horizon-shaped, it is global.** k = 8 at bb40k is worse
than #373's A4 on 95 of 97 configs, k = 16 on 97 of 97, k = 8 at bb100k on
97 of 97. See `per_config_vs_373.csv`.

**The head budget is under test.** 30,000 head steps here against #373's
15,000 default. Two controls run #401's path, at 30,000 head steps, on
known-good backbones. See below.

## Controls in flight

`scripts/diag_path.sh`, launched 12:15 on 2026-08-16.

| control | backbone | published at 15,000 head steps |
|---|---|---|
| c1 pathbound | #379 B5pub bb40k, #373's own G1 subject | 1.2751 |
| c2 k0anchor | #393 `arm6_v2_combab_alignS` bb40k, this cell's k = 0 parent | none |

c1 measures the head budget and nothing else: same runner, same seed, same
backbone, 30,000 steps against 15,000. c2 gives the depth ladder a k = 0
point that #401's own path measured.

Both write to `/home/jupyter/checkpoints_backup/cf-401-diag` and to
`results/diag/`. No score file of the study changes.

## Effect on the card

The card asks whether more iterative rollout improves GM-Relative MASE. As
specified, `k = 8`, `k = 16` and `k = 32` cannot answer it. Each raises the
alignment weight along with the depth, and the run collapses instead of
training a deeper rollout.

To separate the two, hold the f-side weight fixed:
`--align-loss-weight 1/(k+1)`, so 0.111 at k = 8, 0.0588 at k = 16 and
0.0303 at k = 32. #373 already ran this axis on other cells (`aw025`,
`aw4`), so it needs no new code. This is outside the card, so the card
owner decides.

## Files

| file | what |
|---|---|
| `collapse.csv` | the five checkpoints, latent spread, from `scripts/diag_collapse.py` |
| `per_config_vs_373.csv` | 97 configs, #401 against #373's A4 k = 3 bb40k |
| `diag_path.log`, `diag_path.out` | the two controls |
| `score_c1_*.txt`, `score_c2_*.txt` | the controls' GM-Relative MASE |
| `../../plots/collapse_onset.png` | AUC, dimension usage, cos_err_d0 against step |
| `../../plots/latent_rank.png` | the saved checkpoints, through the eval's loader |
