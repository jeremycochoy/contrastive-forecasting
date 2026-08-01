# L_align on the EMA teacher — the 10 L_align cells of #379, rerun (#390)

Parent: [`experiments/2026-07-21_split_pred_rep_small/`](../2026-07-21_split_pred_rep_small/)
· Parent report: [`reports/2026-07-21_split_pred_rep_small/small_long.md`](../../reports/2026-07-21_split_pred_rep_small/small_long.md)

## What changes vs #379

One flag. #379 trained

```
L_align = 2 − 2·cos(f_t, stopgrad(h_{t+1}))          # h = student encoder
```

The intended term targets the EMA teacher:

```
L_align = 2 − 2·cos(f_t, h_teacher_{t+1})
```

`--align-target teacher` selects it. Every #379 run already trains an EMA
teacher (`--ema-embedding --ema-encoder --ema-tau 0.9`), so no new teacher
code and no new compute per step.

Everything else is #379's command line verbatim: `d_model=64, n_heads=8,
num_encoder_layers=3, num_layers=3`, `batch_size=64`, `seed=20260520`,
dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`.
`tests/test_390_launcher_shape.py` checks that claim against #379's
launcher, arm by arm.

## The 10 cells

Only `arm5` and `arm6_v2` pass `--align-loss-weight 1.0`. Each runs in the
five #379 settings:

| arm       | loss recipe                                                        |
|-----------|--------------------------------------------------------------------|
| `arm5`    | `cosine_similarity_batch_rep_only --align-loss-weight 1.0`          |
| `arm6_v2` | the same `+ --moco-rep-keys`                                        |

| setting  | added flags                                        |
|----------|----------------------------------------------------|
| base     | —                                                  |
| `tr1`    | `--tau-rep 1.0`                                    |
| `nse`    | `--sigreg-embedding-weight 0.0`                    |
| `ncpc`   | `--cpc-infonce-weight 0.0`                         |
| `combab` | `--tau-rep 1.0 --cpc-infonce-weight 0.0`           |

`arm1`, `arm3`, `arm4` and `bimoco` carry no L_align term. Their numbers
cannot change, so #379's 20 other cells are copied, not rerun.

## Running an arm

```bash
WT=$HOME/workspaces/contrastive-forecasting BB_GPU=1 \
  TARGET_STEPS=40000 SAVE_EVERY=10000 EXTRA_SAVES=2500,40000 \
  bash scripts/run_arm.sh arm5
```

Checkpoints land in this directory's `runs/`, named with the
`_alignteacher` suffix. Neither the directory nor the name can collide
with a #379 artefact.

Staged waves, as in #379: a wave trains to `TARGET_STEPS`, `_FINAL.pth` is
written only when `TARGET_STEPS ≥ FINAL_STEPS`, and the next wave resumes
from the newest `_<N>k.pth`. The issue's schedule is 40,000 → 100,000 →
200,000, with a q-head and a GIFT-Eval measurement after each wave, and
the third wave restricted to cells that improved from 40k to 100k.
