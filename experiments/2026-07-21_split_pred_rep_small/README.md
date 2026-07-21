# Small-model long-training sweep — 6 arms of #374 × 200k steps (#379)

Report: [`reports/2026-07-21_split_pred_rep_small/small_long.md`](../../reports/2026-07-21_split_pred_rep_small/small_long.md).

## What this experiment adds vs #374

The six contrastive-loss-shape arms from
[`experiments/2026-07-10_split_pred_rep/`](../2026-07-10_split_pred_rep/)
run at a much smaller backbone for ≥4× longer, to separate
"architecture capacity" from "training duration" as the limiting factor
for two observations flagged in the #374 report:

1. The MoCo-on-L_rep arms (arm 6 v2, bimoco) compress h_t (dim usage
   0.23–0.29) while achieving the best eval GM-Relative MASE at 25k.
2. Arm 5 (L_align + L_rep) plateaus at `1 − ff ≈ 0.4` through 50k.

Same six loss recipes (loss-shape + MoCo flags + alignment weights)
copied verbatim from #374; only the backbone architecture and training
length change.

## Backbone architecture (identical across the six arms)

| Field                 | #374           | #379 (this run) |
|-----------------------|----------------|-----------------|
| `d_model`             | 384            | **128**         |
| `n_heads`             | 6 (head_dim=64)| **16 (head_dim=8)** |
| `num_encoder_layers`  | 3              | 3               |
| `num_layers`          | 6              | **3**           |
| `T`, `C`              | 4096, 1        | 4096, 1         |
| `encoder_type`        | gru            | gru             |
| `rev_norm_kind/span`  | ewma / 128     | ewma / 128      |
| Params                | ~17 M          | ~4–6 M          |

## Training schedule (identical across the six arms)

| Field                 | #374                     | #379 (this run)          |
|-----------------------|--------------------------|--------------------------|
| `batch_size`          | 512                      | **128**                  |
| `total_steps`         | 12,500                   | **200,000**              |
| `save_every`          | 2,500                    | **10,000**               |
| `extra_save_steps`    | —                        | **2500,25000** (`_2k.pth`, `_25k.pth` cells) |
| `lr`, `wd`, `betas`   | 1e-3, 0.1, (0.9, 0.98)   | 1e-3, 0.1, (0.9, 0.98)   |
| `seed`                | 20260520                 | 20260520                 |
| SIGReg λ_e, λ_h       | 1.0, 1.0                 | 1.0, 1.0                 |
| EMA teacher τ         | 0.90                     | 0.90                     |
| contrastive τ         | 0.10                     | 0.10                     |
| CPC auxiliary weight  | 1.0                      | 1.0                      |

## Arm table (loss flags copied from #374)

| Arm      | Launcher                          | Loss shape                                              | Extra flags                                          |
|----------|-----------------------------------|---------------------------------------------------------|------------------------------------------------------|
| arm 1    | `elisa_arm1_launch.sh`            | `cosine_similarity_batch_split_pred_rep`                | —                                                    |
| arm 3    | `elisa_arm3_launch.sh`            | `cosine_similarity_batch_split_pred_rep`                | `--moco-negatives`                                   |
| arm 4    | `elisa_arm4_launch.sh`            | `cosine_similarity_batch_full_hh_negs_xshh_allt`        | `--pos-in-denominator --subtract-contrastive-floor --moco-negatives` |
| arm 5    | `elisa_arm5_launch.sh`            | `cosine_similarity_batch_rep_only`                      | `--align-loss-weight 1.0`                            |
| arm 6 v2 | `elisa_arm6_v2_launch.sh`         | `cosine_similarity_batch_rep_only`                      | `--align-loss-weight 1.0 --moco-rep-keys`            |
| bimoco   | `elisa_bimoco_launch.sh`          | `cosine_similarity_batch_split_pred_rep`                | `--moco-negatives --moco-rep-keys`                   |

## Downstream (per arm)

Five backbone-step cells `{2k, 25k, 50k, 100k, 200k}` × two head-layer
sizes `{2L, 6L}` = 10 cells per arm × 6 arms = **60 GIFT-Eval cells**
total. Each cell: 40k-step transformer q-head training
(`head_arch=transformer`, `head_causal=true`, `head_nhead=6`,
`head_ffn_mult=4.0`, `head_train_input=e_then_f`, `forecast_len=16`,
`batch_size=256`, `lr=1e-3`), then full-97 GIFT-Eval B4.

## Reusable code changes

- `experiments/2026-04-27_freq-embedding/scripts/train.py` gains
  `--extra-save-steps` (comma-separated) so a run can snapshot off the
  `--save-every` cadence (needed here for the 2500 and 25000 cells).
  Covered by `tests/test_extra_save_steps.py`.

## Test coverage

- `tests/test_extra_save_steps.py` — schedule union semantics for
  `parse_extra_save_steps` / `should_snapshot`.
- `tests/test_small_long_launcher_shape.py` — verifies every arm
  launcher carries the small-model backbone config, the correct
  arm-specific loss flags, and the 5 backbone-step / 2 head-layer cells.
