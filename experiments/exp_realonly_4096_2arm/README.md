# exp_realonly_4096_2arm — real-data-only training on gift-pretrain-small-4096

## Question

Phases 1–5 trained the contrastive backbone on a 50/50 mix of real
(`contrastive-training-base-bundles/base_mixed_v1`, T=1024, C=4) plus
synthetic (composite-synth recipe). Best result: **v3-prim + EWMA-128 →
GM-MASE 1.621 on GIFT-Eval**.

The open question: **how much of the gain came from the synth recipe vs
from the real-data pretraining?** If real-data-only is competitive, synth
was a useful regulariser but not the main lever; if real-data-only
underperforms, synth was the primary driver.

This experiment trains on the new `jeremycochoy/gift-pretrain-small-4096`
dataset with **no synth** (`mix_ratio=0.0`). The dataset has T=4096
(4× longer than the pretrain bundles used in phases 1–5) and C=1
(single-channel), so this also doubles as a **scale-up test** for the
backbone at 16× attention compute.

The resulting checkpoints are also the **basis for upcoming
architecture-search experiments** (Tiny → Small → Base sweeps).

## Dataset

`jeremycochoy/gift-pretrain-small-4096` — companion to the existing
`gift-pretrain-small`. Description: 10 series uniformly sampled from every
sub-dataset of `Salesforce/GiftEvalPretrain`, cropped into
non-overlapping windows of length 4096, globally shuffled. Series shorter
than 4096 yield zero windows.

* Path: `small_v1/`
* Format: 32 parquet shards (`shard_00000.parquet` … `shard_00031.parquet`)
  + `manifest.json`
* Total size: 2.4 GB
* Per-window: T=4096, C=1
* The `eval/` subdirectory is the standard GIFT-Eval test split (97 configs,
  same as phases 1–5). Used for the post-training eval.

## Architecture changes vs phases 1–5

| Knob              | Phases 1–5         | This exp           | Notes                                                      |
| ----------------- | ------------------ | ------------------ | ---------------------------------------------------------- |
| Source dataset    | `base_mixed_v1`    | `small_v1`         | `gift-pretrain-small-4096` repo                            |
| T_raw             | 1024               | **4096**           | 4× longer raw window                                       |
| W (patch size)    | 16                 | 16                 | unchanged                                                  |
| T_patches         | 64                 | **256**            | 16× more attention compute                                 |
| C (channels)      | 4                  | **1**              | single-channel                                             |
| `--mix-ratio`     | 0.5                | **0.0**            | NO synth                                                   |
| `--synth-kind`    | `composite`        | (omit / "none")    | should bypass synth path entirely                          |
| Norm              | revin / ewma-128   | revin / ewma-128   | both arms, unchanged otherwise                             |
| batch_size        | 24                 | **start 8**        | benchmark first; expect bs=4 fallback if OOM               |
| total_steps       | 30k                | **full single-pass** | with safety checkpoints; report 30k checkpoint AND final |

The code changes required to support C=1 and T=4096 are tracked in a
separate audit (the architecture audit subagent's report). They include
position-embedding cap and any hardcoded C/T assumptions.

## Arms

| arm     | norm     | priority | notes                                                                 |
| ------- | -------- | -------- | --------------------------------------------------------------------- |
| ewma128 | EWMA-128 | first    | reuses surviving instance 35892408 (5090) once #18 SN code is in     |
| revin   | RevIN    | second   | provision a NEW 5090 once EWMA arm is launched and stable            |

Sequential start, parallel runs once both are kicked off.

## Checkpoints to preserve

Per user's explicit request:
1. **30k checkpoint** (`*_30k.pth` + optimizer)
2. **Last / FINAL checkpoint** (end of full single-pass)
3. **Best loss checkpoint** (`*_best_loss.pth` + optimizer)
4. **Best gap checkpoint** (`*_best_gap.pth` + optimizer)
5. **Periodic safety checkpoints**: `--save-every` set so we get one every
   ~1 wall-hour at minimum (probably every 2k–5k steps depending on
   throughput).

These will be used downstream by architecture-search experiments. **Do NOT
delete locally** until #19's eval is done AND the architecture search has
the checkpoints it needs.

## Eval

After each arm finishes:
1. Run GIFT-Eval B4 on the 30k checkpoint AND the FINAL checkpoint.
2. Compare 4 ways:
   - phase 3 v3-prim + EWMA-128 (GM-MASE 1.621, the prior best)
   - this exp realonly EWMA-128 30k
   - this exp realonly EWMA-128 FINAL
   - same triplet for RevIN arm
3. If task #18 (SN-normalized metrics) is done by then: also report
   GM-MAPE_SN and GM-CRPS_SN against Aksu targets (0.882 / 0.642).

## Predicted outcomes

* **Best case**: realonly-EWMA-128 FINAL beats v3-prim+EWMA-128 GM 1.621
  by ≥5%. Means: real-data scale + longer T was the dominant lever; synth
  was a side dish. Architecture-search opens up at this point — scale to
  Small/Base on the same recipe.
* **Neutral**: realonly tracks v3-prim within ~3% (seed noise). Means:
  synth and real-data recipes contribute roughly equally. Both axes worth
  pursuing.
* **Worst case**: realonly-EWMA-128 underperforms v3-prim by >5%. Means:
  the synth recipe was load-bearing (probably regularisation effect) and
  removing it leaves the model under-trained on real-only. Future work:
  longer training, mix-ratio sweep at 0.1 / 0.3, or a curriculum.

## Cost estimate

Rough sketch (refined when training throughput is measured):
* T=4096 vs T=1024 → ~16× attention cost per token.
* bs=8 vs bs=24 → 1/3 effective compute per second.
* Single-pass on the dataset → ~10–50× phase 5's data volume.
* So per arm: 16 × 0.33 × 30 ≈ 160× phase 5 wall time.
* Phase 5 was ~5h × $0.37/h. Per arm: 160 × 5h × $0.37/h ≈ ~$300.
* **Two arms ≈ ~$600**.

This is much bigger than phases 1–5's $20 total. **Confirm budget with
user before going past $50** — if throughput is much worse than estimate,
fall back to single-arm or subset.

## Setup

* Code lives in `experiments/exp_realonly_4096_2arm/` (this dir).
* Sync dirs in main checkout (per CLAUDE.md): `sync_realonly_4096/<arm>/`.
* HF token at `experiments/hf_token.txt` as usual.
* Use raw `vastai create` (vastrun-provision SSH-attach bug, see CLAUDE.md).
* Label format: `realonly-4096-<arm>-<MMDD>`.
* sync_loop.sh per CLAUDE.md "Remote Machine Monitoring" rules:
  - 15-min cadence
  - safe_pull.sh atomic .tmp → mv
  - per-class size thresholds (BB ~80M, optimizer ~150M, head ~2.4M)
  - always sync optimizer files

## Architecture audit findings (2026-04-30)

Hardcoded constants in 4 files need to be CLI-args or configurable:

| File                                                       | Line(s) | Current        | Need                              |
| ---------------------------------------------------------- | ------- | -------------- | --------------------------------- |
| `src/dataloader.py`                                        | 23      | `T_RAW = 1024` | parameterise via factory arg      |
| `experiments/freq-embedding/scripts/train.py`              | 57      | `T_RAW = 1024` | new CLI flag `--t-raw`            |
| `experiments/freq-embedding/scripts/train.py`              | 44      | `C=4`          | new CLI flag `--n-channels`       |
| `experiments/gift-eval/scripts/train_forecasting_head.py`  | 50, 60  | C=4, T=1024    | match backbone via same CLI flags |

Plus one risk to validate before launching:

* **`src/norm.py`** — RevEWMNorm cumsum at T=4096. For span=128 (our planned
  norm), inv_decay max ≈ exp(32) ≈ 7.9e13 → safe in float64 (cap 1.8e308),
  also safe in float32 (cap 3.4e38). For span=32 it WOULD overflow float32
  (≈ exp(129)), but we don't use span=32. Smoke-test span=128 + T=4096 to
  confirm before launching cloud.

Other paths (causal mask cache `src/blocks.py:166`, channel mixing
`src/models.py:131`, freq-embed buckets `src/freq_embedding.py:62`) all
parameterise correctly for C=1, T=4096.

**`mix_ratio=0.0` plumbing**: confirmed working with freq-emb-dim>0 (our
default). The pure-HF fast path is bypassed when emit_freq_ids=True so
`MixedPeriodicLoader` is still used with `synth_bs=0` — yields
`(x, freq_ids, seas_ids)` tuples as expected.

## Memory expectations

Per audit subagent estimate (rough, validate at step 1):
* T_patches=256 vs 64 → 16× attention compute per channel
* C=1 vs C=4 → 1/4 channel multiplier
* bs=8 vs bs=24 → 1/3 batch multiplier
* Net: **~4× memory per backward pass** vs phase 1–5
* RTX 5090 = 32 GB GPU mem
* Plan: **start bs=8, expect bs=4 fallback on OOM**

## Status

- [x] Dataset ready on HF (`jeremycochoy/gift-pretrain-small-4096`,
      `small_v1/` split, 32 parquet shards, 2.4 GB)
- [x] Architecture audit done (4 CLI flags to add; norm/mask paths OK)
- [ ] Add `--t-raw` and `--n-channels` flags to train.py + train_forecasting_head.py
- [ ] Update `create_mixed_periodic_dataloader` factory to accept t_raw arg
- [ ] Smoke test RevEWMNorm(span=128, T=4096) for overflow-free path
- [ ] Local smoke test (bs=2, 50 steps, T=4096, C=1)
- [ ] run.sh + sync_loop.sh
- [ ] EWMA-128 arm launched on instance 35892408
- [ ] RevIN arm launched on a new 5090 (provisioned after EWMA stable)
- [ ] Both arms trained to single-pass completion
- [ ] GIFT-Eval B4 on 30k + FINAL for both arms
- [ ] REPORT.md
