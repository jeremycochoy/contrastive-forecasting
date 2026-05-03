# 2026-05-03_2026-05-02_exp_realonly_full4096_moirai_hp_FINAL — REPORT PLAN (#10)

**Status:** plan only; full REPORT.md lands after STAGE E. Plan file
2026-05-03.

## Goal

One full epoch on `jeremycochoy/gift-pretrain-full-4096` (path `small_v1`,
~42.5M windows per the codebase's prior reports), starting from the
**#9** MOIRAI-HP 30k backbone. #9 won the 30k vs #6 head-to-head
(GM-MASE 1.6391 vs 1.8043, GM-MAPE_SN 1.1850 vs 1.3698, GM-CRPS_SN
1.0155 vs 1.1000), so we lock the optimizer-HP axis and let step
budget vary along it.

## Resume bundle (4 files; pushed by operator pre-launch)

| file                                                          | purpose                                       |
| ------------------------------------------------------------- | --------------------------------------------- |
| `tiny_realonly_full4096_moirai_hp_30k.pth`                    | model state, the resume target                |
| `tiny_realonly_full4096_moirai_hp_30k_optimizer.pth`          | step counter, RNG, AdamW state, hf_rows seen  |
| `tiny_realonly_full4096_moirai_hp_losses.csv`                 | trainer appends (mode "a"); preserves history |
| `run_full4096_moirai_hp.log`                                  | optional but recommended for τ-trajectory grep |

## HP table (unchanged from #9)

| param         | value      |
| ------------- | ---------- |
| arch          | smaller (L=6 H=384 nhead=6, 11.4M) |
| RevIN         | EWMA-128   |
| batch_size    | 96         |
| T_raw         | 4096       |
| n_channels    | 1          |
| mix_ratio     | 0.0        |
| mixup_p       | 0.3        |
| τ             | 0.07 (learnable) |
| loss          | cosine_similarity_batch |
| **lr**        | **1e-3**   |
| **wd**        | **0.1**    |
| **β1, β2**    | **0.9, 0.98** |
| warmup        | none       |
| schedule      | flat       |
| grad-clip     | none       |
| save-every    | **10000** (was 2500 in #9) |
| **total-steps** | **498000** (= ceil(47.8e6/96), upper bound) |

## Wallclock & cost (5090, vast.ai)

- Backbone resume: 498k - 30k = **468k new steps** at ~1.7 sps (#9 rate
  on 5090) → ≈ 76 hours; rounded up: **~78h** including HF stream stalls.
- Q-head: 30k × ~0.6 s/step ≈ **5h**.
- GIFT-Eval: ≈ **5h** (matches #9).
- **Total ≈ 88h ≈ $34** at $0.39/h. Add slack: **add ≥ $30 to vast.ai
  before launch**.

## Watchpoints

1. **HF stream timeouts on a real-epoch loop.** The full-4096 set has
   4274 zstd shards; over a true epoch the streaming dataloader will
   touch every one. Watch the run log for `HfHubHTTPError`,
   long stall warnings, or sps-collapse below 0.5.
2. **Optimizer state replay continuity.** `load_training_state` reads
   `<path>_optimizer.pth` for step counter, RNG, AdamW momentum,
   hf_rows_consumed. After resume the very first 100 steps' loss must
   match the trajectory tail of #9's CSV — if it diverges by >0.05
   abs, something is wrong with state reload, not the run.
3. **τ-trajectory continuity.** #9 settled into τ ≈ 0.07–0.08 by 30k
   under the τ-suppression seen in the comparison plot. Expect this
   to persist throughout the resumed run; record per-checkpoint τ as
   in #9 to verify.
4. **Save churn.** With save-every=10000 over 468k new steps, ~47
   periodic snapshots; sync_loop must keep up at 15-min cadence
   (each snapshot is ~230 MB model+opt, so ≤ ~1 GB / 15min over the
   vast.ai scp proxy — should be fine, but verify after first tick).
5. **No grad-clip.** Project rule. If loss diverges, fix data /
   normalization, do NOT add grad-clip.

## Success criteria

- **Clear improvement:** GM-MASE ≤ 1.50 (from #9's 1.6391, an ~8%
  reduction) at full epoch.
- **Stretch (Aksu et al. baseline):** GM-MASE ≈ 0.882 / GM-CRPS_SN
  ≈ 0.642. Almost certainly out of reach for an 11.4M-param backbone
  on one epoch, but it's the directional target.
- **Floor (run was worth doing):** GM-MASE strictly below #9's 1.6391
  — anything else means step budget wasn't the limiter and we need a
  different axis (arch, data mix, loss).

## Sync target

`sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/` (NOT
`sync_realonly_full4096_moirai_hp/moirai_hp/` — that's #9's). Operator
launches `sync_loop.sh` after vastrun-provision returns SSH host/port.

## Followup

REPORT.md will land after STAGE E with: per-stage trajectory plots
(loss + τ continuous from step 0 to 498k, head loss 0–30k, GIFT-Eval
deltas vs #9 and #6), and a final go/no-go for any further axis
exploration.
