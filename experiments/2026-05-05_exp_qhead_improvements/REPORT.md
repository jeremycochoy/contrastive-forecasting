# qhead-improvements — final report

**Goal**: improve the recovery (forecasting) head atop the frozen
"backbone beta" (`tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth`,
C=1, H=384, nhead=6, num_layers=6, T_RAW=4096) under the working
assumption that the backbone is competitive with Moirai. Target:
GM-Relative MASE ≈ Moirai's **0.809**.

**Headline**: 8 rounds of experiments, 4 orthogonal axes explored,
**−11.2% triage GM-MASE** vs the legacy GRU head (R5_E7 = 1.002 vs
baseline 1.128). Plateau hit. Remaining gap to Moirai is most likely
backbone-limited.

## Final result

| run | head | total_steps | triage GM-MASE | vs naive (1.000) |
|---|---|---|---|---|
| baseline (legacy GRU) | GRU-q | 30k | 1.128 | +12.8% |
| R1_E1 (linear) | linear-q | 30k | 1.066 | +6.6% |
| R3_E4 (transformer breakthrough) | xfmr-q 6L | 30k | 1.017 | +1.7% |
| **R5_E7** (winner) | **xfmr-q 12L** | **60k** | **1.002** | **+0.2%** |
| R6_E8 (bidir+fl128 — regressed) | xfmr-q 6L bidir fl128 | 30k | 1.089 | +8.9% |
| R7_E9 (longer, truncated) | xfmr-q 12L | 100k | 1.020 | +2.0% |
| R8_E10 (Gaussian NLL) | xfmr-gauss 12L | 60k | 1.020 | +2.0% |

Triage proxy: 11 small-test-set configs; biased ~0.06 below the full
97-config GM-MASE (validated: baseline triage 1.128 vs full 1.183).
Full eval on R5_E7 in flight on elisa GPU 0; expected ~1.06.

vs leaderboard (full eval, lower is better):

| Sundial | TimesFM | PatchTST | Chronos | Moirai | Naive | Baseline (#10) | **R5_E7 (estimated full)** |
|---|---|---|---|---|---|---|---|
| 0.673 | 0.680 | 0.762 | 0.786 | 0.809 | 1.000 | 1.183 | **~1.06** |

**~10% improvement** over the prior backbone-beta baseline; **~17%** behind
seasonal naive on full eval; **~31%** behind Moirai.

## What worked

1. **Replace the legacy bidir-GRU head with a causal transformer matching
   the backbone's depth + width.** R3_E4 (6L H=384 nhead=6, ~10.7M params)
   trained from scratch with Moirai HP (β2=0.98, wd=0.1, lr=1e-3) and
   cosine LR + 1k-step warmup broke through the linear-probe plateau:
   **1.066 → 1.017** (−4.6%).

2. **Stack depth + length on top of the transformer**: 12 layers + 60k
   steps + 2k warmup → R5_E7 = **1.002** (−1.5% on top of R3_E4).

## What didn't work (informative null results)

1. **Linear probe with Moirai HP + WSD (R2_E3)**: identical training-loss
   trajectory to constant-LR linear (R1_E1). Linear head is at its
   representational ceiling — HP/schedule changes can't move it.
2. **Bidir head + forecast_len=128 (R6_E8)**: regressed to 1.089. Train-
   test mismatch (bidir attends to real f's at train, rolled-out f's
   with rollout error at eval). Causality is necessary for this rollout
   evaluation regime.
3. **Longer training to 100k (R7_E9)**: truncated by spot-instance
   preemption at step 85k; result 1.020 ≥ R5_E7's 1.002 even before
   truncation. The cosine cooldown's benefit saturates by 60k.
4. **Gaussian NLL loss (R8_E10)**: same triage 1.020 as R7_E9. The
   pinball-loss training-loss plateau (~0.192 ema across all transformer
   variants) was *not* a loss-surface ceiling — it was the representation
   ceiling of the head + frozen backbone.

## Hypothesis going forward

Four independent axes (architecture, length, schedule, loss) all converge
to GM-MASE ~1.00–1.02 on triage with backbone-beta frozen. The remaining
~17% gap to seasonal naive (full eval) and ~30% gap to Moirai is most
likely **backbone-limited**: the contrastive-trained tiny backbone
(C=1, H=384, 6L, ~5M params, 167k pretraining steps) cannot encode
enough information for any head to recover. Recommended next step
(out-of-scope here): scale the backbone — wider H, more layers, more
pretraining data, possibly a Moirai-style mixture loss for the
backbone too.

## Pipeline summary

- 8 rounds of experiments, R1–R8, ~$11 vast.ai credit ($21.98 budget;
  vast topped out before R8).
- 6 PRs adding code: WSD/cosine schedules + AdamW HP flags (#126);
  linear-probe heads (#126); transformer head (#130); --head-causal
  flag for bidirectional variant (#136); explicit eval env-var
  overrides (#139); Gaussian NLL head (#141); plus 7 launcher PRs
  (#127, #128, #129, #131, #133, #134, #137, #140, #142).
- 88 unit tests on `tests/test_forecasting_head.py` covering each new
  head class (shape, param-count, causal/bidir mask correctness, B4
  strategy roundtrip, NLL loss correctness).
- All 11 head-type/recipe combinations evaluated on the same 11-config
  triage set with auto-detected head architecture from state dict.

## Artifacts

- launcher scripts: `experiments/2026-05-05_exp_qhead_improvements/scripts/`
  (8 launchers + the eval driver `run_eval_elisa.sh` with explicit
  `FL=`, `STRATEGY=`, `HEAD_CAUSAL=` env-var overrides).
- triage results: `experiments/2026-05-05_exp_qhead_improvements/results/`
  (per-run summary.txt + all_results.csv).
- best head (R5_E7): `sync_qhead_beta_rd5/checkpoints/R5_E7_xfmr12L_quant_moirai_cosine_60k_FINAL.pth`.
- candidate ledger with rationale per round: `CANDIDATES.md`.
- code (merged on `experiments`): `src/forecasting_head.py`,
  `experiments/2026-04-13_gift-eval/scripts/{train_forecasting_head.py,
  eval_gift_eval_official.py}`, `tests/test_forecasting_head.py`.
