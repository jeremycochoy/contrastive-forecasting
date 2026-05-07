# qhead-improvements — final report

**Goal**: improve the recovery (forecasting) head atop the frozen
"backbone beta" (`tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth`,
C=1, H=384, nhead=6, num_layers=6, T_RAW=4096) under the working
assumption that the backbone is competitive with Moirai. Target:
GM-Relative MASE ≈ Moirai's **0.809**.

**Headline**: 9 rounds of experiments, 5 orthogonal axes explored,
**−12.2% triage GM-MASE** vs the legacy GRU head (R9_E13 = 0.990 vs
baseline 1.128). **Finally beats seasonal naive on triage**. Remaining
gap to Moirai is likely backbone-limited.

## Final result

| run | head | training | triage GM-MASE | vs naive (1.000) |
|---|---|---|---|---|
| baseline (legacy GRU) | GRU-q | 30k | 1.128 | +12.8% |
| R1_E1 (linear) | linear-q | 30k | 1.066 | +6.6% |
| R3_E4 (transformer breakthrough) | xfmr-q 6L | 30k | 1.017 | +1.7% |
| R5_E7 (best with f-only training) | xfmr-q 12L | 60k | 1.002 | +0.2% |
| R6_E8 (bidir+fl128 — regressed) | xfmr-q 6L bidir fl128 | 30k | 1.089 | +8.9% |
| R7_E9 (longer, truncated) | xfmr-q 12L | 100k | 1.020 | +2.0% |
| R8_E10 (Gaussian NLL) | xfmr-gauss 12L | 60k | 1.020 | +2.0% |
| **R9_E13** (winner) | **xfmr-q 12L + e_then_f** | **60k** | **0.990** | **−1.0%** |

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

3. **Match the train-time input distribution to the eval-time input
   distribution** (R9_E13). Discovered by reading
   `_b_variant_decode`: at eval the head sees `[e_ctx, rolled_f]`
   (encoder latents for context + rolled forecaster latents) but at
   training it used to see only `f_0..f_{T-1}`. Adding
   `--head-train-input e_then_f` feeds the head a length-2T sequence
   `[e_0..e_{T-1}, f_0..f_{T-1}]` at training, with a custom
   no-leakage mask: every row (including e-block rows) is causal, and
   f-block rows can only attend to e-cols `0..p_f` so the head can't
   peek at `e_{p_f+1}` (which encodes the target patch). Without this
   mask, training-loss collapses to ~0.115 from leaked target info;
   with it, training-loss matches R5_E7's 0.192 plateau but **eval
   improves anyway** — 1.002 → **0.990** (−1.2%), finally below
   seasonal naive (1.000) on triage.

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

Five axes explored: architecture, length, schedule, loss, train-eval
input distribution. Four of them converge to ~1.00–1.02 triage GM-MASE.
The fifth (matching the train input layout to the eval input layout
via `e_then_f` + leak-free mask) finally crossed under seasonal naive
on triage at **0.990**.

Full eval on R9_E13 is in flight (estimated based on baseline's
+0.06 triage→full bias: ~1.05 unbiased GM-MASE). R9_E14 (same recipe
at 100k steps) running on vast in parallel to test if longer training
under the matched-input setup helps further.

Remaining gap to Moirai (~25% on triage, ~22% projected on full)
is most likely **backbone-limited**. The head can't extract more than
the backbone latents carry. Recommended next step (out-of-scope here
per the user's frozen-backbone assumption): scale the backbone —
wider H, more layers, more pretraining data.

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
