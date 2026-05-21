# #309 execution log

## 2026-05-20 12:24 UTC — vast box up
- Offer 35882331 → instance 37144527 @ ssh9.vast.ai:24526
- 1×RTX 4090 (48GB VRAM displayed), prosumer, $0.55/h, reliability 0.992
- 2-GPU DDP supply was empty at provision time; box_run.sh falls back
  to 1-GPU bs=256 (mathematically identical per train.py:739-740
  "gathered loss (global negatives, == 1-GPU @ global B)").

## 12:27 — α started; diverged by step ~1000
α = (B) recipe + β2=0.98 + forecaster bottleneck removed (d=384 instead
of 128). Loss bottomed at step ~1000 (loss=3.54) then climbed
monotonically: 4.84 (3k) → 5.63 (5k) → 6.00 (6.8k) → 7.90 (10k) →
9.42 (10.6k). AUC collapsed 0.9999→0.97; Top1 collapsed 0.99→0.23;
R²_rand collapsed 0.81→0.34 — clear representation collapse.

Compare baseline (B) cl_hh_50k @ step 10k: loss=2.25, AUC=1.0, Top1=1.0.
α is 3.5× worse and trending upward; (B) was already converged.

**Interpretation (hypothesis, not architectural claim).** The fp16 body
in (B) is load-bearing on the forecaster bottleneck (d=128). Removing
the bottleneck (α) lets the forecaster's residual stream grow unbounded
at fresh init, which fp16's narrow mantissa cannot represent — the same
mechanism documented in `experiments/2026-05-11_exp_encoder_forecaster/
EXPERIMENT_LOG_2026-05-15_fp16_precision.md`:
"residual-stream max-abs amplitude grows unbounded with depth and
training (forecaster block: ~80 @ step 200 → ~1070 @ step 2800)" and
"fresh-init partial-fp16 diverges in every tested combination."
The (B) bottleneck constrains forecaster capacity → constrains residual
growth → fp16 stays stable.

If this is right, γ (no bottleneck, β2=0.95) should diverge the same
way; β (bottleneck kept, β2=0.98) should be stable.

## 13:07 — α SIGTERM'd
α reached step 10,600 (21%) before SIGTERM. best_loss.pth (from step
~1000, the minimum before divergence) copied to FINAL.pth. Spend on
α: ~40 min × $0.55/h ≈ $0.37. Saved ~3h × $0.55 ≈ $1.65 of doomed
compute by stopping early.

## 13:07 — β started (bottleneck kept, β2=0.98)
Initial: step 100 loss 5.85, AUC 1.0, Top1 0.997. Healthy start.
Note: hf_token.txt was missing from the initial code tar (gitignored);
scp'd to box at 13:08. β picked up env at 13:07:34 with empty HF_TOKEN
but ran at 3.8 sps anyway (dataset cached locally from α's run).
γ will pick up the token (re-read on box_run.sh restart).

## 16:19 — β BB DONE @ step 50000
Final β trajectory tracks (B) almost identically:

| step  | (B) loss | β loss | β tracks (B) to ±0.04? |
|------:|---------:|-------:|-----------------------|
| 10000 | 2.25     | 2.24   | yes (-0.01)           |
| 20000 | 2.22     | 2.21   | yes (-0.01)           |
| 30000 | 2.21     | 2.19   | yes (-0.02)           |
| 50000 | 2.17     | 2.13   | yes (-0.04, β slightly lower) |

β2=0.98 (vs (B)'s 0.95) gives a marginal ~2% loss improvement at 50k.
**Bottleneck-kept + β2 change = fp16 stable** — confirms hypothesis
that the bottleneck (not β2) was the load-bearing fp16 ingredient.

## 16:19 — γ started; same divergence pattern as α
γ = no bottleneck + β2=0.95 — the (B) β2, but bottleneck removed.
Trajectory:

| step  | (B)  | α    | γ    | Notes                |
|------:|-----:|-----:|-----:|----------------------|
|  2100 | ~2.45| ~5.0 | 3.96 | γ slower onset than α|
|  5000 | 2.31 | 5.63 | 5.17 | both diverging       |
| 10000 | 2.25 | 7.90 | 5.60 | γ slower but climbing|
| 10100 | —    | —    | 5.57 | SIGTERM              |

γ diverges more slowly than α (β2=0.95 reacts faster to recent
gradients, perhaps partially mitigating the fp16 amplitude blowup
that β2=0.98 amplifies). But the gap collapses (1.13 → 0.97 between
step 5k and 10k) — same representation-collapse signature.

## 16:56 — γ SIGTERM'd at step 10100; vast box destroyed (17:58)
γ killed once divergence vs (B) was unambiguous. best_loss.pth (step
~1000) copied to FINAL.pth. After sync verified all artifacts local
(α + β + γ backbones + optimizers + losses CSVs + per-step logs),
vast instance 37144527 (label bbeta2-conf-309) destroyed. Total spend:
**$2.66** of $20.37 budget.

## 2026-05-21 08:57 — α fp32 continuation (per user)
Resumed α's `best_loss.pth` (optimizer reports step 900, the loss-min
pre-divergence checkpoint) with the body switched to all-fp32, continued
to 50k on elisa GPU 1 (free; #307/other-agent work had vacated GPU 1).
Loss descended 3.27 → 2.30 with no divergence — fp16 was the only thing
blocking training of the no-bottleneck arm. ~2.8h wall (resumed 08:57,
BB DONE 11:43). q-head + GIFT-Eval 11:44 → 14:28.

Result: **full-97 1.3687** — worse than α's step-900 snapshot (1.2767),
worse than v11c (1.292), worse than (B) (1.3572). Triage 1.4498.

## Verdict on the issue's expected outcome

> "α reaches the same or better full-97 GM-MASE than v11c (≤ 1.292)."

**Not met at convergence.** Two layers:
1. Under the (B) fp16 recipe, α diverges by step ~900 — unrecoverable
   in fp16. γ confirms it's bottleneck-removal, not β2.
2. The fp16-blocked training can be done in fp32 (stable to 50k), but
   the converged backbone (1.369) is *worse* than v11c. α's step-900
   snapshot beating v11c (1.277) was an under-training artifact — it
   does not survive full training. More contrastive training of the
   no-bottleneck arm degrades GIFT-Eval transfer.

So the bottleneck was not what held (B) back from v11c; removing it and
training to convergence makes things worse, not better (under dropkey
0.7 + hh-negs). Open: a dropkey-0.9 / plain-loss fp32 continuation —
i.e. the v11c recipe itself.

## Compute note
α fp32 continuation + its downstream ran on elisa (free) — no
additional vast spend. Total vast spend remains **$2.66**.
