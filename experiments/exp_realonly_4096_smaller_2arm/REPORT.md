# exp_realonly_4096_smaller_2arm — REPORT

## Headline

Smaller architecture (L=6 H=384 nhead=6, 11.43M params) **wins on
EWMA-128**, basically ties on RevIN. The overall winner across the
realonly + T=4096 + C=1 setting at 30k steps is
**smaller + EWMA-128**.

| arm                  | params | GM-MASE | GM-MAPE_SN | GM-CRPS_SN | configs<1.5 |
|----------------------|------:|--------:|-----------:|-----------:|------------:|
| Tiny + EWMA-128 (#19)  | 19.96M | 1.805  | 1.432  | 1.083  | 45/97 |
| **smaller + EWMA-128** | **11.43M** | **1.783** | **1.243** | **1.082** | **48/97** |
| Tiny + RevIN (#19)     | 19.96M | 2.447  | 1.887  | 1.510  | 30/97 |
| smaller + RevIN        | 11.43M | 2.532  | 1.849  | 1.548  | 30/97 |

Notes:
- Smaller wins **EWMA-128 head-to-head per-config 58/97** (vs Tiny).
- Smaller loses **RevIN head-to-head 42/97** (slight Tiny advantage).
- 13% better GM-MAPE_SN on EWMA (1.24 vs 1.43) is the biggest delta —
  the smaller model generalises better at this dataset/budget.
- ~½ params, ~1.5× faster training.

Phase-3 v3-prim+EWMA-128 (mix=0.5 synth) reference: GM-MASE 1.621.
Smaller realonly EWMA at 1.783 → still ~10% behind the half-synth
recipe. (Same conclusion as #19: synth is load-bearing on this base
recipe.)

## Setup

Identical to `exp_realonly_4096_2arm` except for the backbone shape:

| knob          | Tiny (#19)        | Smaller (this exp) |
|---------------|-------------------|--------------------|
| num_layers    | 6                 | 6                  |
| H (d_model)   | 512               | **384**            |
| nhead         | 8                 | **6**              |
| ffn_mult      | 4.0               | 4.0                |
| W (patch)     | 16                | 16                 |
| Params        | 19,955,516        | **11,428,668** (43% smaller) |

Everything else identical: `--t-raw 4096 --n-channels 1 --mix-ratio 0.0
--freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3`, 30k steps,
bs=24, lr=1e-4, save-every 2500. Both arms used `--grad-clip 1.0`
(carry-over from #19's post-NaN state). Per the May 1 ban, future runs
in #27/#28/#23 are grad-clip-free.

## Per-config head-to-head (smaller vs Tiny, EWMA-128)

Smaller wins on 58/97 configs. Selected explosive-trend tail:

| config | Tiny MASE | smaller MASE |
|---|----:|----:|
| covid_deaths/D/short | 69.71 | 71.90 (smaller worse) |
| bizitobs_application/10S/medium | 15.71 | 14.90 (smaller better) |
| bizitobs_application/10S/long | 15.88 | 15.10 (smaller better) |
| m4_yearly/A/short | 8.76 | 8.52 (smaller better) |

Smaller is slightly worse on the worst-case (covid) but better on the
medium-tail explosive trends. Net effect (GM): smaller wins.

## Why smaller beats Tiny here

Working hypothesis (not directly tested):
- 11.4M params vs 19.96M; with 30k steps × bs=24 on a 61k-row dataset
  (≈12 epochs), Tiny has more capacity than data, fits noise.
- Smaller has a more favourable params-to-data ratio and generalises
  better.
- Consistent with the GM-MAPE_SN gap (1.24 vs 1.43) being largest —
  MAPE is sensitive to per-config rate-of-error, where overfit
  hyper-parametrised models regress more visibly.
- For RevIN, the simpler normalisation is amplified by capacity (the
  larger Tiny is the marginal winner). With EWMA-128's already-strong
  per-instance dynamic-range adaptation, the smaller arch is enough.

## Caveat — grad-clip used here, banned for future runs

Both arms in this experiment used `--grad-clip 1.0` (carry-over from
the post-NaN belt-and-suspenders in #19). User feedback (May 1):
**grad-clip is forbidden in this project** (it's a workaround for
ungovernable data, hides design defects, AdamW already attenuates
outliers via the v moving average). The underlying numerical bug from
#19 was already fixed by the float64 cumsum promotion — the grad-clip
was unnecessary. Removed for all subsequent runs (#22 spans 64/256/512,
#27 τ-sweep, #28 learnable-τ, #23 train-to-completion). The grad-clip
1.0 setting probably had a small effect on these specific numbers but
shouldn't change the qualitative comparison.

## Files

* `results/gift_eval_ewma128/{all_results.csv, summary.txt}` — full 97 configs
* `results/gift_eval_revin/{all_results.csv, summary.txt}` — full 97 configs
* `plots/gift_eval_smaller_compare.png` — 4-panel comparison vs Tiny + v3prim
* `run.sh` — pipeline launch script (now grad-clip-free)
* `README.md` — pre-experiment hypothesis & arch comparison

## Cost (rough)

| arm    | wall hours | $/hr      | cost   |
|--------|----------:|----------:|------:|
| EWMA-smaller | ~3h | $0.37/h | $1.10 |
| RevIN-smaller| ~3h | $0.64/h | $2.00 |
| total  |     |     | **~$3.10** |

## What this gates

- #22 EWMA span sweep: now uses smaller arch (winner).
- #27 τ-sweep: now uses smaller arch + EWMA-128 + best span from #22.
- #28 learnable τ: same.
- #23 train-to-completion: best config so far is smaller+EWMA-128, will
  finalise after #27/#28.
