# exp_realonly_4096_smaller_2arm — REPORT (partial; RevIN arm in flight)

## Status

| arm | state |
|---|---|
| EWMA-128 | DONE |
| RevIN    | in flight (eval ~95% as of writing — will edit when complete) |

This file will be updated when the RevIN arm finishes; the EWMA section
is final.

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
bs=24, lr=1e-4, save-every 2500. The original run.sh used
`--grad-clip 1.0`; this is now banned per user feedback (May 1) and
removed from the run.sh going forward.

## Results so far

| arm                  | params | GM-MASE | GM-MAPE_SN | GM-CRPS_SN | configs<1.5 |
|----------------------|------:|--------:|-----------:|-----------:|------------:|
| Tiny + EWMA-128 (#19)| 19.96M | 1.805  | 1.432      | 1.083      | 45/97       |
| **smaller + EWMA-128** | **11.43M** | **1.783** | **1.243** | **1.082** | **47/97** |
| Tiny + RevIN (#19)   | 19.96M | 2.448  | 1.887      | 1.510      | 30/97       |
| smaller + RevIN      | 11.43M | TBD    | TBD        | TBD        | TBD         |

(Aksu Moirai-Small reference: GM-MAPE_SN target 0.882, GM-CRPS_SN 0.642.)

## Headline so far

**Smaller wins on EWMA-128.** GM-MASE 1.78 (vs Tiny 1.81, ~1.2% better)
is roughly within seed noise, but the **GM-MAPE_SN gap is meaningful**:
1.24 vs 1.43 = 13% better. With ~half the params and ~1.5× faster
training, smaller is the better Pareto point at this dataset / step
budget.

This signals the model is bottlenecked by **data** (not capacity) at
30k steps on 61k-row gift-pretrain-small-4096 — the bigger Tiny arch
has more capacity to memorise training noise, and the smaller arch
generalises slightly better.

→ #22 (EWMA span sweep) uses the smaller arch.
→ #21 (gift-pretrain-base full pass) and #23 (train-to-completion) will
   pick smaller as the working architecture unless the RevIN-smaller
   result flips the verdict.

## Per-config head-to-head (EWMA-128 only, smaller vs Tiny)

(See `plots/gift_eval_smaller_compare.png` once produced.)

Quick spot-check on the explosive-trend offenders:

| config                          | Tiny MASE | smaller MASE | delta |
|---------------------------------|----------:|------------:|------:|
| covid_deaths/D/short            |   69.71   |   ?         |   ?   |
| bizitobs_application/10S/medium |   15.71   |   ?         |   ?   |
| m4_yearly/A/short               |    8.40   |   ?         |   ?   |

(Will populate once `plot_compare_smaller.py` runs against the full
result set.)

## Caveat — grad-clip used here, banned for future runs

Both arms in this experiment used `--grad-clip 1.0` (carry-over from
the post-NaN belt-and-suspenders in #19). This is now banned in the
project per user feedback (May 1). Future runs in #22 / #23 / #21 will
have grad-clip removed; the run.sh in this directory has been updated
accordingly so any re-runs use the corrected version.

## Files (so far)

* `results/gift_eval_ewma128/{all_results.csv, summary.txt}` — full 97 configs
* `results/gift_eval_revin/{all_results.csv, summary.txt}` — TBD
* `plots/` — TBD
* `run.sh` — pipeline launch script (now grad-clip-free)
* `README.md` — pre-experiment hypothesis & arch comparison
