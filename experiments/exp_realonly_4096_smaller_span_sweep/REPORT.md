# exp_realonly_4096_smaller_span_sweep — REPORT (partial; only span=32 done)

## Status

| span | state |
|------|-------|
| 32   | DONE — pulled to `sync_realonly_4096_smaller/ewma_span32/` |
| 64   | queued (auto-launches after #20 RevIN-smaller frees the RevIN box) |
| 128  | DONE (lives in `exp_realonly_4096_smaller_2arm/results/gift_eval_ewma128/` — used as reference) |
| 256  | in flight (backbone training on EWMA box) |
| 512  | queued (auto-launches after span=64 finishes) |

This file will be updated as each span lands.

## Setup

Smaller arch fixed (L=6 H=384 nhead=6, 11.43M params); EWMA-128
fixed; only `--rev-norm-span` varies.

| span | alpha = 2/(span+1) | "memory" |
|-----:|-------------------:|----------|
| 32   | 0.0606             | ~32 timesteps  |
| 64   | 0.0308             | ~64 timesteps  |
| 128  | 0.0155             | ~128 timesteps (reference, from #20) |
| 256  | 0.0078             | ~256 timesteps |
| 512  | 0.0039             | ~512 timesteps (12.5% of T=4096) |

All other knobs identical to `exp_realonly_4096_smaller_2arm` ewma128:
30k steps, bs=24, lr=1e-4, mix=0.0, T=4096, C=1, freq+seas-emb 3,
mixup-p 0.3. The original run.sh used `--grad-clip 1.0`; this is now
banned per user feedback and removed in the run.sh going forward
(span=32 had it; spans 64/256/512 will not).

## Results so far

| span | GM-MASE | GM-MAPE_SN | GM-CRPS_SN | configs<1.5 | grad-clip? |
|-----:|--------:|-----------:|-----------:|------------:|-----------:|
| 32   | **1.739** | 1.277    | 1.076      | TBD         | yes (legacy) |
| 64   | TBD     | TBD        | TBD        | TBD         | no         |
| 128  | 1.783   | 1.243      | 1.082      | 47/97       | yes (legacy from #20) |
| 256  | TBD     | TBD        | TBD        | TBD         | no         |
| 512  | TBD     | TBD        | TBD        | TBD         | no         |

(Aksu reference: GM-MAPE_SN 0.882, GM-CRPS_SN 0.642.)

## Headline so far (preliminary, only 2 of 5 spans)

**span=32 narrowly beats span=128 on GM-MASE (1.74 vs 1.78), within
seed noise.** GM-MAPE_SN slightly favours span=128 (1.24 vs 1.28),
GM-CRPS_SN tied (1.08 ≈ 1.08). Need 64/256/512 to know whether the
optimum is below 32, between 32 and 128, or above 128.

Complications:
- **grad-clip mismatch**: span=32 and span=128 both used `--grad-clip
  1.0` (legacy), but the next three spans (64, 256, 512) will run
  WITHOUT grad-clip per user feedback. So the comparison may have a
  small confound — to be assessed when all 5 land. If we see a clear
  trend without grad-clip and span=32/128 sit on a smooth curve, the
  effect was negligible.
- **Single seed**: cross-seed variance is ~3–5% on this kind of run, so
  ≤3% gaps are not actionable on their own.

## What we'll do once all 5 are in

* Run `scripts/plot_span_sweep.py` for the GM-curves.
* Pick the optimum span; if a single span clearly wins on GM-MASE AND
  GM-MAPE_SN AND GM-CRPS_SN, that's the new default. If the picture is
  noisy, default to the cheapest span that's not strictly worse
  (smaller alpha → fewer cache buffer recomputes; minimal practical
  difference).
* Flag span=32 + span=128's grad-clip caveat in the final report.

## Files (so far)

* `results/gift_eval_ewma_span32/{all_results.csv, summary.txt}` — full 97 configs
* `scripts/plot_span_sweep.py` — sweep plotter
* `run.sh` — span CLI; grad-clip-free
* `README.md` — pre-experiment plan
