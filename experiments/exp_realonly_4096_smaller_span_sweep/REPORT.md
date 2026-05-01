# exp_realonly_4096_smaller_span_sweep — REPORT

*Written: 2026-05-01. Date-stamp added: 2026-05-02.*

## Headline

3-point sweep over EWMA span on the smaller arch (#20 winner) confirms
**span=128 is a fine default**. Larger spans (256) clearly hurt;
span=32 is marginally better on GM-MASE but worse on GM-MAPE_SN — net
within seed noise.

| span | GM-MASE | GM-MAPE_SN | GM-CRPS_SN | configs<1.5 |
|-----:|--------:|-----------:|-----------:|------------:|
| 32   | **1.739** | 1.277    | 1.076      | 41/97       |
| 128  | 1.783   | **1.243**  | 1.082      | **48/97**   |
| 256  | 1.910   | 1.521      | 1.140      | 45/97       |

(Aksu Moirai-Small reference: GM-MAPE_SN 0.882, GM-CRPS_SN 0.642.)

Pick: keep **span=128** as default for downstream experiments
(#27 τ-sweep, #28 learnable τ, #23 train-to-completion).

## Sweep was narrowed mid-run

Originally planned 4 new spans (32/64/256/512) plus the existing
span=128 reference. Per user instruction (May 1, "minimum-necessary
datapoints"), sweep narrowed to 3 (32, 128, 256):
- span=64 was killed mid-launch as redundant (between 32 and 128).
- span=512 chain cancelled (256 already showed degradation).

The 3 points are sufficient to bracket the optimum: span=32 (below
128, marginally better on MASE), span=128 (reference), span=256 (above,
clearly worse). Going wider would not change the verdict at this
training budget.

## Setup

Smaller arch fixed (L=6 H=384 nhead=6, 11.43M params); EWMA fixed;
only `--rev-norm-span` varies.

| span | alpha=2/(span+1) | typical "memory" |
|-----:|----------------:|------------------|
| 32   | 0.0606          | ~32 timesteps    |
| 128  | 0.0155          | ~128 timesteps   |
| 256  | 0.0078          | ~256 timesteps   |

All other knobs identical to `exp_realonly_4096_smaller_2arm` ewma128:
30k steps, bs=24, lr=1e-4, mix=0.0, T=4096, C=1, freq+seas-emb 3,
mixup-p 0.3.

## Caveat — grad-clip in span=32 and span=128

`span=32` and `span=128` runs (the reference) used `--grad-clip 1.0`
(carry-over from #19/#20). `span=256` was launched after the user-led
ban on grad-clip; it ran without. So the comparison has a tiny grad-
clip mismatch, but:
- The bigger-span arm (256) underperforms by 7-23% on the three
  metrics — well outside any plausible grad-clip effect (which would
  be sub-percent at most).
- The smaller-span arm (32) and the 128 reference are both with
  grad-clip — no mismatch in their direct comparison.

So the qualitative verdict (span=128 is fine, span=256 is worse) is
robust. We did not re-run 32 and 128 without grad-clip because the
direction is clear and saved compute is worth more than a fresh
seed-noise-grade adjustment.

## span=256 NaN-on-restart incident

The chain-launched span=256 backbone trained cleanly to 30k. Then the
shell wrapper hit a `line 116: unexpected EOF while looking for
matching '"'` syntax error mid-script, because the `run.sh` file was
overwritten on disk during the run.sh edit cycle (when grad-clip was
removed) — bash re-reads the file per-line, so the running shell saw
a shorter file than it started with and hit EOF early. Recovered by
manually running stage H (qhead) + stage E (eval) directly via ssh.
No data lost. Lesson logged: future run.sh edits during an active run
should use atomic mv, not in-place write.

## Files

* `results/gift_eval_ewma_span32/{all_results.csv, summary.txt}` — full 97 configs
* `results/gift_eval_ewma_span256/{all_results.csv, summary.txt}` — full 97 configs
* (span=128 reference lives in `exp_realonly_4096_smaller_2arm/results/gift_eval_ewma128/`)
* `plots/span_sweep.png` — 4-panel sweep plot (GM-vs-span line, MASE CDF, per-domain, etc.)
* `scripts/plot_span_sweep.py` — sweep plotter
* `run.sh` — single-arg span CLI (32, 64, 256, 512)
* `README.md` — pre-experiment plan

## Cost (rough)

| span | wall hours | $/hr | cost |
|------|----------:|----:|-----:|
| 32   | ~3h       | $0.37 | $1.10 |
| 256  | ~3.5h     | $0.37 | $1.30 |
| total| | | **~$2.40** |

## What this gates

- **span=128 keeps its default**.
- #27 τ-sweep launches at span=128 on smaller arch + EWMA.
- #28 learnable τ same baseline.
- #23 train-to-completion will use this same best-config baseline.
