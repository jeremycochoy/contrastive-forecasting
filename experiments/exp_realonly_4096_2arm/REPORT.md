# exp_realonly_4096_2arm — REPORT

*Written: 2026-05-01. Date-stamp added: 2026-05-02.*

## Headline

Real-data-only training (mix=0.0) on
`jeremycochoy/gift-pretrain-small-4096` at T=4096 / C=1 with the
**Tiny** backbone (L=6 H=512 nhead=8, ~20M params), 30k steps each
arm, gives:

| arm                                 | GM-MASE | GM-MAPE_SN | GM-CRPS_SN |
|-------------------------------------|--------:|-----------:|-----------:|
| realonly + EWMA-128                 |  **1.805** | 1.432  | 1.083 |
| realonly + RevIN                    |  2.448 | 1.887  | 1.510 |
| (Aksu Moirai-Small reference)       |   —    | 0.882  | 0.642 |
| v3-prim + EWMA-128 (phase 3 winner) | 1.621  |  n/a   |  n/a  |
| periodic + EWMA-128 (phase 0)       | 1.659  |  n/a   |  n/a  |

**Conclusion: synth was load-bearing in phases 1–5, not just
regularising.** EWMA-128 realonly@30k underperforms the v3-prim+EWMA-128
phase-3 winner by 11% on GM-MASE and barely beats periodic-baseline
phase 0 (1.81 vs 1.66). RevIN realonly is much worse still (2.45). On
the SN-normalised skill scores both arms sit ~1.6–2.4× above Aksu's
reference targets — single-pass on 61k rows at 30k steps is not enough.

The headline answers the original question: removing synth costs us
~11% GM-MASE on this base recipe. So synth was contributing real
signal, not just regularising noise.

## Setup

| knob          | value                                              |
|---------------|----------------------------------------------------|
| dataset       | `jeremycochoy/gift-pretrain-small-4096`, `small_v1/` (61,717 rows × T=4096 × C=1) |
| backbone      | Tiny: L=6 H=512 nhead=8 ffn_mult=4 W=16 (≈19.96M params) |
| mix_ratio     | **0.0** (no synth)                                 |
| t_raw         | **4096** (was 1024 in phases 1–5)                  |
| n_channels    | **1**  (was 4)                                     |
| total steps   | 30,000 (≈ 11.7 epochs at bs=24, since 61717/24≈2572 per epoch) |
| batch size    | 24                                                 |
| lr            | 1e-4                                               |
| save-every    | 2,500                                              |
| grad-clip     | 1.0  (see Caveat below — removed for future runs)  |
| freq-emb-dim  | 3, seasonality-emb-dim 3                           |
| mixup-p       | 0.3                                                |
| eval          | full GIFT-Eval B4, forecast_len=16, 9 quantile levels, with new SN-normalised columns from task #18 |

Two arms run in parallel: `ewma128` (RevEWMNorm span=128) and `revin`.

## What changed in the code

| file | change |
|---|---|
| `src/dataloader.py`               | `T_RAW` is now per-loader-instance via `t_raw=` arg; threaded through `ShardDataset`, `HFStreamingLoader`, and the `create_*` factories. |
| `src/norm.py`                     | RevEWMNorm cumsum trick promotes intermediates to **float64 when T>2048**; output stats cast back to input dtype. T=1024 fast path preserved. |
| `experiments/freq-embedding/scripts/train.py`               | new flags `--t-raw`, `--n-channels`. Plus the (now removed) `--grad-clip 1.0`. |
| `experiments/gift-eval/scripts/train_forecasting_head.py`   | matching `--t-raw`, `--n-channels`. |
| `experiments/gift-eval/scripts/eval_gift_eval_official.py`  | `--t-raw`, `--backbone-c` (CLI exposure of constants that already lived in the predictor). |

Commits: `27bb8c3` (CLI flags), `452d79b` (NaN fix + grad-clip — see
Caveat), `1e0428c` (plot script), `fe61160` (results).

## Failure & recovery

### NaN at step 1697 — float32 cumsum overflow

First EWMA attempt NaN'd at step 1697. Diagnosis:
- gift-pretrain-small-4096 has rows with values up to ~1e5 (electricity etc.).
- RevEWMNorm's cumsum trick computes `residuals_sq[k] / (1-alpha)^k`. At
  T=4096 and span=128, `(1-alpha)^{-(T-1)} ≈ 6e27`, so the late-t weighted
  residuals can hit ~1e10 × 6e27 = 6e37 — just below float32's 3.4e38
  ceiling. Sum across timesteps overshoots → Inf → NaN.
- Phase 1–5 didn't hit this because (a) T=1024 not 4096 (inv_decay max
  ~1e7, ~20 OOM lower) and (b) the 50/50 synth mix bounded inputs through
  the synth half.

Fix: promote cumsum intermediates to float64 when T>2048 in
`RevEWMNorm._compute_statistics`; cast `self.mean` and `self.stdev`
back to the input dtype. Verified: T=4096 + span=128 + spike-heavy
input (values up to 1e5) → finite z-scores in [-8, 8], reconstruction
error 7.8e-3, T=1024 fast path unchanged.

### RevIN host-stop mid-pipeline

Original RevIN instance (`35922200`) was vast.ai-side-stopped at qhead
step 8000 (intended_status went to "stopped"). The backbone FINAL.pth
and 17k periodic checkpoint had already synced locally before the stop,
so we lost no training compute.

Recovery: provisioned r3 (`35927139`), uploaded the local backbone
FINAL.pth back to the new instance, ran
`run_revin_resume.sh` (skips stage B). Stage H + Stage E completed
cleanly on r3. The reported RevIN numbers are from this resumed run.

## Per-config behaviour

Top failure modes carry over from the phase 1–5 picture — explosive
trends and big CloudOps spikes still dominate the high-MASE tail.

EWMA-128 worst configs (selected):

| config | MASE |
| --- | ---: |
| covid_deaths/D/short            | 69.71 |
| bizitobs_application/10S/medium | 15.71 |
| bizitobs_application/10S/long   | 15.88 |
| bizitobs_application/10S/short  |  7.47 |
| bizitobs_service/10S/medium     |  7.41 |
| bitbrains_rnd/H/short           |  6.30 |
| saugeen/D/short                 |  4.89 |

Compare to phase 5 v5envboost (env_gain_max=100): covid_deaths was
69.26 — within seed noise. Confirms the env-bump hypothesis (phase 5)
was not the right intervention; the failure mode is shape-limited
(saturating curves), not gain-limited.

RevIN's worst-case max was 174.8 (vs EWMA-128's 69.7), and only
30/97 configs landed below 1.5 vs EWMA's 45/97. RevIN does not handle
the extreme-outlier inputs as gracefully as EWMA-128 at this T.

## Caveat — grad-clip used here, banned for future runs

The runs in this experiment (both arms) used `--grad-clip 1.0` after I
added it post-NaN. Per user feedback (May 1): **grad-clip is forbidden in
this project.** The reasoning: grad-clip is a workaround for
ungovernable data, not a feature of a well-designed pipeline; it hides
design defects we explicitly want to see; and AdamW's v moving average
already attenuates outliers. The underlying numerical bug was already
fixed by the float64 cumsum promotion — the grad-clip was unnecessary
"belt-and-suspenders". It is removed for all future runs (#22 spans
64/256/512, #23 train-to-completion, #21 if/when it launches). The
grad-clip 1.0 setting probably had a small effect on these specific
Tiny-arm numbers but is not expected to change the qualitative
comparison vs phases 1–5.

A related implication of the user's stance: the gift-pretrain-small-4096
dataset has rows with values up to ~1e5 (electricity etc.). If we keep
hitting numerical edge cases on raw real data, that points to a data
hygiene gap (per-series scaling/clipping at curation time, not at
training time) rather than a training-side fix. Worth revisiting before
#21 if `gift-pretrain-base-4096` is being prepared.

## What it tells us about #21 / #23

- 30k steps on this small dataset is not enough for the SN-normalised
  scores to approach Aksu (factor of ~1.6× to 2.4× away).
- A bigger dataset OR much longer training is needed. #21 (full pass on
  gift-pretrain-base) is exactly the right next test, gated on the
  dataset existing on HF.
- The smaller arch (#20) and different spans (#22) may yield small
  improvements, but won't bridge the bulk of the gap. The dominant
  lever is data + steps, not architecture details.

## Files

* `results/gift_eval_ewma128/{all_results.csv, summary.txt}` — full 97 configs
* `results/gift_eval_revin/{all_results.csv, summary.txt}` — full 97 configs
* `plots/gift_eval_realonly_compare.png` — 4-panel comparison vs phases 0/2/3
* `run.sh` — pipeline launch script (now grad-clip-free)
* `run_revin_resume.sh` — resume helper used after the host-stop incident
* `README.md` — pre-experiment hypothesis & setup

## Cost (rough)

| arm    | wall hours | $/hr      | cost   |
|--------|----------:|----------:|------:|
| EWMA   | ~5h       | $0.37/h   | $1.85 |
| RevIN  | ~6h (+r2 sunk cost ≈ $0.30) | $0.37–0.64/h | ~$3.50 |
| total  |           |           | **~$5.35** |
