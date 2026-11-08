# `--forecast-batch-size`: N windows of a config in one pass

Flag: `--forecast-batch-size N` on
`experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py`.
Default: `1`, which forecasts one window at a time, as every score so far.
Strategy B4 only.

## What it changes

A GIFT-Eval config holds many windows. Each window has a context of the same
length and a horizon of the same length. With `N > 1`, the windows of one
config go through the backbone, the rollout and the head `N` at a time
(`forecast_B4_windows` in `src/forecasting_head.py`). The per-window code
prepares each window as before: the fill of the missing values, the cut or
the left padding to the context length, and the gluonts forecast object.

Each window gets the forecast of the per-window path, up to floating-point
noise. These parts of the forecast are per window, in a batch as alone:

- the EWMA statistics of the normaliser and its left-padding mask,
- the scale of the normalised rollout feedback (`--rollout-feedback
  normalised`): the mean norm of the encoder latents of that window,
- the latent sphere (#453, #454),
- the unscaling with the statistics at the last context value.

The last pass of a config holds the windows that are left, fewer than `N`.
A config of one window runs one pass of one window.

## The memory rule

The rollout of a window holds its context latents and its horizon latents.
A horizon with more patches than the context gets fewer windows per pass:
`N` times `2 c / (c + h)` windows, with `c` the context patches and `h`
the horizon patches, and at least one window. So a pass never holds more
latents than `N` windows with a horizon as long as the context, down to one
window per pass. At the 1024 context (64 patches),
the longest GIFT-Eval horizon (900 values, 57 patches) still runs the full
`N`. The script prints the windows per pass of each config.

## Limits

- B4 only. The script refuses the flag with another strategy. Strategy R
  has its own batch of 64 windows.
- The forecasts match the per-window path up to floating-point noise. The
  kernels of a batch differ from the kernels of one window. In the check
  below, the largest change of the MASE of a config was 2.2e-7 relative.
- The memory of a pass grows with `N`, the width and the depth of the
  model, and the context length. `N = 64` of a width-384 model at the 1024
  context took 1.9 GB of GPU memory at its peak.
- The time of a config also holds the load of its dataset and the gluonts
  metrics, which the flag does not change. A config of short series with
  many windows gains less than a config with a long horizon.

## Example

```bash
GIFT_EVAL=~/workspaces/gift-eval-data CUDA_VISIBLE_DEVICES=1 \
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
  --backbone-path <backbone .pth> --head-path <head .pth> \
  --strategy B4 --forecast-len 16 --forecast-batch-size 64 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 8 --num-layers 6 \
  --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
  --head-nhead 8 --head-causal true --rollout-feedback normalised \
  --output-dir <dir>
```

## In the head waves

`run_wave.sh` and `follow_multi.sh` of `reports/2026-10-10_frequency_family_head/scripts/` pass
`FF_EVAL_FORECAST_BATCH_SIZE=N` to each shard of a score (`eval_local.sh` reads it as
`EVAL_FORECAST_BATCH_SIZE`). The protocol of the score then ends with `fbs=N`, so a start with another
value does not take the score of a tag. Unset, each shard forecasts one window at a time.

## The check

One checkpoint of the grid40k runs (`grn_ema099995_f6_rep01_ss256` at 10k
steps, its control head at 30k steps), 97 configs, `N = 64` in one process
on one RTX 4090 that other jobs shared, against its per-window score of
2026-11-06 (code af14ca2b, 4 shards in parallel). The per-window score
took about 1 h 45 min of wall time. The batched score took 36 min 55 s.

| | per window | `N = 64` |
|---|---|---|
| GM-Relative MASE | 1.0345 | 1.0345 |
| largest relative change of a MASE of a config | | 2.2e-7 |
| time, sum over the 97 configs | 24,201 s | 2,212 s |
| electricity/15T/long (long horizon) | 2,963 s | 265 s |
| temperature_rain/D/short (96,216 windows) | 2,004 s | 172 s |
| bitbrains_fast_storage/5T/long | 1,814 s | 77 s |
| loop_seattle/5T/short | 279 s | 134 s |
| peak GPU memory of the process | | 1.9 GB |
