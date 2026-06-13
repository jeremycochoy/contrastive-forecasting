# Execution notes — periodic-synth-mix

Operational journey, cost, and infra incidents. Kept out of the report body
(science, not journey). Nothing here changes a result.

## Cost

- ~$4 of the $10 vast.ai balance.
- ~16 h total wall time: 2.1 h CONTROL + 1.3 h MIX backbone, 1.3 h + 1.8 h
  R1 heads, ~5 h + ~4 h GIFT-Eval.
- One 2 h stall on Stage 2a (prefetch-thread hang, below); resumed cleanly
  from the 10k checkpoint with no lost work.

## Session notes

- **vastrun-kit `attach-ssh` idempotency bug** (filed as kit issue #296) hit
  4 consecutive provisions. Worked around with a direct
  `vastai create instance` + `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`.
- **Prefetch hang:** one 2 h Stage 2a hang on a 233-thread futex wait
  (HF-stream prefetch). Resumed cleanly from the 10k checkpoint; no recurrence.
- **Sync loop:** the local sync loop (5-min then 15-min cadence, atomic
  `.tmp → mv`, ≥70 MB min-size guard) caught every checkpoint without issue.

## Synth-validation extra plots

Two additional eyeballed-synth views beyond the `inspect_grid.png` grid in the
report body. Per-sample synth metadata for both is in
[`../plots/inspect_metadata.txt`](../plots/inspect_metadata.txt).

![Zoomed single-period view of each synth primitive (sinusoid / square / saw).](../plots/inspect_zoom.png)

![Long-period (P > 96 samples) synth samples — the slow end of the log-uniform period range.](../plots/inspect_long_period.png)

## Per-config forecast plots

Each figure is a 10-sample grid; per panel: grey = context, black = truth,
plus seasonal-naive / v3b / MIX forecasts over the 16-step horizon (legend in
panel 1). The report body embeds the representative `ett2_W_short.png` (MIX's
biggest win, −15.5%); the other five focus configs are below. The Δ% is the
MIX−CONTROL change in relative MASE from the focus-subset table.

![ett1/15T/short — MIX −4.1%. A noisy, weakly-periodic electricity-temperature series.](../plots/predictions/ett1_15T_short.png)

![m4_hourly/H/short — MIX +4.3% (regression). Strongly periodic hourly data; the report's hypothesis for the hourly regressions is that real hourly series carry a 168-sample weekly modulation the single-primitive synth never shows.](../plots/predictions/m4_hourly_H_short.png)

![solar/10T/long — MIX −4.0%. Spiky diurnal solar (flat overnight, daytime bump) on the long horizon.](../plots/predictions/solar_10T_long.png)

![solar/10T/medium — MIX −5.0%. Same solar series, medium horizon.](../plots/predictions/solar_10T_medium.png)

![solar/H/short — MIX +5.4% (regression). Hourly solar, where weather-driven irregularity sits on top of the diurnal cycle — the report's stated failure mode for clean-synth transfer.](../plots/predictions/solar_H_short.png)
