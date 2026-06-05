# Data provenance & corrections — window-size-comparison

Operational / bookkeeping detail kept out of the main report.

## Sources

- **gap-vs-step (authoritative):** `logs/window_test_w32.log`,
  `logs/window_test_w16.log`, `logs/window_test_w16_bs28.log`. Each logs
  `gap=` every 1000 steps. `scripts/plot_window_curves.py` parses these
  directly for `plots/gap_vs_step.png`.
- **gap-vs-wall-time:** the logs record `sps` and `vram` but **no
  timestamps**, so a true wall-time series cannot be reconstructed from
  them. The wall-time axis in `plots/gap_vs_walltime.png` is the
  "Results by Wall Time" table that shipped with the original report
  (hard-coded in the plot script and tagged as such). The gap-vs-step
  figure is therefore the authoritative comparison; the wall-time figure
  is illustrative of the speed trade-off.

## Best-gap values (from logs)

| Arm | Best gap | Step | Final step reached |
|-----|----------|------|--------------------|
| W=32 bs=32 | 0.0824 | 9000 | 10000 |
| W=16 bs=24 | 0.0927 | 9000 | 10000 |
| W=16 bs=28 | 0.0820 | 6000 | 6895 (wall-time cap) |

## Correction: bs=28 step table in the original report

The original report's bs=28 follow-up table listed gaps for steps
7k/8k/9k/10k as 0.088 / 0.091 / 0.093 / 0.092. Those are the **W=16
bs=24** values copy-pasted into the bs=28 column. `window_test_w16_bs28.log`
shows the bs=28 run hit its 29.2-min wall-time cap at **step 6895**
(`Wall time limit reached at step 6895` → last logged point 6000,
gap 0.0820), so it never produced 7k–10k points. The rewritten report
drops those phantom rows and marks bs=28 as wall-time-capped.

## Metric framing

- Gap improvement 0.0927 vs 0.0824 = **+12.5%** (reported as "+13%").
- VRAM 14.7 vs 23.4 GB = **−37.2%** (reported "37% less").
- Speed 4.7 vs 5.7 sps = **−17.5% throughput** ≈ **+21% per-step time**;
  reported as "~18% slower per step" using the throughput framing.
- bs=28 throughput 3.9 vs 4.7 sps = **−17%**.

## Caveat

One run per arm, one validation seed (0). W and batch size co-vary
(W=32→bs32, W=16→bs24), so this is not a clean batch-matched A/B. The
~0.011 gap margin is small vs visible run-to-run wobble. Treat as
directional, not conclusive; not architectural.
