# Execution log — RevEWMNorm span search

Operational / journey detail kept out of the main report.

## Run layout

Two rounds, each launched as `python scripts/span_search.py <span>` with two
runs sharing a GPU (`CUDA_VISIBLE_DEVICES=0/1`, ~4.5 GB VRAM each) on a 2x RTX
4090 box. One run per span, 3,000 steps, no seeds varied (single-seed sweep).

- **Initial round:** `none`, `32`, `128`, `512`.
- **Refined round:** `8`, `16`, `45`, `64`, `91` (fills in the sub-patch and
  near-patch region around the initial peak at 32).

Raw stdout for every run is committed under `logs/span_<span>.log` and is the
authoritative source for the figures (`scripts/plot_span_sweep.py` parses it).

## Log buffering — missing intermediate ticks

The validation line is printed every 500 steps. In the `span=128` and
`span=512` logs a couple of intermediate ticks were overwritten mid-line by a
carriage-return / buffering artifact (the throughput banner and the next
`[ step]` line collided on one physical line). The affected ticks:

- `span=128`: steps 2000 and 2500 missing (500/1000/1500/3000 present).
- `span=512`: step 1500 missing (500/1000/2000/2500/3000 present).

These are **display-only losses** — the run completed all 3,000 steps and the
`DONE ... Best gap` summary line plus the step-3000 record are intact, so every
reported best-gap@3k value is recoverable. `parse_log()` in the plot script
extracts every well-formed record and ignores the mangled fragments, so the
gap-vs-step curves simply skip the missing points for those two spans.

## span=8 divergence

`span=8` produced `loss=nan` from the very first validation at step 500 and
`best=-inf` throughout (`logs/span_8.log`). With a span of 8 the EMA window is
so short its running variance collapses toward zero on the near-constant
stretches of integrated ARIMA, and the normalize-by-std step divides by ~0.
This is consistent with the project rule of fixing divergence via
data/normalization rather than grad-clip: the fix here is "don't use a span
that small", not clipping.
