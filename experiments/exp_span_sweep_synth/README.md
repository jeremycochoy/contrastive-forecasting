# exp_span_sweep_synth

**What**: RevEWMNorm span sweep on synth-only (mix=1.0). Spans tested:
32, 64, 128, 256, 512, 1024 (extended after monotonic improvement at
256). Each arm: 30k bb + 30k qhead + 1024-sample held-out synth eval.

**When**: Late April 2026.

**Status**: Success. Inverted-U with peak at span=512 (GM-MASE 0.848,
2.8× better than the previous span=32 default). Both metrics agree on
the optimum. Single seed per arm — open question on second-seed
validation noted in REPORT.

**Run script**: `run.sh` (formerly `run_span_sweep_synth.sh` at repo
root). Covers spans 64/128/256; spans 512 and 1024 were launched
ad-hoc on remote (not preserved as scripts here, but the for-loop
in run.sh extends trivially).

**Code referenced**:
- `../freq-embedding/scripts/train.py`
- `../gift-eval/scripts/train_forecasting_head.py`
- `../freq-embedding/scripts/synth_eval.py`
