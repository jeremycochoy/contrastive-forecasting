# 2026-04-27_exp_revin_synth

**What**: RevIN backbone + qhead on mix=1.0 (60k bb, 30k qh).
Counterpart to `2026-04-27_exp_revin_repro` but on synth-only (in-distribution)
to isolate the normaliser comparison from OOD transfer.

**When**: Late April 2026.

**Status**: Completed. RevIN-synth was best of the *original* 4 synth
arms (~5.7% better MASE than fe+mu @ 60k), but was later dominated by
EWMA span=64+ when the span sweep revealed the previous EWMA span=32
default was the bottleneck. Single seed.

**Run script**: not preserved as a `run.sh` here — original was
`/tmp/run_revin_synth.sh` on the remote vast.ai instance and was lost
when the instance was destroyed. See `REPORT.md` for the setup table
and use `../2026-04-27_exp_synth_only_redo/run.sh` as a template (substitute
`--rev-norm-kind revin`, drop span, set bb to 60k).

**Code referenced**:
- `../freq-embedding/scripts/train.py`
- `../gift-eval/scripts/train_forecasting_head.py`
- `../freq-embedding/scripts/synth_eval.py`
