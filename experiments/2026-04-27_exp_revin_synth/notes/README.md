# 2026-04-27_exp_revin_synth

**What**: RevIN backbone + qhead on mix=1.0 (60k bb, 30k qh).
Counterpart to `2026-04-27_exp_revin_repro` but on synth-only (in-distribution)
to isolate the normaliser comparison from OOD transfer.

**When**: Late April 2026.

**Status**: Completed. RevIN-synth was best of the *original* 5 synth
arms — it beat all four EWMA span=32 arms (~5.75% better MASE than
fe+mu @ 60k) — but was later dominated by EWMA span=64+ when the span
sweep revealed the previous EWMA span=32 default was the bottleneck.
Single seed; the intra-group ordering is not seed-separated.

**Run script**: not preserved as a `run.sh` here — original was
`/tmp/run_revin_synth.sh` on the remote vast.ai instance and was lost
when the instance was destroyed. See `../exp_revin_synth.md` for the setup table
and use `../2026-04-27_exp_synth_only_redo/scripts/run.sh` as a template (substitute
`--rev-norm-kind revin`, drop span, set bb to 60k).

**Code referenced**:
- `../2026-04-27_freq-embedding/scripts/train.py`
- `../2026-04-13_gift-eval/scripts/train_forecasting_head.py`
- `../2026-04-27_freq-embedding/scripts/synth_eval.py`

**Reproduction (run script lost)**: the run was launched ad-hoc on the
remote (`/tmp/run_revin_synth.sh` on the vast.ai instance) and was lost
when the instance was destroyed; no `run.sh` is preserved here. Use
`../2026-04-27_exp_synth_only_redo/scripts/run.sh` as a template:
substitute `--rev-norm-kind revin`, drop the span flag, set backbone
steps to 60k. The Protocol table in `../exp_revin_synth.md` carries the
full knob set.

**Run timeline / wallclock (operational, not science)**:
- Backbone `tiny_femu_revin_synth60k`: ~1.2h on a single GPU shared with
  the span sweep; best gap ~0.77; best loss decreased through 60k.
- Quantile head `R1q_femu_revin_synth60k`: 30k steps, ~20 min on
  synth-only (~28 sps).

**Dropped from the report (no figure, not retained)**: an earlier draft
referenced a `synth_qhead_grid_*` visual-similarity claim for the RevIN
variants (amplitude damping / phase drift "like the other arms"). That
was a single-seed eyeball, the PNG was never saved, and the checkpoint
is not in git — so it cannot be regenerated. The claim is dropped rather
than cited as a missing image.
