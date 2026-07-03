# B=1024 retrain of the #366 τ=0.90 last-ckpt winner (with trajectory checkpoints)

Scope: retrain the arm minimising GM-Rel MASE on both `2L / last-ckpt`
and `6L / last-ckpt` on #366's τ=0.90 A–I grid, under one change only —
`--batch-size 1024` instead of `--batch-size 512`. Preserve trajectory
checkpoints every 500 steps up to the parent's 12,500-step budget so
heads can be trained from either the parent's `best-loss` step or from
`last`.

**Status**: implementation landed; awaiting launch. The winning arm and
the parent's best-loss step number are locked at launch via the
`winners.sh` manifest (see
[`experiments/2026-07-03_b1024_traj_ckpts/README.md`](../../experiments/2026-07-03_b1024_traj_ckpts/README.md)).

Results, plots, and the final verdict will land here once the run
completes. Success criteria (from #369): if BOTH retrained cells beat
their parents' `last-ckpt` on GM-Rel MASE, extend to ≈ 25,000 steps and
re-evaluate at the same two loci; otherwise stop and report the delta.
