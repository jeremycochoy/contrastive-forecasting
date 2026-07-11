# B=512 arm C (seed 2) extracts

Both files extracted on 2026-07-11 from the in-flight #371 experiment
(`2026-07-07_b512_armC_seed2_traj`; the seed-1 parent's raw telemetry
was not retained, its run directory predated durable-path
checkpointing).

- `steps_loss.csv` — `step,loss`, legs base + r2 + r4 deduplicated on
  step, covering steps 1–50,000. B=512 reference curve for the
  report's backbone-loss plot.
- `gm.csv` — `head,step,gm`: GM-Rel MASE of every completed seed-2
  trajectory cell (steps 12,500–50,000 at extraction time). B=512
  re-run points for the report's GM plot.
