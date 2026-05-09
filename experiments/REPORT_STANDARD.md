# Report standard

A "report" here means a single canonical Markdown file per experiment (typically `RESULTS.md`). The list below is what a sub-agent should check before any report is finalised — feed it the report + the underlying data CSVs and ask "which boxes don't tick?". Address every flagged item.

## Checklist

- [ ] **One file per experiment.** A single canonical Markdown file (e.g. `RESULTS.md`) is the report. Information that doesn't fit can be recorded elsewhere in the experiment directory — scripts, docstrings, an `EXECUTION_LOG.md`, a notebook — but not in additional report files.

- [ ] **Structure: goal → protocol → what we did → what we learned.** The reader who arrives cold should understand the question, the design, and the conclusion in that order.

- [ ] **Facts only; flag extrapolation.** State measurements directly. Anything that goes beyond the data is labelled as a hypothesis. Spearman ρ at n=5 is "directional", not a prediction.

- [ ] **Science, not journey.** Preemptions, retries, infrastructure incidents belong in an `EXECUTION_LOG.md` or scripts/comments, not the main report. If the same experiment were re-run cleanly, would this sentence still belong? If not, it's journey.

- [ ] **Each metric labelled with what it measures.** AUC / Top-1 = representation quality (can the encoder distinguish a target from negatives?). R² = forecast match vs random / naive baselines (prediction error). U = dimension usage / spread (necessary, not sufficient).

- [ ] **Plots embedded inline.** Any plot referenced in the report is in a `plots/` subdirectory and embedded with `![alt](plots/<name>.png)`. No orphan images.

- [ ] **Multi-sample held-out eval** when comparing arms whose differences are smaller than single-batch noise (~0.005 AUC). Report mean ± stdev with N ≥ 10 samples; show error bars.

- [ ] **Sub-agent review pass.** Before merging the report, dispatch a sub-agent: pass it the report + underlying data; ask which claims aren't directly supported. Address every flagged item.
