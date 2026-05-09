# Report standard

A "report" here means a single canonical Markdown file per experiment (typically `RESULTS.md`). The list below is what a sub-agent should check before any report is finalised — feed it the report + the underlying data CSVs and ask "which boxes don't tick?". Address every flagged item.

## Checklist

- [ ] **One file per experiment.** A single canonical Markdown file is the report. Supporting information lives elsewhere in the experiment directory (scripts, docstrings, execution logs, notebooks), never in additional report files.

- [ ] **Structure: goal → protocol → what we did → what we learned.** The reader who arrives cold should understand the question, the design, and the conclusion in that order.

- [ ] **Facts only; flag extrapolation.** State measurements directly. Anything that goes beyond what the data directly supports is labelled as a hypothesis.

- [ ] **Science, not journey.** Operational events (retries, infrastructure incidents, preemptions, debugging detours) belong outside the main report. If the same experiment were re-run cleanly, would this sentence still belong? If not, it's journey.

- [ ] **Each metric labelled with what it measures.** The reader should understand what each metric quantifies and what it does NOT — without prior knowledge of the project.

- [ ] **Plots embedded inline.** Any plot referenced in the report is stored in a `plots/` subdirectory and embedded inline in the Markdown. No orphan images.

- [ ] **Multi-sample held-out eval** when comparing arms whose differences could plausibly be within single-batch noise. Report mean and variance over multiple samples; show uncertainty visually.

- [ ] **Sub-agent review pass.** Before merging the report, dispatch a sub-agent: pass it the report + underlying data; ask which claims aren't directly supported. Address every flagged item.
