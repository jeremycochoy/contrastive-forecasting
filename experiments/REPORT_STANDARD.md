# Report standard

A "report" here means a single canonical Markdown file per experiment, named `{experiment_name}.md`. The list below is what a sub-agent should check before any report is finalised — feed it the report + the underlying data CSVs and ask "which boxes don't tick?". Address every flagged item.

## Checklist

- [ ] New report filenames follow **`{experiment_name}.md`**.

- [ ] **One file per experiment.** A single canonical Markdown file is the report. Supporting information lives elsewhere in the experiment directory (scripts, docstrings, execution logs, notebooks), never in additional report files.

- [ ] Structure follows **question → result → setup detail**: question first, then plots + inline result tables + bottom-line bullets, then setup detail (arm rationale, metric definitions, big raw-number tables) for readers who want depth.

- [ ] Prefer **pictures over prose** — use a plot wherever one can carry the meaning, and place each plot directly above its interpretation sentence.

- [ ] **Result tables go inline** with their plot; **setup tables go to the back** (arm rationale, metric definitions, big raw-number tables).

- [ ] The reader follows a **single forward thread** from question to verdict — no backtracking, no open questions held in the head.

- [ ] **Facts only; flag extrapolation.** State measurements directly. Anything that goes beyond what the data directly supports is labelled as a hypothesis.

- [ ] **Science, not journey.** Operational events (retries, infrastructure incidents, preemptions, debugging detours) belong outside the main report. If the same experiment were re-run cleanly, would this sentence still belong? If not, it's journey.

- [ ] **Less is more**: removing a sentence that doesn't carry clear signal is a positive change, especially if the sentence is confusing.

- [ ] **Define specialized vocabulary** where it first appears, including each metric: what it measures and what it does NOT.

- [ ] **Plots embedded inline.** Any plot referenced in the report is stored in a `plots/` subdirectory and embedded inline in the Markdown. No orphan images.

- [ ] **Multi-sample held-out eval** when comparing arms whose differences could plausibly be within single-batch noise. Report mean and variance over multiple samples; show uncertainty visually.

- [ ] **Sub-agent review pass.** Before merging the report, dispatch a sub-agent: pass it the report + underlying data; ask which claims aren't directly supported. Address every flagged item.
