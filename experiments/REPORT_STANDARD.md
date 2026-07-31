# Report standard

A "report" here means a single canonical Markdown file per experiment (typically `{experiment_name}.md`).

## Checklist

- [ ] **One file per experiment.** A single canonical Markdown file is the report. Supporting information lives elsewhere in the experiment directory (scripts, docstrings, execution logs, notebooks), never in additional report files.

- [ ] **Structure: question → result → protocol → what we learned → optional follow-up / hypothesis.** The reader who arrives cold should understand the question, the design, and the conclusion in that order.

- [ ] **Pictures over prose.** Use a plot wherever one can carry the meaning, with each plot directly above its interpretation sentence. Do not write a sentence the plot already shows.

- [ ] **Result tables go inline** with their plot; **additional detail / annex tables** (arm rationale, metric definitions, big raw-number tables) go to the back.

- [ ] **Facts only.** State measurements directly.

- [ ] **Science, not journey.** Operational events (retries, infrastructure incidents, preemptions, debugging detours) belong outside the main report. If the same experiment were re-run cleanly, would this sentence still belong? If not, it's journey.

- [ ] The reader follows a **single forward thread** from question to verdict, no backtracking, no open questions left hanging.

- [ ] **Less is more**: removing a sentence that doesn't carry clear signal is a positive change, especially if the sentence is confusing.

- [ ] **Define specialized vocabulary** where it first appears. Do not define terms standard in the field — standard means used often in other reports.

- [ ] **Plots embedded inline.** Any plot referenced in the report is stored in a `plots/` subdirectory and embedded inline in the Markdown. No orphan images.

- [ ] **Multi-sample held-out eval** when comparing arms whose differences could plausibly be within single-batch noise. Report mean and variance over multiple samples; show uncertainty visually.
