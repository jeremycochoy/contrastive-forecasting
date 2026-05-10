# Report standard

A "report" here means a single canonical Markdown file per experiment, named `{experiment_name}.md`. The list below is what a sub-agent should check before any report is finalised — feed it the report + the underlying data CSVs and ask "which boxes don't tick?". Address every flagged item.

## Checklist

- [ ] New report filenames follow **`{experiment_name}.md`**.

- [ ] Each experiment has **one canonical Markdown report**; supporting information lives elsewhere in the experiment directory (scripts, docstrings, execution logs, notebooks).

- [ ] Structure follows **question → result → setup detail**: question first, then plots + inline result tables + bottom-line bullets, then setup detail (arm rationale, metric definitions, big raw-number tables) for readers who want depth.

- [ ] Prefer **pictures over prose** — use a plot wherever one can carry the meaning, and place each plot directly above its interpretation sentence.

- [ ] **Result tables go inline** with their plot; **setup tables go to the back** (arm rationale, metric definitions, big raw-number tables).

- [ ] The reader follows a **single forward thread** from question to verdict — no backtracking, no open questions held in the head.

- [ ] **Define specialized vocabulary** where it first appears, including each metric: what it measures and what it does NOT.

- [ ] **Facts only**; anything beyond what the data directly supports is labelled a **hypothesis**.

- [ ] **Science, not journey** — operational events (retries, incidents, preemptions, debugging detours) belong outside the main report.

- [ ] **Less is more**: removing a sentence that doesn't carry clear signal is a positive change, especially if the sentence is confusing.

- [ ] Every plot is **embedded inline**, stored in a `plots/` subdirectory; no orphan images.

- [ ] Use **multi-sample held-out eval** when differences could plausibly be within single-batch noise; report mean and variance over multiple samples, show uncertainty visually.

- [ ] Before merging, **sub-agent review pass**: pass the report + underlying data to a sub-agent, ask which claims aren't directly supported, address every flagged item.
