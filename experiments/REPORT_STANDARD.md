# Report standard

A "report" here means a single canonical Markdown file per experiment (typically `{experiment_name}.md`).

## Checklist

- [ ] **One canonical Markdown report per experiment.** Supporting information lives elsewhere in the experiment directory (scripts, docstrings, execution logs, notebooks), never in additional report files.

- [ ] **Structure:** Order by value to the reader, highest first. A cold reader sees the conclusion in the first screen, in a plot, not in prose. Details go last.

- [ ] **Pictures over prose.** Use a plot wherever one can carry the meaning, with each plot directly above its interpretation sentence. Do not write a sentence the plot already shows.

- [ ] **Tables go to the back**, after the plots, most of the time.

- [ ] **Facts only.** State measurements directly.

- [ ] **Science, not journey.** Operational events (retries, infrastructure incidents, preemptions, debugging detours) belong outside the main report.

- [ ] The reader follows a **single forward thread** from question to verdict, no backtracking, no open questions left hanging.

- [ ] **Low cognitive load.** Short, direct sentences: active voice, name the subject. Cut a sentence that needs a second read.

- [ ] **Less is more**: removing a sentence that doesn't carry clear signal is a positive change, especially if the sentence is confusing.

- [ ] **Define specialized vocabulary** where it first appears. Do not define terms standard in the field (ex: terms used often in other reports).

- [ ] **Plots embedded inline.** Store plots in a `plots/` subdirectory and embed them inline. No orphan images.

- [ ] **Present the uncertainty.** If two arms are close, one eval run is not enough to decide which is better. Show several evals as error bars with the mean and the std.
