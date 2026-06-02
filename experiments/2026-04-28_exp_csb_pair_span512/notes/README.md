# 2026-04-28_exp_csb_pair_span512 — clean A/B of contrastive losses on the span=512 best arm

**What**: side-by-side comparison of two contrastive losses on the
otherwise-frozen best arm from `2026-04-27_exp_span_sweep_synth` (fe+mu mix=1.0
synth-only, EWMA span=512, 30k bb + 30k qhead):

| Arm | Loss |
|---|---|
| A | `cosine_similarity_batch_no_time_neg` (the previous default; re-runs the lost baseline) |
| B | `cosine_similarity_batch` (paper-matching, includes within-time and cross-time negatives; re-runs `2026-04-27_exp_csb_synth` clean to drop its multi-resume confound) |

**When**: Apr 28 2026.

**Why redo both arms**:
- The original span=512 baseline checkpoint was lost when an auxiliary
  worktree was torn down (CLAUDE.md rule 4 added in PR #82 to prevent
  recurrence).
- The original CSB run (`2026-04-27_exp_csb_synth`) had three remote-instance failures
  during backbone training and was a multi-resume composite, leaving the
  loss-vs-resume effects confounded.
- Both arms now use `_best_loss → _FINAL.pth` for the FINAL backbone (gap
  saturates near step ~1600 deterministically on synth, so `_best_loss`
  is the right selector).

**Setup**: see `run.sh`. Both arms share every hyperparameter except
`--loss-shape`. Single-shot (no `--resume`). `_best_gap.pth` and periodic
`_Nk.pth` snapshots are preserved on disk for later re-eval if we want to
swap the selector.

**Eval**: same held-out 1024-sample synth set (`synth_eval.py`,
`seed=99999999`) used by every prior arm in
`../2026-04-27__aggregate/results/synth_eval.csv`. Numbers are directly comparable
against the historical rows (the SN baseline values in the CSV are
byte-identical for any arm using the same eval, which proves the test set
is the same).

**Status**: in flight (training launched Apr 28 2026).

**Code referenced**:
- `../2026-04-27_freq-embedding/scripts/train.py`
- `../2026-04-13_gift-eval/scripts/train_forecasting_head.py`
- `../2026-04-27_freq-embedding/scripts/synth_eval.py`
- `../2026-04-27_freq-embedding/scripts/synth_compare_grid.py` (for the 2-arm comparison plot)

**Artefacts (when complete)**:
- `plots/synth_compare_pair.png`: 12-panel × 2-arm forecast grid on the
  same fixed-seed samples.
- `results/synth_eval.csv`: local copy of the per-arm eval rows.
- Aggregate CSV updated with the two new rows.
- Backbone + qhead checkpoints (~80 MB + 2.5 MB each) not tracked in git.
  Pulled to the project root's `sync_csb_pair_ewma/checkpoints/` (NOT in
  any worktree, per CLAUDE.md rule 4).
