# 2026-04-28_exp_csb_pair_revin — RevIN counterpart of the contrastive-loss A/B

**What**: same A/B as `2026-04-28_exp_csb_pair_span512` but with **RevIN** as the
reversible normaliser instead of `RevEWMNorm span=512`. Two arms, single
axis difference (loss flag).

| Arm | `--rev-norm-kind` | `--loss-shape` |
|---|---|---|
| C | `revin` | `cosine_similarity_batch_no_time_neg` |
| D | `revin` | `cosine_similarity_batch` |

**Why**: combined with `2026-04-28_exp_csb_pair_span512` this gives a **2 × 2 grid**
covering EWMA × RevIN normalisers × old × new losses. The EWMA pair
showed CSB beats no_time_neg by 4.5% MASE; the question is whether the
loss-flag direction is the same under RevIN, or whether it depends on
the normaliser.

**When**: Apr 28 2026.

**Status**: in flight.

**Setup**: see `run.sh`. Same protocol as `2026-04-28_exp_csb_pair_span512` (30k bb
+ 30k qh, mix=1.0, freq_emb=3, mixup=0.3, single-shot, `_best_loss →
_FINAL.pth`) but with RevIN. RevIN has no span parameter; it normalises
each input window with its own statistics.

**Eval**: `synth_eval.py` extended to accept `--rev-norm-kind revin`
(small patch in this PR). Same 1024-sample held-out synth set
(seed=99999999). Numbers directly comparable with the EWMA pair via the
shared `2026-04-27__aggregate/results/synth_eval.csv`.

**Code referenced**:
- `../2026-04-27_freq-embedding/scripts/train.py`
- `../2026-04-13_gift-eval/scripts/train_forecasting_head.py`
- `../2026-04-27_freq-embedding/scripts/synth_eval.py` (patched here for RevIN)
- `../2026-04-27_freq-embedding/scripts/synth_compare_grid.py` (already supports
  RevIN; used for the final 4-arm grid).

**Artefacts (when complete)**:
- `plots/synth_compare_pair_revin.png`: 12-panel × 2-arm grid for the
  RevIN pair on its own.
- `plots/synth_compare_grid_4arm.png`: 12-panel × 4-arm grid (EWMA
  ntn/csb + RevIN ntn/csb) for the cross-cutting visual.
- `results/synth_eval.csv`: local copy of the 2 RevIN rows.
- 2 new rows in `../2026-04-27__aggregate/results/synth_eval.csv`.
- Backbone + qhead checkpoints (~80 MB + 2.5 MB each) under the
  project-root `sync_csb_pair_revin/checkpoints/` (NOT in any worktree,
  per CLAUDE.md rule 4).

**Reusing the EWMA-pair instance**: this run uses the same Vast.ai
instance that ran `2026-04-28_exp_csb_pair_span512` (deps already installed, raw
`vastai create` + manual SSH key attach to bypass the
vastrun-provision destroy bug). Saves ~10 min of provisioning + setup.
