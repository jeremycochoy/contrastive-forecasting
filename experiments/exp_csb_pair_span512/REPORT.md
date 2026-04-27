# exp_csb_pair_span512 — clean A/B of contrastive losses on the span=512 best arm

**Status: in flight.** Training launched Apr 28 2026.

## Why

Two reasons converge here:

1. **The span=512 baseline (no_time_neg) checkpoint was lost.** During
   the wrap-up of `exp_csb_synth`, an auxiliary git worktree was torn
   down with `git worktree remove --force` — that deletes all untracked
   files in the worktree directory, including `sync_multiexp/checkpoints/`
   where the only local copy of `tiny_femu_span512_synth30k_FINAL.pth`
   lived. The corresponding remote vast.ai instance had been destroyed
   shortly before, so the checkpoint was unrecoverable. CLAUDE.md rule 4
   (PR #82) was added to prevent this happening again.

2. **`exp_csb_synth` had a multi-resume confound.** The CSB run suffered
   three remote-instance failures during backbone training; the FINAL
   checkpoint is the result of `8k → resume → 24k → resume → 30k`,
   which is not a clean single-shot training. A clean rerun is needed
   before treating the result as definitive.

This experiment retrains both arms cleanly (single-shot, no `--resume`)
under matched protocols, so the only axis between them is the loss flag.

## Setup

| Knob | Value (both arms) |
|---|---|
| Steps (backbone) | 30000 |
| Steps (qhead) | 30000 |
| Batch size | 24 |
| LR (bb / qh) | 1e-4 / 3e-4 |
| Mix ratio | 1.0 (synth-only) |
| Freq emb dim | 3 |
| Mixup | 0.3 |
| Reversible norm | RevEWMNorm span=512 |
| Backbone selector | `_best_loss → _FINAL.pth` (gap saturates near step ~1600 deterministically on synth) |
| Qhead selector | `_best.pth → _FINAL.pth` (qhead val loss) |
| Save-every (bb / qh) | 2000 / 1000 |
| `--resume` | NOT used (single-shot) |
| Eval | 1024 held-out synth samples (`synth_eval.py`, seed=99999999) |

Single-axis difference between the arms:

| Arm | `--loss-shape` |
|---|---|
| A | `cosine_similarity_batch_no_time_neg` |
| B | `cosine_similarity_batch` |

## Results

_Pending — fill in when the run completes._

| Arm | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| A (`no_time_neg`, this run) | TBD | TBD | TBD | TBD |
| B (`cosine_similarity_batch`, this run) | TBD | TBD | TBD | TBD |
| Seasonal Naive (held-out) | 0.497 | 0.344 | 0% | 0% |

For reference, the historical (lost-checkpoint and multi-resume) numbers
on the same eval:

| Arm | GM-MASE | GM-WQL | Notes |
|---|---:|---:|---|
| `fe+mu @ 30k span=512` (lost baseline, _best_gap selector, exp_span_sweep_synth) | 0.848 | 0.413 | Selector was `_best_gap`, which on synth saturates near step ~1600 |
| `fe+mu @ 30k span=512 +cosine_similarity_batch` (exp_csb_synth) | 0.886 | 0.434 | Multi-resume run (8k → 24k → 30k) |

Direct comparison of the new arms with the historical numbers will show
how much of the 4.5% gap was loss-flag vs selector vs resume.

## Caveats

_Pending._

## Open questions

_Pending._

## Artefacts

- Backbones (~80 MB each) and qheads (~2.5 MB each) not tracked in git;
  pulled to `<repo>/sync_csb_pair_ewma/checkpoints/` (NOT in any worktree,
  per CLAUDE.md rule 4).
- Eval rows in `experiments/_aggregate/results/synth_eval.csv`.
- Plots in `plots/`:
  - `synth_compare_pair.png` — 12-panel × 2-arm forecast grid.
- `run.sh` — the actual driver script (committed).
