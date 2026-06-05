# Run log — per-arm timelines, crashes/resumes, checkpoint inventory

*Pure operational journey for `exp_realonly_4096_smaller_tau_sweep`. Demoted out of the report (§6/§7 of the original) per REPORT_STANDARD ("science, not journey"). The reported metrics derive from the external sync directories named below; those CSVs are **not committed to git**, so they cannot be recomputed from this repo.*

All times UTC. Checkpoint paths are absolute on the laptop (`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/`).

## Per-arm details

### τ = 0.05 (#27 arm 005)

- Hyperparams: τ=0.05 fixed, AdamW, lr=1e-4 (BB) / 3e-4 (head), bs=96, 30k BB + 30k head steps.
- Timeline: BB started Fri May 1 09:12; completed without crashes; STAGE H (head) started Fri May 1 ~16:55; STAGE E (gift_eval) ran Fri May 1 ~20:59 after a partial-CSV resume (the credit-restore window forced a re-launch of eval with `--resume`, which cleanly skipped the 38 already-done configs); full pipeline DONE Sat May 2 00:58.
- Backbone: final ema_loss 5.7084 at step 30,000.
- Head: final ema_loss 0.06923 at step 30,000.
- Eval results CSV: `sync_realonly_4096_smaller_tau_sweep/tau005/results/all_results.csv`
- Key checkpoints (all under `sync_realonly_4096_smaller_tau_sweep/tau005/checkpoints/`):
  - `tiny_realonly_4096_smaller_tau005_FINAL.pth` (backbone)
  - `tiny_realonly_4096_smaller_tau005_best_loss.pth` + `_best_loss_optimizer.pth`
  - `tiny_realonly_4096_smaller_tau005_30k.pth` + `_30k_optimizer.pth` (last periodic)
  - `R1q_realonly_4096_smaller_tau005_FINAL.pth` (head)
  - `R1q_realonly_4096_smaller_tau005_best.pth` + `_best_optimizer.pth`

### τ = 0.07 (#27 arm 007 — halted)

- Hyperparams: τ=0.07 fixed, otherwise identical.
- Timeline: hit two operational interruptions during BB — one at ~step 2k forcing a `_2k.pth` resume after the original machine3 was host-stopped, and a second around step ~3,600 during the credit-restore window forcing `_3600.pth` resume; BB finally reached step 30k as `STAGE B DONE` Sat May 2 ~05:47. STAGE H (head) started ~05:47; the qhead training process **stopped writing log/checkpoints at step ~11.5k** (Sat May 2 ~08:55) — root cause not preserved in local logs. User chose **not to resume** because the small-data ranking was deemed unreliable (47-epoch regime caveat). The dead process was confirmed gone by ssh inspection at ~10:53; instance was destroyed at ~10:57 after final artifact pull.
- Backbone: final ema_loss 5.7208 at step 30,000.
- Head: ema_loss 0.07467 at step 11,800 (last CSV row); CSV has 11,800 rows, no FINAL.pth produced.
- Eval results: **none** (the `results/` dir is empty).
- Key checkpoints (all under `sync_realonly_4096_smaller_tau_sweep/tau007/checkpoints/`):
  - `tiny_realonly_4096_smaller_tau007_FINAL.pth` (backbone — usable for resume into a future head retrain)
  - `tiny_realonly_4096_smaller_tau007_best_loss.pth` + `_best_loss_optimizer.pth`
  - `tiny_realonly_4096_smaller_tau007_30k.pth` + `_30k_optimizer.pth`
  - `R1q_realonly_4096_smaller_tau007_best.pth` + `_best_optimizer.pth` (best of the truncated qhead run, ema_loss=0.0749 at step ~11.5k)
  - **No `R1q_*_FINAL.pth`** — the run never completed; to finish this arm, resume the qhead from `R1q_..._best.pth` + `_best_optimizer.pth` and re-launch eval.

### τ = 0.20 (#27 arm 020)

- Hyperparams: τ=0.20 fixed, otherwise identical.
- Timeline: BB ran with one resume around step 24k (`STAGE H (RESUME)` line at step 24,000, best_loss=0.0709) — the credit-restore window also touched this arm; recovered cleanly. STAGE B / H / E all completed; ALL DONE Fri May 1 23:20.
- Backbone: final ema_loss 6.3888 at step 30,000 (notably higher than τ=0.05/0.07 — softer contrast yields a higher absolute InfoNCE; expected, not a deficiency).
- Head: final ema_loss 0.07005 at step 30,000.
- Eval results CSV: `sync_realonly_4096_smaller_tau_sweep/tau020/results/all_results.csv`
- Key checkpoints (all under `sync_realonly_4096_smaller_tau_sweep/tau020/checkpoints/`):
  - `tiny_realonly_4096_smaller_tau020_FINAL.pth` (backbone)
  - `tiny_realonly_4096_smaller_tau020_best_loss.pth` + `_best_loss_optimizer.pth`
  - `tiny_realonly_4096_smaller_tau020_30k.pth` + `_30k_optimizer.pth`
  - `R1q_realonly_4096_smaller_tau020_FINAL.pth` (head)
  - `R1q_realonly_4096_smaller_tau020_best.pth` + `_best_optimizer.pth`

### learnable τ (#32)

- Hyperparams: `--tau 0.07 --learnable-tau` (CLIP-style log_inv_tau, init τ=0.07, clamp [0.01, 1.0] post optimizer step). Otherwise identical to #27.
- Timeline: BB had two resumes visible in the local `run.log` (resume at step 17,100 from `_resume.pth`; second resume at step 21,700 after a 21k httpx-class crash — recovered through the same `_resume.pth` mechanism); STAGE B DONE Sat May 2 02:18; STAGE H DONE / STAGE E DONE; ALL DONE Sat May 2 04:18.
- Backbone: final ema_loss 5.7039 at step 30,000 (the lowest of the four arms, but only marginally; τ ended at 0.0525 — slightly *looser* than the τ=0.05 fixed arm, not tighter).
- Head: final ema_loss 0.06818 at step 30,000.
- Final τ at end of training: 0.0526 (`log_inv_tau=2.9453`, auto-detected by both head trainer and eval).
- Eval results CSV: `sync_realonly_4096_smaller_learnable_tau/learnable/results/all_results.csv`
- Key checkpoints (all under `sync_realonly_4096_smaller_learnable_tau/learnable/checkpoints/`):
  - `tiny_realonly_4096_smaller_learnable_tau_FINAL.pth` (backbone — embeds the final learned τ in the state dict)
  - `tiny_realonly_4096_smaller_learnable_tau_best_loss.pth` + `_best_loss_optimizer.pth`
  - `tiny_realonly_4096_smaller_learnable_tau_30k.pth` + `_30k_optimizer.pth`
  - `R1q_realonly_4096_smaller_learnable_tau_FINAL.pth` (head)
  - `R1q_realonly_4096_smaller_learnable_tau_best.pth` + `_best_optimizer.pth`

## Data-loss / operational notes

τ=0.07 had two httpx-class crashes mid-BB (one ~step 2k, one ~step 3,600) and a third event mid-resume (around step 21k — visible as a `Resumed from ... at step 21100` line). Each was recovered through periodic-save anchors (`_2k.pth`, `_3600.pth`) shipped via the `resume_source/` mechanism. The qhead then died at step ~11.5k post-BB-completion; not resumed. The learnable τ run had two resumes for similar credit-restore-class events (step 17,100 and 21,700) and recovered through `_resume.pth` anchors. PR #94 (`fix-dataloader-resume-mod-wrap`) was developed in flight across these two experiments to harden the dataloader stream-position handling around the dataset epoch boundary; future readers should confirm it lands before #6/#9/#10. All four arms have a complete sync_loop tick log under `<sync_dir>/sync.log` and rotated `*.prev` copies; the data-loss surface is operationally clean.

## Operational status of the follow-up runs (as of 2026-05-02)

The science forward-pointer (which runs settle τ) is in the report. The operational placement at the time of writing:

- **#6** — 30k learnable-τ on the full 42.5M-window dataset. Was running on machine5 (vast.ai 35985578); machine5 went offline mid-eval at config 1/97 with FINAL backbone + FINAL qhead saved locally; eval continuation provisioning on machine7 (vast.ai 36005039).
- **#9** — MOIRAI-HP variant (`lr=1e-3, wd=0.1, β=(0.9, 0.98)`) on the full 42.5M-window dataset, same arch + learnable τ as #6. Provisioning on machine6 (vast.ai 36004921).
- **#10** — 1-full-epoch FINAL retrain. Decision pending #6/#9; will resume from whichever 30k checkpoint wins between {#6, #9} and continue ~413k more steps (~1 full epoch of full-4096).

## Pre-existing plot copy

A pre-existing version of the loss-curves plot lives at `<repo>/plots/tau_sweep_and_learnable_loss.png` (rendered earlier in the main checkout); the in-tree `plots/loss_curves.png` here is the canonical one going forward.
