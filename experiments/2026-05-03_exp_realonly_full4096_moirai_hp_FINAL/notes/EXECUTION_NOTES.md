# Execution notes — #10 real-only full-4096 MOIRAI-HP FINAL

Operational journey for the run behind
[`../exp_realonly_full4096_moirai_hp_FINAL.md`](../exp_realonly_full4096_moirai_hp_FINAL.md).
None of this changes the result; it records how the run was produced and recovered.

## Cost & instance

- **$33.48** on vast.ai instance **36055545** (RTX 5090, **51h30m**, $0.65/h).
- The eval (STAGE E) was re-run **locally on elisa (RTX 4090)** after the vast
  instance went terminal (credit-out) during the on-instance eval retry. Backbone
  and quantile-head checkpoints were unaffected — only the deterministic GIFT-Eval
  scoring step was repeated, which produced the committed
  `results/gift_eval_resume50k_local/{all_results.csv, summary.txt}`.

## Pipeline / resume timeline

1. **STAGE B — backbone** (`scripts/run_resume50k.sh`): resumed from the FRESH run's
   `tiny_full4096_moirai_hp_FRESH_50k.pth` at step 50,001, ran to step 167,000
   (one full epoch on `gift-pretrain-full-4096`, MOIRAI HP). ~12.6h on the RTX 5090.
   `best_loss.pth` (saved at step 166,800) copied to `..._FRESH_RESUME50k_FINAL.pth`.
2. **STAGE H — quantile head** (`scripts/run_qhead_eval.sh`): 30k steps,
   forecast_len=16, lr=3e-4, bs=256, on the frozen RESUME50k backbone.
   final ema_loss = 0.0606. ~2.3h.
3. **STAGE E — GIFT-Eval** (`scripts/run_eval_only.sh`): 97 configs, B4 strategy,
   forecast_len=16. The first on-instance attempt failed all 97 configs with
   "argument should be a str or PathLike, not NoneType" because `GIFT_EVAL` env var
   was not exported; `run_eval_only.sh` exports it and re-runs. The instance then
   credited out, so the eval was finished on elisa.

The run originally used a doubled-date directory string on remote
(`2026-05-03_2026-05-02_...`), preserved inside the launcher scripts for sync_loop
compatibility; the committed experiment dir uses the single-date name.

## Bugs fixed during the run

| PR   | issue |
|------|-------|
| #120 | HF `httpx` client closure mid-stream killed the FRESH run at step 52,400. Now retried + tested. |
| #94  | `skip_rows >= total_rows` on resume → `StopIteration`. Now mod-wraps. |
| #122 | repo-wide reorg into per-experiment dirs. |
| #123 | `PrefetchIterator` early-exit leaked the producer thread → process abort at shutdown. |
| #124 | `train_forecasting_head.py` and `eval_gift_eval_official.py` had hard-coded backbone arch (C=4, H=512, nhead=8); CLI overrides added so the C=1/H=384/nhead=6 backbone evaluates correctly. |

PR #110 (deterministic resume: `hf_rows_consumed` fast-skip + RNG cast fix) is the
code under test in the report's secondary result, not a bug fixed mid-run.

## Artifacts / checkpoint inventory

- backbone end-of-train: `tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth`
  (synced to `sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/`).
- quantile head end-of-train: `R1q_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth` (same dir).
- eval outputs (committed): `results/gift_eval_resume50k_local/{all_results.csv, summary.txt}`.
- launchers (committed, `scripts/`): `run_resume50k.sh`, `run_qhead_eval.sh`, `run_eval_only.sh`, `run.sh`.

The `.pth` checkpoints (~80–230 MB each) and the HF `gift-pretrain-full-4096`
dataset are **not** in git.

## Plots / stats whose generators are not committed

The GIFT-Eval figures (leaderboard / domain-breakdown / progression) **are**
regenerable — their scripts read the committed `results/gift_eval_resume50k_local/`
CSVs (`scripts/plot_leaderboard.py`, `plot_domain_breakdown.py`, `plot_progression.py`),
and each asserts the recomputed overall equals 1.1828.

The resume-continuity figure (`resume50k_continuity.png`) and its scalar statistics
(the mean/std deltas, the Welch/Levene p-values, the v1/v2 reference levels) were
produced from the backbone training-loss CSV / run log, which are **not** committed in
this experiment dir; the figure is kept as a committed artifact but cannot be
regenerated from committed data here.

Two earlier backbone training-curve figures (`full4096_v3_4arm.png`,
`full4096_gaps_only.png`) were **removed** from the report: their step labels
(resumed ≈188,800 / fresh ≈9,800, plus reference arms) could not be reconciled with this
run's 50k→167k bs-256 trajectory, and their source CSV is not committed, so the captions
were not verifiable.
