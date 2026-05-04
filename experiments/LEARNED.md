# Project-Specific Empirical Learnings

Cross-cutting findings and conventions accumulated across experiments. Each
entry should cite the experiment / PR that produced it. New entries go here
rather than in CLAUDE.md so that root-level instructions stay short.

## Training hyperparameters

- **MOIRAI HP wins.** lr=1e-3, wd=0.1, β2=0.98, no warmup, no cosine beats
  default HP (lr=1e-4, wd=0.01, β2=0.999) on 30k steps. Use MOIRAI HP for
  any new run unless otherwise specified.
  - MOIRAI-HP arm (run name `tiny_realonly_full4096_moirai_hp`,
    head `R1q_realonly_full4096_moirai_hp`):
    [`2026-05-02_exp_realonly_full4096_moirai_hp/REPORT.md`](2026-05-02_exp_realonly_full4096_moirai_hp/REPORT.md),
    config in
    [`2026-05-02_exp_realonly_full4096_moirai_hp/run.sh`](2026-05-02_exp_realonly_full4096_moirai_hp/run.sh).
  - Default-HP baseline (run name
    `tiny_realonly_full4096_learnable_tau`):
    [`2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md`](2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md).
  - Head-to-head (MOIRAI vs default at 30k, PR #104):
    GM-MASE 1.6391 vs 1.8043, GM-MAPE_SN 1.1850 vs 1.3698,
    GM-CRPS_SN 1.0155 vs 1.1000.
  - Full-epoch follow-up (resumes from MOIRAI-HP 30k bundle):
    [`2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/REPORT_PLAN.md`](2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/REPORT_PLAN.md).

## Training infrastructure

- **HF dataloader resume fix (PR #112).** Pre-sweeping all 4274 parquet
  shards' metadata to find the resume start-shard takes 18 min on cold cache.
  Now reads shard 0's metadata only and assumes uniform sizing — works for
  the gift-pretrain bundles whose shards are uniformly 10000 rows each at
  upload. Don't revert.
- **PrefetchIterator doesn't actually parallelize.** The Python thread is
  CPU-bound and holds the GIL during HF/parquet decode. To get >2×
  throughput we'd need multiprocessing workers (PyTorch
  `DataLoader(num_workers>0)` pattern). Not landed yet.

## Conventions

- Plot scripts use a consistent color scheme: blue / green / red / orange
  across all panels for the four-arm comparison.
- `B<batch>` convention in plot labels for batch-size-dependent runs.
- Run-name convention: `tiny_<dataset>_<arm>_<suffix>` for backbones,
  `R1q_<dataset>_<arm>_<suffix>` for quantile heads.
