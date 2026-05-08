# Project-Specific Empirical Learnings

Cross-cutting findings and conventions from experiments. Each entry cites the experiment / PR that produced it. New entries go here, not in CLAUDE.md.

## Training hyperparameters

- **MOIRAI HP wins.** lr=1e-3, wd=0.1, β2=0.98, no warmup, no cosine beats default HP (lr=1e-4, wd=0.01, β2=0.999) on 30k steps — head-to-head (PR #104) GM-MASE 1.6391 vs 1.8043, GM-MAPE_SN 1.1850 vs 1.3698, GM-CRPS_SN 1.0155 vs 1.1000. Use MOIRAI HP for new runs unless otherwise specified.
  - MOIRAI-HP arm (run `tiny_realonly_full4096_moirai_hp`, head `R1q_realonly_full4096_moirai_hp`): [`2026-05-02_exp_realonly_full4096_moirai_hp/REPORT.md`](2026-05-02_exp_realonly_full4096_moirai_hp/REPORT.md), config in [`run.sh`](2026-05-02_exp_realonly_full4096_moirai_hp/run.sh).
  - Default-HP baseline (`tiny_realonly_full4096_learnable_tau`): [`2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md`](2026-05-02_exp_realonly_full4096_learnable_tau/REPORT.md).
  - Full-epoch follow-up (resumes from MOIRAI-HP 30k bundle): [`2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/REPORT_PLAN.md`](2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/REPORT_PLAN.md).

## Training infrastructure

- **HF dataloader resume fix (PR #112).** Pre-sweeping all 4274 parquet shards' metadata to find the resume start-shard takes 18 min on cold cache. Now reads shard 0's metadata only and assumes uniform sizing — works because gift-pretrain bundles are uniformly 10000 rows/shard at upload. Don't revert.
- **PrefetchIterator doesn't actually parallelize.** The Python thread is CPU-bound and holds the GIL during HF/parquet decode. >2× throughput needs multiprocessing workers (`DataLoader(num_workers>0)`). Not landed.

## Conventions

- Plot scripts use blue / green / red / orange across all panels for four-arm comparisons.
- `B<batch>` convention in plot labels for batch-size-dependent runs.
- Run-name convention: `tiny_<dataset>_<arm>_<suffix>` for backbones, `R1q_<dataset>_<arm>_<suffix>` for quantile heads.
