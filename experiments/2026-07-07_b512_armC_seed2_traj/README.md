# 2026-07-07 — B=512 arm-C retrain, new seed, trajectory checkpoints (#371)

Recovers the B=512 thread of #371. The original #366 arm-C run
(λ_e=1, λ_h=1, τ=0.90, B=512, seed 20260520) cannot be prolonged: its
checkpoints were written to `runs/` inside the #366 worktree and were
deleted with that worktree after PR #368 merged — no weights+optimizer
pair survives anywhere on this machine (full-disk sweep 2026-07-07).

Per owner instruction: retrain the same recipe from scratch with a
**different seed** (control), keep fine trajectory checkpoints, extend
past the original budget, then score heads + GM-Rel MASE at each locus.

**Reporting**: this experiment produced the B=512 half of the batch-size
comparison; the joint conclusion is written up in the #369 report, which
references the `plots/` and `results/gm_table.csv` here directly. No
standalone report file lives under this directory.

## Design

- Recipe identical to #366's `train_backbone_sigreg.sh` arm C flags
  (B=512, 3-layer GRU patch encoder + 6-layer encoder, SIGReg λ_e=1,
  λ_h=1, EMA τ=0.90, CPC weight 1, LR 1e-3, dropkey 0.70).
- **Seed 20260707** (original: 20260520) — a seed-control replica: the
  12,500-step locus re-measures the #366 arm-C cells under a fresh seed.
- Backbone budget: **50,000 total steps** across three concatenated
  runs, each optimizer-state resumed from the previous last checkpoint.
  - 0 → 25,000 in `train_b512_seed2.sh` (13.6 h, one run).
  - 25,000 → 37,500 in `extend_b512_to_37500.sh` (6.5 h, resumed).
  - 37,500 → 50,000 in `extend_b512_to_40000.sh` invoked with
    `STEPS=50000` (5.9 h, resumed).
- `--traj-save-every 500`, `--save-every 2500` throughout.
- Save dir: `/home/jupyter/contrastive-forecasting/sync_b512_armC_seed2/`
  (main checkout, never a worktree — checkpoint-safety directive from
  the incident above).
- Run name: `bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2`.

## Downstream cells

2L and 6L quantile heads + full-97 GIFT-Eval at 10 loci:
12,500 / 15,000 / 20,000 / 25,000 / 30,000 / 35,000 / 37,500 / 40,000 /
45,000 / 50,000. Protocol mirrors #369's `dl_at_step.sh` — a fresh 30k
head at step 12,500 (2k warmup) serves as the parent-head for every
curve cell, which is then a 10k re-adapt from that head (1k warmup).
20 cells total; every cell's `summary.txt` and `all_results.csv` live
under `results/gift_eval_full_*/`.

## Deliverables in this directory

- `results/gm_table.csv` — headline GM-Relative MASE (full-97) per
  cell, plus the two original-seed B=512 reference rows.
- `plots/gm_curve_512_vs_1024.png` — 2L and 6L GM-Rel MASE trajectory,
  B=512 seed2 (mine) vs B=1024 (from #369), 12.5k → 50k.
- `results/gift_eval_full_*/` — per-cell full-97 all_results.csv +
  summary.txt.
- `scripts/` — all launchers used to produce the cells (backbone train,
  extensions, per-cell head+eval, parallel-worker queue, orchestrator).
- Run logs under `results/`.

## Layout

- `scripts/train_b512_seed2.sh` — 0 → 25k backbone.
- `scripts/extend_b512_to_37500.sh` — 25k → 37.5k backbone.
- `scripts/extend_b512_to_40000.sh` — 37.5k → 40k default; passed
  `STEPS=50000` for the final 37.5k → 50k run.
- `scripts/dl_one_cell.sh` — train+eval one (head-depth, backbone-step)
  cell. Atomic per-cell claim so parallel workers on the same cell list
  don't race.
- `scripts/downstream_seed2.sh` — the original serial downstream for
  loci 12.5k / 15k / 20k / 25k. Kept for reproducibility of the first
  wave of cells.
- `scripts/queue_remaining.sh` — parallel worker queue. Runs a fixed
  CELLS list on one GPU; multiple workers can share it via the atomic
  claim in `dl_one_cell.sh`.
- `scripts/orchestrate_40_45_50.sh` — waits for each 40k/45k/50k
  backbone trajectory checkpoint to land, then launches 2L+6L per locus.
