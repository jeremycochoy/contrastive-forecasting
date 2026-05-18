# Architecture-improvement plan — optimise information per HUMAN-HOUR

Standing task (user 2026-05-18): with any free compute, improve this
architecture's held-out GM-Relative MASE, **as fast as possible in
wall-clock**. Starts only when the eval/report work (task #10, PR #296)
is done and GPUs are free. Fully autonomous.

## Target & what we know
- Goal: full GM-MASE **< 1.292** (beat best prior v11c); stretch **< 1.0**
  (beat seasonal naive — no arm ever has).
- Anchors (full, 97 cfg): v11c 1.292 · v16 1.335 · v13 bottleneck 1.451 ·
  this run (50k) **1.4377** · seasonal-naive 1.0.
- **More training does NOT help** (50k→150k triage 1.561→1.574, flat) →
  the ceiling is architectural, not steps/seed. Tweaking horizon is wasted
  compute.
- **Contrastive fit ≠ MASE** (project learning: v11c had higher loss but
  best MASE). The objective MUST be GM-MASE (triage proxy), never
  loss/AUC/top1.

## Speed doctrine (information per wall-clock hour)
1. **Frozen-backbone first.** The backbone is the expensive part and 3 are
   already trained (50k/100k/150k). Eval-only and q-head-only experiments
   cost minutes–1h with **zero backbone retrain** → do these first.
2. **Parallel, one variable per arm.** 2 GPUs → ≥2 arms at once; never
   confound two changes in one arm.
3. **Triage(11, ~3 min) screens; full(97) only for a triage improver.**
4. **Proxy-calibration gate before trusting any short proxy:** a candidate
   proxy must rank the known anchors (v11c / v16 / this-50k) consistently
   with full eval; if not, lengthen it. No proxy → no proxy-gated claims.
5. Reuse the lock + watcher + idempotent-skip infra. Never grad-clip;
   stable recipes only (fp32 residual, or 1L/fp16). Don't disturb other
   sessions' GPU0 notebooks. Findings → leaderboard md → small PRs.

## Tier 0 — frozen 150k backbone, NO retrain (minutes; do first, parallel)
Eval-time knobs on the existing backbone+q-head (each triage ≈ 3 min):
decoding `--strategy`, `--forecast-len`, `--head-causal`. Then q-head-only
sweeps (retrain just the cheap head, 15k ≈ 30 min): head layers {2,3},
lr, `--head-train-input {e_then_f,f}`, `--reconstruction` mode. Highest
info/hour; the project already shows head/eval choices move MASE.

## Tier 1 — short backbone proxy (~15k steps + 10k head + triage ≈ ~1 h, parallel)
Highest-leverage backbone levers, one per arm, vs a fixed control:
- **Normalisation** (usually the biggest TS lever): `--rev-norm-span
  {64,128,256}`, RevIN vs EWMA, `--patch-stats {diff,raw}`.
- **dropkey {0.8,0.9}** under the new full_fh_negs loss (only 0.70 tested).
- **Bottleneck off vs on** under the new loss (v16-style no-bottleneck +
  full_fh_negs+normInfoNCE — isolates whether the bottleneck is the drag).
- Patch encoder variant.
Keepers (triage < best) → 50k confirm + full eval.

## Tier 2 — ShinkaEvolve autonomous search (overnight, once a proxy passes the gate)
Scaffold a Shinka task (shinka-setup/convert): evolve the recipe/arch
block; objective = triage GM-MASE via the calibrated ≤~20 min proxy;
parallel workers across GPUs. This is the autonomous engine for the
long tail; Tiers 0–1 de-risk the proxy and seed good genomes first.

## Tracking
`LEADERBOARD.md` (arm → change → triage → full → keep/kill), updated each
arm. Stop a direction after 2 consecutive non-improvers. Each real
improver = its own small PR with the before/after + curve.
