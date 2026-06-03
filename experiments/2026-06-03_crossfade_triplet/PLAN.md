# PLAN — Crossfade (A, B, C) triplet on allt·0.8%, no bottleneck, 3-layer encoder (#328)

Follow-up to #326 (regime crossfade). Base recipe: the all-time 0.8%-fork arm of
#322 (`allt·0.8%`) plus #326's regime crossfade, with **three changes applied
together**, all expected to lift the downstream forecast:

1. **Crossfade as an explicit (A, B, C) triplet** (replaces #326's 10% C-only
   slice). Per step, draw two distinct real windows A, B from the step's real
   sub-batch, z-normalise each per series, and blend
   `C = (1 − s(t))·A_norm + s(t)·B_norm` with #326's monotone ramp s(t)
   (midpoint `m ~ U(0,T)`, width `w ~ LogUniform(T/128, T)`). Append **all three**
   rows — `A_norm`, `B_norm`, `C` — on top of the natural batch (additive). #326
   added only `C` and left the raw real rows as its in-batch parents; here both
   parents are z-normalised and present alongside the blend.
2. **No bottleneck** — drop `--forecaster-d-model 128 --forecaster-n-heads 4`; the
   forecaster runs at the encoder width (384, 6 heads).
3. **3-layer encoder** — `--num-encoder-layers 3` (was 6).

#322's stabilised batch-1024 recipe is kept throughout: `--qk-norm`,
`--attn-out-norm`, `--subtract-contrastive-floor`, `--pos-in-denominator`.

## Arm

| knob | allt·0.8% base (#322) | this arm (#328) |
|---|---|---|
| forked-arma mix | 0.8% | 0.8% (`--mix-ratio 0.0078125`) |
| crossfade | none | 1 explicit (A_norm,B_norm,C) triplet / step (`--crossfade-triplets 1`) |
| forecaster width | 128 (bottleneck) | 384 = encoder width (no bottleneck) |
| encoder layers | 6 | 3 |
| backbone params | 12.7M | 16.7M |

Single backbone seed (20260520), global batch 1024, 12500 steps, lr 1e-3 — the
same training budget as #322 / #326.

## Evaluation

Each backbone is frozen and scored by training a fresh quantile forecasting head
on top of it, once with 2 layers and once with 6 (the 2L / 6L q-heads of
#322 / #326): 30k steps, batch 256. The heads then run GIFT-Eval
(`--strategy B4`) on the full 97-task benchmark and the 11-task triage subset.
Forecast quality is **GM-Relative MASE**: the geometric mean, over the 97 tasks,
of model error / seasonal-naive error (lower is better; 1.0 = seasonal-naive).

## Comparison

Primary baseline = `allt·0.8%` (#322), the stated base, evaluated with the same
2L / 6L heads (per-config summaries on disk). Δ = gm(this arm) − gm(allt·0.8%),
**paired bootstrap** over the 97 shared configs (and per domain). Reference
points carried for context: `allt·10%` (2L 1.222, 6L 1.191) and the #326 C-only
`crossfade·10%` (2L 1.208, 6L 1.178). Baseline GM (full-97): 2L 1.213, 6L 1.198.

Because three things change at once, a delta measures their **joint** effect and
cannot be attributed to any single change.

## Layout

- code (this worktree, `WT`): `src/synthetic_crossfade.py` (new
  `generate_crossfade_triplets`), `src/dataloader.py` (`cross_triplets`),
  `scripts/.../train.py` (`--crossfade-triplets`).
- outputs (elisa `~/workspaces/...`, gitignored `*.pth`): `runs/`, raw
  `results/gift_eval_*/`.
- committed: PLAN, scripts, RESULTS.md, `plots/*.png`, derived
  `results/*.csv`.
