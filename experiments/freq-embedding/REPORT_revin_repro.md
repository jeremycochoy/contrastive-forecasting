# EXP1 — RevIN reproduction (#28 redo)

## Why

The previous session (late-Apr 2026) trained a RevIN backbone+qhead
(ablation #28) on mix=0.5 + GIFT-Eval, but lost both checkpoints to a
partial-transfer SSH drop. The eval CSV survived (`results/R1q_femu_revin/`)
but no weights, so the RevIN-arm synth-grid plot the user asked for
couldn't be generated.

Reproducing the run was queued first in HANDOFF.md TL;DR: ~$1.50,
~6h on a 4090 with the PR #47 sync fixes that were merged in the
intervening session.

## Setup

| Knob | Value |
|---|---|
| Script | `experiments/freq-embedding/scripts/train.py` |
| Steps | 30k backbone, 30k qhead |
| Batch size | 24 |
| LR | 1e-4 (backbone), 3e-4 (qhead) |
| Architecture | Tiny: H=512, L=6, W=16, GRU encoder, 8 heads |
| Data | mix=0.5 (50% HF base-bundles + 50% periodic synth) |
| Freq emb | dim=3, mixup=0.3 |
| Reversible norm | RevIN (single per-instance z-score) |
| Patch stats | none |
| Loss | `cosine_similarity_batch_no_time_neg` (matches #28) |

## Results

### Backbone (`tiny_femu_revin`)

- Wallclock: 1.4h
- Best gap: **0.4693** at step 21800
- Best loss: 0.4254 at step 27900
- 5.5–6.0 sps throughout (HF-bottlenecked at mix=0.5)
- Behavior matched the previous session's #28 numbers within noise (gap
  was around 0.47 there).

### Quantile head (`R1q_femu_revin_v2`)

- Wallclock: 1.9h
- Best loss: **0.0522** at step 29000
- 4.3 sps

### Synth grid

`experiments/freq-embedding/plots/synth_qhead_grid_revin.png` (12-panel).

Visually indistinguishable from the EWMA grid we had on file
(`synth_qhead_grid.png`): same amplitude damping and phase drift on
clean periodics, same panels work / don't work across the two
normalisers. Confirms (qualitatively, single seed) that switching
RevEWMNorm → RevIN does **not** by itself fix the periodic-tracking
problem — the bottleneck is upstream of the normaliser.

## Caveats

- Single seed.
- mix=0.5 setup; in this session's later experiments we re-ran on
  mix=1.0 (synth-only) where the comparison is clearer.
- The "RevIN better than EWMA on periodics" finding from the previous
  session was confounded by the EWMA arm using span=32; later in this
  session the synth span sweep showed EWMA at span=512 beats RevIN by
  a wide margin on synth (see `REPORT_synth_span_sweep.md`).

## Artefacts

- Backbone: `checkpoints/tiny_femu_revin_best_gap.pth` (note: gap is no
  longer the recommended selector — see HANDOFF for the rationale).
- Qhead: `checkpoints/R1q_femu_revin_v2_best.pth`.
- Plot: `experiments/freq-embedding/plots/synth_qhead_grid_revin.png`.
- Loss CSVs: `checkpoints/tiny_femu_revin_losses.csv`,
  `R1q_femu_revin_v2_losses.csv`.
