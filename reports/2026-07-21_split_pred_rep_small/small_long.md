# Small-model long-training sweep — 6 arms × 200k steps (#379)

*v0 — implementation-only. Fills in as backbone / downstream cells finish.*

## Question

Do the between-arm effects observed in #374
([`reports/2026-07-10_split_pred_rep/split_pred_rep.md`](../2026-07-10_split_pred_rep/split_pred_rep.md))
at 12.5k–50k steps on a 17M-parameter backbone hold when the backbone
is ~4–6M parameters and trained ≥4× longer?

Specifically:

1. Does the between-arm ranking at 25k in this small-model run match
   #374's ranking at 25k?
2. Does bimoco / arm 6 v2's `1 − cos(f̂, h_{t+1})` continue to climb
   through 200k, or plateau, or reverse?
3. Does arm 5's alignment plateau (`1 − ff ≈ 0.4` at 50k in #374) break
   through at 100k or 200k?

## Design

Six arms, same loss recipes as #374 (see arm table in
[`../../experiments/2026-07-21_split_pred_rep_small/README.md`](../../experiments/2026-07-21_split_pred_rep_small/README.md)).
Only backbone architecture and training length change:

- Backbone: `d_model=128, n_heads=16, num_encoder_layers=3, num_layers=3`
  — ~4–6M parameters, encoder-to-body ratio 1:1 (vs 1:2 in #374).
- Training: `batch_size=128, total_steps=200,000` — 25.6M samples over
  200k steps, no revisit against `gift-pretrain-full-4096`'s 42.7M rows.
- Downstream: 5 backbone-step cells `{2k, 25k, 50k, 100k, 200k}` × 2
  head-layer sizes `{2L, 6L}` × 6 arms = 60 full-97 GIFT-Eval B4 cells.

## Results

*Filled in as cells complete.*

### GM-Relative MASE per arm × backbone step

Placeholder: `plots/gm_curve_per_arm_2L.png`,
`plots/gm_curve_per_arm_6L.png`.

### `1 − cos(f̂, h_{t+1})` (cos_error) per arm

Placeholder: `plots/cos_error_per_arm.png`.

### Dim usage per arm (u_batchtime for `h_t` and `e_t`)

Placeholder: `plots/dim_usage_per_arm.png`.

### Per-run training-loss curves

Placeholder: `plots/per_run_loss.png`.

### Side-by-side vs #374 finals

*Placeholder.*

## Answers to the three questions

*Filled in once all cells complete.*
