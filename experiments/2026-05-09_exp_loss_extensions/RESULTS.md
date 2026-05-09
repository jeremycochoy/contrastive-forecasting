# Loss extensions — Exp 3 (REJECT) and Exp 4-only (REJECT)

τ-sweep Exp 1 picked **τ = 0.20 / gru** as the recipe to extend (R²_random
= 0.7731 on the held-out batch). This experiment tested two loss-shape
extensions on that recipe at τ = 0.20.

## Setup

All three arms reuse the Exp 1 recipe (gru, τ = 0.20, 15k steps, B = 256,
EWMA RevIN span 128, freq+seas emb dim 3, mixup 0.30). Only
`--loss-shape` varies:

- **baseline τ=0.20** — `cosine_similarity_batch` (Exp 1 winner).
- **Exp 3** — `cosine_similarity_batch_add_pos_htft` (PR #181 cumulative):
  adds an `(h_t, f_t)` positive on top of the baseline `(h_t, f_{t-1})`
  positive. Run on elisa, 15k steps.
- **Exp 4-only** — `cosine_similarity_batch_add_f_cross_negs` (PR #182
  non-cumulative): adds f-side cross-(b, c) negatives. Run on vast.ai
  5090 spot 36374874; preempted at step 12,900 / 15,000 (86 %).
  `best_loss.pth` (step 12,800, training loss 8.6948 — the same
  checkpoint the launcher would have promoted at end-of-run) is used as
  FINAL.pth. Trajectory CSV: 12,900 rows.

## Held-out eval (B=256, skip=50M wrap, seed=0, FINAL.pth)

| name                       | loss_shape                                | R²_random | R²_naive | U_temp | U_batch |  AUC   | Top-1  |
|----------------------------|-------------------------------------------|-----------|----------|--------|---------|--------|--------|
| baseline τ=0.20            | cosine_similarity_batch                   | 0.7731    | 0.7256   | 0.0386 | 0.0850  | 0.8938 | 0.7470 |
| Exp 3 +(h_t,f_t) pos       | cosine_similarity_batch_add_pos_htft      | 0.0129    | -0.0005  | 0.2529 | 0.3011  | 0.5079 | 0.2146 |
| Exp 4-only +f-cross-bc neg | cosine_similarity_batch_add_f_cross_negs  | 0.7708    | 0.7118   | 0.0319 | 0.0677  | 0.8902 | 0.7389 |

(Source: `results/loss_extensions_metrics.csv`.)

## Trajectory observations

(`plots/loss_extensions_trajectories.png`, 1000-step MA, per-step training-batch metrics.)

- **Exp 3** sits at AUC ≈ 0.51 / Top-1 ≈ 0.22 across the entire 15k
  trajectory (first-1k mean AUC = 0.538, last-1k = 0.511) — chance-level
  retrieval throughout. U_batch rises near-monotonically from 0.014 to
  0.289 (1k MA); held-out 0.301 vs baseline 0.085; held-out U_temporal
  = 0.253 vs baseline 0.039. The encoder spreads mass across many
  dimensions, but that spread doesn't translate into prediction signal.
- **Exp 4-only** tracks the partial baseline trajectory closely.
  Final-1k mean AUC = 0.898 vs 0.888 for the baseline at step 4,300.
- **Baseline** trajectory is partial: only 4,300 steps survived locally
  from the original Exp 1 sync. The trajectory comparison is therefore
  short-baseline-vs-long-extension.

## Conclusions

- **Exp 3 — REJECT.** The `(h_t, f_t)` positive is degenerate: the
  forecaster has e_t in its causal context (input at step t includes the
  encoder output at step t), so f_t can copy h_t directly. The optimiser
  exploits this — the new positive term is trivially satisfied while
  the original `(h_t, f_{t-1})` term collapses. Held-out R²_random
  = 0.013 vs baseline 0.773; AUC at chance.
- **Exp 4-only — REJECT (no improvement).** On the held-out eval, the
  f-cross-(b, c) negative term is indistinguishable from the baseline
  (AUC 0.8902 vs 0.8938; Top-1 0.7389 vs 0.7470; R²_random 0.7708 vs
  0.7731 — small deltas, all unfavourable). The apparent trajectory
  advantage (~0.01 AUC) is an artefact of a 12.9k-step extension vs a
  4.3k-step baseline trajectory. **Loss shape stays at
  `cosine_similarity_batch`.**

### Note on "high-U → good"

Exp 3 has held-out U_batch ≈ 0.30 and U_temporal ≈ 0.25 — much higher
than the baseline's 0.085 / 0.039 — yet AUC is at chance. Direct
counter-evidence to "high U means useful structure": a degenerate
positive can drive U up without delivering prediction signal. U is
necessary but not sufficient.

## Deviations / caveats

- Exp 4-only spot preempted at step 12,900 / 15,000. We use the existing
  `best_loss.pth` (step 12,800) as FINAL.pth — exactly the launcher's
  end-of-run promotion. Trajectory CSV is 12,900 rows, not 15,000. We
  did not resume the last ~2.1k steps because best_loss tracks lowest
  training loss; the end-of-run cp would not have changed the FINAL.pth.
- Cost: vast spot 36374874 destroyed after 1h 49m for $0.41 total.
