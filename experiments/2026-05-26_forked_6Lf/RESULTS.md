# #320 — Forked arms × 6-layer forecaster

**Verdict.** No forked-6Lf config beats β. The best 6Lf cell (β·0.8% on the 6L
q-head) lands at **1.4006** — above β's 2-seed range **[1.3272, 1.4591]** and well
above v11c (1.292). 6 of 10 cells are reliably worse with 6Lf, 3 reliably better,
1 inconclusive (paired bootstrap, 90% CI over the 97 shared configs).

![Figure 1 — full-97 GM-Relative MASE per arm × q-head, 1L vs 6Lf (β shown as
2-seed range)](plots/gm_summary.png)

## Question

#318 ran the same 5 forked arms with a **1-layer** forecaster; only one cell
(β·10%) beat β. This card asks whether a **6-layer** forecaster
(`--num-layers` 1 → 6; encoder unchanged) moves any arm past β.

## Result

![Figure 2 — Δ(6Lf − 1L) per (arm, head), paired-bootstrap 90 % CI; green = 6Lf
better than 1L, red = worse](plots/forecaster_delta.png)

- **No 6Lf config beats β.** Best 6Lf cell: β·0.8% 6L = 1.4006; β seed-1 = 1.3272.
- **6Lf reliably *hurts* 6 / 10 cells**, including both β·10% cells (+0.263 / +0.394
  on 2L / 6L) and both allt·0.8% cells (+0.813 / +0.338) — the arms where the 1L
  fork already scored best (β·10% overall; allt·0.8% 2L within the all-time arms).
- **6Lf reliably *helps* 3 / 10 cells**: β·0.8% on both heads (−0.093 / −0.041)
  and allt·50% on the 2L head (−0.176). None of these gains cross β.
- The remaining cell (allt·10% 2L) is inconclusive (90 % CI straddles 0).
- Triage-11 and full-97 sometimes disagree on individual cells; only full-97
  numbers are cited above.

## Scoreboard

*Full-97 GM-Relative MASE — geometric mean over GIFT-Eval's 97 configs of
(model MASE) ÷ (seasonal-naive MASE). **Lower is better.** triage-11 is the noisy
fast subset, kept for continuity with #318. **Δ = 6Lf − 1L** on the same 97
configs; the 90 % CI on Δ is a paired bootstrap over those 97 configs (so config
difficulty cancels). **1L** columns reused verbatim from #318. Single backbone
seed (20260520) per cell — paired-bootstrap CIs capture config spread, **not seed
noise**. **Bold** = reliable (whole CI on one side of 0).*

| Arm | head | 1L full | **6Lf full** | **Δ full** | 90% CI on Δ | 1L triage | 6Lf triage |
|---|:--:|---:|---:|---:|---|---:|---:|
| β·10%    | 2L | 1.3030 | 1.5662 | **+0.263** | (+0.210, +0.318) | 1.4559 | 1.6581 |
| β·10%    | 6L | 1.2889 | 1.6832 | **+0.394** | (+0.320, +0.472) | 1.4747 | 1.8429 |
| β·0.8%   | 2L | 1.5302 | 1.4369 | **−0.093** | (−0.146, −0.041) | 1.4376 | 1.6348 |
| β·0.8%   | 6L | 1.4412 | 1.4006 | **−0.041** | (−0.080, −0.0005)| 1.4027 | 1.5752 |
| allt·50% | 2L | 1.6366 | 1.4608 | **−0.176** | (−0.231, −0.124) | 1.8824 | 1.7214 |
| allt·50% | 6L | 1.4065 | 1.5387 | **+0.132** | (+0.085, +0.186) | 1.5339 | 1.7264 |
| allt·10% | 2L | 1.6130 | 1.5973 | −0.016 | (−0.083, +0.050) | 2.0115 | 1.6333 |
| allt·10% | 6L | 1.5304 | 1.6939 | **+0.163** | (+0.098, +0.228) | 1.9293 | 1.9596 |
| allt·0.8% | 2L | 1.4049 | 2.2180 | **+0.813** | (+0.697, +0.942) | 1.6083 | 2.2589 |
| allt·0.8% | 6L | 1.5100 | 1.8483 | **+0.338** | (+0.284, +0.398) | 1.6348 | 2.0528 |

**References** (lower better): β = **[1.3272, 1.4591]** (2L, n=2 seeds) /
**[1.3702, 1.4489]** (6L, n=2); v11c = 1.292; seasonal-naive = 1.0.

## Protocol

Each arm is byte-identical to its #318 counterpart **except** the forecaster
depth (`--num-layers 1 → 6`; the 6L causal *encoder*
`--num-encoder-layers 6` is unchanged). Backbone: 50 k steps, batch 256,
seed 20260520, β-loss arms use `cosine_similarity_batch_full_hh_negs`, all-time-loss
arms use `…_xshh_allt`. Forks are in the **data**
(`--synth-kind forked-arma --mix-ratio MIX`); injection fractions 0.8 % / 10 % /
50 % match #318. Eval: fresh 2L and 6L quantile q-head (30 k, transformer,
causal, `--head-train-input e_then_f`, `--reconstruction forecaster`,
`--amp-dtype none`) + GIFT-Eval `--strategy B4` on full-97 and triage-11. Full
recipe (every flag): see [#318](https://github.com/jeremycochoy/contrastive-forecasting/pull/319).

## Annex — exact negatives (per anchor, C=1; pooled N = B·Σ)

Forks add no loss term, so each arm's negatives are its base loss's (unchanged
from #318). B=256, T=64 latent:

| family | repels | β loss | all-time loss |
|---|---|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 |
| `hh_all` within-series ∀ℓ | `cos(h_t, h_ℓ)`, ℓ≠t | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 |
| `xs_allt` cross-series ∀ℓ | `cos(h_{b,t}, h_{b',ℓ})` | — | (B−1)·T |
| **pooled N** | | **81 920** | **4 259 584** (52×) |
