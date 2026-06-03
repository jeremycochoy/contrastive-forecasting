# #320 — Forked arms × 6-layer forecaster

**Verdict.** Mixed, and the direction depends on which arm you treat as the
test. The fork was put in to break an **indexing-shortcut mode collapse**
during training. The arm #318 used to *measure* that denial — forked β·0.8%,
where the latent participation ratio jumped from β=3.17 to 10.19 — improves
reliably on both 6Lf q-heads, and is the only one to do so. The
arm that *won* GIFT-Eval at 1L — forked β·10% — regresses sharply on both
6Lf heads. No 6Lf cell clearly beats β (#309 baseline recipe; full-97 2-seed
range ≈ 1.33–1.46); β·0.8% on both heads lands inside that range, matching β
within seed variance. So 6Lf helps the arm that anchored the shortcut
hypothesis and hurts the arm that anchored the GIFT-Eval lift — which of
those two is "the fork doing its job" is the open question.

![Figure 1 — full-97 GM-Relative MASE per arm × q-head, 1L vs 6Lf forecaster.
Whisker = bootstrap 90 % CI on the GM over its 97 configs. β shown as 4
horizontal dashed lines = the bounds of each head's 2-seed range.](plots/gm_summary.png)

## What we asked

A contrastive backbone can satisfy its training objective with an **indexing
shortcut**: a per-step positional code, *shared across all series*, that buys
β's cross-time distinctness `cos(h_t, h_l) ≠ 1` at no forecastable-content
cost. When the encoder takes it, the latent space collapses to a low-rank
mode — in #318's PR table, β's latent participation ratio is 3.17 vs 384
available dimensions.

#318 introduced two ways to deny that shortcut. One was data-side: pair each
sample with a **forked-ARIMA continuation** — same past, different future —
so position alone can no longer encode the future, and the encoder is pushed
off the positional code onto actual content. In #318 the denial was visible
in the latent: forked β·0.8% PR=10.19 vs β=3.17, and that same configuration
was the one arm that cleared β on GIFT-Eval.

The forecaster is the small transformer between the encoder and the q-head;
its depth is the only thing this card changes (1 → 6 layers, encoder
unchanged). The question: does giving the forecaster more capacity change
*whether*, and *where*, the fork holds the encoder off the shortcut?

## What happened

![Figure 2 — Δ(6Lf − 1L) per (arm, q-head) on full-97; whisker =
paired-bootstrap 90 % CI; green = reliably better, red = reliably worse,
grey = inconclusive.](plots/forecaster_delta.png)

Across the GM, the deeper forecaster moves *which* (arm × mix × head) cell
the fork's signal shows up in, rather than removing it. Two clean
opposites split the matrix.

The arm #318 *measured* the shortcut denial on — forked β·0.8% — is the
only one that improves on both 6Lf q-heads. The arm that *won* GIFT-Eval
at 1L — forked β·10% — drops from only-cell-above-β at 1L to a reliable
regression on both 6Lf heads. The other all-time arms mostly regress;
allt·0.8%·2L (1L's best all-time cell) lands past seasonal-naive — the
1.0 floor a trivial prior beats — and is the worst single cell of the 6Lf
matrix. Of the 10 (arm, head) cells: 6 reliably worse, 3 better, 1
inconclusive.

(One caveat: this card measures GIFT-Eval GM, not the encoder participation
ratio. "Where the fork's signal shows up" is the simplest reading of which
cells move which way; a direct PR re-measurement under 6Lf would settle it.)

**Forward-looking (author's reading, not in the data).** β·0.8% on the 6L
q-head is 6Lf's closest approach to v11c, and the only reliably-better cell
on the 6L head. A longer schedule + a second seed + a finer mix fraction here
might cross v11c next.

![Figure 3 — Per-domain GM-Relative MASE on full-97, log radial. One colour
per arm, at its own best head; dashed line = 1L #318 reference, solid =
6Lf this card; green band = β 2-seed range (best head per seed); purple ring =
v11c; black dashed ring = seasonal-naive.](plots/perdomain.png)

The radar makes the per-arm shifts legible: four of the five arms move
outward (worse) on most domains under 6Lf at their best head; forked β·0.8%
is the one arm that moves inward — consistent with the split in Figures 1–2.

## Scoreboard

*Full-97 GM-Relative MASE = GM over GIFT-Eval's 97 configs of (model MASE) ÷
(seasonal-naive MASE). **Lower is better.** triage-11 is the noisy fast subset,
kept for continuity with #318. **Δ = 6Lf − 1L**; the 90 % CI on Δ is a paired
bootstrap over the 97 shared configs (config difficulty cancels). 1L columns
reused verbatim from #318. Single backbone seed per cell — the paired CI
captures config-set spread, not seed noise. **Bold** = reliable (whole CI on
one side of 0).*

| Arm | head | 1L full | **6Lf full** | **Δ full** | 90 % CI on Δ | 1L triage | 6Lf triage |
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

**References** (lower better): β · 2L = [1.3272, 1.4591] (n = 2 seeds); β · 6L
= [1.3702, 1.4489] (n = 2); v11c = 1.292; seasonal-naive = 1.0.

**Paired bootstrap.** For each (arm, head) Δ: resample 97 config indices
jointly for 1L and 6Lf, recompute the two GMs, take the difference; 2 000
iter; 5–95 percentiles = 90 % CI. Joint resample cancels per-config
difficulty. Figure 1 whiskers use the un-paired form per GM. Neither captures
seed variance.

## Protocol

Byte-identical to each arm's #318 counterpart except `--num-layers 1 → 6`
(6L encoder unchanged). 50 k backbone (batch 256, seed 20260520); fresh 30 k
2L and 6L q-head (`e_then_f` / `reconstruction forecaster` / `amp-dtype none`);
GIFT-Eval `--strategy B4`. Full recipe:
[#318 / PR #319](https://github.com/jeremycochoy/contrastive-forecasting/pull/319).

## Annex — exact negatives (per anchor, C = 1; pooled N = B·Σ)

Forks add no loss term, so each arm's negatives are its base loss's (unchanged
from #318). B = 256, T = 64 latent:

| family | repels | β loss | all-time loss |
|---|---|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 |
| `hh_all` within-series ∀ℓ | `cos(h_t, h_ℓ)`, ℓ≠t | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 |
| `xs_allt` cross-series ∀ℓ | `cos(h_{b,t}, h_{b',ℓ})` | — | (B−1)·T |
| **pooled N** | | **81 920** | **4 259 584** (52×) |
