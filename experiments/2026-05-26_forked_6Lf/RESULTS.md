# #320 — Forked arms × 6-layer forecaster

**Verdict.** A deeper forecaster doesn't rescue #318's data-side fork. No 6Lf
cell crosses β. The deepening hits hardest exactly where the 1L fork was
strongest, and only modestly helps the arms where the 1L fork was already
weakest.

![Figure 1 — full-97 GM-Relative MASE per arm × q-head, 1L vs 6Lf forecaster.
Whisker = bootstrap 90 % CI on the GM over its 97 configs. β shown as 4
horizontal dashed lines = the bounds of each head's 2-seed range.](plots/gm_summary.png)

## What we asked

#318 ran five forked arms with a **1-layer** forecaster and found exactly one
winner: the fork at ≈10 % injection on the β loss, on both q-heads. Every
other forked cell either tied β or worsened it.

Does deepening the forecaster change that map? `--num-layers 1 → 6`, the 6L
causal *encoder* untouched, every other flag fixed to its #318 value, the 5
forked arms re-trained from scratch, the same protocol re-run.

## What happened

![Figure 2 — Δ(6Lf − 1L) per (arm, q-head) on full-97; whisker = paired-bootstrap
90 % CI; green = whole CI < 0 (6Lf better), red = whole CI > 0 (worse),
grey = inconclusive.](plots/forecaster_delta.png)

The deeper forecaster *un-finds* the 1L fork's wins.

The arm that was the only 1L cell to beat β — β·10% — becomes the worst-hit
cell of the whole matrix. The same on the 2L head for allt·0.8%, which was
the strongest 1L cell among the all-time-loss arms: 6Lf there walks the GM
well past seasonal-naive. Wherever the 1L fork had focused its advantage, the
deeper forecaster scatters it.

The mirror is true at the bottom of the 1L distribution: the arms where the
1L fork already lagged β — β·0.8% on both heads, allt·50% on the 2L head —
all improve with 6Lf. But the gains stay small, and none of them reach β: the
best 6Lf cell still falls outside the wider of β's two seed bounds.

In counts: 6 cells reliably worse with 6Lf, 3 reliably better, 1 inconclusive
(paired-bootstrap 90 % CI on the per-cell Δ).

**Forward-looking (author's reading, not in the data).** The **β·0.8% on the
6L q-head** cell is the closest 6Lf has come to v11c so far, and the only cell
where 6Lf improves the 6L head reliably. It is plausible — but **not shown
here** — that this combination has further room with more iteration (longer
schedule, second seed, finer mix fraction) and could be the one that crosses
the v11c threshold next.

## Scoreboard

*Full-97 GM-Relative MASE = GM over GIFT-Eval's 97 configs of
(model MASE) ÷ (seasonal-naive MASE). **Lower is better.** triage-11 is the
noisy fast subset, kept for continuity with #318. **Δ = 6Lf − 1L**; the 90 %
CI on Δ is a paired bootstrap over the 97 shared configs (so config difficulty
cancels). The 1L columns are reused verbatim from #318. Single backbone seed
per cell — the paired CI captures config-set spread, not seed noise. **Bold**
= reliable (whole CI on one side of 0).*

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

**References** (lower better): β · 2L = [1.3272, 1.4591] (n = 2 seeds);
β · 6L = [1.3702, 1.4489] (n = 2); v11c = 1.292; seasonal-naive = 1.0.

**Paired bootstrap, in one paragraph.** Each cell has 97 per-config
Relative-MASE values. To estimate the CI on Δ = GM(6Lf) − GM(1L) for one arm
× head, we resample the **97 config indices jointly** (same indices for both
arms), recompute the two GMs, take the difference, and repeat 2 000 times. The
5th and 95th percentiles of that distribution are the 90 % CI. Pairing the
indices is what makes config difficulty cancel and isolates the
forecaster-depth effect. The whiskers on Figure 1 use the un-paired form per
GM (each cell's 97 values alone); neither flavour captures seed variance, and
only β has more than one seed here.

## Protocol

Each arm is byte-identical to its #318 counterpart **except**
`--num-layers 1 → 6` (the 6L causal *encoder*
`--num-encoder-layers 6` is unchanged). Backbone: 50 k steps, batch 256, seed
20260520; the β-loss arms use `cosine_similarity_batch_full_hh_negs`, the
all-time-loss arms `…_xshh_allt`. Forks are in the **data**
(`--synth-kind forked-arma --mix-ratio MIX`) at 0.8 % / 10 % / 50 %, matching
#318. Eval: a fresh 2L and 6L quantile q-head (30 k, transformer, causal,
`--head-train-input e_then_f`, `--reconstruction forecaster`,
`--amp-dtype none`) + GIFT-Eval `--strategy B4` on full-97 and triage-11. The
full recipe lives in [#318](https://github.com/jeremycochoy/contrastive-forecasting/pull/319).

## Annex — exact negatives (per anchor, C=1; pooled N = B·Σ)

Forks add no loss term, so each arm's negatives are its base loss's
(unchanged from #318). B=256, T=64 latent:

| family | repels | β loss | all-time loss |
|---|---|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 |
| `hh_all` within-series ∀ℓ | `cos(h_t, h_ℓ)`, ℓ≠t | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 |
| `xs_allt` cross-series ∀ℓ | `cos(h_{b,t}, h_{b',ℓ})` | — | (B−1)·T |
| **pooled N** | | **81 920** | **4 259 584** (52×) |
