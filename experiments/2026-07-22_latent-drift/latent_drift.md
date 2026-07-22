# Latent-drift across #374 arm training

Six-arm sweep from
[#374](https://github.com/jeremycochoy/contrastive-forecasting/issues/374)
(PR
[#376](https://github.com/jeremycochoy/contrastive-forecasting/pull/376))
— retroactively measured. Runs the offline probe in
[`scripts/latent_drift.py`](../../scripts/latent_drift.py) over the
saved backbone snapshots.

The arms differ only in the contrastive loss recipe applied to the
same backbone. `h_t` is the encoder output (patch-embedding + causal
encoder-layer stack) — the tensor a downstream forecasting head reads.
`L_pred`, `L_rep`, `L_align`, MoCo, "arm 6 v2", "bimoco", and arm 2's
absence (arm 2 was cut in #374) are defined in PR
[#376](https://github.com/jeremycochoy/contrastive-forecasting/pull/376);
the six-arm mapping is repeated in the panel titles below.

## Question

How much does the encoder latent `h_t` move between consecutive
checkpoints during #374 training, and how much of that movement is a
pure feature-axis rotation vs a real change in the token geometry a
downstream head sees?

## Method

Given the encoder output `h_t^A`, `h_t^B` of two checkpoints of the
same arm, evaluated on ONE fixed ARMA probe batch (`B=64`, `T=4096`,
`C=1`, seed `20260722`), rows unit-normalised (the loss lives on the
unit sphere), we compute:

| Symbol | Definition | Reads |
|---|---|---|
| `drift_cos` | `1 − mean_i cos(A_i, B_i)` | raw per-token movement |
| `drift_cos_aligned` | `1 − (1/N) Σ_k σ_k(AᵀB)` | movement AFTER Procrustes rotation on the feature axis — the part a linear head cannot absorb |
| `rot_gap` | `drift_cos − drift_cos_aligned` | pure feature-axis rotation; movement a linear head absorbs |
| `cka` | linear centered CKA | cross-check on token-Gram similarity, rotation- and global-scale-invariant |

Split logic. If `A = B Q` for some orthogonal `Q ∈ ℝ^(H×H)` on the
feature axis, then row cosines flip but the Procrustes optimum removes
`Q` exactly and `drift_cos_aligned = 0`; the whole raw movement lands
in `rot_gap`. A rising `drift_cos_aligned` (or falling `cka`) is the
signal we cannot explain away as reparameterization — the geometry the
head sees is being rewritten.

Manifest: 68 snapshots, 6 arms — arm 1 (10 pts, 2k→25k), arm 3 (3 pts,
2k, 12.5k, 25k), arm 4/5/6v2 (15 pts each, 2k→50k), bimoco (10 pts,
2k→25k). See [`results/manifest_374.csv`](results/manifest_374.csv);
raw metric [`results/drift_374.csv`](results/drift_374.csv). Arm 3's
step-2000 snapshot is a mid-experiment retrain; its step-25000 uses a
fresh-optimizer extension — the 12.5k → 25k interval crosses that
seam. Regenerate:
`python3 scripts/latent_drift.py --manifest results/manifest_374.csv
--out results/drift_374.csv` on a host with the backbones on disk;
`python3 plots/_make_plots.py` for the PNGs.

Probe-noise band. The metric is a comparison, so both operands must
share ONE probe batch — but the CHOICE of batch is a nuisance. Two
extra runs at seeds `20260723` and `20260724` on the identical
manifest sit in
[`results/drift_374_seed20260723.csv`](results/drift_374_seed20260723.csv)
and
[`results/drift_374_seed20260724.csv`](results/drift_374_seed20260724.csv);
across-seed standard deviation is `≤ 0.029` for `drift_cos`, `≤ 0.015`
for `drift_cos_aligned`, `≤ 0.023` for `rot_gap`, `≤ 0.038` for `cka`
at the worst interval, means an order of magnitude smaller. Every
inter-arm and regime difference discussed below is `> 5×` the worst
probe-noise sigma.

## Results

Adjacent-pair curves for all four metrics. X axis: step at the END of
the interval (arm 5's step 15k point is the `12500→15000` interval).
Y axis linear, arm-panel scales shared across arms within a metric.

### Total per-token drift

![drift_total](plots/drift_total.png)

Every arm's `h_t` keeps moving substantially across the whole training
window. Arm 6 v2 (`0.35–0.63`) and bimoco (`0.39–0.61`) are the two
lowest-drift arms; arm 3 is flat at `0.91–0.92`; arm 5 stays high
(`0.78–0.98`) — successive checkpoint pairs are close to orthogonal on
the unit sphere.

### Uninformative rotation

![rot_gap](plots/rot_gap.png)

`rot_gap > 0` at the LAST recorded interval of every arm:
arm 1 `0.20` at 25k, arm 3 `0.27` at 25k, arm 4 `0.25` at 50k, arm 5
`0.46` at 50k, arm 6 v2 `0.21` at 50k, bimoco `0.24` at 25k. A steady
fraction of the per-checkpoint movement remains pure feature-axis
rotation deep into training in every arm — never decays to zero.
Arm 5 is a category apart, sustaining `≥ 0.46` through 50k.

### Informative drift

![drift_informative](plots/drift_informative.png)

Same-token movement remaining after Procrustes rotation. Two regimes
in the late-training (`step_b ≥ 20k`) window (adjacent-interval means):
arm 6 v2 (`0.23`) and bimoco (`0.18`) sit low — the geometry a head
would see is nearly frozen; arms 1 (`0.36`), 4 (`0.38`), and 5
(`0.39`) sit roughly twice as high. Arm 5 reverses its own
trajectory: informative drift is `0.12` at step 15k, then climbs to
`0.38` at step 25k and stays in `0.43–0.48` through step 50k — the
arm reorganizes MORE late in training than early.

### Linear CKA

![cka](plots/cka.png)

Cross-check on token-Gram similarity. Adjacent-pair CKA ranges: arm 1
`0.50–0.67`, arm 3 `0.26–0.57`, arm 4 `0.46–0.73`, arm 5 `0.35–0.91`,
arm 6 v2 `0.63–0.79`, bimoco `0.66–0.82`. Arm 5 collapses from
`0.91` at step 15k to `0.36` from step 30k on — matches the
`drift_cos_aligned` climb; the token geometry is being rewritten, not
rotated.

## What we learned

- **Uninformative rotation never stops in any #374 arm.** `rot_gap` is
  bounded away from zero at every arm's terminal interval. The
  encoder frame keeps rotating deep into training even when a linear
  head would see a nearly-stable geometry (arm 6 v2, bimoco — the two
  arms with the lowest late-training informative drift).
- **Arm 5 has a two-regime trajectory** the training logs don't
  surface. At step 15k, `drift_cos_aligned = 0.12` and `cka = 0.91`
  — most of the movement between consecutive checkpoints is
  reparameterization. Through step 50k, `drift_cos_aligned` climbs to
  `0.43–0.48` and CKA drops to `0.35–0.38` — the token geometry is
  being rewritten. `L_align + L_rep` continues reshaping the encoder
  long after #374's `12.5k`-step budget ended.
- **Bimoco and arm 6 v2 are the geometrically-quietest arms.** Late-
  training mean informative drift `0.18` (bimoco) and `0.23` (arm 6 v2)
  — roughly half of the other L_rep-bearing arms (arm 1 `0.36`, arm 4
  `0.38`, arm 5 `0.39`). The two arms share `L_rep(MoCo)`; consistent
  with #374's report that MoCo-on-the-representation term compresses
  `h_t`. The head-visible axis stabilizes even while the frame keeps
  spinning.

For every future run, the same probe is written by the training loop
into `<run>_latent_drift.csv` automatically (probe cadence defaults to
`--save-every`), so this curve becomes a free diagnostic on every arm
we launch.
