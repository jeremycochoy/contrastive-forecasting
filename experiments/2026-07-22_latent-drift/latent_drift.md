# Latent-drift across #374 arm training

Six-arm sweep from
[#374](https://github.com/jeremycochoy/contrastive-forecasting/issues/374)
(PR
[#376](https://github.com/jeremycochoy/contrastive-forecasting/pull/376))
— retroactively measured. Loss recipes per arm are defined in #376 and
repeated in the plot legends. Runs
[`scripts/latent_drift.py`](../../scripts/latent_drift.py) over 68
saved backbone snapshots on a fixed ARMA probe batch (`B=64`,
`T=4096`, `C=1`; seed `20260722`). Two extra probe seeds
([`20260723`](results/drift_374_seed20260723.csv),
[`20260724`](results/drift_374_seed20260724.csv)) give a max
across-seed stdev of `0.038` for `cka` and `≤ 0.029` for the other
three metrics.

## Adjacent-pair drift

![drift_total](plots/drift_total.png)

Arm 5 stays in `0.78–0.98` across every interval; arm 3 sits at
`0.91–0.92` (2 intervals); the other four arms terminate in
`0.40–0.69`.

![rot_gap](plots/rot_gap.png)

Arm 5's terminal rotational drift is `0.46` (never falls below);
the other five arms terminate in `0.20–0.27`.

![drift_residual](plots/drift_residual.png)

Arm 5 climbs from `0.12` at step 15k to `0.44–0.48` through step 50k;
arm 4 sits at `0.30–0.47`; arm 6 v2 and bimoco stay lowest at
`0.16–0.28`.

![cka](plots/cka.png)

Arm 5's CKA drops from `0.91` at step 15k to `0.35–0.38` through
step 50k; the other five arms terminate at `0.50–0.82`.

## GM-Relative MASE reference (from #374)

![gm_mase_374](plots/gm_mase_374.png)

At step 12,500 (6L quantile head): bimoco `1.11` (lowest);
arm 3 `1.15`; arm 4 `1.14`; arm 1 `1.16`; arm 6 v2 `1.18`;
arm 5 `1.22` (highest). Arm 5 is the highest-MASE arm at every
eval step.

## Metric derivation

Two encoder-latent tensors `A, B ∈ ℝ^{N × H}` — same probe batch, two
training steps, `N` tokens, `H` features. Row-normalise; the
contrastive loss lives on `S^{H-1}`.

**Total drift** — mean cosine distance:

```
drift_cos = 1 − (1/N) Σᵢ ⟨Aᵢ, Bᵢ⟩.
```

A linear head absorbs any feature-axis rotation `R ∈ O(H)`: its first
weight matrix `W` becomes `W R`. Quotient the best such `R` out via
Procrustes:

```
R* = argmax_{R ∈ O(H)} Tr(R AᵀB).
```

The SVD `AᵀB = U Σ Vᵀ` gives `R* = V Uᵀ` and `max = Σₖ σₖ(AᵀB)`.
Hence

```
drift_cos_aligned = 1 − (1/N) Σₖ σₖ(AᵀB).
```

**Split**:

- `rot_gap = drift_cos − drift_cos_aligned` — the part `R*` removed;
  a linear head absorbs this into its first weight matrix.
- `drift_cos_aligned` — the residual after removing `R*`; a linear
  head cannot absorb it by right-multiplication.

**Cross-check** via centered linear CKA (columns of `A, B`
mean-subtracted, subscript `c`):

```
cka = ‖A_cᵀ B_c‖_F² / (‖A_cᵀ A_c‖_F · ‖B_cᵀ B_c‖_F).
```

CKA is invariant under right-multiplication by any `R ∈ O(H)` and any
global rescaling; `drift_cos` is not. `cka ≈ 1` with `drift_cos > 0`
means all raw drift lives in `rot_gap`.

## Regenerate

- Manifest of the 68 #374 checkpoints:
  [`results/manifest_374.csv`](results/manifest_374.csv).
- Drift CSVs (three seeds):
  `python3 scripts/latent_drift.py
  --manifest experiments/2026-07-22_latent-drift/results/manifest_374.csv
  --out experiments/2026-07-22_latent-drift/results/drift_374.csv
  [--seed <S>]` on a host with the #374 backbones on disk.
- GM-MASE reference CSV
  [`results/gm_mase_374.csv`](results/gm_mase_374.csv) parsed from
  `Aggregate GM-Relative MASE` in the `summary.txt` files at
  `origin/feature/contrastive-forecasting-374:experiments/2026-07-10_split_pred_rep/{results,results_arm4,results_arm5,results_arm6_v2,results_bimoco_v2}/gift_eval_full_*/summary.txt`.
- Plots:
  `python3 experiments/2026-07-22_latent-drift/plots/_make_plots.py`.
