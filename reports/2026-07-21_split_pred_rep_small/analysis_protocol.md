# GM-MASE deep-extraction protocol

Reusable procedure for squeezing maximum signal out of a sparse GM-MASE eval set
(N cells observed, many arm × variant cells missing). Assumes the split-pred/rep
factorial design: each cell is characterised by an arm (loss recipe) and a
treatment vector over {tr1, nse, ncpc} where combab is decomposed per-arm.

## When to run

- After a batch of new GIFT-Eval cells lands under `results/eval_gm_mase/`.
- Before deciding which additional cells to prioritise for evaluation.
- Whenever a reviewer asks "which modification reliably improves MASE?"

## Inputs

- `results/eval_gm_mase/<arm>_<variant>_bb<N>k_hd<K>s_summary.txt` — observed
  GM-MASE per cell. Parse the "Aggregate GM-Relative MASE (97 configs)" line.
- `results/wave_d_metrics.csv` — arm, variant, ff40 (1−ff at target step),
  mv_h_mean / mv_e_mean (training-side drift proxies).
- Actual step-K drift on the shared probe batch: compute drift on the
  (K−10k, K) adjacent-checkpoint pair, cache alongside the CSV. Use
  `_latent_movement_batch.pt` as the fixed input; `mean_one_minus_cos` from
  `src.eval_latent_movement`.

## Treatment decomposition

Standard mapping (base = (0,0,0)):

| variant | tr1 | nse | ncpc |
|---------|-----|-----|------|
| base    | 0   | 0   | 0    |
| tr1     | 1   | 0   | 0    |
| nse     | 0   | 1   | 0    |
| ncpc    | 0   | 0   | 1    |
| combab  | 1   | (arm ∈ {arm1, arm3, arm4} ? 1 : 0) | 1 |

## Steps

### 1. Rank the observed cells and per-arm winners

Sort observed cells by GM-MASE. For each arm with ≥2 evaluated variants report
its within-arm winner and loser. This gives the honest raw picture before any
model.

### 2. Enumerate clean matched pairs

Same-arm cell pairs (with, without) that differ in exactly one treatment
(Hamming distance 1 over the treatment vector). At sparse eval this may reduce
to a handful of pairs (e.g. only tr1 ↔ combab for arm 5/6_v2/bimoco isolates
"add ncpc while holding τ=1"). Report per-pair Δ and count-positive; Wilcoxon
only when n ≥ 3.

### 3. Univariate correlation with training-side proxies

For every proxy in {ff40, mv_h_mean, mv_e_mean, drift_h@K, drift_e@K}, compute
Pearson and Spearman vs GM-MASE across the observed cells. Flag any proxy with
|r| ≥ 0.5 and p < 0.05 as a candidate predictor. Note whether Pearson-and-
Spearman agree (linear-and-monotone) or whether the correlation is driven by
extremes.

### 4. Within-arm residualised correlation

Demean GM-MASE and each proxy by arm, then repeat the Pearson / Spearman.
Distinguishes true within-arm predictive power from arm-baseline confounding.

### 5. Per-arm rank concordance

For every within-arm pair of variants, does the proxy rank them the same way
as GM-MASE? Report `concordant / total` per proxy. This is the strictest test
of proxy usefulness — insensitive to magnitudes.

### 6. OLS with arm fixed effects + treatment dummies

Fit `gm_rel_mase ~ C(arm) + tr1 + nse + ncpc` on all observed cells. Report
each treatment's coefficient, 95 % CI, p. All coefficients being same-signed
but individually non-significant is informative — treat as a directional
consensus, not per-modification evidence.

### 7. Extrapolate to unobserved cells

Use the OLS to predict GM-MASE for all cells whose arm has at least one eval.
Report predicted values with prediction SE (= √(var_pred_mean + scale)).
Cells whose arm has zero eval anchors are unpredictable — list them and stop
there. Include predictions in the full 30-cell ranking table, tagged
OBS / PRED / MISS.

### 8. Actionable next-cell suggestion

Identify the cell(s) whose observation would most collapse OLS residual
variance and separate treatment marginals. Usually this means one base cell
(to anchor "no-mod") plus the combab cells for any arms not yet tested.

## Interpretation guardrails

- **Small-sample OLS at N = 11 has df_resid = 3.** Coefficients are directional
  hints, not tests. Never quote a p-value from such a fit without n and df.
- **Pearson r without Spearman ρ agreement means an outlier is driving.** Do
  not claim a "monotone relationship" from Pearson alone.
- **Additive OLS ignores interaction.** If the residuals for combab are large
  and same-signed by arm, the design has recipe-specific interaction terms the
  model cannot fit at this N.
- **Do not describe unpredicted cells (no arm anchor) as "worse than X".**
  They are simply unknown.

## Ready-to-run scripts

- `/tmp/claude-1000/-home-jupyter-rnd/726c8ee6-ed50-46f8-9a6c-d8a239ed8af9/scratchpad/orch-379/drift_at_40k.py` — recompute drift at exactly step K from adjacent checkpoints.
- `/tmp/claude-1000/-home-jupyter-rnd/726c8ee6-ed50-46f8-9a6c-d8a239ed8af9/scratchpad/orch-379/mase_stats.py` — steps 1, 2 (ranking + clean matched pairs).
- `/tmp/claude-1000/-home-jupyter-rnd/726c8ee6-ed50-46f8-9a6c-d8a239ed8af9/scratchpad/orch-379/mase_deep.py` — steps 3–7 (correlations + OLS + prediction).

Both mase scripts read the eval `_summary.txt` files and `wave_d_metrics.csv`
directly; drop new cells in and rerun.
