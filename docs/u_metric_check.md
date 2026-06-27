# `U` (dimension-usage) formula — math check + verdict

> Issue [#363](https://github.com/jeremycochoy/contrastive-forecasting/issues/363).
> Empirical companion: `experiments/2026-06-24_sigreg_lambda_sweep/scripts/verify_u_formula.py`.
> Test pins: `tests/test_metrics.py::test_dim_usage_*`.

## 1. Formula (verbatim from `src/metrics.py`)

```python
# dim_usage(z, axis)  with feature dim d = z.shape[-1]
z_norm   = F.normalize(z, p=2, dim=-1, eps=1e-12)
sim      = z_norm @ z_norm.transpose(-1, -2)              # cos(ẑ_i, ẑ_j)
sq       = sim ** 2
off_mean = (sq.sum() - diag(sq).sum()) / (n * (n - 1))    # mean_{i≠j} cos²
U        = clamp_max(1 / (d * off_mean), 1.0)
```

So, in math:

$$U(z) \;=\; \min\!\Big(1,\; \frac{1}{d \cdot \overline{\cos^2}}\Big),
\qquad \overline{\cos^2} \;=\; \frac{1}{n(n-1)} \sum_{i\neq j} \frac{(z_i \cdot z_j)^2}{\|z_i\|^2 \|z_j\|^2}.$$

The `clamp_max(1.0)` only ever fires when the finite-sample $\overline{\cos^2}$ dips
below $1/d$ — i.e. the "isotropic" regime; it never lifts a low value.

## 2. Analytic limits

**Isotropic on the sphere.**
If $\hat z_i$ is uniform on $S^{d-1}$ (e.g. unit-normalised $z \sim \mathcal{N}(0, I_d/d)$),
then for $i \neq j$ independent, $\mathbb{E}[(\hat z_i \cdot \hat z_j)^2] = 1/d$
(symmetry: pick $\hat z_i$, the dot product is $\hat z_{j,1}$, and $\mathbb{E}[\hat z_{j,1}^2] = 1/d$).
So $\overline{\cos^2} \to 1/d$ and

$$U \;\to\; \frac{1}{d \cdot (1/d)} \;=\; 1.$$

**Rank-1 collapse.**
If $z_i = \alpha_i v$ for a fixed unit $v$, then $\hat z_i = \mathrm{sgn}(\alpha_i)\,v$,
so $(\hat z_i \cdot \hat z_j)^2 \equiv 1$, $\overline{\cos^2} = 1$, and

$$U \;=\; \frac{1}{d \cdot 1} \;=\; \frac{1}{d}.$$

The two limits are **isotropic → 1**, **rank-1 → 1/d** — *not* the inverse
suggested in the report Vocabulary entry (see §5).

## 3. Is U a per-dimension usage estimator?

### 3a. `1/U` is **not** the effective number of dimensions.

Build a $(N, d)$ tensor whose rows are isotropic in an exact $r$-dim
subspace of $\mathbb{R}^d$. After unit-normalising, each row is uniform
on $S^{r-1}$ embedded into $r$ of the $d$ axes, so
$\mathbb{E}[(\hat z_i \cdot \hat z_j)^2] = 1/r$ and

$$U \;=\; \frac{1}{d \cdot (1/r)} \;=\; \frac{r}{d},
\qquad \text{hence}\qquad \frac{1}{U} \;=\; \frac{d}{r}.$$

So $1/U$ over-counts the effective rank by a factor of $d/r^2$.
Empirically (`verify_u_formula.py`, $d=384$, $N=4096$):

|   r | U       | d·U    | 1/U     | r itself | d/r    |
|----:|--------:|-------:|--------:|---------:|-------:|
|   1 | 0.00260 |   1.00 |  384.00 |        1 | 384.00 |
|   4 | 0.01041 |   4.00 |   96.07 |        4 |  96.00 |
|  16 | 0.04167 |  16.00 |   24.00 |       16 |  24.00 |
|  64 | 0.16670 |  64.02 |    6.00 |       64 |   6.00 |
| 192 | 0.49990 | 191.96 |    2.00 |      192 |   2.00 |
| 384 | 0.99966 | 383.87 |    1.00 |      384 |   1.00 |

**`d · U` tracks `r` exactly. `1/U` tracks `d/r`, not `r`.**

### 3b. `d · U` is the participation ratio of the unit-normalised second-moment.

Let $G = \tfrac{1}{N}\sum_i \hat z_i \hat z_i^\top$ ($d \times d$, $\mathrm{Tr}\,G = 1$).
Then

$$\sum_{i,j}(\hat z_i \cdot \hat z_j)^2 \;=\; \|ZZ^\top\|_F^2
   \;=\; \mathrm{Tr}\big((ZZ^\top)^2\big)
   \;=\; N^2 \cdot \mathrm{Tr}(G^2),$$

so $\overline{\cos^2} = \tfrac{N \cdot \mathrm{Tr}(G^2) - 1}{N - 1}
\xrightarrow{N \to \infty} \mathrm{Tr}(G^2)$.
Since $\mathrm{Tr}\,G = 1$, the participation ratio
$\mathrm{PR}(G) = (\mathrm{Tr}\,G)^2 / \mathrm{Tr}(G^2) = 1 / \mathrm{Tr}(G^2)$.
Substituting:

$$U \;\xrightarrow{N \to \infty}\; \frac{1}{d \cdot \mathrm{Tr}(G^2)}
   \;=\; \frac{\mathrm{PR}(G)}{d},
\qquad\text{i.e.}\qquad \boxed{\,d \cdot U \;\approx\; \mathrm{PR}(G)\,}.$$

Empirical check (`verify_u_formula.py`, $d=384$, $N=4096$, several prescribed spectra):

| spectrum             |       U |    d·U |     PR | match  |
|:---------------------|--------:|-------:|-------:|:-------|
| decay 1/k            |  0.0837 |  32.13 |  31.89 | ✓ |
| decay 1/k²           |  0.0108 |   4.13 |   4.13 | ✓ |
| step-32 (top heavy)  |  0.0853 |  32.77 |  32.52 | ✓ |
| step-64 (mid)        |  0.1843 |  70.77 |  69.58 | ✓ |
| uniform (full rank)  |  0.9997 | 383.87 | 351.05 | (saturating; PR(Ĝ) finite-N-biased) |

The uniform full-rank entry is a measurement artefact, not a U bias:
the empirical $\mathrm{Tr}(\hat G^2)$ under-estimates the population value at
$N/d \approx 10$, so the empirical $\mathrm{PR}$ runs low. The U value
itself is correct ($\to 1$). Tightens cleanly at larger $N$.

## 4. Verdict + interpretation

**U passes (3).** It is a valid per-dimension usage estimator with the
following precise interpretation:

> $U$ is the **participation-ratio fraction** of the L2-normalised samples:
> $U \approx \mathrm{PR}(G)\,/\,d$, where $G = \tfrac{1}{N}\sum_i \hat z_i \hat z_i^\top$.
> Equivalently, the **effective number of dimensions in use is $K \cdot U$**, not $1/U$.

So for $K=384$:
- $U = 1.00$ ⇔ all 384 dims equally used (isotropic).
- $U = 0.04$ ⇔ ≈ 15 effective dimensions in use.
- $U = 1/K \approx 0.0026$ ⇔ collapsed to a single direction.

## 5. The report vocabulary (now correct)

`reports/2026-06-24_sigreg_lambda_sweep/sigreg_lambda_sweep.md` Vocabulary
entry now reads:

> $U \in [1/K, 1]$ with $K{=}384$. $U{=}1$ means all $K$ dimensions
> equally used (isotropic); $U{=}1/K$ means collapsed onto a single
> direction. **Higher = more dimensions in use.** Effective number of
> dimensions in use $\approx K \cdot U$.

This matches `dim_usage` in `src/metrics.py`, the docstring there, and
the existing unit tests `test_dim_usage_orthonormal_is_one` /
`test_dim_usage_collinear_is_one_over_d`.

Historical record: an earlier revision of the report glossed $U$ in the
opposite direction ("`1/K` = maximally spread; near 1 = collapse;
**Lower = more dimensions in use**"). That gloss was replaced in the
report iteration that inverted the U direction in the Vocabulary entry
and split the `dim_usage` panel per latent.

Under the correct direction, the §Annex D numbers read as:
embedding-side tail-50 $u\_batch\_e \le 0.06$ at high $\lambda_e$ means
the embedding has been concentrated to $\approx K \cdot 0.06 \approx 23$
effective dimensions out of $K{=}384$, i.e. SIGReg is concentrating the
embedding side at this λ range, not spreading it.

## 6. Code fix?

None to `src/metrics.py`. The formula, the docstring, and the existing
unit tests are mutually consistent and match the analytic limits.
The fix is to the report vocabulary text (see §5).
