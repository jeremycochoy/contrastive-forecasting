# U-metric audit: per-slice vs global vs inverse-of-mean

## Reference-implementation agreement

A slow nested-loop reference (`dim_usage_ref` in `scripts/u_audit.py`) was
checked against `src.metrics.dim_usage` on five small inputs (random `(5, 7,
3, 11)` tensor at axes 0/1/2, a collinear `(6, 8)` set, and an orthonormal
`8×8` identity).

- **Max abs diff: 5.96e-08.** Collinear case returns `1/d = 0.125` (matches
  docstring "collinear → 1/d"); orthonormal returns 1.0. The torch
  implementation is faithful to its docstring.

## Three variants × three Ws (means over 3 seeds, n=B=32, d=H=384)

`u_b_*` uses `axis=0` (n=32 batch); `u_t_*` uses `axis=1` (n=256 time).

| setup                | W   | u_b ps | u_b gp | u_b iom | u_t ps | u_t gp | u_t iom |
|----------------------|-----|--------|--------|---------|--------|--------|---------|
| isotropic_synthetic  |  -  | 0.9775 | 1.0000 | 1.0000  | 0.9968 | 1.0000 | 0.9997  |
| encoder_init         |  16 | 0.0427 | 0.0426 | 0.0425  | 0.0426 | 0.0426 | 0.0426  |
| encoder_init         |  64 | 0.1449 | 0.1447 | 0.1443  | 0.1452 | 0.1447 | 0.1449  |
| encoder_init         | 192 | 0.3316 | 0.3309 | 0.3302  | 0.3336 | 0.3309 | 0.3311  |

ps = per_slice (current), gp = global_pooled, iom = inverse_of_mean.

## What the data shows

On `torch.randn(32, 256, 1, 384)` (isotropic), `u_b_per_slice` reads
0.9775 while `u_b_global_pooled` reads 1.0000 and `u_b_inverse_of_mean`
reads 1.0000 — a 0.022 gap between per-slice and the other two on this
input. Under `u_t` (n=256) the same dataset gives 0.9968 per-slice vs
1.0000 global vs 0.9997 inverse-of-mean — gap 0.003.

On encoder-init outputs, the three variants are within ~0.003 of each
other at every W tested. At W=192 the `u_b` triplet is (ps 0.3316, gp
0.3309, iom 0.3302); at W=64 (0.1449, 0.1447, 0.1443); at W=16 (0.0427,
0.0426, 0.0425). The `u_t` triplet at W=192 is (0.3336, 0.3309, 0.3311).
The W-sweep numbers from the launch (W=16: 0.0427, W=64: 0.1449, W=192:
0.3316) are reproduced by `u_b_per_slice` here.

## Bug or design choice?

**No bug** in the sense of "returns a value inconsistent with its own
docstring": the docstring says U is computed per slice and slices are
averaged into a scalar, and that is what the code does. The
reference-loop agreement (max diff 5.96e-08) confirms the math is
implemented correctly.

The metric's averaging order (per-slice clamp, then mean) is a design
choice. On the isotropic input, the per-slice variant differs from
global/iom by 0.022 at `u_b` (n=32) and 0.003 at `u_t` (n=256). On the
encoder-init data tested here, the three variants differ by ≤0.003 at
all three Ws, and the trend `U_b` grows monotonically with W
(0.0426/0.0427 → 0.1443/0.1449 → 0.3302/0.3316 for iom/per-slice)
holds for all three variants.
