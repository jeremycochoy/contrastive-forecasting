# Init U_b vs batch size B — sweep

Re-measure dim_usage U at init for `o_lat` at training B=256, comparing
per-slice (axis=0) vs global-pooled (over B·T·C samples), with an
isotropic-synthetic control. Default init, W=16, freq=3, seas=3,
T_raw=4096 → T=256, H=384, CPU.

## Mean ± std (sorted by setup, then B)

| setup               |   B | n |   u_b_per_slice | u_b_global_pooled |   u_t_per_slice |
|:--------------------|----:|--:|----------------:|------------------:|----------------:|
| encoder_init        |  32 | 3 | 0.0427 ± 0.0002 |   0.0426 ± 0.0001 | 0.0426 ± 0.0001 |
| encoder_init        |  64 | 3 | 0.0426 ± 0.0001 |   0.0426 ± 0.0001 | 0.0426 ± 0.0001 |
| encoder_init        | 128 | 3 | 0.0425 ± 0.0001 |   0.0426 ± 0.0001 | 0.0426 ± 0.0000 |
| encoder_init        | 256 | 1 | 0.0425 ± 0.0000 |   0.0425 ± 0.0000 | 0.0425 ± 0.0000 |
| isotropic_synthetic |  32 | 3 | 0.9749 ± 0.0005 |   1.0000 ± 0.0000 | 0.9963 ± 0.0004 |
| isotropic_synthetic |  64 | 3 | 0.9874 ± 0.0005 |   0.9999 ± 0.0001 | 0.9966 ± 0.0007 |
| isotropic_synthetic | 128 | 3 | 0.9941 ± 0.0006 |   0.9999 ± 0.0000 | 0.9967 ± 0.0004 |
| isotropic_synthetic | 256 | 3 | 0.9975 ± 0.0001 |   1.0000 ± 0.0000 | 0.9968 ± 0.0005 |

(n=1 for encoder_init B=256: budgeted single seed for the slowest cell.)

## What the data shows

For `encoder_init`, the per-cell means of all three U variants sit in
0.0425–0.0427 across every B; per-slice and global-pooled means agree
within ≤ 0.0001 at every B, including B=32; per-cell seed std ≤ 2e-4.
The per-cell mean does not change with B over this range, and the
per-slice/global-pooled gap is already negligible at B=32.

For `isotropic_synthetic`, per-slice batch U rises from 0.9749 (B=32)
to 0.9874, 0.9941, 0.9975 (B=256), while global-pooled stays at
0.9999–1.0000. The per-slice/global-pooled gap shrinks from ≈0.025
(B=32) to ≈0.003 (B=256). The temporal control stays at 0.9963–0.9968
across all B.

## Hypothesis check

"Need B≈256 to measure U meaningfully" is consistent with the
isotropic-synthetic column but not with the encoder_init column: for
`o_lat` at default init, the B=32 mean (0.0427) and B=256 single-seed
value (0.0425) differ by ≈1× the B=32 seed σ (0.0002), so on this
rank-collapsed signal the small-B bias is not detectable above seed
noise (caveat: B=256 only has 1 seed).
