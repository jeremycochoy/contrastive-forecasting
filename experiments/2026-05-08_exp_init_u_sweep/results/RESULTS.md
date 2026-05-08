# Results — init_u_sweep

Encoder-latent dimension usage at step 0 (no training), `B=32, T_raw=4096, C=1`,
mean ± std over 3 model seeds (42, 43, 44). Sorted by `u_b_o_mean` desc.

| scheme              |              u_b_o (o_lat batch) |              u_t_o (o_lat time) |              u_b_f (f_lat batch) | u_t_f (f_lat time) |
| :------------------ | -------------------------------: | ------------------------------: | -------------------------------: | -----------------: |
| **ortho_patch**     |              **0.0435 ± 0.0002** |             **0.0434 ± 0.0001** |              **0.0101 ± 0.0009** |    0.0032 ± 0.0000 |
| ortho_all           |                  0.0435 ± 0.0002 |                 0.0434 ± 0.0001 |                  0.0103 ± 0.0006 |    0.0031 ± 0.0000 |
| proj_smallgain      |                  0.0428 ± 0.0003 |                 0.0428 ± 0.0001 |                  0.0098 ± 0.0011 |    0.0032 ± 0.0000 |
| default (PyTorch)   |                  0.0427 ± 0.0002 |                 0.0426 ± 0.0001 |                  0.0097 ± 0.0014 |    0.0032 ± 0.0000 |
| gpt_scaled          |                  0.0168 ± 0.0008 |                 0.0168 ± 0.0008 |                  0.0042 ± 0.0003 |    0.0030 ± 0.0000 |
| xavier_small_patch  |                  0.0107 ± 0.0001 |                 0.0107 ± 0.0001 |                  0.0036 ± 0.0002 |    0.0030 ± 0.0000 |
| skip_smallgain      |                  0.0029 ± 0.0000 |                 0.0029 ± 0.0000 |                  0.0027 ± 0.0000 |    0.0029 ± 0.0000 |
| skip_zero           |                  0.0027 ± 0.0000 |                 0.0027 ± 0.0000 |                  0.0026 ± 0.0000 |    0.0030 ± 0.0000 |

**Winner: `ortho_patch`** (orthogonal init on `proj` and `skip` of the GRU
patch head, gain=√2, zero biases). It edges out the default by ~2% relative
on `U_b(o_lat)` (0.0435 vs 0.0427) and ties `ortho_all` within seed-noise — so
the patch head dominates and orthogonalising the transformer linears /
MHA in/out projections adds nothing measurable on top.

## Interpretation

Init alone cannot rescue the U_b ≈ 0.003 collapse the user observed at step
100. At step 0, default init already gives `U_b(o_lat) ≈ 0.043` — about 14×
higher than the post-100-step value. So the collapse is being **driven by
optimisation** in the first 100 steps, not baked in by init: the model
quickly drives the encoder output near the only direction the LayerNorm-on-
skip-plus-residual structure rewards. Two observations sharpen this. (1) The
`skip_*` schemes (zero / small-gain) drop init U_b directly to ~0.003 — i.e.
they reproduce the post-training collapse just by killing the skip path,
confirming the skip path is what gives `o_lat` whatever batch-axis spread it
has. (2) `xavier_small_patch` and `gpt_scaled` (smaller proj+skip / smaller
residual-stream weights) also degrade init U; small weights here are the
*opposite* of what we want. Conclusion: switch the patch-head init to
**ortho_patch** for a small free win, but the real fix for the step-100
collapse will need a loss / architecture change (e.g. anti-collinearity
regulariser on `o_lat`, or removing the LayerNorm-after-residual that pins
all batch members onto a shared direction). Reference: backbone-beta default
init reaches U_b = 0.0762 at step 167k, so the optimiser does eventually
recover — ortho_patch should let it start from a marginally better point.
