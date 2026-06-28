# SIGReg λ-sweep — embedding-side weight pushed further, encoding-side weight varied

## Question

The prior arm (`λ_e=1.0, λ_h=0.1`) had a lower GM-Rel MASE than the `λ_e=λ_h=0.1` arm in all 4 (q-head depth, backbone checkpoint) cells (point Δ_GM range `[−0.014, −0.007]`), but every paired-bootstrap 95% CI vs that anchor straddled zero. Continue the sweep on the same recipe (SIGReg + EMA-target, B=512, enc3+CPC, 12 500 steps, seed 20260520) to find either a clear-of-noise improvement or a ceiling on the SIGReg-weight axis.

The 6 arms run:

| arm | `λ_e` | `λ_h` |
| --- | ---: | ---: |
| 1 | 10.0 | 0.1 |
| 2 | 10.0 | 1.0 |
| 3 | 10.0 | 10.0 |
| 4 | 1.0 | 1.0 |
| 5 | 100.0 | 0.1 |
| 6 | 1000.0 | 1.0 |

## Result

**Verdict.** Two baselines, 6 sweep arms.

Vs the `λ_e=1.0, λ_h=0.1` anchor (#359), only arm 2 (`λ_e=10.0, λ_h=1.0`) on `2L/best` is CI-clean (Δ = −0.017, 95% CI `[−0.029, −0.005]`, P(Δ<0) = 0.997); every other (arm, head, ckpt) cell straddles zero. Arm 6 (`λ_e=1000.0, λ_h=1.0`) on `2L/last` reaches Δ = −0.014, 95% CI `[−0.030, +0.001]`, P(Δ<0) = 0.965 — the upper end just touches zero, so not CI-clean by the 95% rule.

Vs arm 1 (`λ_e=10.0, λ_h=0.1`) — the `λ_h` 0.1 → 1.0 isolation contrast — arm 2 is CI-clean on both best-ckpt cells (`2L/best` Δ = −0.017, 95% CI `[−0.029, −0.006]`, P(Δ<0) = 0.9995; `6L/best` Δ = −0.015, 95% CI `[−0.028, −0.002]`, P(Δ<0) = 0.985). Arm 5 (`λ_e=100.0`) regresses CI-clean on both last-ckpt cells (Annex B).

Among the 6 SIGReg arms:
- Arm 6 holds the lowest `last`-ckpt GM-Rel MASE on both heads (2L/last = 1.1537 vs the next-best SIGReg arm 1 at 1.1610; 6L/last = 1.1415 beats the `enc3+CPC, B=1024` anchor #344 at 1.1436 by 0.0021). 2L/last 1.1537 is 0.0006 above #344's 1.1531.
- Arm 4 (interior point, `λ_e=1.0, λ_h=1.0`) lands mid-tier on `best`: 2L/best = 1.1435 sits between arm 1 (1.1474) and arm 2 (1.1302); 6L/best = 1.1449 is within 0.0002 of arm 1's 1.1447.
- Arm 6 has the smallest positive best-vs-last drift of any SIGReg arm (2L drift = +0.005, 6L drift = +0.002); the runner-up is arm 1 on 6L at +0.003.

![GIFT-Eval full-97 GM-Rel MASE bars across the 4 anchors and the 6 sweep arms, faceted by (q-head depth, backbone checkpoint); whiskers on the sweep bars = paired-bootstrap 95% CI vs the `λ_e=1.0, λ_h=0.1` anchor; per-cell horizontal lines mark each anchor at its published value (grey dotted = enc3+CPC, blue dotted = EMA enc3+CPC, red dashed = SIGReg λ_e=λ_h=0.1, green solid = SIGReg λ_e=1.0/λ_h=0.1); bar labels = GM-Rel MASE.](plots/gm_rel_mase.png)

### (λ_e, λ_h) heatmap

![4-panel GM-Rel MASE heatmap over (λ_e, λ_h), one panel per (q-head depth, backbone checkpoint) cell. X axis = λ_e ∈ {0.1, 1.0, 10.0, 100.0, 1000.0} on a log grid; Y axis = λ_h ∈ {0.1, 1.0, 10.0} on a log grid. Diverging colormap centred on the per-cell SIGReg + EMA-target, B=512, λ_e=1.0, λ_h=0.1 anchor (#359): red = worse than that anchor, blue = better. Tile text = GM-Rel MASE. Hatched tiles = (λ_e, λ_h) points not run.](plots/heatmap.png)

The 2D view now spans 15 grid tiles (5 λ_e × 3 λ_h); 8 are populated (#355 + #359 + arms 1/2/3/4/5/6) and 7 are hatched. Inside the original 4 λ_e × 3 λ_h sub-grid that the sweep started from, only the high-`λ_h` corner at λ_e=100 and the `λ_e=0.1, λ_h≥1` corner remain unrun; arm 6 sits in a new λ_e=1000 column with its λ_h=0.1 and λ_h=10 neighbours unrun. The λ_h=1.0 row reads (arm 4: 1.143 / arm 2: 1.130 / arm 6: 1.149) on 2L/best — minimum at arm 2 — and (arm 4: 1.168 / arm 2: 1.176 / arm 6: 1.154) on 2L/last — minimum at arm 6.

### Training trajectory

![Log-log total training loss (50-step rolling mean) from step 100 onwards for the 6 sweep arms and the 2 prior λ_h=0.1 anchors. Cutting the first 100 warm-up steps and log axes keep the converged regime readable.](plots/loss_curve.png)

Tail-50 mean total loss ordering (steps 12 451–12 500, table in §Annex D) is arm 6 (3.64) < arm 5 (3.88) < arm 1 (4.20) < arm 3 (4.26) < arm 4 (4.47) < arm 2 (4.55); the two regulariser-side terms make total loss not directly comparable to the GM-Rel MASE ordering.

![Log-y trajectories of L_SIGReg(e_t), L_SIGReg(h_t), U_batch(e_t), U_temporal(e_t) from step 100 onwards for the 6 sweep arms and the 2 anchors; rolling 50-step mean. The bottom row is the embedding-side dimension-usage metric U; its 1/K ≈ 0.00260 dotted floor marks rank-1 collapse (one effective dim out of K=384). Higher U = more dimensions in use; K · U ≈ effective number of dims.](plots/sigreg_e_inspection.png)

Arms 5 and 6 (the two `λ_e ≥ 100` arms) drive `L_SIGReg(e_t)` to tail-50 floors 1.5 to 2.5 orders of magnitude below the four `λ_e ≤ 10` arms (per-arm values in §Annex D).

![Dimension-usage U split by latent — top panel: `e_t` (embedding side) on log-y `[1/K, 0.1]`; bottom panel: `h_t` (encoder side) on log-y `[0.05, 1]`. Per panel: solid = `u_batch` (cross-batch), dashed = `u_temporal` (cross-time), dotted line with `●` markers = `u_batchtime` per-checkpoint trajectory (every 2 500 steps, retroactively scored on the saved backbones; only available for the 4 first sweep arms + 2 anchors — see §Annex D), `★` marker at the actual best-loss step = `u_batchtime` on the canonical `best_loss.pth` / `FINAL.pth` checkpoint (cross-(batch × time) pooled — same `(B·T, K)` sample axis SIGReg uses). Colour = arm; dotted line at `1/K ≈ 0.0026` marks rank-1 collapse (one effective dim out of K=384). Higher U = more dimensions in use; K · U ≈ effective number of dims.](plots/dim_usage.png)

The `u_batchtime` trajectories now cover the 4 first sweep arms + 2 anchors across all saved checkpoints. On `h_t`, arms 1 and 2 climb from `≈ 0.38` at step 2 500 to `0.72 / 0.77` at step 12 500; arm 3 starts higher at `0.59` and plateaus near `0.75`; arm 5 stays in the `0.25–0.34` band throughout. The `★` marker for arms 1/2/3 sits well below the step-12 500 `●` because their `best_loss.pth` (= `FINAL.pth`) landed at step ≈ 1 400, before the late-training U climb; for arm 5 the `★` (step 1 700) and the step-12 500 `●` are close because U barely moves. Arms 4 and 6 are not in the trajectory or retro CSVs yet — those scoring passes are queued.

### GM-Rel MASE — B=512 sweep family

Among the 6 SIGReg sweep arms, arm 2 (`λ_e=10.0, λ_h=1.0`) holds the row-minimum on both `best` cells and arm 6 (`λ_e=1000.0, λ_h=1.0`) holds the row-minimum on both `last` cells. The 4 anchor rows are kept for reference; column-bold marks the row-minimum among the B=512 family (`λ_e=λ_h=0.1` (#355), `λ_e=1.0, λ_h=0.1` (#359), and the 6 sweep arms). The B=1024 cells (#344, #353) are not comparable to the B=512 family and are never bolded, regardless of their value.

| head / ckpt | `enc3+CPC`, B=1024 (#344) | `EMA enc3+CPC`, B=1024 (#353) | `λ_e=λ_h=0.1`, B=512 (#355) | `λ_e=1.0, λ_h=0.1`, B=512 (#359) | arm 1 (10.0, 0.1) | arm 2 (10.0, 1.0) | arm 3 (10.0, 10.0) | arm 4 (1.0, 1.0) | arm 5 (100.0, 0.1) | arm 6 (1000.0, 1.0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2L / best | 1.1846 | 1.1614 | 1.1610 | 1.1470 | 1.1474 | **1.1302** | 1.1540 | 1.1435 | 1.1554 | 1.1488 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 | 1.1681 | 1.1610 | 1.1756 | 1.1682 | 1.1679 | 1.1828 | **1.1537** |
| 6L / best | 1.1584 | 1.1576 | 1.1543 | 1.1408 | 1.1447 | **1.1294** | 1.1465 | 1.1449 | 1.1462 | 1.1397 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 | 1.1482 | 1.1473 | 1.1515 | 1.1538 | 1.1517 | 1.1682 | **1.1415** |

### λ_e ladders at λ_h=0.1 and λ_h=1.0

![Two λ_e ladders. Left: λ_h=0.1, 0.1 (#355) → 1.0 (#359) → 10.0 (arm 1) → 100.0 (arm 5). Right: λ_h=1.0, 1.0 (arm 4) → 10.0 (arm 2) → 1000.0 (arm 6). Both axes log; 4 line curves per panel (2L/6L × best/last); shaded bands = paired-bootstrap 95% CI vs the `λ_e=1.0, λ_h=0.1` anchor (#359).](plots/lambda_e_ladder.png)

Walking `λ_e` from 0.1 → 1.0 → 10.0 → 100.0 at `λ_h=0.1`, the GM drops between 0.1 and 1.0 on all 4 cells, moves by Δ ∈ [−0.0071, +0.0039] between 1.0 and 10.0 (within the bootstrap CI on every cell), then rises at `λ_e=100.0` on all 4 cells.

The `λ_h=1.0` sub-ladder has 3 points (arm 4 → arm 2 → arm 6 at λ_e = 1.0 → 10.0 → 1000.0) and #359 is *not* on this row. On the two `best` cells it is U-shaped, minimum at arm 2 (2L: 1.1435 → 1.1302 → 1.1488; 6L: 1.1449 → 1.1294 → 1.1397). On the two `last` cells the minimum is at arm 6 (2L: 1.1679 → 1.1756 → 1.1537 — non-monotonic, dip at λ_e=1000; 6L: 1.1517 → 1.1515 → 1.1415 — flat then drop).

### Best-vs-last divergence

The desirable direction at step 12 500 is `last ≤ best` — i.e. the model is still improving at the end of training, so picking `last` would be at least as good as picking `best`. Positive `last − best` is therefore the drift / over-fitting signal.

Only the `enc3+CPC, B=1024` anchor (#344) shows negative drift on both heads (still improving at step 12 500). Every SIGReg arm has positive drift on both heads. Arm 6 has the smallest positive drift on both heads (2L = +0.005, 6L = +0.002); arm 2's `2L` drift of `+0.045` is the largest in the table.

Drift is a within-arm diagnostic; cross-arm last-ckpt CIs vs arm 1 are tabulated separately in §Annex B.

![Drift = last − best GM-Rel MASE per arm, split by 2L vs 6L q-head. Positive = last is worse than best (model stopped improving by step 12 500); negative = last is better than best (model still improving). Only the `enc3+CPC, B=1024` anchor sits below zero on both heads.](plots/best_vs_last_drift.png)

## Protocol

Single seed `20260520`, 12 500 steps. Launcher: [`scripts/train_backbone_sigreg.sh`](../../experiments/2026-06-24_sigreg_lambda_sweep/scripts/train_backbone_sigreg.sh). Backbone: GRU patch-embed → 3-layer transformer encoder (`K`=384, 6 heads). The 6 arms change exactly two flags vs the `λ_e=1.0, λ_h=0.1` anchor:

| flag | arm 1 | arm 2 | arm 3 | arm 4 | arm 5 | arm 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `--sigreg-embedding-weight` (`λ_e`) | 10.0 | 10.0 | 10.0 | 1.0 | 100.0 | 1000.0 |
| `--sigreg-encoding-weight` (`λ_h`) | 0.1 | 1.0 | 10.0 | 1.0 | 0.1 | 1.0 |

All other flags identical to the prior arm: `--batch-size 512`, `--sigreg-embedding --sigreg-encoding`, `--sigreg-n-chunk 2048`, `--sigreg-post-normalization` OFF, `--ema-embedding --ema-encoder --ema-tau 0.99`, `--cpc-infonce-weight 1.0`, `--encoder-dropkey 0.70`, `--mix-ratio 0.0078125`, `--sigreg-m 1024`, `--sigreg-t-knots 17`, same dataset (`gift-pretrain-full-4096` / `small_v1`), dtypes.

### Head-matched downstream

For each arm, each backbone checkpoint (`best` = best train-loss, `last` = step 12 500) trains a 2-layer and a 6-layer quantile head; each head is evaluated on GIFT-Eval full-97 via `scripts/run_gift_eval_full.sh`. Per-cell summaries: `results/gift_eval_full_<tag>{,_last}_{2L,6L}/summary.txt`.

### Anchors

The 4 anchors that share the GM table:

| label | recipe |
| --- | --- |
| `enc3+CPC, B=1024` (#344) | non-SIGReg, non-EMA baseline |
| `EMA enc3+CPC, B=1024` (#353) | EMA-target only, no SIGReg |
| `SIGReg λ_e=λ_h=0.1, B=512` (#355) | `λ_e=λ_h=0.1` (per-config rel-MASE re-read from `reports/2026-06-20_lejepa_sigreg/`) |
| `SIGReg λ_e=1.0, λ_h=0.1, B=512` (#359) | the bootstrap baseline (per-config rel-MASE re-read from `reports/2026-06-22_lejepa_sigreg_emb10/`) |

Anchor data provenance (GM-table source, per-config rel-MASE files) is listed in §Annex C.

## Vocabulary

| term | definition |
| --- | --- |
| `K` | latent dimensionality, 384 throughout this report. `1/K ≈ 0.00260`. |
| `enc3` | 3-layer transformer encoder (hidden size `K`, 6 heads). |
| `CPC` | InfoNCE auxiliary head on the encoder, `--cpc-infonce-weight 1.0`. |
| **EMA-target** | exponential-moving-average teacher on the encoder + patch-embed, `--ema-tau 0.99`. |
| `e_t` | output of the GRU patch-embed at position (batch, time, channel); dimension `K`. |
| `h_t` | output of the 3-layer transformer encoder at the same position. |
| **SIGReg** | LeJEPA spherical regulariser. Epps–Pulley test statistic averaged over `M`=1024 random unit-direction 1-D projections of the pooled latent, trapezoidal-integrated on `[−6/√K, 6/√K]` against `N(0, 1/K)`. Two terms: `L_SIGReg(e_t)` weighted by `λ_e`, `L_SIGReg(h_t)` weighted by `λ_h`. |
| `U`, `u_*` | **dimension usage** statistic of the latent, in `[1/K, 1]` with `K`=384. `U = 1 / (K · E[cos²(z_i, z_j)])` clipped at 1. The `1/K` floor = rank-1 collapse, i.e. one effective dim; values near 1 = all K dims used (isotropic). **Higher = more dimensions in use.** `K · U ≈` effective number of dimensions in use; e.g. `U = 0.79` at `K=384` corresponds to `≈ 303` effective dims, `U = 0.013` to `≈ 5` effective dims. Three pooling axes — cross-batch (`u_batch`, pools `B` per time slice), cross-time (`u_temporal`, pools `T` per batch slice), and cross-(batch × time) (`u_batchtime`, pools `B·T` jointly — the same `(B·T, K)` sample axis SIGReg's random-projection statistic uses). Each axis has an encoding-side variant on `h_t` (no suffix) and an embedding-side variant on `e_t` (`_e` suffix). Full set: `u_batch`, `u_temporal`, `u_batchtime`, `u_batch_e`, `u_temporal_e`, `u_batchtime_e`. Same `[1/K, 1]` range and rank-1-collapse floor interpretation across all six. Math check + verification: [`docs/u_metric_check.md`](../../docs/u_metric_check.md). |
| **GM-Rel MASE** | GIFT-Eval full-97 aggregate: geometric mean over 97 configs of (model MASE ÷ seasonal-naive MASE). Lower = better; 1.0 = seasonal-naive parity. |
| **best-ckpt / last-ckpt** | `best` = backbone checkpoint at lowest train-loss step; `last` = backbone at step 12 500. The q-head is trained from each backbone separately. The preferred end-state is `last ≤ best` (model still improving at step 12 500). |
| **paired bootstrap** | resample the 97 per-config rel-MASE values with replacement (B=10 000 draws, seed 20260624), recompute the statistic `mean(log(rel_arm) − log(rel_baseline))`, take its 2.5/97.5 quantiles, convert back to absolute GM scale via `GM_baseline · (exp(quantile) − 1)`. Aligned per config. |

## Annex

### A. Δ vs `λ_e=1.0, λ_h=0.1` (#359) with paired-bootstrap 95% CI

B=10 000 bootstrap draws, n=97 configs, paired on per-config rel-MASE. Δ on the absolute GM-Rel MASE scale via `GM_anchor · (exp(quantile) − 1)`.

| arm | head / ckpt | Δ_GM | 95% CI | P(Δ<0) |
| --- | --- | ---: | --- | ---: |
| arm 1 (10.0, 0.1) | 2L / best | +0.0004 | `[−0.0081, +0.0106]` | 0.494 |
| arm 1 (10.0, 0.1) | 2L / last | −0.0071 | `[−0.0230, +0.0087]` | 0.807 |
| arm 1 (10.0, 0.1) | 6L / best | +0.0039 | `[−0.0036, +0.0130]` | 0.188 |
| arm 1 (10.0, 0.1) | 6L / last | −0.0010 | `[−0.0156, +0.0175]` | 0.583 |
| **arm 2 (10.0, 1.0)** | **2L / best** | **−0.0168** | **`[−0.0293, −0.0045]`** | **0.997** |
| arm 2 (10.0, 1.0) | 2L / last | +0.0075 | `[−0.0061, +0.0220]` | 0.140 |
| arm 2 (10.0, 1.0) | 6L / best | −0.0114 | `[−0.0258, +0.0067]` | 0.906 |
| arm 2 (10.0, 1.0) | 6L / last | +0.0032 | `[−0.0110, +0.0185]` | 0.328 |
| arm 3 (10.0, 10.0) | 2L / best | +0.0070 | `[−0.0035, +0.0193]` | 0.102 |
| arm 3 (10.0, 10.0) | 2L / last | +0.0002 | `[−0.0143, +0.0142]` | 0.479 |
| arm 3 (10.0, 10.0) | 6L / best | +0.0057 | `[−0.0042, +0.0173]` | 0.142 |
| arm 3 (10.0, 10.0) | 6L / last | +0.0056 | `[−0.0105, +0.0221]` | 0.242 |
| arm 4 (1.0, 1.0) | 2L / best | −0.0035 | `[−0.0113, +0.0048]` | 0.810 |
| arm 4 (1.0, 1.0) | 2L / last | −0.0002 | `[−0.0145, +0.0155]` | 0.510 |
| arm 4 (1.0, 1.0) | 6L / best | +0.0042 | `[−0.0038, +0.0134]` | 0.169 |
| arm 4 (1.0, 1.0) | 6L / last | +0.0035 | `[−0.0114, +0.0213]` | 0.347 |
| arm 5 (100.0, 0.1) | 2L / best | +0.0084 | `[−0.0010, +0.0188]` | 0.042 |
| arm 5 (100.0, 0.1) | 2L / last | +0.0148 | `[−0.0055, +0.0363]` | 0.083 |
| arm 5 (100.0, 0.1) | 6L / best | +0.0055 | `[−0.0034, +0.0147]` | 0.116 |
| arm 5 (100.0, 0.1) | 6L / last | +0.0199 | `[−0.0045, +0.0458]` | 0.051 |
| arm 6 (1000.0, 1.0) | 2L / best | +0.0018 | `[−0.0144, +0.0204]` | 0.429 |
| arm 6 (1000.0, 1.0) | 2L / last | −0.0144 | `[−0.0299, +0.0013]` | 0.965 |
| arm 6 (1000.0, 1.0) | 6L / best | −0.0011 | `[−0.0151, +0.0129]` | 0.551 |
| arm 6 (1000.0, 1.0) | 6L / last | −0.0067 | `[−0.0254, +0.0146]` | 0.753 |

Bold = the one cell whose 95% CI excludes zero.

### B. Δ vs arm 1

Arm 1 differs from the anchor by a single `λ_e` factor of 10×, so Δ vs arm 1 isolates the `λ_h` and `λ_e ∈ {1.0, 100.0, 1000.0}` axes.

| arm | head / ckpt | Δ_GM | 95% CI | P(Δ<0) |
| --- | --- | ---: | --- | ---: |
| arm 2 (10.0, 1.0) | 2L / best | −0.0172 | `[−0.0291, −0.0062]` | 0.9995 |
| arm 2 (10.0, 1.0) | 2L / last | +0.0146 | `[−0.0009, +0.0306]` | 0.033 |
| arm 2 (10.0, 1.0) | 6L / best | −0.0153 | `[−0.0280, −0.0018]` | 0.985 |
| arm 2 (10.0, 1.0) | 6L / last | +0.0042 | `[−0.0121, +0.0207]` | 0.294 |
| arm 3 (10.0, 10.0) | 2L / best | +0.0066 | `[−0.0065, +0.0200]` | 0.164 |
| arm 3 (10.0, 10.0) | 2L / last | +0.0072 | `[−0.0072, +0.0215]` | 0.160 |
| arm 3 (10.0, 10.0) | 6L / best | +0.0018 | `[−0.0073, +0.0106]` | 0.339 |
| arm 3 (10.0, 10.0) | 6L / last | +0.0066 | `[−0.0089, +0.0219]` | 0.194 |
| arm 4 (1.0, 1.0) | 2L / best | −0.0039 | `[−0.0156, +0.0068]` | 0.756 |
| arm 4 (1.0, 1.0) | 2L / last | +0.0069 | `[−0.0061, +0.0201]` | 0.149 |
| arm 4 (1.0, 1.0) | 6L / best | +0.0003 | `[−0.0081, +0.0095]` | 0.483 |
| arm 4 (1.0, 1.0) | 6L / last | +0.0045 | `[−0.0099, +0.0202]` | 0.272 |
| arm 5 (100.0, 0.1) | 2L / best | +0.0080 | `[−0.0022, +0.0179]` | 0.057 |
| arm 5 (100.0, 0.1) | 2L / last | +0.0218 | `[+0.0047, +0.0410]` | 0.005 |
| arm 5 (100.0, 0.1) | 6L / best | +0.0016 | `[−0.0100, +0.0107]` | 0.360 |
| arm 5 (100.0, 0.1) | 6L / last | +0.0209 | `[+0.0026, +0.0411]` | 0.011 |
| arm 6 (1000.0, 1.0) | 2L / best | +0.0014 | `[−0.0125, +0.0150]` | 0.408 |
| arm 6 (1000.0, 1.0) | 2L / last | −0.0073 | `[−0.0215, +0.0091]` | 0.828 |
| arm 6 (1000.0, 1.0) | 6L / best | −0.0050 | `[−0.0188, +0.0081]` | 0.760 |
| arm 6 (1000.0, 1.0) | 6L / last | −0.0057 | `[−0.0160, +0.0048]` | 0.858 |

### C. Plot and CI provenance

- **Training CSVs.** `experiments/2026-06-24_sigreg_lambda_sweep/runs/bb_<tag>_<arm>_losses.csv` (12 500 rows each, seed 20260520).
- **Per-config rel-MASE for sweep arms.** `experiments/2026-06-24_sigreg_lambda_sweep/results/gift_eval_full_<tag>_<arm>{,_last}_{2L,6L}/summary.txt`.
- **Per-config rel-MASE for the bootstrap baseline (`sigreg10`, #359).** `reports/2026-06-22_lejepa_sigreg_emb10/results/gift_eval_full_<tag>_emb10{,_last}_{2L,6L}/summary.txt`.
- **Per-config rel-MASE for the other SIGReg anchor (`sigreg01`, #355).** `reports/2026-06-20_lejepa_sigreg/results/gift_eval_full_<tag>{,_last}_{2L,6L}/summary.txt`.
- **Anchor GM-Rel MASE values.** Transcribed verbatim from `reports/2026-06-22_lejepa_sigreg_emb10/results/gm_table.csv`, which carries `cpc_enc3` / `ema_enc3` / `sigreg01_enc3` / `sigreg10_enc3` rows alongside the prior arm.
- **CI computation.** [`scripts/compute_bootstrap.py`](scripts/compute_bootstrap.py); outputs `results/bootstrap_ci_vs_359.csv` and `results/bootstrap_ci_vs_arm1.csv`.
- **Plot script.** [`scripts/build_plots.py`](scripts/build_plots.py). Trajectory plots cut the first `PLOT_START_STEP = 100` steps so the warm-up regime does not dominate the y-range; the loss curve uses log x and log y, the SIGReg-inspection panels and the split `dim_usage.png` panels use log y throughout.
- **Bar-chart colour map.** grey = `enc3+CPC, B=1024`; blue = `EMA enc3+CPC, B=1024`; red = `SIGReg λ_e=λ_h=0.1`; green = `SIGReg λ_e=1.0, λ_h=0.1`; purple = arm 1; brown = arm 2; pink = arm 3; cyan = arm 4; olive = arm 5; orange = arm 6.

### D. Numeric annex — diagnostics tables

#### Tail-50 trajectories per arm

Tail-50 mean = mean over steps 12 451–12 500 (last 50 of 12 500 logged steps) of each per-arm `experiments/2026-06-24_sigreg_lambda_sweep/runs/bb_<tag>_losses.csv`. The same Tail-50 values are also written to [`results/final_trajectories.txt`](results/final_trajectories.txt) (whose first per-arm line records the `final_step` index 12 500 alongside the Tail-50 means).

| arm | `loss` | `sigreg_e` | `sigreg_h` | `u_batch_e` | `u_temporal_e` | `u_batch` (`h_t`) | `u_temporal` (`h_t`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arm 1 (10.0, 0.1) | 4.203 | 1.90e-4 | 5.07e-4 | 0.0190 | 0.0164 | 0.7734 | 0.4825 |
| arm 2 (10.0, 1.0) | 4.550 | 4.49e-4 | 3.31e-4 | 0.0506 | 0.0340 | 0.7964 | 0.6291 |
| arm 3 (10.0, 10.0) | 4.256 | 4.46e-4 | 3.20e-4 | 0.0588 | 0.0400 | 0.7853 | 0.6138 |
| arm 4 (1.0, 1.0) | 4.467 | 8.49e-4 | 3.41e-4 | 0.0363 | 0.0259 | 0.7822 | 0.5977 |
| arm 5 (100.0, 0.1) | 3.878 | 7.10e-6 | 9.61e-5 | 0.0177 | 0.0160 | 0.6161 | 0.2705 |
| arm 6 (1000.0, 1.0) | 3.636 | 4.50e-6 | 1.64e-4 | 0.0318 | 0.0267 | 0.5930 | 0.2562 |

#### `u_batchtime` per-checkpoint trajectory (retroactive)

`u_batchtime` (cross-(batch × time) pooled `U` on `h_t`) and `u_batchtime_e` (same on `e_t`) were added to the training-loop metrics after the 6 sweep arms and the 2 prior B=512 anchors had already trained, so no in-training trajectory exists in their losses CSVs. The values below are computed retroactively from each saved backbone checkpoint over a single fixed batch (gift-pretrain-full-4096 / small_v1, seed 20260520, B=512). All-checkpoint trajectories were scored for the 4 first sweep arms + 2 anchors; the same scoring pass for arms 4 and 6 is queued but not yet committed, so they appear in this table only at the FINAL retroactive row.

`FINAL` rows (single-step retro on the `FINAL.pth` saved at training end):

| arm / anchor | recipe | `u_batchtime` (`h_t`) | `u_batchtime_e` (`e_t`) |
| --- | --- | ---: | ---: |
| #355 anchor | `λ_e=0.1, λ_h=0.1` | 0.3897 | 0.0136 |
| #359 anchor | `λ_e=1.0, λ_h=0.1` | 0.3535 | 0.0133 |
| arm 1 | `λ_e=10.0, λ_h=0.1` | 0.3527 | 0.0132 |
| arm 2 | `λ_e=10.0, λ_h=1.0` | 0.3535 | 0.0131 |
| arm 3 | `λ_e=10.0, λ_h=10.0` | 0.3663 | 0.0137 |
| arm 5 | `λ_e=100.0, λ_h=0.1` | 0.3144 | 0.0154 |

Sources: [`results/u_batchtime_retro.csv`](results/u_batchtime_retro.csv) (FINAL rows), [`results/u_batchtime_trajectory.csv`](results/u_batchtime_trajectory.csv) (all-checkpoint rows for the 4 first sweep arms + 2 anchors); retroactive computation scripts `experiments/2026-06-24_sigreg_lambda_sweep/scripts/compute_u_batchtime_retro.py` and `compute_u_batchtime_trajectory.py`.

### E. Scope notes

- Reported GIFT-Eval aggregate is GM-Rel MASE only. GM-MASE, GM-MAPE_SN, GM-CRPS_SN are out of scope: the per-config evaluation output does not carry the seasonal-naive denominators required to compute them.
- Single seed; the bootstrap CIs above describe sampling variability across the 97 GIFT-Eval configs, not run-to-run seed variability.
