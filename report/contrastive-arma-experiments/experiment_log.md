# Experiment Log

Complete record of all training runs for research paper reference.
Compute: NVIDIA RTX 4090 (24GB) on `elisa`. Single GPU throughout.

---

## 1. Backbone Architecture Search (Phase 1: Encoder, Phase 2: Transformer config)

All Phase 1/2 runs used **6 layers**, **H=512**, **batch_size=16**, **lr=1e-4**, **50,000 steps**, **temperature=0.07**, ARMA dimension=4. Only JSON results saved (no .log files).

### Phase 1: Encoder comparison (5 configs, 50k steps each)

| Exp | Encoder | Params | Duration | Step/s | Peak gap | Final gap |
|-----|---------|--------|----------|--------|----------|-----------|
| E1 | mlp (int=64) | 13.2M | 30.5 min | 27.3 | 0.0727 | 0.0710 |
| E2 | mlp_wide (int=256) | 13.3M | 29.8 min | 27.9 | 0.0750 | 0.0736 |
| E3 | residual_silu | 13.7M | 29.9 min | 27.9 | 0.0838 | 0.0824 |
| **E4** | **gru** | **13.7M** | **61.6 min** | **13.5** | **0.1146** | **0.1146** |
| E5 | conv | 13.2M | 33.1 min | 25.2 | 0.0747 | 0.0747 |

**Total Phase 1 duration**: ~185 min (~3.1 h). Winner: GRU encoder.

### Phase 2: Transformer config comparison (7 configs, 50k steps each, gru encoder fixed)

| Exp | Variation | Params | Duration | Step/s | Peak gap | Final gap |
|-----|-----------|--------|----------|--------|----------|-----------|
| T1 | baseline (nhead=8, ffn=2x, gelu, k=3) | 13.7M | 61.6 min | 13.5 | 0.1190 | 0.1169 |
| T2 | nhead=16 | 13.7M | 62.7 min | 13.3 | 0.1102 | 0.1102 |
| **T3** | **ffn_mult=4** | **20.0M** | **69.2 min** | **12.0** | **0.1250** | **0.1250** |
| T4 | ffn_mult=1 | 10.5M | 57.8 min | 14.4 | 0.1150 | 0.1150 |
| T5 | activation=silu | 13.7M | 61.6 min | 13.5 | 0.1059 | 0.1047 |
| T6 | depthwise_conv=0 | 13.7M | 64.3 min | 13.0 | 0.1043 | 0.1030 |
| T7 | nhead=16 + silu | 13.7M | 62.6 min | 13.3 | 0.1176 | 0.1158 |

**Total Phase 2 duration**: ~440 min (~7.3 h). Winner: FFN 4x. Depthwise conv confirmed essential (T6 is worst).

**Phase 3 (scaling)** was skipped. Decision: proceed with 12L H=1024.

**Note**: No 4L or 8L ablation was run. All Phase 1/2 experiments used 6 layers.

---

## 2. Phase 4: Full training of best architecture (500k steps)

| Field | Value |
|-------|-------|
| Architecture | gru encoder, H=1024, 12 layers, nhead=8, ffn_mult=4.0, gelu, conv_k=3 |
| Parameters | **153,849,600** (153.8M) |
| batch_size | 8 |
| lr | 7e-5 |
| total_steps | 500,000 |
| Duration | **1211.3 min (~20.2 h)** |
| Step/s | 6.9 |
| **Peak gap** | **0.1862** at step ~494,000 |
| Final gap | 0.1788 |

Log: `arch_search_phase4.log`, Results: `arch_search_phase4_gru_ffn4x_H1024_results.json`

---

## 3. Extended 2M training of 12L model (3 segments)

The 12L V2 model was trained incrementally across multiple segments due to resource/crash issues.

### Segment A: Phase 4 (500k initial)
See Section 2 above. Total: 20.2 h.

### Segment B: Resume to 2M
| Field | Value |
|-------|-------|
| Log | `v2_2M_training_resumed.log` |
| Resumed from | Phase 4 best checkpoint (step ~12000) |
| Total steps configured | 2,000,000 |
| Actual steps reached | **1,967,000** (crashed before completion) |
| Duration | ~79.2 h (estimated from step count / 6.9 step/s) |
| Step/s | 6.9 |
| **Peak gap** | **0.2015** |
| Final gap | 0.1938 |
| Notes | Optimizer state lost on earlier resume (pre-checkpoint.py), crashed at step 1,967,000 |

### Segment C: Final 50k refinement
| Field | Value |
|-------|-------|
| Log | `v2_2M_final.log`, Results: `arch_search_v2_2M_final_results.json` |
| Resumed from | `v2_2M_model.pth` (step 1,967,000) |
| total_steps | 50,000 |
| Duration | 121.2 min (~2.0 h) |
| **Peak gap** | **0.2028** (step 21,000 of continuation) |
| Final gap | 0.197 |
| Best val FF | 0.5974 @ step 32,000 |

**Total 12L training**: ~500k + 1.97M + 50k = **~2.02M effective steps, ~101 h wall clock**. Peak gap 0.2028.

---

## 4. Recovery Architecture Search (Mar 27-29, 2026)

Backbone: V2 12L H=1024 (`v2_2M_model_best.pth`). Recovery heads trained on frozen embeddings. Default: bs=32, lr=1e-3, MSE loss, 4 AR + 4 MA coefficients, ARMA dim=4.

### Phase 1: 7 head architectures on V2 backbone (5k epochs each)
Total: **135 min** (~8141s).

| Head | Params | Best val loss | Improvement | Sign AR | Sign MA |
|------|--------|---------------|-------------|---------|---------|
| mlp | 396K | 0.0440 | 3.79x | 88.2% | 85.9% |
| gru | 2.4M | 0.0246 | 5.99x | 91.8% | 89.5% |
| resmlp | 1.3M | 0.0424 | 3.99x | 88.3% | 87.2% |
| attention | 1.4M | 0.0257 | 5.54x | 90.8% | 88.2% |
| **grupool** | **2.4M** | **0.0237** | 5.66x | 90.4% | 88.8% |
| deepgru | 3.6M | 0.0252 | 5.79x | 91.8% | 90.0% |
| deepgrupool | 3.0M | 0.0259 | 5.81x | 90.5% | 89.3% |

### Phase 2: Same 7 heads on V1 backbone (5k epochs each)
Total: **92 min** (~5571s).

| Head | Best val loss | Improvement |
|------|---------------|-------------|
| mlp | 0.0511 | 3.26x |
| gru | 0.0257 | 5.68x |
| resmlp | 0.0455 | 3.73x |
| attention | 0.0275 | 5.37x |
| grupool | 0.0252 | 5.87x |
| deepgru | 0.0258 | 5.81x |
| deepgrupool | 0.0310 | 4.70x |

### Phase 3: Hyperparameter sweep (24 configs, 5k epochs each)
Total: **483 min** (~29,029s, ~20 min per run).

Tested: `gru` and `deepgru` × hidden_dim ∈ {128, 256, 512} × num_gru_layers ∈ {1, 2, 3, 4}. Best val loss range: 0.0233 – 0.0268.

**BUG**: `create_recovery_head()` did not forward `num_gru_layers` to `GRU`/`GRUPool` heads. All Phase 3 GRU runs used default 2 layers. DeepGRU unaffected. Fixed in commit `ca6663f`.

### Phase 4: Loss sweep on deepgru (4 configs, 5k epochs each)
Total: **80 min** (~4807s).

| Loss | Best val loss | Improvement |
|------|---------------|-------------|
| MSE | 0.0234 | 5.65x |
| Huber | 0.0072* | 5.60x |
| L1 | 0.1236* | 5.50x |
| weighted_mse | 0.0400* | 5.62x |

*Val loss not comparable across different training losses. MSE is best by improvement ratio.

### Phase 5: Full training of best configs (20k epochs each)
Total: **316 min** (~18,967s, ~63 min per run).

| Run | Backbone | Head | Loss | Best val loss @ epoch | Improvement |
|-----|----------|------|------|----------------------|-------------|
| **v2_gru_h128_l2** | **V2** | **gru h=128 l=2** | **mse** | **0.0223 @ 19711** | **6.96x** |
| v2_grupool | V2 | grupool h=256 l=2 | mse | 0.0216 @ 12396 | 6.43x |
| v2_gru_h128_l2_huber | V2 | gru h=128 l=2 | huber | 0.0055 @ 19686 | 6.87x |
| v1_gru_h128_l2 | V1 | gru h=128 l=2 | mse | 0.0221 @ 17796 | 6.59x |
| v1_grupool | V1 | grupool h=256 l=2 | mse | 0.0218 @ 19689 | 6.38x |

(Labels show "l3" in log names but actual layer count is 2 due to the num_gru_layers bug.)

### Follow-up: Corrected GRU layer sweep (post-bug-fix)

After fixing the bug, gru_h128 layer variants were rerun (5k epochs each):

| Layers | Params | Best val loss | Improvement |
|--------|--------|---------------|-------------|
| l1 | 380K | 0.0250 | 6.00x |
| l2 | 676K | 0.0241 | 5.90x |
| l3 | 973K | 0.0239 | 5.76x |
| l4 | 1.27M | 0.0242 | 5.88x |

Also `gru_h128_l1` at 20k epochs: 0.0225 @ 18117, improvement 6.64x (worse than l2 at 20k).

### Loss sweep on gru_h128_l2 (post-search)
| Loss | Best val loss | Improvement |
|------|---------------|-------------|
| MSE | 0.0247 | 5.90x |
| Huber | 0.0072 | 5.84x |
| L1 | 0.1273 | 5.65x |
| weighted_mse | 0.0382 | 5.77x |

### Dim 2x2 comparison (2 AR + 2 MA, against old 7.3x record)
Backbone: V2, Head: gru h=128 l=2, MSE.

| Config | Epochs | Best val loss @ epoch | Improvement |
|--------|--------|----------------------|-------------|
| dim2 | 5k | 0.0475 @ 3931 | 7.65x* |
| **dim2_20k** | **20k** | **0.0425 @ 12364** | **8.34x*** |

*Baseline for dim=2 (0.4305) differs from dim=4 (0.1963). Numbers not directly comparable across dimensions.

### v2_2M_recovery (separate from Phase 5, original DeepGRU run)
| Field | Value |
|-------|-------|
| Head | deepgru h=256, 3-layer GRU, ~4.7M params |
| Best val loss | 0.02161 @ epoch 19,588 |
| Improvement | 6.64x |

**Total recovery search duration**: ~1100 min (~18.3 h) across all phases.

---

## 5. Scaling Search (Apr 6-11, 2026, ongoing)

All runs: encoder=gru, nhead=8, ffn_mult=4, gelu, conv_k=3, batch_size=8, lr=7e-5 (unless noted), ARMA dim=4. Started with 200k step comparison, then full 2M on best.

### Quick comparison phase (200k steps each)

| Config | Params | Duration | Step/s | Peak gap | Final gap |
|--------|--------|----------|--------|----------|-----------|
| 12L H=1024 (baseline) | 153.8M | 482.4 min (~8.0 h) | 6.9 | 0.1617 | 0.1613 |
| 16L H=1024 | 204.2M | 621.0 min (~10.4 h) | 5.4 | **0.1662** | 0.1609 |
| 20L H=1024 | 254.6M | 748.6 min (~12.5 h) | 4.5 | 0.1535 | 0.1504 |
| 12L H=1280 | 240.0M | <5 min | — | — | — (aborted, see next section) |

Winner at 200k: 16L (+0.004 over 12L), but 20L's slope is steepest — expected to overtake with more training.

**H=1280 and H=1536 configs were NOT run** — decision made to focus on depth scaling only (H=1024 → 20L) since width scaling is confounded by LR choice without muP. The 12L H=1280 experiment was aborted at step 3 when the scaling search script was killed to redirect compute.

### 20L H=1024 full training attempts

**Attempt 1 (FAILED — gap collapse)**:
- Log: `scaling_20L_H1024_2M.log`
- Config: lr=7e-5, fresh training from scratch
- Reached step 127,000 with gap ≈ 0.00 (training collapse, non-contrastive solution)
- Duration before abort: ~470 min

**Attempt 2 (COMPLETE — successful)**:
- Log: `scaling_20L_H1024_2M_resumed.log`
- Config: lr=7e-5, resumed from `scaling_20L_H1024_best.pth` (step 12000 of 200k search)
- Total steps: 2,000,000
- Duration: **7554.8 min (125.9 h, ~5.2 days)**
- Step/s: 4.4
- **Peak gap**: **0.2019**
- Final gap: 0.1975
- **CHECKPOINT LOST**: `scaling_20L_H1024_2M.pth` was accidentally overwritten by a subsequent low-LR continuation run that used the same `--save-path`. The trained model weights are gone.

**Attempt 3 (COMPLETE)**:
- Log: `scaling_20L_2M_lr54.log`
- Config: **lr=5.4e-5** (= 7e-5 × sqrt(12/20), depth-scaling heuristic)
- Resumed from `scaling_20L_H1024.pth` (step 200000 of 200k search, with optimizer state)
- Note: first 200k steps trained at lr=7e-5, remaining at lr=5.4e-5
- Stopped at step ~2,286,000 (manually, gap plateaued)
- **Peak gap**: **0.2033**
- Final gap: ~0.198
- Step/s: 4.4
- Permanent checkpoint: `scaling_20L_2M_lr54_FINAL.pth` (972M)
- Save path: `scaling_20L_2M_lr54.pth` (separate to prevent overwrite)

**20L Recovery (GRU h128 l2, MSE, 20k epochs, 4 AR + 4 MA)**:
- Backbone: `scaling_20L_2M_lr54_FINAL.pth`
- Log: `scaling_search_logs/recovery_20L_gru_h128_l2.log`
- Best val loss: 0.020739 @ epoch 11159
- Mean AR Error: 0.0142, Mean MA Error: 0.0148, Total: 0.0290
- **Improvement: 6.77x**
- Sign Agreement AR: 92.0%, MA: 90.8%
- Correlation AR: 0.929, MA: 0.929

### Aborted/failed scaling runs

| Run | Reason | Steps before abort |
|-----|--------|--------------------|
| scaling_12L_H1280 | Killed to redirect compute | 3 |
| scaling_20L_H1024_2M (from scratch) | Training collapsed (gap stayed at 0) | 127,000 |
| scaling_20L_H1024_lowlr | Wrong resume source (_best was early checkpoint, gap 0.15→0 on resume) | 125,000 |

---

## 6. Summary of Compute Budget

| Stage | Experiments | Wall-clock time |
|-------|-------------|-----------------|
| Phase 1 encoder search | 5 runs × 50k | ~3.1 h |
| Phase 2 transformer search | 7 runs × 50k | ~7.3 h |
| Phase 4 full 500k | 1 run × 500k | ~20.2 h |
| 12L extension to 2M | 2 resume segments | ~81 h |
| Recovery search (5 phases) | 47+ runs | ~18.3 h |
| Scaling 200k comparison | 3 runs × 200k | ~31 h |
| 20L 2M (Attempt 2, lost) | 1 run × 2M | ~126 h |
| 20L 2M retraining (lr=5.4e-5) | 1 run × 2.3M | ~145 h |
| 20L recovery training | 1 run × 20k epochs | ~1 h |
| **TOTAL approx** | | **~433 h (~18 days)** |

---

## 7. Key Questions for Paper

- **Was a 4L or 8L variant trained?** No. All Phase 1/2 used 6 layers. Only 12L, 16L, 20L were tested at H=1024 in the scaling search.
- **What is the lightest successfully trained config?** Phase 1 E4 (gru, 6L, H=512): 13.7M params, 50k steps, peak gap 0.115.
- **Does gap keep climbing with more training?** Yes, logarithmically. 12L: 0.186 @ 500k → 0.203 @ 2M (+9%). 20L plateaued at 0.203 by 2.3M.
- **Depth vs width scaling**: Only depth was tested systematically due to LR/muP concerns. 20L matches 12L peak gap (0.203) but took ~45% more wall time. Depth alone gives diminishing returns with standard init.
- **Did adding depth improve recovery?** No. 20L recovery = 6.77x vs 12L's 6.96x, at matched peak gap (0.203). More depth at same gap slightly hurts recovery.
- **Is gap correlated with recovery?** Yes. V1 gap=0.105 → 6.59x, V2 gap=0.203 → 6.96x. Higher gap → better recovery across architectures. But within the same gap, shallower is better for recovery.

---

## 8. File Paths (on elisa `~/workspaces/contrastive-forecasting/`)

### Backbone logs/results (root dir)
- Phase 1: `arch_search_E{1..5}_*_results.json`
- Phase 2: `arch_search_T{1..7}_*_results.json`
- Phase 4: `arch_search_phase4.log`, `arch_search_phase4_gru_ffn4x_H1024_results.json`
- 12L 2M: `v2_2M_training_resumed.log`, `v2_2M_final.log`, `arch_search_v2_2M_final_results.json`

### Scaling search: `scaling_search_logs/`
- `scaling_12L_H1024_baseline.log`, `.pth`, `_best.pth`, `_optimizer.pth`, `_best_optimizer.pth`
- `scaling_16L_H1024.log`, `.pth`, `_best.pth`, `_optimizer.pth`, `_best_optimizer.pth`
- `scaling_20L_H1024.log`, `.pth`, `_best.pth`, `_optimizer.pth`, `_best_optimizer.pth` (the 200k search)
- `scaling_20L_H1024_2M_resumed.log` (the completed 2M, checkpoint lost)
- `scaling_20L_2M_lr54.log`, `.pth`, `_best.pth` (ongoing retraining)

### Recovery search: `recovery_search_logs/`
- `phase{1..5}_stdout.log` — driver scripts
- `recovery_search_p{1..4}_*.log` / `.pth` — individual runs (47+ files each)
- `recovery_p5_*.log` / `.pth` — Phase 5 full training
- `recovery_fixed_*.log` / `.pth` — post-bug-fix reruns
- `recovery_loss_*.log` / `.pth` — loss sweep on best arch
- `recovery_v2_*_dim2*.log` / `.pth` — dim 2x2 experiments
