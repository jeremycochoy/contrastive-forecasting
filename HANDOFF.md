# Session Handoff — 2026-04-22

## What just finished (feat/v3b-continuation, PR merged)

### R1v3b eval on 120k-step v3b backbone
- v3b backbone trained from scratch on `base-bundles` to step 120k (target was 500k; shelved early — see next section).
- R1v3b head: 30k steps, forecaster reconstruction, W=16, GRU h=128 l=2, MSE.
- GIFT-Eval B4, 97 configs, C=1.

**Result: aggregate GM-Relative MASE = 1.1865.**

Reference points:
| Model | GM-Rel MASE |
|---|---|
| Sundial | 0.673 |
| TimesFM | 0.680 |
| PatchTST | 0.762 |
| Chronos | 0.786 |
| Moirai | 0.809 |
| Naive (baseline) | 1.000 |
| **v2 R1 (500k backbone, 2026-04)** | **1.168** |
| **v3 R1v3 (v2-resumed, 500k, base-bundles)** | **1.188** |
| **v3b R1v3b (from-scratch, 120k, base-bundles) — THIS RUN** | **1.186** |

v3b-120k ≈ v3-500k ≈ naive overall. **Still worse than v2**, despite training on real-data bundles. The comparison is not clean because v3b only ran 120k/500k, but the failure pattern is now very concrete.

### Failure pattern: periodic datasets
Biggest deficits (our MASE vs seasonal-naive):

| Dataset | Ours | Seasonal-Naive | Ratio |
|---|---|---|---|
| m4_hourly/H/short | 5.22 | 1.19 | 4.38× worse |
| solar/10T/medium | 3.15 | 0.93 | 3.40× worse |
| solar/10T/long | 2.08 | 0.87 | 2.39× worse |
| solar/H/short | 2.07 | 0.95 | 2.18× worse |
| ett2/W/short | 1.64 | 0.78 | 2.11× worse |
| ett1/15T/short | 1.78 | 0.93 | 1.91× worse |

Every one of these is a strongly-seasonal dataset. Wins are concentrated on non-periodic datasets (hierarchical_sales, hospital, sz_taxi, jena_weather short horizons).

This directly supports the working hypothesis: **synthetic + base-bundles training data lacks real-world periodic structure**, so the model never learns to extrapolate seasonality.

### Artifacts landed in the repo
- `results/R1v3b/all_results.csv` — full 97-config MASE breakdown
- `results/R1v3b/summary.txt` — aggregate + per-config, with leaderboard reference
- `results/R1v3b/v3b_head_eval.log` — elisa training log for the 30k head + eval

Checkpoints (NOT in git — gitignored `*.pth`):
- `sync_v3b_final/checkpoints/tiny_v3b_r2_120k.pth` + `_optimizer.pth` (backbone, 120k)
- `checkpoints/v3b/R1v3b_best.pth` + `_optimizer.pth`, `R1v3b_final.pth`, `R1v3b_losses.csv`

### Session notes worth keeping
- Filed `jeremycochoy/vastrun-kit#296` cataloguing 7 distinct vastrun-kit reliability issues hit during the v3b backbone-training phase (gpu_name substring bug, provision hangs, rsync 300s timeout, cu124/cu128 confusion, attach-ssh idempotency, cancel-vs-billing race, ghost instances from auto-retry).
- Vast.ai budget spent ~$14 on v3b backbone; shelved at 120k because each retry after ran ≤3h before dying. Elisa used ONLY for the cheap/bounded head+eval pass — the CLAUDE.md rule still holds: no continued elisa use for long jobs (coworker queue).

## What's next (task #8, queued)

**Conditional mini-experiment — now unconditionally triggered.** The periodic-dataset failure is unambiguous.

Plan (full detail in TaskList #8):
- New dir: `experiments/periodic-synth-mix/`
- Same tiny arch (C=4 H=512 W=16 GRU×6), FROM SCRATCH, 30k steps only.
- Mixed data: 50% base-bundles HF stream, 50% on-the-fly simple synthesizer. Mix at row level inside each batch.
- Synthesizer primitives: sinusoid, square (random up/down phase), saw (random slope sign). Draw sampling step first from a real-world set (10s, 1min, 5min, 10min, 15min, 30min, 1h, 1d, 1w), then draw the process period larger than the step, aiming for balanced samples-per-period coverage (~[8, 256]). With p≈0.3, multiply by `exp(±λt)` envelope capped to ~[0.1×, 10×] total gain.
- No additive noise.
- **Before training**: save ~100 synthetic plots and eyeball them. Make sure they look like things a seasonal-naive baseline would predict well and that values stay safely in float32.

**Expected signal**: 30k steps is short; we need a CONTROL. Train a paired 30k-step v3b-base (same from-scratch, no synthetic mix) so we can say "adding 50% synthetic periodic data at matched compute changes the periodic-dataset MASE by X". Without that control, the 30k run alone is under-trained and uninterpretable.

**Compute**: vast.ai (use vastrun-provision/sync/run via the new split CLI). Lessons from #296:
- `--gpu-model "RTX 4090"` (SPACE, not underscore).
- `--on-demand` + `--max-bid ~0.8` to avoid preemption.
- If `vastrun-sync --resume-from` times out at 300s on the optimizer, fall back to manual `scp -P <port> <file> root@<host>:/workspace/app/checkpoints/`.
- cu128 torch wheels only — DON'T use cu124 (fails with CUDA error 804 on driver ≥565).
- Destroy stale instances yourself with `yes | vastai destroy instance <id>` if `vastrun-cancel` seems to lie.
- Before each run, add a step that CONFIRMS `torch.cuda.is_available()` before spending time downloading data.

## Currently active infrastructure
- Branch: `experiments` (PR for this session merged).
- No running vast.ai instances. No elisa jobs.
- Vast balance: ~$10. Head+eval on elisa cost $0.

## Useful commands for next session
```bash
# Verify checkpoints still on disk
ls -la sync_v3b_final/checkpoints/tiny_v3b_r2_120k*
ls -la checkpoints/v3b/R1v3b*

# See the full eval breakdown
less results/R1v3b/summary.txt
```
