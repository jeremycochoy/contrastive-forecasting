# Encoder+forecaster q-head plan

## Backbone

`/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_tau_0_10_50k_FINAL.pth` — GRU patch embed → 6 causal encoder layers → 6 causal forecaster layers, H=384, n_heads=6, ffn_mult=4, RevEWMNorm span=128, freq+seasonality emb dim=3, τ=0.10. `load_state_dict` matches all keys (22.06M params).

## Budget: 30k (vs R9_E13's 60k)

R9_E13 losses CSV (`sync_qhead_beta_rd9/checkpoints/R9_E13_*_losses.csv`, ±500-step window means):

| step | 10k | 20k | **30k** | 45k | 60k |
|---|---:|---:|---:|---:|---:|
| loss | 0.19435 | 0.19363 | **0.19310** | 0.19270 | 0.19220 |

Δ(30k→60k) = 0.0009 = 0.5 % reduction in the second half — training loss is essentially flat past 30k. Cutting to 30k halves wall-clock on the (slower) RTX 4090 with no real signal loss. Cosine schedule compressed to 30k (2k warmup, decay to lr×0.1).

## Code changes (worktree only, uncommitted)

Both edits mirror the existing `freq_emb_dim`/`learnable_tau` auto-detect: read state_dict keys matching `transformer.encoder_layers.<N>.*`, set `BACKBONE_CONFIG["num_encoder_layers"] = max(N)+1`. Falls back to 0 when absent.

- `experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py` lines 357–372.
- `experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py` lines 459–473.

Smoke-tested: backbone builds + loads + does a forward pass with `num_encoder_layers=6` auto-set on the new checkpoint.

## Run commands

```bash
# Q-head training (GPU 1, ~30k steps, expandable_segments)
bash experiments/2026-05-10_exp_encoder_forecaster/scripts/run_qhead_training.sh

# GIFT-Eval triage (~5 min, 11-config subset)
bash experiments/2026-05-10_exp_encoder_forecaster/scripts/run_gift_eval_triage.sh
```

Compare the triage GM-MASE vs R9_E13's **0.990 triage / 1.029 full**. Decide on the full ~6 h eval after the triage number lands.
