# encoder-forecaster v2 — dropkey + new metric + full budget

GRU patch encoder → 6L causal transformer encoder → 6L causal transformer forecaster, all HPs identical to the τ=0.10 baseline arm (`experiments/2026-05-08_exp_tau_sweep/` τ=0.10), plus:

- backbone: `--num-encoder-layers 6 --encoder-dropkey 0.7 --total-steps 50000 --amp-dtype bf16`.
- q-head: R9_E13 recipe + `--amp-dtype bf16`, 30k steps.
- gate on `auc_bt` from `retrieval_auc_topk_batch_temporal`, not legacy `auc`.
- triage gate: GM-Relative MASE < 1.0 → full GIFT-Eval; else stop.
