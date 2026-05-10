# encoder-forecaster v2 — dropkey + new metric + full budget

Same backbone as `2026-05-10_exp_encoder_forecaster_failed` (PR #270); apply the three landed fixes and run to budget.

- backbone: `--encoder-dropkey 0.7 --total-steps 50000 --amp-dtype bf16`; all other HPs unchanged.
- q-head: R9_E13 recipe + `--amp-dtype bf16` (PR #264), 30k steps.
- gate on `auc_bt` from `retrieval_auc_topk_batch_temporal` (PR #272), not legacy `auc`.
- triage gate: GM-Relative MASE < 1.0 → full GIFT-Eval; else stop.
