# Pipeline, code, and artifact inventory (operational)

Operational/journey detail moved out of
[`../exp_qhead_improvements.md`](../exp_qhead_improvements.md) per the
report standard (science, not journey).

## Pipeline summary

- 9 rounds of head experiments (R1–R9), plus the R10 backbone-proxy
  side-investigation (see [`BACKBONE_DIAGNOSTICS.md`](BACKBONE_DIAGNOSTICS.md)).
  ~$11 of a $21.98 vast.ai budget; the budget topped out around R8, and
  R7_E9 was truncated at step ~85k by a vast spot-instance preemption.
- Code PRs: WSD/cosine schedules + AdamW HP flags (#126); linear-probe
  heads (#126); transformer head (#130); `--head-causal` flag for the
  bidirectional variant (#136); explicit eval env-var overrides (#139);
  Gaussian NLL head (#141). Plus launcher PRs #127, #128, #129, #131,
  #133, #134, #137, #140, #142.
- 88 unit tests in `tests/test_forecasting_head.py` covering each new
  head class (shape, param-count, causal/bidir mask correctness, B4
  strategy roundtrip, NLL loss correctness).
- All head-type/recipe combinations evaluated on the same 11-config
  triage set with head architecture auto-detected from the state dict.

## Artifacts

- Launcher scripts: `../scripts/` (per-round `run_round*.sh` launchers +
  the eval driver `run_eval_elisa.sh` with explicit `FL=`, `STRATEGY=`,
  `HEAD_CAUSAL=` env-var overrides).
- Triage + full results: `../results/` (per-run `summary.txt` +
  `all_results.csv`). Note: the legacy GRU `#10` **full** baseline
  (GM-MASE 1.1828) lives in the prior `#10` RESUME50k report, not in
  this dir — see [`CANDIDATES.md`](CANDIDATES.md). The full-eval runs
  committed here are `R5_E7_..._full` (28-config partial; `all_results.csv`
  only, no aggregate summary) and `R9_E13_..._full` (97-config, full
  `summary.txt`).
- Best head (R5_E7): `sync_qhead_beta_rd5/checkpoints/R5_E7_xfmr12L_quant_moirai_cosine_60k_FINAL.pth`
  (80 MB `.pth`, not in git).
- Candidate ledger with per-round rationale: [`CANDIDATES.md`](CANDIDATES.md).
- Triage-subset bias analysis: [`TRIAGE_NOTE.md`](TRIAGE_NOTE.md).
- Code (merged on `experiments`): `src/forecasting_head.py`,
  `experiments/2026-04-13_gift-eval/scripts/{train_forecasting_head.py,
  eval_gift_eval_official.py}`, `tests/test_forecasting_head.py`.
