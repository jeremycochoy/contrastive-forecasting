# 2026-04-27_exp_revin_repro

**What**: Reproduction of the previous-session ablation #28 — RevIN as a
drop-in replacement for RevEWMNorm on the fe+mu backbone, mix=0.5
(50% HF base-bundles + 50% periodic synth), 30k bb + 30k qhead.

**When**: Late April 2026 (this session).

**Status**: Success (reproduction). Backbone gap=0.469, qhead loss=0.052
matched the previous-session run within noise. Synth grid plot
generated (`plots/synth_qhead_grid_revin.png`).

**Run script**: not preserved as a `run.sh` — the run was launched via
inline bash. Setup parameters are documented in `REPORT.md` and the
shared trainer at `../freq-embedding/scripts/train.py` was invoked
with the flags listed there.

**Code referenced**: `../freq-embedding/scripts/train.py`,
`../freq-embedding/scripts/plot_synth_qhead.py`. See repo-root
README on shared scripts.
