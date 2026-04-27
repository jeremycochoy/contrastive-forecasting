# exp_span_sweep_real

**What**: RevEWMNorm span sweep on pure real data (mix=0.0,
base-bundles HF). Spans tested: 32, 64, 128, 256. 4 backbones × 30k
(originally 20k — see REPORT) steps. Backbones-only (no qhead, no
downstream eval).

**When**: Late April 2026.

**Status**: Partial. Loss is U-shaped at span=128; gap decreases
monotonically with span. The two metrics disagree on the optimum, and
20k steps may be insufficient — open question (see REPORT). The synth
counterpart (`exp_span_sweep_synth`) showed the optimum continues to
span=512 on in-distribution data.

**Run script**: `run.sh` (formerly `run_span_sweep_real.sh` at repo
root).

**Code referenced**: `../freq-embedding/scripts/train.py`.

**Bug caught and fixed during this run**:
`create_mixed_periodic_dataloader(mix_ratio=0.0)` short-circuited to
`create_hf_dataloader` which doesn't yield freq_ids, crashing
`train.py::main` with "too many values to unpack" when freq_emb_dim>0.
Fixed in `src/dataloader.py`.
