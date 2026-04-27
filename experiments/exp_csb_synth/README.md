# exp_csb_synth (in-flight)

**What**: cosine_similarity_batch loss (paper-matching, re-includes
within-time negative `h[b, t-1, c] vs h[b, t, c]`) on the best arm
from the span sweep: fe+mu mix=1.0 synth-only, 30k bb + 30k qhead,
ewma span=512. Single-axis ablation vs the established
`cosine_similarity_batch_no_time_neg` loss.

**When**: started late April 2026 (in-flight at time of writing).

**Status**: in-flight. Output will arrive later — placeholder REPORT
documents the design.

**Run script**: `run.sh` — copy of `/tmp/run_wtn_v2.sh` from the
remote vast.ai instance at SSH ssh2.vast.ai:16198 (snapshot at launch
time). `run_v1.sh` is the earlier variant (`run_within_time_neg.sh`
at repo root) that used the deprecated `cosine_similarity_batch_with_within_time_neg`
loss-shape — superseded by v2.

**Code referenced**:
- `../freq-embedding/scripts/train.py`
- `../gift-eval/scripts/train_forecasting_head.py`
- `../freq-embedding/scripts/synth_eval.py`

**Where the output will go**:
- Eval CSV row appended to `../_aggregate/results/synth_eval.csv`.
- Plots in `plots/`, checkpoints (not tracked in git) in repo
  `checkpoints/`.
