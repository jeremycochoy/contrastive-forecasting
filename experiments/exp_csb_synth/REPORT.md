# exp_csb_synth — cosine_similarity_batch + RevIN follow-up

**Status: results pending — run in flight at time of writing.**

## Why

After the synth span sweep, the established `cosine_similarity_batch_no_time_neg`
loss is what every reported arm has used. The paper-matching variant
`cosine_similarity_batch` re-introduces the within-series, within-channel
time negative `h[b, t-1, c] vs h[b, t, c]` that was removed during ARMA-era
tuning (lag-1 autocorrelation made it counter-productive on ARMA, but
periodic data is different). See `../freq-embedding/FOLLOWUP.md` for
the full proposal.

This is the single-axis ablation on the otherwise-frozen best arm:
fe+mu, mix=1.0 synth-only, 30k bb + 30k qhead, ewma span=512.

## Setup

| Knob | Value |
|---|---|
| Steps | 30k bb + 30k qhead |
| Mix ratio | 1.0 |
| Freq emb | dim=3, mixup=0.3 |
| Reversible norm | RevEWMNorm span=512 |
| Loss | `cosine_similarity_batch` (re-includes within-time negative) |
| Backbone selector | `best_loss` (gap saturates early on synth) |
| Eval | 1024 held-out synth samples (same protocol/seed as span sweep) |

## Provenance / status notes

- `run.sh` is a copy of `/tmp/run_wtn_v2.sh` from the remote vast.ai
  instance at the time the run was launched.
- `run_v1.sh` is the earlier variant (`run_within_time_neg.sh` at repo
  root, `--loss-shape cosine_similarity_batch_with_within_time_neg`)
  that was superseded by v2. Preserved for provenance.
- Output (eval CSV row, plots, checkpoints) will land in this dir's
  `results/` and `plots/` once the remote run finishes; until then this
  REPORT and README record the design and what to expect.

## Results

**Pending.** Will be filled in once remote run completes.

## Artefacts (expected once run completes)

- Backbone: `checkpoints/tiny_femu_span512_synth30k_csb_FINAL.pth`
  (won't be tracked in git; 80MB).
- Qhead: `checkpoints/R1q_femu_span512_synth30k_csb_FINAL.pth` (won't
  be tracked).
- Eval CSV row: "fe+mu @ 30k span=512 +cosine_similarity_batch" in
  `../_aggregate/results/synth_eval.csv` (append).
- Synth grid plot: TBD if produced.
