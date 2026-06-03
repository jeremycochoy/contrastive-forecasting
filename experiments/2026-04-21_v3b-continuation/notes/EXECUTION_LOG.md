# v3b execution log (operational — not part of [../v3b-continuation.md](../v3b-continuation.md))

Operational record of the v3b backbone run, kept for traceability. None of this would
belong in the report if the run were re-done cleanly.

## Preemption chain

The backbone was trained on Vast.ai across **seven** successive instances (the local archive
dir names `v3b-train_3531…` / `v3b-final_3535…` are those contract IDs). Each retry after the
first few ran ≤ 3 h before being preempted, so the run was **shelved at ~120k of the 500k
target** rather than chased to completion. ~$14 of Vast.ai credit spent on the backbone phase.

The head training (30k steps) and the GIFT-Eval pass were run on **elisa** only — bounded,
cheap ($0) work — consistent with the project rule not to hold elisa for long jobs.

## vastrun-kit reliability bugs

Seven distinct vastrun-kit issues were hit during the backbone phase and filed together as
**`jeremycochoy/vastrun-kit#296`**: a `gpu_name` substring bug, provision hangs, a 300 s rsync
timeout on the optimizer file, cu124/cu128 image confusion, attach-ssh idempotency, a
cancel-vs-billing race, and ghost instances left by auto-retry.

Mitigations adopted for later runs: `--on-demand` + a bid ceiling to avoid preemption; manual
`scp` fallback when `--resume-from` times out; confirm `torch.cuda.is_available()` before
downloading data.

## Lost artifacts

The backbone checkpoints (`tiny_v3b_r2_120k.pth` + optimizer) and the backbone loss CSVs were
gitignored and never committed; the on-disk archive dirs are now empty. Only the downstream
eval outputs survive, copied into [../results/R1v3b/](../results/R1v3b/) (they were originally
filed under `2026-04-27_periodic-synth-mix/results/R1v3b/`, where the follow-up experiment
cites them as the predecessor baseline).
