#!/usr/bin/env bash
# Autonomous solver for the stop-grad follow-up card, in the contrastive-forecasting
# repo, on model `fable`. Built on the rnd claude_solve_issue.sh pattern (sources the
# shared standards preamble) but card-driven and pointed at the cf repo.
set -euo pipefail
source /tmp/rnd-328/scripts/claude_standards_preamble.sh   # RND_ROOT + standards_system_prompt
CF_ROOT=/home/jupyter/contrastive-forecasting
cd "$CF_ROOT"

read -r -d '' OPERATING <<'EOF' || true
You are a fully autonomous agent running a follow-up experiment end-to-end in the
contrastive-forecasting repo (your current working directory). The user is NOT
available and will not answer — make reasonable decisions and proceed; never ask
questions. Operating rules:
- First read this repo's CLAUDE.md, README.md, CODING_STYLE.md, docs/report_writing.md
  and docs/ISSUE_WORKFLOW.md, and follow them. Work on a feature/ branch in a worktree;
  drive the result to a report PR via the repo's issue/PR workflow.
- The single code change is the stop-gradient described in the goal. Keep everything
  else in the #328 best recipe identical, INCLUDING --subtract-contrastive-floor.
- elisa GPUs are SHARED with other users and agents. Before launching any heavy
  compute (backbone training, GIFT-Eval) check nvidia-smi and never preempt jobs
  already running. If no GPU is free, do every no-GPU prep step first, then WAIT:
  sleep on the 1-hour wake-up loop and re-check each hour. Waiting hours or days is fine.
- The report focuses on the science, not the journey; high human readability. It MUST
  include downstream GM-Relative MASE vs the reference at both head sizes (2L and 6L)
  and the log-log training-dynamics curves of both models (with vs without stop-grad).
- Run the PR checklist and the report checklist before yielding; re-run the report
  checklist after any change to the report.
- Run a mandatory 1-hour wake-up loop until the goal is met.
EOF

GOAL='A report PR in this (contrastive-forecasting) repo implements and evaluates the follow-up card below, is open and ready to merge, every reviewer comment is addressed, and both the PR checklist and the report checklist pass in full.

CARD — L3+nobn+triplet with stop-gradient on the encoder positive term:
- Why: the #328 best arm (L3 + no-bottleneck + triplet) reliably beats base at full training. Test whether a stop-gradient on the encoder side of the positive pair changes the learning dynamics and downstream transfer (SimSiam/BYOL-style target stop-grad).
- What: re-run the #328 best recipe (L3 + no-bottleneck + triplet; keep --subtract-contrastive-floor) with a single change. In the positive similarity term sim(h_{t+1}, f_{t+1}), stop-gradient the encoder term: sim(stopgrad(h_{t+1}), f_{t+1}). Apply the stop-gradient everywhere h_{t+1} appears in this positive term: BOTH numerator and denominator.
- Report: downstream GM-Relative MASE vs the reference (#328 L3+nobn+triplet, no stop-grad) at both head sizes (2L and 6L); plus the log-log training-dynamics curves of both models (with vs without stop-grad).'

exec claude \
  --model fable \
  --append-system-prompt "$(standards_system_prompt)

$OPERATING" \
  --effort max \
  --dangerously-skip-permissions \
  --add-dir /home/jupyter/workspaces/contrastive-forecasting \
  --add-dir "$RND_ROOT" \
  --name "Contrastive Forecasting - Stop grad the encoder ground truth" \
  "/goal $GOAL"
