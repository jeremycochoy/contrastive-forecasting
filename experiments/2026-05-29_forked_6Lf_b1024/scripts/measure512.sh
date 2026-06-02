#!/bin/bash
set -uo pipefail
HERE=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/scripts
export MEMPROBE="$HERE/memprobe"
R=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/results
echo "### measuring β@512 (12 steps) ###"
bash "$HERE/smoke.sh" beta 0.10 512 12 1
echo "### measuring allt@512 chunk=4 (12 steps) ###"
bash "$HERE/smoke.sh" alltime 0.10 512 12 1 4
echo "===== MEMPROBE β ====="; grep '\[memprobe\]' "$R/smoke_beta_b512.log" | tail -2
echo "===== MEMPROBE allt ====="; grep '\[memprobe\]' "$R/smoke_alltime_b512.log" | tail -2
echo "===== β sps ====="; grep -oE '\] .*sps' "$R/smoke_beta_b512.log" | tail -3
echo "===== allt sps ====="; grep -oE '\] .*sps' "$R/smoke_alltime_b512.log" | tail -3
echo "===== allt timing ====="; grep -E 'timing:' "$R/smoke_alltime_b512.log" | tail -2
echo "ALL_DONE"
