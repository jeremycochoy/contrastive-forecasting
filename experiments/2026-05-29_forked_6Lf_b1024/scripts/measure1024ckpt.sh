#!/bin/bash
set -uo pipefail
HERE=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/scripts
R=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/results
export MEMPROBE="$HERE/memprobe" PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
echo "### β @1024 single-GPU CKPT (30 steps, GPU1) ###"
bash "$HERE/smoke.sh" beta 0.10 1024 30 1
echo "### allt @1024 single-GPU CKPT chunk-loss=2 (30 steps, GPU1) ###"
bash "$HERE/smoke.sh" alltime 0.10 1024 30 1 2
echo "===== MEMPROBE β ====="; grep '\[memprobe\]' "$R/smoke_beta_b1024.log" | tail -1
echo "===== MEMPROBE allt ====="; grep '\[memprobe\]' "$R/smoke_alltime_b1024.log" | tail -1
echo "===== β sps/timing ====="; grep -E 'sps|timing:' "$R/smoke_beta_b1024.log" | tail -3
echo "===== allt sps/timing ====="; grep -E 'sps|timing:' "$R/smoke_alltime_b1024.log" | tail -3
echo "===== OOM? ====="; grep -ciE 'out of memory' "$R/smoke_beta_b1024.log" "$R/smoke_alltime_b1024.log"
echo "ALL_DONE_1024CKPT"
