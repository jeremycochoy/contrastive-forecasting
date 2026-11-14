#!/bin/bash
# Starts one head follower on elisa for the width-64 runs that E_RUNS names (full run names; their saves in
# $HP64_ROOT/<run>/ckpt). At each stop of E_STOPS: one head wave on GPU E_GPU for all these runs, then their scores
# one at a time (slot_0, one shard, the protocol 'device=cuda configs=97 filter= fbs=64'). The followers of one GPU
# share its stream lock ($HP64_ROOT/ff/locks/stream_gpu<E_GPU>.lock); all followers share slot_0. Give a (run, stop)
# pair to one follower only. The head code is the deployed copy in $HP64_ROOT/ff/code (follow_multi.sh).
#   E_RUNS="<run> ..." E_STOPS="0 10 25" E_GPU=1 E_LOG=follow_e1_<name>.log bash start_follow_elisa_runs.sh
H=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}
GPU=${E_GPU:?E_GPU: the elisa GPU of the head waves}
RUNS=""
for N in ${E_RUNS:?E_RUNS: the full run names}; do
  [ -d $H/$N/ckpt ] || { echo "no $H/$N/ckpt"; exit 2; }
  RUNS="$RUNS $N=$H/$N/ckpt"
done
cd $H/ff/code/reports/2026-10-10_frequency_family_head/scripts || exit 2
FF_BASE=$H/ff FF_MULTI_RUNS="${RUNS# }" FF_STOPS="${E_STOPS:?E_STOPS: the stops}" FF_GPU=$GPU \
  FF_BB_SHAPE="--d-model 64 --n-heads 8 --num-layers 3" FF_STREAM_LOCK=$H/ff/locks/stream_gpu$GPU.lock \
  FF_EVAL_DEVICE=cuda FF_EVAL_SLOTS=1 FF_EVAL_SHARDS=1 FF_EVAL_FORECAST_BATCH_SIZE=64 FF_WAVE_VRAM_MIB=8000 \
  FF_VRAM_WAIT=86400 \
  setsid nohup bash follow_multi.sh >>"$H/ff/results/${E_LOG:?E_LOG: the log name}" 2>&1 </dev/null &
echo "$(date -u +%FT%TZ) follow_multi.sh started (pid $!, stops $E_STOPS, GPU $GPU, log $E_LOG)"
