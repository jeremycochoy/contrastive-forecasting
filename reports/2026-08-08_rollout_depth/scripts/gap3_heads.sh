#!/bin/bash
# #373 review gap 3 — the two heads and the two evals of `G_B1_k0_aw4`.
#
# `gap_worker.sh` runs a row's two heads one after the other, and each head
# is followed by its own 97-config eval on the cores. That order idles the
# card for the whole of the first eval:
#
#   student head (GPU)  -> student eval (CPU) -> teacher head (GPU) -> teacher eval (CPU)
#
# The card is free for the 1.5 h the student eval spends on the cores, and
# the teacher head that wants it waits. This runs both `head_eval_bb.sh`
# calls at once instead:
#
#   student head (GPU)  -> student eval (CPU)
#                          teacher head (GPU) -> teacher eval (CPU)
#
# Nothing here has to sequence the heads. `head_eval_bb.sh` already takes an
# exclusive `flock` on a per-card file for the head-training half and drops
# it before the eval, so two calls on one card train one at a time and their
# evals overlap. The saving is one eval, ~1.5 h.
#
# BOTH HEADS TRAIN ON ONE CARD, and that is not a choice. elisa's card 0
# holds another session's 15 GB notebook and reports 1.6 GB free, under the
# 6 GB a head needs. Card 1 carries the backbone this row is waiting on,
# and frees ~8 GB when it exits: one head, not two.
#
# Everything that touches the measurement is `gap_worker.sh`'s, unchanged:
# 15,000 head steps, head seed 20260722, `--grad-clip 1.0`, batch 256,
# lr 1e-3, forecast-len 16, then 97 GIFT-Eval configs at strategy B4. The
# pair this control is read against, B1 at k = 0 and at k = 3, ran that
# protocol on this machine.
#
# Usage: nohup bash scripts/gap3_heads.sh > results/gap3_heads.out 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$HERE/../results"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
ID="${ID:-G_B1_k0_aw4}"
NAME="bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0_aw4"
CK="$CF373_ROOT/$NAME/${NAME}_40k.pth"
HEAD_GPU="${HEAD_GPU:-1}"
DEADLINE="${DEADLINE:-28800}"   # 8 h from launch: 2.6 h of backbone left, then heads and evals

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [gap3] $*" | tee -a "$RES/gaps_driver.log"; }

# The backbone. It is a separate process with its own parent, so this waits
# on the file it writes rather than on a PID.
waited=0
until [ -f "$CK" ]; do
  if [ "$waited" -ge "$DEADLINE" ]; then
    log "TIMEOUT after ${waited}s: no checkpoint at $CK"; exit 1
  fi
  if ! pgrep -f "run-name $NAME" >/dev/null 2>&1; then
    log "ABORT: the backbone process is gone and $CK does not exist"; exit 1
  fi
  [ $(( waited % 1800 )) -eq 0 ] && log "waiting for the backbone, ${waited}s so far"
  sleep 60; waited=$(( waited + 60 ))
done
log "backbone in hand: $(basename "$CK") $(stat -c %s "$CK") bytes"

one_enc(){ # <student|teacher>
  local enc="$1" rc
  for attempt in 1 2; do
    log "HEAD START $ID $enc (attempt $attempt, gpu $HEAD_GPU)"
    BB_GPU="$HEAD_GPU" HEAD_VRAM_MIB="${HEAD_VRAM_MIB:-6000}" \
    EVAL_SHARDS="${EVAL_SHARDS:-6}" CF393_EVAL_SLOTS="${CF393_EVAL_SLOTS:-5}" \
      bash "$HERE/head_eval_bb.sh" "${ID}_bb40k_${enc}" "$CK" "$enc" 15000
    rc=$?
    log "HEAD END   $ID $enc rc=$rc"
    # head_eval_bb.sh skips a head whose final checkpoint exists and its
    # eval resumes per shard, so a retry costs only what did not finish.
    [ "$rc" -eq 0 ] && return 0
    [ "$attempt" -eq 2 ] && log "GIVING UP on $ID $enc"
  done
  return 1
}

one_enc student &
S=$!
one_enc teacher &
T=$!
wait "$S"; src=$?
wait "$T"; trc=$?
log "student rc=$src  teacher rc=$trc"
log "student $(cat "$WT/reports/2026-08-08_rollout_depth/results/score_${ID}_bb40k_student.txt" 2>/dev/null || echo MISSING)"
log "teacher $(cat "$WT/reports/2026-08-08_rollout_depth/results/score_${ID}_bb40k_teacher.txt" 2>/dev/null || echo MISSING)"
log "gap 3 done"
