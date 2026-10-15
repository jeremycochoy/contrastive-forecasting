#!/bin/bash
# #425 — the five B4 forecast scores that the overlay pairs with a
# reconstruction score. Each one is a kept checkpoint of an old-data run that
# has no forecast score.
#
# The code and the protocol are those of the #412 B4 scores: the box folder
# /workspace/cf-412bm (937f824b), `head_eval_bb.sh`, a head of 30,000 steps,
# then the 97-config B4 score on the CPU. The predictor reads a context of
# 1,024 values, as every B4 score of #412 did.
#
# Three lanes, so at most three jobs run at the same time. `head_eval_bb.sh`
# holds one head lock for each lock directory, so each lane takes its own
# directory and the heads of the three lanes share the GPU.
#
# LNG 665k keeps the complete head that #414 trained on 09-22 with the same
# protocol. Its score stopped on a shard merge (121 rows for 97 configs). The
# lane links that head into a new directory, so the score starts clean and no
# old file changes.
#
# Usage, on the box:
#   nohup setsid bash forecast_scores.sh >>/workspace/results/cf-425/forecast_scores.log 2>&1 &
set -uo pipefail

CODE="${CF425_FC_CODE:-/workspace/cf-412bm}"
CK="${CF425_CK:-/workspace/ckpt}"
ROOT="${CF425_FC_ROOT:-$CK/cf-425/forecast}"
RES="${CF425_RES:-/workspace/results/cf-425}"
RUNNER="$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh"
A=arm6_v2_combab_alignT
P=k3_r100_09_lr56_fix09_dec10k
N=cf393_${A}_cf373k3_cf412_${P}

# <lane> <arm suffix> <stop in thousands> <backbone checkpoint>
JOBS="
0 cos665k 665 $CK/${P}_cos665k/$A/leg_665k/${N}_cos665k_665k.pth
0 cos200k 1140 $CK/${P}_cos200k/$A/leg_1330k/${N}_cos200k_1140k.pth
1 lr30x 665 $CK/${P}_lr30x/$A/leg_665k/${N}_lr30x_665k.pth
1 lr10xb 420 $CK/${P}_lr10xb/$A/leg_665k/${N}_lr10xb_420k.pth
2 lr100x 1080 $CK/${P}_lr100x/$A/leg_1330k/${N}_lr100x_1080k.pth
"
OLD_LNG_HEAD="$CK/${P}_cos665k/eval/${P}_cos665k_bb665k_h30k_student/qhead_${P}_cos665k_bb665k_h30k_student_s20260722_final"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 forecast] $*"; }

[ -f "$RUNNER" ] || { log "ABORT: no runner at $RUNNER"; exit 2; }
mkdir -p "$ROOT" "$RES"

# The old LNG head, linked into the directory that head_eval_bb.sh reads.
# A link adds a name and changes no byte of the old file.
link_lng_head(){
  local tag="${P}_cos665k_bb665k_h30k_student" dir
  dir="$ROOT/eval/$tag"
  mkdir -p "$dir"
  for f in "$OLD_LNG_HEAD.pth" "${OLD_LNG_HEAD}_encoder_source.txt"; do
    [ -s "$f" ] || { log "ABORT: no old LNG head file $f"; return 1; }
    [ -e "$dir/$(basename "$f")" ] || ln "$f" "$dir/$(basename "$f")" || return 1
  done
}

run_job(){  # <lane> <arm suffix> <stop k> <backbone>
  local lane="$1" arm="${P}_$2" stop="$3" bb="$4"
  local tag="${arm}_bb${stop}k_h30k_student"
  [ -s "$bb" ] || { log "lane $lane: SKIP $tag, no backbone at $bb"; return 1; }
  log "lane $lane: start $tag"
  WT="$CODE" CF373_ROOT="$ROOT" CF_RESULTS="$RES" CF_STOP_K="$stop" \
    CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" \
    GIFT_EVAL=/workspace/gift-eval-data EVAL_SHARDS="${CF425_FC_SHARDS:-3}" \
    BB_GPU=0 HEAD_VRAM_MIB=9000 GPU_GATE_LOCKDIR="/tmp/cf425_fc_lane$lane" \
    bash "$RUNNER" "$tag" "$bb" student 30000
  log "lane $lane: $tag rc=$?"
}

run_lane(){  # <lane>
  local lane="$1" l arm stop bb
  mkdir -p "/tmp/cf425_fc_lane$lane"
  while read -r l arm stop bb; do
    [ "$l" = "$lane" ] || continue
    run_job "$lane" "$arm" "$stop" "$bb"
  done <<<"$JOBS"
}

link_lng_head || exit 3
log "start: 5 jobs in 3 lanes, code $CODE"
for lane in 0 1 2; do
  run_lane "$lane" &
  sleep 20
done
wait
log "FORECAST_SCORES_END"
