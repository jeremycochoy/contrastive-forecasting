#!/bin/bash
# #412 pass 2 — one card's backbone legs, in order, and no head.
#
# WHY NOT `phase1.sh`. That script trains a head and runs its 97-config
# GIFT-Eval between two legs of the same arm. The evaluation is 2.9 hours of
# CPU, and the card sits idle through it. R1 is two legs of one arm, so
# `phase1.sh` would idle a 4090 for about 4.6 hours between them.
#
# WHY NOT `queue_backbones.sh`. That queue starts its arms TOGETHER on one
# card. Pass 2 runs R2, then R4, then R3 on the second card, in that order,
# because that is the order of PR #413's plan and because a k = 32 arm beside
# a k = 3 arm leaves under 3 GB on a card that two other projects share.
#
# WHAT IT DOES. It takes an ordered list of `<arm>:<stop>` legs. For each leg
# it waits for the free memory that arm needs, runs `run_arm.sh`, and WAITS for
# it to end before the next leg. `head_sweep.sh` on the other card trains every
# head, so no leg waits for one.
#
# A leg whose checkpoint is on disk is a no-op, so a re-fire after a crash
# costs only what did not finish.
#
# Usage:  BB_GPU=0 CF412_LEGS="k3_r100_09_lr56:100000 k3_r100_09_lr56:200000" \
#           bash scripts/pass2_lane.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

GPU="${BB_GPU:-0}"
LEGS="${CF412_LEGS:?usage: CF412_LEGS=\"<arm>:<stop> ...\" pass2_lane.sh}"
LANE="${CF412_LANE:-pass2}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 $LANE] $*" \
  | tee -a "$CF412_RESULTS/${LANE}.log"; }

for leg in $LEGS; do
  cf412_require_arm "${leg%%:*}" || exit $?
  cf412_require_stop "${leg##*:}" || exit $?
done
log "gpu $GPU  legs: $LEGS"

failed=0
for leg in $LEGS; do
  arm="${leg%%:*}"; stop="${leg##*:}"
  # An arm that lost the contrastive task does not climb: a higher stop trains
  # the same collapse. `run_arm.sh` deletes this note when the arm runs again,
  # so the read happens here, before the leg.
  if [ -f "$(cf412_collapse_file "$arm")" ]; then
    log "$arm at $stop SKIPPED — it lost the contrastive task"
    continue
  fi
  need="$(cf412_leg_vram_mib "$arm")"
  cf412_wait_for_vram "$GPU" "$need" "backbone $arm" \
    2>&1 | tee -a "$CF412_RESULTS/${LANE}.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "$arm at $stop SKIPPED — gpu $GPU never had ${need} MiB free"
    failed=$(( failed + 1 )); continue
  fi
  log "$arm -> $stop steps on gpu $GPU (needs ${need} MiB)"
  BB_GPU="$GPU" bash "$HERE/run_arm.sh" "$arm" "$stop" \
    >>"$CF412_RESULTS/${LANE}_${arm}_${stop}.log" 2>&1
  rc=$?
  if [ "$rc" -eq "$CF412_RC_COLLAPSED" ]; then
    log "$arm at $stop LOST the contrastive task — see" \
        "$(cf412_collapse_file "$arm")"
    continue
  fi
  [ "$rc" -eq 0 ] || { log "$arm at $stop FAILED rc=$rc"
                       failed=$(( failed + 1 )); continue; }
  log "$arm at $stop DONE — $(cf412_bb_ckpt "$arm" "$stop")"
done
log "lane done — $failed failure(s)"
[ "$failed" -eq 0 ] || exit 1
