#!/bin/bash
# #373 — start the four A1/B3 reproduction evals, one process each.
#
# The heads land on the box one after another, about 20 min apart for a
# 15,000-step head and 40 min for a 30,000-step one. Each worker below waits
# only for its own head, so eval three starts while head four still trains.
# Serial, the four cost about 5.6 h; overlapped, about 3 h.
#
# Usage: repro_eval_all.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
R2=/home/jupyter/cf373_r2

A1_40="$R2/A1/sync/arm5_combab_alignS/leg_40k/cf393_arm5_combab_alignS_cf373k3_40k.pth"
A1_100="$R2/A1/sync/arm5_combab_alignS/leg_100k/cf393_arm5_combab_alignS_cf373k3_100k.pth"
B3D="$R2/B3/sync/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
B3_40="$B3D/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3_40k.pth"
B3_100="$B3D/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3_r2_100k.pth"

start(){ # <tag> <stop_k> <backbone>
  local tag="$1"
  if pgrep -f "repro_eval_one.sh $tag " >/dev/null 2>&1; then
    echo "  $tag already running"; return
  fi
  setsid nohup bash "$HERE/repro_eval_one.sh" "$@" \
    >"$RES/repro_eval_$tag.out" 2>&1 </dev/null &
  echo "  $tag started pid $!"
}

echo "[$(date '+%m-%d %H:%M:%S')] repro evals, four workers"
start A1rep_k3_bb40k_student   40  "$A1_40"
start B3rep_k3_bb40k_student   40  "$B3_40"
start A1rep_k3_bb100k_student 100  "$A1_100"
start B3rep_k3_bb100k_student 100  "$B3_100"
