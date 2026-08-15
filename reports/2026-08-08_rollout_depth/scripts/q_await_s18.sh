#!/usr/bin/env bash
# #373 session 18 — block until the round's last numbers land.
#
# Left at 10:30Z: A4's bb200k student score, and the two bb100k A1/B3 student
# reproductions. The two bb40k reproductions are already in.
#
# Exits 0 when every wanted score file exists AND the queue holds no job that
# is neither done nor failed. Exits 2 on timeout. One line per event.
set -u
STUDY="/home/jupyter/wt-cf-373-run2/reports/2026-08-08_rollout_depth"
RES="$STUDY/results"
Q="$STUDY/scripts/q_queue.tsv"
MAXH="${1:-6}"
WANT="A4_k3_bb200k_student A1rep_k3_bb40k_student B3rep_k3_bb40k_student \
A1rep_k3_bb100k_student B3rep_k3_bb100k_student"

qleft(){ # jobs neither done nor failed
  local n=0 id s
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
    s="$(cat "$RES/queue/$id.state" 2>/dev/null || echo queued)"
    case "$s" in done|failed) ;; *) n=$(( n + 1 ));; esac
  done; echo "$n"
}

t0=$(date +%s); last_hb=0
declare -A seen
while true; do
  left=0
  for w in $WANT; do
    f="$RES/score_${w}.txt"
    if [ -s "$f" ]; then
      [ -n "${seen[$w]:-}" ] || { seen[$w]=1; echo "[$(date -u +%H:%M:%SZ)] SCORE $w = $(cat "$f")"; }
    else
      left=$((left+1))
    fi
  done
  ql="$(qleft)"
  [ "$left" -eq 0 ] && [ "$ql" -eq 0 ] && {
    echo "[$(date -u +%H:%M:%SZ)] ROUND COMPLETE — every score in, queue drained"; exit 0; }
  now=$(date +%s); el=$((now-t0))
  if [ $((el-last_hb)) -ge 1800 ]; then
    last_hb=$el
    cr=$(PATH="$HOME/.local/bin:$PATH" timeout 60 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')
    bb=$(tail -1 /home/jupyter/cf373_r3/sync/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k3_r2_losses.csv 2>/dev/null | cut -d, -f1)
    echo "[$(date -u +%H:%M:%SZ)] HEARTBEAT scores_left=$left queue_left=$ql credit=${cr:-?} A4_bb_step=${bb:-?}"
  fi
  [ "$el" -gt $((MAXH*3600)) ] && { echo "[$(date -u +%H:%M:%SZ)] TIMEOUT scores_left=$left queue_left=$ql"; exit 2; }
  sleep 60
done
