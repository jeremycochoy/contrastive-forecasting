#!/bin/bash
# sync_loop: pull arm5 (allt·0.8%) checkpoints from the vast ON-DEMAND instance to the
# elisa MAIN checkout every 15 min. Atomic (.tmp + size-gate + mv). Pulls: latest periodic
# _Nk.pth + _Nk_optimizer (resume net if credit runs out), losses CSV + run log (monitor),
# and FINAL.pth (+optimizer) when it appears — landing it in elisa's runs dir under the exact
# name so the orchestrator's idempotent check SKIPS arm 5 once it's synced.
set -uo pipefail
PORT="${PORT:-30546}"; HOST="${HOST:-ssh2.vast.ai}"; REMOTE=/root/runs
LOCAL=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/runs
RES=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/results
NAME=bb_xshh_allt_forked2_qk_aon_6Lf_b1024
LOG="$RES/sync_arm5.log"
SSHO="-p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"
SCPO="-P $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
pull(){ local f="$1" minb="$2"
  scp $SCPO "root@$HOST:$REMOTE/$f" "$LOCAL/$f.tmp" 2>/dev/null || { return 1; }
  local sz=$(stat -c%s "$LOCAL/$f.tmp" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$minb" ]; then [ -f "$LOCAL/$f" ] && cp -f "$LOCAL/$f" "$LOCAL/$f.prev" 2>/dev/null; mv "$LOCAL/$f.tmp" "$LOCAL/$f"; log "ok $f ($sz)"; else rm -f "$LOCAL/$f.tmp"; log "small $f ($sz<$minb)"; fi
}
log "SYNC start arm5 from $HOST:$PORT -> $LOCAL"
while true; do
  # FINAL first — if it's there, pull it (+optimizer) and we're done
  if ssh $SSHO "root@$HOST" "test -f $REMOTE/${NAME}_FINAL.pth" 2>/dev/null; then
    pull "${NAME}_FINAL.pth" 40000000 && pull "${NAME}_FINAL_optimizer.pth" 40000000 2>/dev/null
    pull "${NAME}_losses.csv" 100
    [ -f "$LOCAL/${NAME}_FINAL.pth" ] && { log "FINAL synced — arm5 done, sync exiting"; exit 0; }
  fi
  # latest periodic checkpoint (resume net) + companions
  latest=$(ssh $SSHO "root@$HOST" "ls -t $REMOTE/${NAME}_*k.pth 2>/dev/null | head -1 | xargs -r basename" 2>/dev/null || true)
  [ -n "$latest" ] && { pull "$latest" 40000000; pull "${latest%.pth}_optimizer.pth" 70000000; }
  pull "${NAME}_losses.csv" 100
  scp $SCPO "root@$HOST:$RES/../run_arm5.log" "$RES/run_arm5_vast.log" 2>/dev/null || scp $SCPO "root@$HOST:/root/run_arm5.log" "$RES/run_arm5_vast.log" 2>/dev/null || true
  sleep 900
done
