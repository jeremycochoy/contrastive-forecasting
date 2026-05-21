#!/bin/bash
# Variance downstream orchestrator. Watches for backbones synced from
# vast, launches the corresponding elisa_variance_run.sh downstream phase
# (skips backbone — already exists; runs q-head + GIFT-Eval) on the
# first free GPU. Up to 2 downstreams run concurrently (GPU 0 + GPU 1).
#
# Queue is the variance_*.env files in scripts/state/ EXCEPT the one
# whose elisa_variance is already running locally (B-s2). The orchestrator
# is conservative: it never launches if a previous variance-run proc is
# alive for that (arm, seed), or if both GPUs are busy.
#
# Usage: downstream_orchestrator.sh [start|stop|tick]
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
EXP="$WT/experiments/2026-05-19_crossed_loss_xbranch_ablation"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation
LOG="$EXP/downstream_orchestrator.log"
PIDF="$EXP/scripts/state/downstream_orchestrator.pid"

# What downstreams are queued:
#   hh 20260518     — B-s2, runs in-line via elisa_variance_run.sh (orchestrator
#                     just observes "running")
#   hh 20260519     — B-s3, backbone on vast Box B → orchestrator launches ds on free GPU
#   hhxbf 20260519  — B-xfree-s3, backbone on vast Box C → ditto
# Note: hhxbf 20260518 (Box A) destroyed mid-training due to slow throttled
# 4070S Ti causing budget overshoot — N=2 for B-xfree (s17 of-record + s19).
queue(){ cat <<EOF
hh 20260518
hh 20260519
hhxbf 20260519
EOF
}

is_done(){ # arm seed -> 0 if full-eval summary exists
  local a="$1"
  local s="$2"
  local ss="s${s:(-2)}"
  local name="cl_${a}_50k_${ss}"
  [ -f "$MAIN/variance/${a}_seed${s}/results/gift_eval_full_${name}/summary.txt" ]
}

is_running(){ # arm seed -> 0 if elisa_variance_run.sh proc alive for it
  pgrep -af "elisa_variance_run.sh $1 $2" >/dev/null
}

backbone_synced(){ # arm seed -> 0 if FINAL.pth + ≥49k losses rows local
  local a="$1"
  local s="$2"
  local ss="s${s:(-2)}"
  local name="cl_${a}_50k_${ss}"
  local loc="$MAIN/variance/${a}_seed${s}"
  local final="$loc/runs/${name}_FINAL.pth"
  local lossfile="$loc/runs/${name}_losses.csv"
  local rows=0
  [ -f "$lossfile" ] && rows=$(wc -l < "$lossfile" 2>/dev/null || echo 0)
  [ -f "$final" ] && [ "$rows" -gt 49000 ]
}

# Which GPU is occupied by which Python compute process. Returns "" if
# none of our procs on it. Ignores small parked allocs (<2GB).
gpu_busy(){
  local g="$1"
  # nvidia-smi process list with GPU index
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', *' -v g="$g" '$3 > 2000 {print $1}')
  # Map our compute-heavy procs to a GPU; check if any of our train/eval is on $g
  for p in $pids; do
    local cve
    cve=$(grep -aoE 'CUDA_VISIBLE_DEVICES=[0-9,]+' /proc/$p/environ 2>/dev/null|head -1|cut -d= -f2)
    # If env has the GPU among visible devices and it's one of our wrappers, busy
    if [ -n "$cve" ]; then
      case ",$cve," in *,$g,*) return 0;; esac
    else
      # Couldn't read env: be conservative and consider it busy
      return 0
    fi
  done
  return 1
}

pick_free_gpu(){
  for g in 0 1; do gpu_busy "$g" || { echo "$g"; return 0; }; done
  return 1
}

launch(){ # arm seed gpu
  local a="$1"
  local s="$2"
  local g="$3"
  local ss="s${s:(-2)}"
  local name="cl_${a}_50k_${ss}"
  local DSLOG="$MAIN/variance/${a}_seed${s}/results/downstream_orch_${name}.log"
  mkdir -p "$(dirname "$DSLOG")"
  echo "[$(date '+%H:%M:%S')] launching downstream $a/$ss on GPU$g -> $DSLOG"
  cd /home/jupyter/contrastive-forecasting
  setsid bash -c "
    bash $EXP/scripts/elisa_variance_run.sh $a $s $g > $DSLOG 2>&1
  " < /dev/null > /dev/null 2>&1 &
  sleep 2
}

tick(){
  echo "=== orchestrator tick $(date '+%m-%d %H:%M:%S') ==="
  queue|while read a s; do
    [ -z "$a" ] && continue
    if is_done "$a" "$s"; then
      echo "  ✓ $a/$s already DONE"
      continue
    fi
    if is_running "$a" "$s"; then
      echo "  · $a/$s running"
      continue
    fi
    if ! backbone_synced "$a" "$s"; then
      echo "  - $a/$s backbone not yet synced (waiting)"
      continue
    fi
    local g
    if g=$(pick_free_gpu); then
      launch "$a" "$s" "$g"
    else
      echo "  - $a/$s ready but both GPUs busy (waiting)"
    fi
  done
}

case "${1:-start}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "already running pid=$(cat "$PIDF")"; exit 0
    fi
    setsid bash -c "
      while true; do
        bash $0 tick >> $LOG 2>&1
        # exit when all queued are done
        all_done=1
        while read a s; do
          [ -z \"\$a\" ] && continue
          name=\"cl_\${a}_50k_s\${s:(-2)}\"
          [ -f \"$MAIN/variance/\${a}_seed\${s}/results/gift_eval_full_\${name}/summary.txt\" ] || all_done=0
        done < <(bash $0 queue 2>/dev/null)
        if [ \"\$all_done\" = 1 ]; then
          echo \"orchestrator: all queued downstreams DONE, exiting\" >> $LOG
          break
        fi
        sleep 120  # 2-min check
      done
      rm -f $PIDF
    " < /dev/null > /dev/null 2>&1 &
    echo $! > "$PIDF"
    echo "orchestrator started pid=$!"
    ;;
  stop)
    [ -f "$PIDF" ] && { kill "$(cat "$PIDF")" 2>/dev/null||true; rm -f "$PIDF"; echo stopped; }
    ;;
  tick)    tick ;;
  queue)   queue ;;
  *) echo "usage: $0 {start|stop|tick|queue}"; exit 2;;
esac
