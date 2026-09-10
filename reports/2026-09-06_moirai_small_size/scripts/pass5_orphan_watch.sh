#!/bin/bash
# #412 — one line when a head or a trainer starts with NO LANE ABOVE IT.
#
# WHY. `head_busy.sh` and `arm_busy.sh` are the only guards this card has
# against two processes on one arm, and a lane calls them before it starts
# anything. A process started BY HAND calls neither. On 2026-09-10 at 23:33 a
# session started a second head for `k3_r100_09_lr56_dec` at 40,000 steps by
# hand, while a first driver for the SAME tag held the GPU 1 flock. The
# `flock` is per CARD, so the two would not have serialized, and both name one
# head checkpoint and one eval directory.
#
# Nothing was corrupt that time, because the first driver had started no
# python. The next one may not be so lucky.
#
# WHAT IT READS. Every #412 head trainer and every #412 backbone trainer, and
# the chain of parents above it. A lane script in that chain means a guard ran
# first. No lane script means nobody checked, and the line goes out.
#
# It writes into the pass-5 watch log, so the session that tails that log sees
# it with no second monitor.
#
# Usage:  nohup bash scripts/pass5_orphan_watch.sh >>results/pass5_watch.log 2>&1 &
set -uo pipefail
POLL="${CF412_ORPHAN_POLL:-60}"
# Every script that runs a guard before it starts a head or a leg.
LANES='phase1.sh|pass2_lane.sh|pass5_lane.sh|head_sweep.sh|head_claim.sh|missing_heads.sh|run_arm.sh'

# The lane script that owns this process, or nothing. It walks up to eight
# parents, which covers driver, wrapper and lane.
#
# AN AGENT SHELL IS NOT A LANE. A session runs each command through a wrapper
# whose own arguments hold the WHOLE command text, so the wrapper that started
# the hand-run head of 23:33 carries the string `phase1.sh` and read as a
# lane. Every wrapper carries `shell-snapshots` and no lane does, which is the
# same test `cf412_count_real` uses. Without this line the watcher reports
# nothing and looks healthy.
lane_of(){  # <pid>
  local p="${1:?pid}" i args
  for i in 1 2 3 4 5 6 7 8; do
    case "$p" in ''|0|1) return 1 ;; esac
    args="$(ps -o args= -p "$p" 2>/dev/null)"
    case "$args" in
      *shell-snapshots*) ;;
      *)
        if printf '%s' "$args" | grep -qE "$LANES"; then
          printf '%s' "$args" | cut -c1-50
          return 0
        fi ;;
    esac
    p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
  done
  return 1
}

# Every #412 head trainer and backbone trainer, as `<pid> <kind> <name>`.
targets(){
  ps -eo pid,args --no-headers 2>/dev/null | awk '
    $2 !~ /python/ { next }
    /train_forecasting_head/ {
      for (i = 1; i <= NF; i++) if ($i == "--run-name") { print $1, "head", $(i+1); next } }
    /--run-name cf393_.*_cf412_/ {
      for (i = 1; i <= NF; i++) if ($i == "--run-name") { print $1, "leg", $(i+1); next } }'
}

seen=" "
while :; do
  while read -r pid kind name; do
    [ -n "${pid:-}" ] || continue
    case "$seen" in *" $pid "*) continue ;; esac
    seen="$seen$pid "
    if lane="$(lane_of "$pid")"; then
      : # a lane owns it, so a guard ran first
    else
      echo "[event] ORPHAN $kind pid $pid started with no lane above it —" \
           "$name. No guard ran. Check for a second process on this tag."
    fi
  done <<<"$(targets)"
  sleep "$POLL"
done
