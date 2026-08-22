#!/bin/bash
# #407 review gap 6 — restart the driver when it dies, and mirror the
# numbers off `/tmp` every hour.
#
# `run_pass.sh` is one detached process for 47 hours, on a box that carries
# other sessions' work. `save_every` is 20,000 steps, so a crash costs at
# most 1.6 h of a leg — but only if somebody re-fires the driver. Nothing
# did that before this file.
#
# A re-fire is safe. `run_leg_k.sh` skips a leg whose checkpoint is on disk,
# `stop_k.sh` skips a head that already scored, and `eval_local.sh` resumes
# per config. So a re-fired driver costs only what died. The continuity
# gates in `full_pass.py` run again from the top and refuse a leg that would
# start at step 0.
#
# The watchdog decides on TWO signals, and needs both to fire:
#
#   no driver   `pgrep` finds no `run_pass.sh`.
#   no progress the last step in the train log has not moved, or the last
#               line of `run_pass.log` has not moved.
#
# A driver inside a 72-minute GIFT-Eval writes no train step, so progress
# alone would re-fire a healthy study. A driver that is alive but wedged
# writes no progress, so the process check alone would never fire. The AND
# of the two is the condition this study wants.
#
# It also runs `read_back.sh` and `teacher_check.sh` every tick. Those cover
# review gaps 5 and 4, and they belong here because the same hourly loop
# already exists and both are cheap and idempotent. `read_back.sh` ends with
# `mirror_durable.sh`, so the mirror still runs every hour.
#
# Usage:
#   WT=<checkout> RUNS=<durable root> BB_GPU=0 \
#     nohup setsid bash watchdog.sh > watchdog.out 2>&1 &
#
# It also decides the head-seed band at the last stop. See
# `band_at_last_stop`. That is review gap 1's second half, and the compute
# audit of 2026-08-22 made it conditional. `band_decision.py` holds the
# rule.
#
# WATCHDOG_PERIOD  seconds between ticks (default 3600).
# WATCHDOG_STALL   ticks with no progress before a re-fire (default 2).
# WATCHDOG_MAX_FIRES  give up after this many re-fires (default 10).
# WATCHDOG_ONCE    run one tick and exit. The test seam.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
export RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"
export BB_GPU="${BB_GPU:-0}"
export HEAD_GPU="${HEAD_GPU:-$BB_GPU}"

PERIOD="${WATCHDOG_PERIOD:-3600}"
STALL="${WATCHDOG_STALL:-2}"
MAX_FIRES="${WATCHDOG_MAX_FIRES:-10}"
# Run one tick and exit. The test seam, the same one `band_queue.sh` holds
# as `QUEUE_ONCE`. The band decision at the last stop runs inside the loop,
# so a test cannot reach it without this.
ONCE="${WATCHDOG_ONCE:-0}"
LOG="$RES/watchdog.log"
TRAIN_LOG="$WT/reports/2026-08-08_rollout_depth/results/run_cf393_arm6_v2_combab_alignS_cf373k3.log"
DRIVER_LOG="$RES/run_pass.log"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-watchdog] $*" | tee -a "$LOG"; }

# The last train step the backbone reached. `train.py` prints `[ 205400] `.
last_step(){
  grep -oE '^\[ *[0-9]+\]' "$TRAIN_LOG" 2>/dev/null | tail -1 | tr -dc '0-9'
}
# How far the driver's own log has got. It moves on every leg, head and eval.
driver_mark(){ stat -c %s "$DRIVER_LOG" 2>/dev/null || echo 0; }

driver_alive(){ pgrep -f 'run_pass\.sh' >/dev/null 2>&1; }

# Which stops still owe a score. The driver takes them as arguments, so a
# re-fire does not repeat a stop that is already drained, and the study
# stops when the last one lands.
open_stops(){
  python3 - "$WT" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.environ["CF407_HERE"]), "scripts"))
import full_pass
wt = sys.argv[1]
parent = full_pass.parent_results(wt)
open_ = [s for s in full_pass.STOPS
         if any(full_pass.score(s, h, parent) is None for h in full_pass.HEADS)]
print(" ".join(str(s) for s in open_))
PY
}
export CF407_HERE="$HERE"

REPLICATE_SCRIPT="$(readlink -f "$HERE/replicate_heads.sh" 2>/dev/null)"

# Is a replicate run for one stop already up? Same `argv[1]` test as
# `driver_alive`, and `argv[2]` carries the stop.
#
# It then resolves `argv[1]` against the process's own working directory and
# demands THIS checkout's copy. A band launched by a relative path, which is
# how this study launches one, would otherwise let a second worktree of the
# repo read as a band of this one. `band_queue.sh` holds the same test.
replicate_alive(){ # <stop>
  local p a1 a2 cwd full
  for p in $(pgrep -f 'replicate_heads\.sh' 2>/dev/null); do
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 2p)
    a2=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 3p)
    case "$a1" in */replicate_heads.sh) ;; *) continue;; esac
    [ "$a2" = "$1" ] || continue
    case "$a1" in
      /*) full="$a1" ;;
      *)  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null) || continue
          [ -n "$cwd" ] || continue
          full="$cwd/$a1" ;;
    esac
    full=$(readlink -f "$full" 2>/dev/null)
    [ -n "$full" ] && [ "$full" = "$REPLICATE_SCRIPT" ] && return 0
  done
  return 1
}

# Review gap 1, second half, and the compute audit of 2026-08-22.
#
# The first half of the band runs at 200,000 steps, where the card's
# comparison starts. This one covers 665,000 steps, where it ends.
#
# It used to be ARMED. It fired on the 665,000-step CHECKPOINT, whatever
# the stop then scored, and it cost about 8 GPU-hours. It is now
# CONDITIONAL. It fires on the SCORE.
#
# `band_decision.py` holds the rule and the two constants. Read that file
# for the reasoning. Its exit codes:
#
#   0   FIRE  the student score is within 0.01 of 1.0651, the mean of the
#             200,000-step student band. One draw cannot decide the card's
#             comparison inside that window, so the card buys two more
#             head seeds.
#   10  SKIP  the score is outside the window. This card's measured pooled
#             standard deviation is 0.0029, so a clearly high or clearly
#             low point reads on its own. The card keeps the 8 GPU-hours.
#   20  WAIT  that stop has no student score yet.
#
# The verdict goes to `band_<k>k_decision.txt`. A SKIP latches on that
# file, so the watchdog decides once and does not re-read the rule every
# tick. A FIRE latches on `replicate_<k>k.log`, which the band writes at
# once, exactly as before.
#
# The student score lands about two hours before the teacher one, and the
# watchdog ticks every 30 minutes. So the decision happens while the
# driver still scores the teacher head. `band_at_last_stop` also runs
# BEFORE the loop tests `open_stops`, so the last tick of the study still
# decides before the watchdog exits.
#
# A fired band runs on the card the driver is NOT using, so it takes no
# GPU time from the card's own heads.
band_at_last_stop(){
  local last="${BAND_STOP:-665000}" k gpu rc dec
  k=$(( last / 1000 ))
  dec="$RES/band_${k}k_decision.txt"
  [ -s "$RES/replicate_${k}k.log" ] && return 0    # fired once already
  grep -q '^SKIP' "$dec" 2>/dev/null && return 0   # decided against, once
  replicate_alive "$last" && return 0

  WT="$WT" python3 "$HERE/band_decision.py" --stop "$last" \
    --results "$WT/reports/2026-08-08_rollout_depth/results" \
    --csv "$RES/head_band.csv" --write "$dec" >>"$LOG" 2>&1
  rc=$?
  case "$rc" in
    20) return 0 ;;
    10) log "BAND at ${k}k: SKIP. The score reads on its own, so the card keeps the draws."
        return 0 ;;
    0)  ;;
    *)  log "WARN: band_decision rc=$rc"; return 0 ;;
  esac

  gpu="${BAND_GPU:-$(( 1 - BB_GPU ))}"
  log "BAND at ${k}k: FIRE. The score sits inside the window, so two more head seeds start on GPU $gpu"
  WT="$WT" CF373_ROOT="$RUNS" BB_GPU="$gpu" \
    HEAD_VRAM_MIB="${BAND_VRAM_MIB:-6400}" \
    nohup setsid bash "$HERE/replicate_heads.sh" "$last" \
      >>"$RES/replicate_${k}k.out" 2>&1 &
}

log "start period=${PERIOD}s stall=$STALL wt=$WT runs=$RUNS bb_gpu=$BB_GPU"
prev_step=""; prev_mark=""; quiet=0; fires=0

while :; do
  # Review gap 5 plus the read-back. `read_back.sh` brings whatever drained
  # since into the checkout, refreshes the band, the teacher track and the
  # figure, and ends with `mirror_durable.sh`. It runs here because a band
  # drains hours after the agent that fired it has gone.
  bash "$HERE/read_back.sh" >>"$LOG" 2>&1 || log "WARN: read_back rc=$?"
  # Review gap 4. It costs seconds, it skips a pair it already
  # answered, and it needs the checkpoint rather than the score, so
  # the answer for a stop lands before that stop finishes scoring.
  CF373_ROOT="$RUNS" bash "$HERE/teacher_check.sh" >>"$LOG" 2>&1 || \
    log "WARN: teacher_check rc=$?"

  band_at_last_stop

  step="$(last_step)"; mark="$(driver_mark)"
  stops="$(open_stops 2>/dev/null)"
  alive=no; driver_alive && alive=yes

  if [ -z "$stops" ]; then
    log "every stop has both scores — watchdog stops"
    exit 0
  fi

  moved=yes
  if [ "$step" = "$prev_step" ] && [ "$mark" = "$prev_mark" ]; then moved=no; fi
  prev_step="$step"; prev_mark="$mark"

  if [ "$moved" = no ]; then quiet=$(( quiet + 1 )); else quiet=0; fi
  log "tick driver=$alive step=${step:-?} quiet=$quiet open='$stops'"

  if [ "$alive" = no ] && [ "$quiet" -ge "$STALL" ]; then
    if [ "$fires" -ge "$MAX_FIRES" ]; then
      log "GIVING UP: $fires re-fires already, and the driver is still down"
      exit 1
    fi
    fires=$(( fires + 1 ))
    log "RE-FIRE $fires: no driver and no progress for $quiet ticks — stops $stops"
    # shellcheck disable=SC2086
    WT="$WT" RUNS="$RUNS" BB_GPU="$BB_GPU" HEAD_GPU="$HEAD_GPU" \
      nohup setsid bash "$HERE/run_pass.sh" $stops \
        >>"$RES/run_pass.out" 2>&1 &
    quiet=0
  fi

  [ "$ONCE" = 1 ] && exit 0
  sleep "$PERIOD"
done
