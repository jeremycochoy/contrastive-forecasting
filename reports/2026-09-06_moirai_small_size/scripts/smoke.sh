#!/bin/bash
# #412 — the wiring, and the cost, before the arms spend days of GPU time.
#
# It runs each arm for a few tens of steps through the SAME wrapper, the same
# #373 runner and the same guards the study runs, then reads four things back
# out of the artefacts:
#
#   the shape      off the trainer's own command line, per arm. The card is
#                  the width, and `run_leg_k.sh` states 64 in its own block.
#   the parameters the count train.py prints for the model it built. It must
#                  be near 11.4 million.
#   the depth      the count of `cos_err_dj` columns in the losses CSV. A
#                  k-depth run writes k + 1 of them.
#   the step time  and the peak memory of this arm. The run plan is sized from
#                  them, and a 384-wide model is about six times the published
#                  one.
#
# ---- Why the cost numbers need this many steps -------------------------------
#
# The first reading of this table came from 40-step arms and a 5-second
# sampler. An arm ran 25 to 41 seconds, so the sampler took five readings and
# could miss the peak: `k3_r100_09b` read 5,419 MiB where the same
# configuration read 6,439.
#
# Three things fix it. The sampler polls every CF412_SMOKE_POLL second, and
# the memory it keeps is THIS arm's own, from `nvidia-smi
# --query-compute-apps`, not the whole card's. Both GPUs of this box carry
# other work, and a card reading holds theirs too. CF412_SMOKE_STEPS is long
# enough that the Hub stream is warm well before the timing line the table
# reads.
#
# The third is below: a smoke DELETES this arm's trial leg first. The runner
# is idempotent, so a second smoke would train nothing and the table would
# then carry a memory of 0 beside the step time of the run before it. A cost
# table that a re-run silently invalidates is worse than no cost table.
#
# The step time still reads high. Both cards carry other work, so it is an
# upper bound on a card of this box, not the cost on an idle card.
#
# It writes nowhere the study writes: `CF412_TRIAL` moves the root and the
# results directory (see study.sh).
#
# Usage:  BB_GPU=0 bash scripts/smoke.sh [steps]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `study.sh` holds the default step count, so read it before the trial budget
# is set and then source the file again under that budget.
STEPS="${1:-}"
[ -n "$STEPS" ] || STEPS="$(. "$HERE/study.sh"; printf '%s' "$CF412_SMOKE_STEPS")"
export CF412_TRIAL="$STEPS"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-0}"
ARMS="${ARMS:-$CF412_ARMS}"
# A smoke of a few tens of steps must print loss lines, and the runner's own
# default of 200 would print none.
export LOG_EVERY="${LOG_EVERY:-10}"
mkdir -p "$CF412_RESULTS"
OUT="$CF412_RESULTS/smoke.csv"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 smoke] $*" \
  | tee -a "$CF412_RESULTS/smoke.log"; }

# How many `cos_err_dj` columns a run's losses CSV carries. A k-depth run
# writes k + 1 of them, so this is the proof the depth reached the trainer.
#
# `tr -d '\r'` is not defensive: the trainer's CSV writer ends every line CRLF,
# so the LAST field of the header carries a trailing \r and an anchored match
# misses it. Without it the count reads k, which is off by one and plausible.
depth_cols(){  # <arm>
  local csv
  csv="$(ls "$(cf412_leg_dir "$1" "$STEPS")"/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$csv" ] || { echo ""; return; }
  head -1 "$csv" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*$' || true
}

# The live weight on `L_rep` at the last row of this arm's losses CSV. A decay
# arm crosses its whole ramp inside a smoke (see `cf412_ramp`), so it must read
# 0.0 here and an arm with no decay must read 1.0. This is the proof the decay
# reached the trainer, beside the guard that reads the command line.
last_rep_w(){  # <arm>
  local csv
  csv="$(ls "$(cf412_leg_dir "$1" "$STEPS")"/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$csv" ] || { echo ""; return; }
  awk -F',' 'NR == 1 { for (i = 1; i <= NF; i++) { gsub(/\r/, "", $i)
                                                  col[$i] = i }
                       next }
             col["rep_w"] { v = $col["rep_w"]; gsub(/\r/, "", v) }
             END { print v }' "$csv"
}

# The parameter count train.py prints for the model it built. It is the
# TRAINABLE count, `src.models.count_parameters`, so it compares to
# Moirai-2-Small's 11.4 million directly.
trained_params(){  # <arm>
  grep -ho 'Params: *[0-9,]*' "$(cf412_leg_log "$1")" 2>/dev/null \
    | tail -1 | tr -dc '0-9'
}

# The memory this arm takes on the card, in MiB.
#
# train.py prints no peak, so the number comes from outside. Both cards of
# this box carry other work, so the whole card reading is not this arm's: the
# sampler sums the rows of `--query-compute-apps` whose pid is under this
# script, and the other tenants of the card drop out.
#
# The run plan needs it. Two arms share one 24 GB card only if two of these
# fit beside whatever else holds the card.
gpu_used_mib(){  # <gpu index>
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    2>/dev/null | tr -dc '0-9'
}

# Every pid under one pid, that pid included.
pid_tree(){  # <pid>
  local pid="${1:?pid}" child
  printf '%s\n' "$pid"
  for child in $(pgrep -P "$pid" 2>/dev/null); do pid_tree "$child"; done
}

# The memory the processes under this script hold on one card, in MiB.
# Prints nothing when no process of ours is on the card yet.
own_gpu_mib(){  # <gpu index>
  local mine
  mine="$(pid_tree $$ | tr '\n' ' ')"
  nvidia-smi --id="$1" --query-compute-apps=pid,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v mine=" $mine " '
        index(mine, " " $1 " ") { total += $2 }
        END { if (total) print total }'
}

# The steady-state cost of one step, in ms, off the trainer's own timing line.
#
# The `seconds` column is wall clock and the first step of a run waits tens of
# seconds for the Hub stream, so it does not divide into a step time. This
# number does: it is the LAST timing line, after the stream is warm.
step_ms(){  # <arm>
  grep -ho 'total=[0-9.]*ms' "$(cf412_leg_log "$1")" 2>/dev/null \
    | tail -1 | tr -dc '0-9.'
}

# The largest reading, kept in a file so the parent reads it after the kill.
sample_peak(){  # <gpu index> <out file>
  local peak=0 now
  while :; do
    now="$(own_gpu_mib "$1")"
    [ -n "$now" ] && [ "$now" -gt "$peak" ] && { peak="$now"; echo "$peak" >"$2"; }
    sleep "$CF412_SMOKE_POLL"
  done
}

echo "arm,arch_wanted,arch_seen,params,depth_cols,rep_w,gpu_mib,step_ms,seconds" \
  >"$OUT"
failed=0
for arm in $ARMS; do
  cf412_require_arm "$arm" || exit $?
  # Measure, never inherit. The path is the trial root, which `study.sh` gives
  # a `-trial` suffix, and the check below refuses anything else.
  leg="$(cf412_leg_dir "$arm" "$STEPS")"
  case "${CF412_ROOT%/}" in
    *-trial) rm -rf "$leg" ;;
    *) echo "ABORT: a smoke must write under a trial root, not $CF412_ROOT" >&2
       exit 2 ;;
  esac
  log "arm $arm -> $STEPS steps on gpu $BB_GPU"
  card="$(gpu_used_mib "$BB_GPU")"
  peak_file="$(mktemp)"; echo 0 >"$peak_file"
  sample_peak "$BB_GPU" "$peak_file" & sampler=$!
  t0=$(date +%s)
  BB_GPU="$BB_GPU" bash "$HERE/run_arm.sh" "$arm" "$STEPS"
  rc=$?
  t1=$(date +%s)
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  mib="$(cat "$peak_file")"; rm -f "$peak_file"
  log "arm $arm peak ${mib} MiB of its own, beside ${card:-0} MiB on the card"
  want="$(cf412_arch_sig)"
  seen="$(cf412_last_cmdline "$(cf412_leg_log "$arm")" 2>/dev/null \
          | cf412_arch_of_cmdline)"
  cols="$(depth_cols "$arm")"
  want_cols=$(( $(cf412_depth "$arm") + 1 ))
  rep_w="$(last_rep_w "$arm")"
  echo "$arm,\"$want\",\"${seen:-}\",$(trained_params "$arm"),${cols:-},${rep_w:-},$mib,$(step_ms "$arm"),$(( t1 - t0 ))" \
    >>"$OUT"
  if [ $rc -ne 0 ]; then
    log "arm $arm rc=$rc"; failed=$(( failed + 1 )); continue
  fi
  if [ "$seen" != "$want" ]; then
    log "arm $arm FAIL: trained shape '$seen', wanted '$want'"
    failed=$(( failed + 1 ))
  fi
  if [ -n "$cols" ] && [ "$cols" -ne "$want_cols" ]; then
    log "arm $arm FAIL: $cols cos_err columns, wanted $want_cols"
    failed=$(( failed + 1 ))
  fi
  want_rep_w=0.0
  [ "$(cf412_decay_ramp "$arm")" = "-" ] && want_rep_w="$CF412_REP_W_START"
  if [ -n "$rep_w" ] && ! cf412_num_eq "$rep_w" "$want_rep_w"; then
    log "arm $arm FAIL: rep_w $rep_w at the last row, wanted $want_rep_w"
    failed=$(( failed + 1 ))
  fi
done

log "smoke done — $failed failure(s); table in $OUT"
column -s, -t <"$OUT" 2>/dev/null || cat "$OUT"
[ "$failed" -eq 0 ] || exit 1
