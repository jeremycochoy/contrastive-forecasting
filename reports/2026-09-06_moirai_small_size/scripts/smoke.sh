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
#   the step time  and the peak memory of this card. The run plan is sized
#                  from them, and a 384-wide model is about six times the
#                  published one.
#
# It writes nowhere the study writes: `CF412_TRIAL` moves the root and the
# results directory (see study.sh).
#
# Usage:  BB_GPU=0 bash scripts/smoke.sh [steps]
set -uo pipefail

STEPS="${1:-60}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# The parameter count train.py prints for the model it built. It is the
# TRAINABLE count, `src.models.count_parameters`, so it compares to
# Moirai-2-Small's 11.4 million directly.
trained_params(){  # <arm>
  grep -ho 'Params: *[0-9,]*' "$(cf412_leg_log "$1")" 2>/dev/null \
    | tail -1 | tr -dc '0-9'
}

# The memory this arm takes on the card, in MiB.
#
# train.py prints no peak, so the number comes from outside: a sampler reads
# `memory.used` while the arm trains and keeps the largest reading. The
# baseline before the start is taken away, because both cards of elisa carry
# other work and the whole reading is not this arm's.
#
# The run plan needs it. Two arms share one 24 GB card only if two of these
# fit beside whatever else holds the card.
gpu_used_mib(){  # <gpu index>
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    2>/dev/null | tr -dc '0-9'
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

sample_peak(){  # <gpu index> <out file>
  local peak=0 now
  while :; do
    now="$(gpu_used_mib "$1")"
    [ -n "$now" ] && [ "$now" -gt "$peak" ] && { peak="$now"; echo "$peak" >"$2"; }
    sleep 5
  done
}

echo "arm,arch_wanted,arch_seen,params,depth_cols,gpu_mib,step_ms,seconds" >"$OUT"
failed=0
for arm in $ARMS; do
  cf412_require_arm "$arm" || exit $?
  log "arm $arm -> $STEPS steps on gpu $BB_GPU"
  base="$(gpu_used_mib "$BB_GPU")"
  peak_file="$(mktemp)"; echo "${base:-0}" >"$peak_file"
  sample_peak "$BB_GPU" "$peak_file" & sampler=$!
  t0=$(date +%s)
  BB_GPU="$BB_GPU" bash "$HERE/run_arm.sh" "$arm" "$STEPS"
  rc=$?
  t1=$(date +%s)
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  mib=$(( $(cat "$peak_file") - ${base:-0} )); rm -f "$peak_file"
  want="$(cf412_arch_sig)"
  seen="$(cf412_last_cmdline "$(cf412_leg_log "$arm")" 2>/dev/null \
          | cf412_arch_of_cmdline)"
  cols="$(depth_cols "$arm")"
  want_cols=$(( $(cf412_depth "$arm") + 1 ))
  echo "$arm,\"$want\",\"${seen:-}\",$(trained_params "$arm"),${cols:-},$mib,$(step_ms "$arm"),$(( t1 - t0 ))" \
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
done

log "smoke done — $failed failure(s); table in $OUT"
column -s, -t <"$OUT" 2>/dev/null || cat "$OUT"
[ "$failed" -eq 0 ] || exit 1
