#!/bin/bash
# #404 — the wiring, before the arms spend 11 hours of GPU time.
#
# It runs each arm for a few hundred steps through the SAME wrapper, the same
# #373 runner and the same guards the study runs, then reads three things back
# out of the artefacts:
#
#   the momentum   off the trainer's own command line, per arm. Four arms that
#                  share a configuration differ in alpha alone, so an arm whose
#                  alpha did not arrive is a silent duplicate.
#   the depth      the count of `cos_err_dj` columns in the losses CSV. A
#                  k = 32 run writes 33 of them.
#   the step time  what one k = 32 step costs on this card, which is what the
#                  run plan is sized from.
#
# It writes nowhere the study writes: `CF404_TRIAL` moves the root and the
# results directory (see study.sh).
#
# Usage:  BB_GPU=0 bash scripts/smoke.sh [steps]
set -uo pipefail

STEPS="${1:-300}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CF404_TRIAL="$STEPS"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-0}"
ARMS="${ARMS:-$CF404_ARMS}"
mkdir -p "$CF404_RESULTS"
OUT="$CF404_RESULTS/smoke.csv"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 smoke] $*" \
  | tee -a "$CF404_RESULTS/smoke.log"; }

# How many `cos_err_dj` columns a run's losses CSV carries. A k-depth run
# writes k + 1 of them, so this is the proof the depth reached the trainer.
#
# `tr -d '\r'` is not defensive: the trainer's CSV writer ends every line CRLF,
# so the LAST field of the header carries a trailing \r and an anchored match
# misses it. Without it the count reads k, which is off by one and plausible.
depth_cols(){  # <arm>
  local csv
  csv="$(ls "$(cf404_leg_dir "$1" "$STEPS")"/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$csv" ] || { echo ""; return; }
  head -1 "$csv" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*$' || true
}

echo "arm,ema_wanted,ema_seen,depth_cols,seconds" >"$OUT"
failed=0
for arm in $ARMS; do
  cf404_require_arm "$arm" || exit $?
  log "arm $arm -> $STEPS steps on gpu $BB_GPU"
  t0=$(date +%s)
  BB_GPU="$BB_GPU" bash "$HERE/run_arm.sh" "$arm" "$STEPS"
  rc=$?
  t1=$(date +%s)
  want="$(cf404_ema_sig "$arm")"
  seen="$(cf404_last_cmdline "$(cf404_leg_log "$arm")" 2>/dev/null \
          | cf404_ema_of_cmdline)"
  cols="$(depth_cols "$arm")"
  echo "$arm,\"$want\",\"${seen:-}\",${cols:-},$(( t1 - t0 ))" >>"$OUT"
  if [ $rc -ne 0 ]; then
    log "arm $arm rc=$rc"; failed=$(( failed + 1 )); continue
  fi
  if [ "$seen" != "$want" ]; then
    log "arm $arm FAIL: trained '$seen', wanted '$want'"
    failed=$(( failed + 1 ))
  fi
  if [ -n "$cols" ] && [ "$cols" -ne $(( CF404_K + 1 )) ]; then
    log "arm $arm FAIL: $cols cos_err columns, wanted $(( CF404_K + 1 ))"
    failed=$(( failed + 1 ))
  fi
done

log "smoke done — $failed failure(s); table in $OUT"
column -s, -t <"$OUT" 2>/dev/null || cat "$OUT"
[ "$failed" -eq 0 ] || exit 1
