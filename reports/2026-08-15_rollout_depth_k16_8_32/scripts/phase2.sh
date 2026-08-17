#!/bin/bash
# #401 phase 2 — the head trained as long as the backbone.
#
# The card's second question: does a head that trains for as many steps as
# its backbone beat the fixed 30,000-step head? Phase 1 answers it for no
# arm; phase 2 answers it for the two best, at all three stops, with the head
# budget set to the backbone's own step count.
#
# The backbones already exist. Phase 2 trains heads only, so it costs three
# head budgets per arm and three GIFT-Evals — no backbone GPU time.
#
# Which two arms: `pick_phase2_arms.py`, which refuses an incomplete phase 1.
# Set ARMS to override it, for a card that changes after the fact.
#
# Usage:  bash phase2.sh
#         ARMS="8 32" bash phase2.sh
#         CF401_DRY_RUN=1 bash phase2.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

SCORES="${CF401_SCORES:-$CF401_RESULTS/scores.csv}"
HEAD_BG="${HEAD_BG:-0}"
BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF401_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 phase2] $*" \
  | tee -a "$CF401_RESULTS/phase2.log"; }

if [ -n "${ARMS:-}" ]; then
  arms="$ARMS"
else
  [ -f "$SCORES" ] || { echo "ABORT: no phase-1 scores at $SCORES" >&2; exit 2; }
  arms="$(python3 "$HERE/pick_phase2_arms.py" --scores "$SCORES")" || exit $?
fi
for k in $arms; do cf401_require_depth "$k" || exit $?; done

[ -n "${CF401_DRY_RUN:-}" ] || log "arms: $arms"

heads=(); head_names=(); inline_failed=0
for k in $arms; do
  for stop in $CF401_STOPS; do
    # The card's rule: head steps = backbone steps.
    steps="$stop"
    if [ -n "${CF401_DRY_RUN:-}" ]; then
      echo "head k=$k stop=$stop steps=$steps enc=$CF401_ENC"
      continue
    fi
    if [ "$HEAD_BG" = "1" ]; then
      log "head k=$k stop=$stop steps=$steps (background)"
      # `nohup`, not `nohup setsid` — see phase1.sh: setsid can fork, and the
      # `wait` below would then return on a PID that is already gone.
      BB_GPU="$BB_GPU" nohup bash "$HERE/head_eval.sh" "$k" "$stop" "$steps" \
        >>"$CF401_RESULTS/head_k${k}_bb$(cf401_steps_label "$stop")_h$(cf401_steps_label "$steps").out" 2>&1 &
      heads+=($!); head_names+=("k=$k stop=$stop steps=$steps")
    else
      log "head k=$k stop=$stop steps=$steps"
      BB_GPU="$BB_GPU" bash "$HERE/head_eval.sh" "$k" "$stop" "$steps"
      rc=$?
      log "head k=$k stop=$stop steps=$steps rc=$rc"
      [ $rc -eq 0 ] || inline_failed=$(( inline_failed + 1 ))
    fi
  done
done

[ -n "${CF401_DRY_RUN:-}" ] && exit 0

# Every rc is logged, named by its (k, stop, head budget) — see phase1.sh.
failed="$inline_failed"
if [ "${#heads[@]}" -gt 0 ]; then
  log "waiting for ${#heads[@]} head+eval job(s)"
  for i in "${!heads[@]}"; do
    wait "${heads[$i]}"; rc=$?
    if [ $rc -eq 0 ]; then
      log "head ${head_names[$i]} rc=0"
    else
      failed=$(( failed + 1 ))
      log "head ${head_names[$i]} rc=$rc — see head_k*.out in $CF401_RESULTS"
    fi
  done
fi

bash "$HERE/collect.sh"
log "phase 2 drained — $failed head(s) failed"
[ "$failed" -eq 0 ] || exit 1
