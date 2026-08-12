#!/bin/bash
# #373 round 2 — one cell's whole ladder, on one box.
#
# Usage (on the box):  r2_cell_worker.sh <cell> <stop> [stop...]
#   e.g.               r2_cell_worker.sh B5 40000 100000
#
# One cell per box, and every stop of that cell on the same box. The study
# already knows the box moves a number: round 1's baseline gate read 1.3917
# for B5 at k = 0 on a rented card against 1.2748 published, and the same
# code and seed on elisa read 1.2751. That is a nuisance variable and this
# study does not measure it — but the EXTEND RULE compares a cell's 100k
# against its own 40k, so those two must not straddle two machines.
#
# The heads of a stop run beside the next backbone wave, not after it. A
# d_model=64 backbone at batch 64 is launch-bound, not compute-bound
# (measured 136 ms/step on a 5090 with 79 ms of it in the forward), so a
# second and third process on the card buy close to their own throughput.
# VRAM: 5.4 GB backbone + 7 GB per head against 32 GB.
set -uo pipefail

CELL="${1:?usage: r2_cell_worker.sh <cell> <stop> [stop...]}"
shift
[ $# -gt 0 ] || { echo "ABORT: no stops"; exit 2; }

K="${K:-3}"
# Stops whose heads already exist from round 1, comma separated. A3, B1, B5
# and B9 carry a k = 3 checkpoint at 40k AND both of its heads' scores, so
# re-training those heads here would spend an hour of the card to reproduce
# a number the study already has.
#
# The default lives here rather than in the launcher: the fleet loop passes
# it, and a fleet already running when the launcher changed did not. The box
# is the only place that knows which cell it is, so it decides.
case " A3 B1 B5 B9 " in
  *" $CELL "*) SKIP_HEAD_STOPS="${SKIP_HEAD_STOPS:-40000}" ;;
  *)           SKIP_HEAD_STOPS="${SKIP_HEAD_STOPS:-}" ;;
esac
WT=/root/cf
HERE="$WT/reports/2026-08-08_rollout_depth/scripts"
RES="$WT/reports/2026-08-08_rollout_depth/results"
mkdir -p "$RES"
export CF373_ROOT=/root/cf373_runs

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [cell $CELL] $*" | tee -a "$RES/worker_$CELL.log"; }

row="$(awk -F'\t' -v c="$CELL" '$1==c {print; exit}' "$HERE/cells.tsv")"
[ -n "$row" ] || { log "ABORT: no cell '$CELL' in cells.tsv"; exit 2; }
launcher=$(cut -f3 <<<"$row"); arg=$(cut -f4 <<<"$row")

log "start k=$K stops: $* launcher=$launcher arg=$arg"
head_pids=()

for steps in "$@"; do
  args=("$arg")
  case "$launcher" in run_leg_k.sh) args+=("$steps") ;; esac

  log "WAVE start -> $steps steps"
  K="$K" TARGET_STEPS="$steps" FINAL_STEPS=200000 \
  SAVE_EVERY=20000 EXTRA_SAVES="$steps" \
  WT="$WT" BB_GPU=0 RUNS="$CF373_ROOT" CF373_RUNS="$CF373_ROOT" \
    bash "$HERE/$launcher" "${args[@]}"
  rc=$?
  log "WAVE end -> $steps rc=$rc"
  if [ "$rc" -ne 0 ]; then
    log "wave failed; not queueing its heads, and not attempting deeper stops"
    break
  fi

  case ",$SKIP_HEAD_STOPS," in
    *",$steps,"*)
      log "heads for bb$(( steps / 1000 ))k SKIPPED — round 1 already scored them"
      continue ;;
  esac

  # The heads of this stop, in the background, beside the next wave.
  #
  # HEAD_ENCS is normally both. It narrows on an extension the card's rule
  # ran with one head down: "extend and keep that head". The head the rule
  # stopped is not a deliverable at the deeper stop, and training it would
  # spend half an hour of the card producing a number the rule already
  # ended.
  for enc in ${HEAD_ENCS:-student teacher}; do
    ( bash "$HERE/r2_head_box.sh" "$CELL" "$K" "$steps" "$enc" ) &
    head_pids+=($!)
    sleep 20   # stagger the two allocation ramps
  done
  log "queued heads for bb$(( steps / 1000 ))k"
done

log "waiting on ${#head_pids[@]} head(s)"
fail=0
for p in "${head_pids[@]}"; do wait "$p" || fail=1; done
log "heads done (fail=$fail)"
touch "$RES/CELL_${CELL}_DONE"
log "CELL DONE"
