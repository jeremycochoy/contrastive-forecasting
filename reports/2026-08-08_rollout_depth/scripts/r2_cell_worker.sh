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

# The deepest step a checkpoint on this box reaches, in thousands. One box
# carries one cell, so the maximum over the whole run tree is that cell's.
# `0` when the box holds none.
latest_ck_k(){
  find "$CF373_ROOT" -name "*_[0-9]*k.pth" ! -name "*optimizer*" 2>/dev/null \
    | sed -E 's|.*_([0-9]+)k\.pth$|\1|' | sort -n | tail -1 | grep . || echo 0
}

# The HuggingFace stream is the failure this study keeps meeting. Give its
# reads room before the retry loop has to earn its keep.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

row="$(awk -F'\t' -v c="$CELL" '$1==c {print; exit}' "$HERE/cells.tsv")"
[ -n "$row" ] || { log "ABORT: no cell '$CELL' in cells.tsv"; exit 2; }
launcher=$(cut -f3 <<<"$row"); arg=$(cut -f4 <<<"$row")

log "start k=$K stops: $* launcher=$launcher arg=$arg"
head_pids=()

for steps in "$@"; do
  args=("$arg")
  case "$launcher" in run_leg_k.sh) args+=("$steps") ;; esac

  log "WAVE start -> $steps steps"
  # The stream, not the model, is what ends a wave early. Both round-3 boxes
  # died on `httpx.ReadTimeout` from the HuggingFace stream, 3 hours into a
  # 5-hour leg, and each left a good 140k checkpoint behind. The launcher
  # resumes from the newest `_<N>k.pth` on the box, so a retry costs one
  # save interval, not the leg.
  #
  # The guard is progress, not a retry count: an attempt that ends with the
  # newest checkpoint no further than the one before it has stopped for a
  # reason a retry cannot fix, and the wave gives up on the second such.
  attempt=0; rc=1; stalls=0
  while [ "$attempt" -lt "${WAVE_TRIES:-6}" ]; do
    attempt=$(( attempt + 1 ))
    before="$(latest_ck_k)"
    K="$K" TARGET_STEPS="$steps" FINAL_STEPS=200000 \
    SAVE_EVERY=20000 EXTRA_SAVES="$steps" \
    WT="$WT" BB_GPU=0 RUNS="$CF373_ROOT" CF373_RUNS="$CF373_ROOT" \
      bash "$HERE/$launcher" "${args[@]}"
    rc=$?
    [ "$rc" -eq 0 ] && break
    after="$(latest_ck_k)"
    log "WAVE attempt $attempt rc=$rc, checkpoint ${before}k -> ${after}k"
    if [ "$after" -le "$before" ]; then
      stalls=$(( stalls + 1 ))
      [ "$stalls" -ge 2 ] && { log "two attempts with no new checkpoint; wave gives up"; break; }
    else
      stalls=0
    fi
    sleep 60
  done
  log "WAVE end -> $steps rc=$rc after $attempt attempt(s)"
  if [ "$rc" -ne 0 ]; then
    log "wave failed; not queueing its heads, and not attempting deeper stops"
    wave_failed=1
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

# The DONE marker is the reaper's only gate: it destroys the box once every
# artefact this marker claims is final has landed locally. A cell that broke
# out of its wave loop has not delivered its stops, and marking it done sent
# the reaper after a box the study still needs. Write the marker only when
# every wave and every head succeeded.
if [ "${wave_failed:-0}" -ne 0 ] || [ "$fail" -ne 0 ]; then
  log "CELL INCOMPLETE (wave_failed=${wave_failed:-0} head_fail=$fail) — no DONE marker"
  exit 1
fi
touch "$RES/CELL_${CELL}_DONE"
log "CELL DONE"
