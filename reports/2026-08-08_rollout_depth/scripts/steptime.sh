#!/bin/bash
# #373 — GPU step-time overhead of the depth, on the first cell.
#
# The card requires this number before the other cells launch, and PR #400
# defers it here: its table is CPU only, where the fp16 sub-blocks the 14
# cells run have no effect.
#
# Method. Run the cell's own launcher for STEPS steps at k = 0 and at k = 3,
# ALTERNATING, REPS times each. Both 4090s on elisa carry another session's
# training, so a single k = 0 run followed by a single k = 3 run would fold
# whatever that neighbour did in between into the overhead. Alternating and
# taking the median over repetitions puts the same contention on both arms.
#
# The trainer prints its own `timing: data=… fwd=… bwd=… total=…` line every
# --log-every steps, averaged over that window. `data` is the input pipeline
# and carries no depth; fwd + bwd is where a depth lands. Both are reported.
# The first window of a run is warm-up (CUDA context, autotune, stream fill)
# and is dropped.
#
# Usage: bash steptime.sh [cell id] [steps] [reps]
set -uo pipefail

CELL_ID="${1:-B5}"
STEPS="${2:-600}"
REPS="${3:-3}"
LOG_EVERY="${LOG_EVERY:-200}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$WT/reports/2026-08-08_rollout_depth/results"
SCRATCH="${CF373_VERIFY_RUNS:-/home/jupyter/checkpoints_backup/cf-373/steptime}"
OUTCSV="$RES/steptime_${CELL_ID}.csv"
mkdir -p "$RES" "$SCRATCH"

row="$(awk -F'\t' -v c="$CELL_ID" '$1==c {print; exit}' "$HERE/cells.tsv")"
[ -n "$row" ] || { echo "ABORT: no cell '$CELL_ID' in cells.tsv" >&2; exit 2; }
SLUG=$(cut -f2 <<<"$row"); LAUNCHER=$(cut -f3 <<<"$row"); ARG=$(cut -f4 <<<"$row")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [steptime $CELL_ID] $*" \
  | tee -a "$RES/steptime_${CELL_ID}.log"; }

echo "cell,rep,k,window,data_ms,fwd_ms,bwd_ms,total_ms,sps,gpu_mem_mib,neighbour_mib" >"$OUTCSV"

run_one(){ # <k> <rep>
  local k="$1" rep="$2"
  local root="$SCRATCH/${CELL_ID}_k${k}_r${rep}"
  rm -rf "$root"; mkdir -p "$root"
  local args=("$ARG")
  case "$LAUNCHER" in run_leg_k.sh) args+=("$STEPS") ;; esac
  local before after
  before=$(nvidia-smi --id="${BB_GPU:-1}" --query-gpu=memory.used \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  K="$k" LOG_EVERY="$LOG_EVERY" TARGET_STEPS="$STEPS" FINAL_STEPS=200000 \
  EXTRA_SAVES="$STEPS" SAVE_EVERY=1000000 \
  WT="$WT" BB_GPU="${BB_GPU:-1}" RUNS="$root" CF373_RUNS="$root" \
    bash "$WT/reports/2026-08-08_rollout_depth/scripts/$LAUNCHER" "${args[@]}" \
    >>"$RES/steptime_${CELL_ID}.log" 2>&1
  # The trainer log the launcher appended to, in this scratch tree's RES.
  local tlog; tlog="$(ls -t "$RES"/run_*"$(basename "$root")"* 2>/dev/null | head -1)"
  [ -n "$tlog" ] || tlog="$(ls -t "$RES"/run_*.log 2>/dev/null | head -1)"
  after=$(nvidia-smi --id="${BB_GPU:-1}" --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  # Pair each `timing:` line with the `sps` on the step line above it.
  awk -v cell="$CELL_ID" -v rep="$rep" -v k="$k" -v mem="$after" -v nb="$before" '
    /sps  ETA/ { for (i = 1; i <= NF; i++) if ($i == "sps") sps = $(i-1) }
    /^  timing:/ {
      d = f = b = t = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^data=/)  { d = $i; sub(/data=/,  "", d); sub(/ms/, "", d) }
        if ($i ~ /^fwd=/)   { f = $i; sub(/fwd=/,   "", f); sub(/ms/, "", f) }
        if ($i ~ /^bwd=/)   { b = $i; sub(/bwd=/,   "", b); sub(/ms/, "", b) }
        if ($i ~ /^total=/) { t = $i; sub(/total=/, "", t); sub(/ms/, "", t) }
      }
      w++
      print cell "," rep "," k "," w "," d "," f "," b "," t "," sps "," mem "," nb
    }' "$tlog" | tail -n +1 >>"$OUTCSV"
  rm -rf "$root"
}

log "START slug=$SLUG steps=$STEPS reps=$REPS log_every=$LOG_EVERY gpu=${BB_GPU:-1}"
for rep in $(seq 1 "$REPS"); do
  # A fresh trainer log per (rep, k), so the awk above reads only this run.
  for k in 0 3; do
    mv -f "$RES"/run_*.log "$SCRATCH/" 2>/dev/null
    run_one "$k" "$rep"
    log "rep=$rep k=$k done"
  done
done

python3 "$HERE/steptime_summary.py" "$OUTCSV" | tee -a "$RES/steptime_${CELL_ID}.log"
