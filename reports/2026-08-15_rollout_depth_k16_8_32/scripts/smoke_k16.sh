#!/bin/bash
# #401 — what k = 16 costs, before nine legs are launched against it.
#
# The card's run plan assumes k = 16 and k = 32 run at all. #373 measured
# 230 ms/step at k = 0 and 537 ms/step at k = 3 on one 4090, with a 15.8 GiB
# peak. A linear extrapolation from those two points puts k = 16 near
# 1.9 s/step and k = 32 near 3.5 s/step, and puts the peak above 24 GiB. An
# extrapolation is not a measurement, and a 200,000-step leg launched against
# a wrong one costs days. So this measures it.
#
# Three numbers, all required:
#
#   step time    the trainer's own `timing: ... total=` line, which is the
#                mean over a --log-every window. The first window is warm-up
#                (CUDA context, autotune, stream fill) and is dropped. The
#                median over the rest, so one window slowed by a neighbour
#                does not carry the number.
#   peak memory  `scripts/gpu_peak_probe.py`, which sums `nvidia-smi
#                --query-compute-apps` over THIS run's process tree. elisa
#                shares both cards with other sessions, so the card total is
#                not this run's memory and a floor subtracted before the run
#                does not stay true during it.
#   depth cols   how many `cos_err_dj` columns the run wrote. A k-depth run
#                writes k + 1 of them (docs/train_rollout_depth.md), so this
#                is the proof that the flag reached the depth it was given
#                rather than being clamped or ignored. A step time alone
#                would not say which depth produced it.
#
# k = 0 runs first, on the same card, in the same session. It is the
# reference the #373 numbers are quoted against, so the depth's cost is a
# ratio measured here rather than a difference across two machines.
#
# An out-of-memory failure is a result, not an error. It is recorded with
# rc != 0 and the peak the probe saw, because "k = 32 does not fit in 24 GiB"
# is what the run plan needs to know.
#
# The table is the study's committed measurement of four depths, and `bash
# run.sh` runs this stage again. So a depth already in the table is a no-op,
# the same rule the other stages follow, and the table is appended to, never
# truncated. `CF401_SMOKE_FORCE=1` measures a depth again and replaces its
# row.
#
# Usage:  bash smoke_k16.sh [steps]
#         BB_GPU=0 DEPTHS="0 16" bash smoke_k16.sh 400
#         CF401_SMOKE_FORCE=1 DEPTHS=16 bash smoke_k16.sh 400
set -uo pipefail

STEPS="${1:-400}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

DEPTHS="${DEPTHS:-0 16}"
LOG_EVERY="${LOG_EVERY:-100}"
BB_GPU="${BB_GPU:-0}"
PROBE="$CF401_REPO/scripts/gpu_peak_probe.py"
RUNNER="$CF401_PARENT/scripts/run_leg_k.sh"
# Durable, and NOT this study's own root: these checkpoints are throwaway and
# a leg dir named leg_0k under the study root would be read as a real stop.
SCRATCH="${CF401_SMOKE_ROOT:-/home/jupyter/checkpoints_backup/cf-401-smoke}"
OUT="$CF401_RESULTS/smoke_k16.csv"
LOG="$CF401_RESULTS/smoke_k16.log"

[ -f "$PROBE" ]  || { echo "ABORT: no probe at $PROBE" >&2; exit 2; }
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

# `run_one` does `rm -rf "$SCRATCH/k<k>"` twice per depth. An empty or a
# one-level override makes that `rm -rf /k16`, so the value is checked once,
# here, before anything runs. Ten characters is longer than every path that
# could be reached by accident and shorter than every real checkpoint root.
case "$SCRATCH" in
  /) echo "ABORT: CF401_SMOKE_ROOT=/ — this script removes its own subdirs" >&2
     exit 2 ;;
  /*) [ "${#SCRATCH}" -ge 10 ] || {
        echo "ABORT: CF401_SMOKE_ROOT='$SCRATCH' is too short to remove from" >&2
        exit 2; } ;;
  *) echo "ABORT: CF401_SMOKE_ROOT='$SCRATCH' is empty or not absolute" >&2
     exit 2 ;;
esac
mkdir -p "$CF401_RESULTS" "$SCRATCH"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 smoke] $*" | tee -a "$LOG"; }

log "scratch root: $SCRATCH"

HEADER="cell,k,steps,windows,data_ms,fwd_ms,bwd_ms,total_ms,sps,peak_mib,depth_cols,card_free_mib,rc"
if [ -s "$OUT" ]; then
  # Two formats in one file is a table nothing can read. It is cheaper to
  # refuse it here than to find it in a plot.
  [ "$(head -1 "$OUT")" = "$HEADER" ] || {
    echo "ABORT: $OUT has another header — move it away first" >&2; exit 2; }
else
  echo "$HEADER" >"$OUT"
fi

# Is this depth in the table already?
measured(){  # <k>
  awk -F, -v k="$1" 'NR > 1 && $2 == k { hit = 1 } END { exit !hit }' "$OUT"
}

# Drop one depth's row, for a forced re-measure. The new row is appended by
# run_one, so the depth keeps ONE row and the other depths keep theirs.
drop_row(){  # <k>
  awk -F, -v k="$1" 'NR == 1 || $2 != k' "$OUT" >"$OUT.tmp" \
    && mv -f "$OUT.tmp" "$OUT"
}

log "card at start: $(nvidia-smi --id="$BB_GPU" \
  --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)"
log "neighbours at start: $(nvidia-smi --id="$BB_GPU" \
  --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"

run_one(){  # <k>
  local k="$1"
  local root="$SCRATCH/k${k}"
  local suffix="_smoke401_k${k}"
  local free tlog peak rc
  rm -rf "$root"; mkdir -p "$root"
  # The runner APPENDS to its trainer log. A second probe of the same depth
  # would otherwise leave the first probe's windows in the file, the median
  # would run over both, and only the first probe's warm-up window would be
  # dropped. One probe, one log.
  tlog="$CF401_RESULTS/run_$(cf401_run_name "$k")${suffix}.log"
  rm -f "$tlog"
  free=$(nvidia-smi --id="$BB_GPU" --query-gpu=memory.free \
           --format=csv,noheader,nounits | tr -d ' ')

  # The launcher, backgrounded, so the probe can follow its process tree.
  # A fresh scratch root per depth means the leg starts at step 0 and its
  # `leg_0k` skip check finds nothing.
  K="$k" RUNS="$root" CF_STUDY_DIR="$CF401_STUDY" CF_RESULTS="$CF401_RESULTS" \
  WT="$CF401_WT" \
  BB_GPU="$BB_GPU" LOG_EVERY="$LOG_EVERY" SAVE_EVERY=1000000 \
  EXTRA_SAVES="$STEPS" RUN_SUFFIX="$suffix" \
    bash "$RUNNER" "$CF401_CELL" "$STEPS" >>"$LOG" 2>&1 &
  local pid=$!
  # The probe and the reader run as two steps, and both send their errors to
  # this run's log. One pipeline with `2>/dev/null` on each half discarded
  # every cause, so a broken probe left an empty peak column, a `peak=?` line
  # and nothing to read. Now each half's rc is kept and named.
  local probe_json="$CF401_RESULTS/smoke_k${k}_peak.json"
  local probe_out probe_rc read_rc
  probe_out="$(python3 "$PROBE" --root-pid "$pid" --gpu "$BB_GPU" --interval 2 \
                 --out "$probe_json" 2>>"$LOG")"
  probe_rc=$?
  peak="$(printf '%s' "$probe_out" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["peak_mib"])' \
            2>>"$LOG")"
  read_rc=$?
  [ -n "$peak" ] || log "k=$k: the peak probe wrote no number" \
    "(probe rc=$probe_rc, read rc=$read_rc) — cause in $LOG"
  wait "$pid"; rc=$?
  local cols; cols="$(cf401_depth_cols "$root")"

  [ -f "$tlog" ] || { log "k=$k: no trainer log at $tlog (rc=$rc)"
                      echo "$CF401_CELL,$k,$STEPS,0,,,,,,${peak:-},${cols},${free},$rc" >>"$OUT"
                      return 0; }
  # Every `timing:` window after the first, and the `sps` on the step line
  # above it. The median over the windows, so one slow window from a
  # neighbour's burst does not carry the number.
  awk -v cell="$CF401_CELL" -v k="$k" -v steps="$STEPS" -v peak="${peak:-}" \
      -v cols="$cols" -v free="$free" -v rc="$rc" '
    /sps  ETA/ { for (i = 1; i <= NF; i++) if ($i == "sps") sps = $(i-1) }
    /^  timing:/ {
      w++
      if (w == 1) next                              # warm-up window
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        v = kv[2]; sub(/ms$/, "", v)
        if (kv[1] == "data")  d[++nd] = v + 0
        if (kv[1] == "fwd")   f[++nf] = v + 0
        if (kv[1] == "bwd")   b[++nb] = v + 0
        if (kv[1] == "total") t[++nt] = v + 0
      }
      s[++ns] = sps + 0
    }
    # An insertion sort, because elisa`s awk is mawk and mawk has no asort.
    function med(a, n,   i, j, c, tmp) {
      if (n == 0) return ""
      for (i = 1; i <= n; i++) c[i] = a[i]
      for (i = 2; i <= n; i++) {
        tmp = c[i]
        for (j = i - 1; j >= 1 && c[j] > tmp; j--) c[j + 1] = c[j]
        c[j + 1] = tmp
      }
      return (n % 2) ? c[(n + 1) / 2] : (c[n / 2] + c[n / 2 + 1]) / 2
    }
    END {
      printf "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", cell, k, steps, nt,
        med(d, nd), med(f, nf), med(b, nb), med(t, nt), med(s, ns),
        peak, cols, free, rc
    }' "$tlog" >>"$OUT"
  # k + 1 columns at depth k, none at k = 0. Anything else means the flag
  # did not reach the trainer, and the step time above measures a run the
  # study is not going to launch.
  if [ "$k" -gt 0 ] && [ "$cols" != "$(( k + 1 ))" ]; then
    log "k=$k WARNING: $cols cos_err_dj column(s), want $(( k + 1 ))"
  fi
  log "k=$k rc=$rc peak=${peak:-?} MiB cols=${cols:-?} — $(tail -1 "$OUT")"
  rm -rf "$root"
}

for k in $DEPTHS; do
  if measured "$k"; then
    if [ -z "${CF401_SMOKE_FORCE:-}" ]; then
      log "SKIP k=$k — $OUT holds it (CF401_SMOKE_FORCE=1 to measure again)"
      continue
    fi
    log "k=$k is in $OUT and CF401_SMOKE_FORCE is set — replacing its row"
    drop_row "$k"
  fi
  log "START k=$k steps=$STEPS gpu=$BB_GPU log_every=$LOG_EVERY"
  run_one "$k"
done

log "card at end: $(nvidia-smi --id="$BB_GPU" \
  --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)"
echo "--- $OUT ---"; cat "$OUT"
