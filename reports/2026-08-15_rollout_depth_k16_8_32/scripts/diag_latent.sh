#!/bin/bash
# #401 — the three latent probes, over every backbone this study wrote.
#
# The probes read SAVED CHECKPOINTS through the loader the GIFT-Eval uses.
# They train nothing and they touch no GPU, so they run beside the head queue.
#
#   diag_collapse.py --all   rank and pair cosine ACROSS SERIES  -> collapse_all.csv
#   diag_time_rank.py        rank and pair cosine ALONG TIME     -> time_rank.csv
#   diag_scalar_readout.py   what the top direction still carries -> scalar_readout.csv
#
# `diag_collapse.py` also runs once WITHOUT `--all`, over the narrow subject
# list, because `plots/latent_rank.png` draws that file and one bar per
# periodic checkpoint would be 53 bars.
#
# Three probes, one after the other, not three at once. Each holds every
# checkpoint's forward pass on the CPU, and the 97-config GIFT-Eval of the
# head queue runs on those same CPUs. `nice` and the thread cap are what keep
# this job from taking time the queue is spending.
#
# Usage:  bash scripts/diag_latent.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

# `$CF401_STUDY/results/diag`, not `$CF401_RESULTS/diag`. One probe run covers
# BOTH arms in one table — that is the whole point of the `reduce` column — so
# the tables belong to the study, not to one arm's results directory. The
# plots and `make_collapse_table.py` read this path.
DIAG="$CF401_STUDY/results/diag"
mkdir -p "$DIAG"

# Four threads, not 32. The eval shards own the rest.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 latent] $*" \
  | tee -a "$DIAG/diag_latent.log"; }

run(){  # <name> <script> <args...>
  local name="$1"; shift
  log "start $name"
  nice -n 10 python3 "$@" >"$DIAG/$name.out" 2>&1
  local rc=$?
  log "done $name rc=$rc rows=$(( $(wc -l <"$DIAG/$name.csv" 2>/dev/null || echo 1) - 1 ))"
  return $rc
}

rc_all=0
run collapse_all "$HERE/diag_collapse.py" --all --out "$DIAG/collapse_all.csv" \
  || rc_all=$?
run collapse "$HERE/diag_collapse.py" --out "$DIAG/collapse.csv" || rc_all=$?
run time_rank "$HERE/diag_time_rank.py" --out "$DIAG/time_rank.csv" || rc_all=$?
run scalar_readout "$HERE/diag_scalar_readout.py" \
  --out "$DIAG/scalar_readout.csv" || rc_all=$?

log "probes drained rc=$rc_all"
exit $rc_all
