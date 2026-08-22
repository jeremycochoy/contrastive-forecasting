#!/bin/bash
# #407 review gap 2 — a confidence interval on each stop, at no GPU cost.
#
# Every (stop, head) keeps its 97 per-config rows in `gift/all_results.csv`.
# The stops share those 97 configs, so a delta between two stops is paired
# per config, and a bootstrap over the configs costs only CPU seconds.
#
# The resampling unit is the DATASET, not the config. #373's
# `paired_bootstrap.py` explains why: `m_dense/H/short`, `m_dense/H/medium`
# and `m_dense/H/long` are three configs of one series, and treating them as
# independent draws makes the interval too narrow. This study calls that
# script rather than writing a second implementation of the study's metric.
#
# WHAT THIS MEASURES: the spread over the 97 GIFT-Eval configs.
# WHAT IT DOES NOT: the spread over head seeds, which `replicate_heads.sh`
# and `head_band.py` cover, and the spread over backbone seeds, which no run
# in this study or its parents has replicated. A narrow interval here does
# not make a difference real.
#
# The baseline is the 200,000-step stop, which is where #373 stopped and
# where the card's question starts.
#
# Usage: [WT=<checkout>] [CF373_ROOT=<root>] stop_bootstrap.sh [stop_k ...]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
GIT_ROOT="$(dirname "$(dirname "$STUDY")")"
RES="${CF407_RESULTS:-$STUDY/results}"

WT="${WT:-$GIT_ROOT}"
RUNS="${CF373_ROOT:-${RUNS:-/home/jupyter/cf373_r3/sync}}"
BOOT="$WT/reports/2026-08-08_rollout_depth/scripts/paired_bootstrap.py"
[ -f "$BOOT" ] || { echo "ABORT: no paired_bootstrap.py at $BOOT" >&2; exit 2; }

BASE_K="${BASE_K:-200}"
ITERS="${ITERS:-10000}"
OUT="$RES/stop_bootstrap.csv"
STOPS_K=("$@")
[ "${#STOPS_K[@]}" -gt 0 ] || STOPS_K=(300 450 665)

# The 97 rows of one (stop, head). `collect.sh` copies them into this study,
# and the durable root holds the original. Take whichever is there.
rows_of(){ # <tag>
  local tag="$1" f
  for f in "$RES/eval/$tag/all_results.csv" "$RUNS/eval/$tag/gift/all_results.csv"; do
    [ -f "$f" ] || continue
    [ "$(awk -F, 'NR>1 && $1!="" && !seen[$1]++ {n++} END {print n+0}' "$f")" -eq 97 ] || continue
    printf '%s\n' "$f"; return 0
  done
  return 1
}

# Start the CSV clean: the bootstrap APPENDS, so a re-run would double every
# row and a reader would average one stop twice.
rm -f "$OUT"
n=0
for head in student teacher; do
  base="$(rows_of "A4_k3_bb${BASE_K}k_${head}")" || {
    echo "[cf407-bootstrap] no ${BASE_K}k $head rows yet — skipping $head"; continue; }
  for stop_k in "${STOPS_K[@]}"; do
    now="$(rows_of "A4_k3_bb${stop_k}k_${head}")" || {
      echo "[cf407-bootstrap] no ${stop_k}k $head rows yet"; continue; }
    echo "[cf407-bootstrap] ${BASE_K}k -> ${stop_k}k  $head"
    python3 "$BOOT" --k0 "$base" --k3 "$now" \
      --label "bb${BASE_K}k_to_bb${stop_k}k_${head}" \
      --iters "$ITERS" --out "$OUT" || exit 1
    n=$(( n + 1 ))
  done
done
echo "[cf407-bootstrap] $n comparisons -> $OUT"
