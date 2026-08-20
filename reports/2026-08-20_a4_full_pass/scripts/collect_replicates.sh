#!/bin/bash
# #407 — bring the head-seed replicates into the git checkout.
#
# `collect.sh` copies the six points the card asks for. Their tags carry no
# seed, because the card's protocol seed is 20260722 and it is implicit.
# The replicate draws of review gap 1 carry `_s<seed>`, so they need their
# own sweep. Same rule as `collect.sh`: a pair crosses only when its eval
# holds all 97 configs, so no figure can average over fewer.
#
# Usage: [WT=<checkout>] [CF373_ROOT=<durable root>] collect_replicates.sh [stop_k ...]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
GIT_ROOT="$(dirname "$(dirname "$STUDY")")"
RES="${CF407_RESULTS:-$STUDY/results}"

WT="${WT:-$GIT_ROOT}"
RUNS="${CF373_ROOT:-${RUNS:-/home/jupyter/cf373_r3/sync}}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"
SEEDS="${SEEDS:-20260723 20260724}"
WANT_CONFIGS=97

STOPS_K=("$@")
[ "${#STOPS_K[@]}" -gt 0 ] || STOPS_K=(200 300 450 665)

mkdir -p "$RES/eval"
say(){ echo "[cf407-collect-rep] $*"; }
n=0 skip=0

eval_configs(){  # <all_results.csv>
  [ -f "$1" ] || { echo 0; return; }
  awk -F, 'NR > 1 && $1 != "" && !seen[$1]++ { n++ } END { print n + 0 }' "$1"
}

for stop_k in "${STOPS_K[@]}"; do
  for head in student teacher; do
    for seed in $SEEDS; do
      tag="A4_k3_bb${stop_k}k_${head}_s${seed}"
      src="$PARENT_RES/score_${tag}.txt"
      d="$RUNS/eval/$tag"
      [ -s "$src" ] || [ -f "$d/gift/all_results.csv" ] || continue
      rows=$(eval_configs "$d/gift/all_results.csv")
      if [ "$rows" -ne "$WANT_CONFIGS" ]; then
        say "skip $tag: $rows configs, want $WANT_CONFIGS"
        skip=$(( skip + 1 )); continue
      fi
      [ -s "$src" ] && cp -f "$src" "$RES/"
      mkdir -p "$RES/eval/$tag"
      for f in gift/all_results.csv gift/summary.txt eval_local.log stop.log; do
        [ -f "$d/$f" ] && cp -f "$d/$f" "$RES/eval/$tag/$(basename "$f")"
      done
      n=$(( n + 1 ))
    done
  done
done
say "replicate pairs $n  skipped $skip"
