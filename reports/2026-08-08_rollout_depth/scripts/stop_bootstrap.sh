#!/usr/bin/env bash
# #373 — what the second 100,000 backbone steps buy, with an interval.
#
# The round's new contrast is not depth against k = 0. It is stop against
# stop: the SAME cell, the SAME head recipe and the SAME 97 configs, with
# the backbone stopped at 100,000 steps or at 200,000. So the delta is
# paired per config exactly as the depth contrast is, and it takes the same
# dataset-cluster bootstrap.
#
# `--k0` is the 100k arm, `--k3` the 200k arm, so a NEGATIVE delta means the
# longer backbone scored better.
#
# Usage: bash stop_bootstrap.sh [out.csv]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
# The eval CSVs live in the git checkout once r3_collect.sh has copied them
# there. A score lands before that tick does, so a freshly finished eval is
# still only in its sync tree, under `gift/`. Look in both, newest first.
EVAL="${EVAL:-/tmp/contrastive-forecasting-373/reports/2026-08-08_rollout_depth/results/eval}"
SYNC_R3="${SYNC_R3:-/home/jupyter/cf373_r3/sync/eval}"
OUT="${1:-$RES/stop_bootstrap.csv}"

# csv_for <tag> — the 97-config CSV of one eval, wherever it currently is.
csv_for(){
  local tag="$1" p
  for p in "$EVAL/$tag/all_results.csv" \
           "$SYNC_R3/$tag/gift/all_results.csv" \
           "$SYNC_R3/$tag/all_results.csv"; do
    [ -s "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

rm -f "$OUT"
n=0
for cell in A2 A3 A4 B1 B2 B4 B6 B10; do
  for head in student teacher; do
    a="$(csv_for "${cell}_k3_bb100k_${head}" || true)"
    b="$(csv_for "${cell}_k3_bb200k_${head}" || true)"
    if [ -z "$a" ] || [ -z "$b" ]; then
      echo "SKIP ${cell} ${head}: one stop is missing"
      continue
    fi
    # 97 configs each, or the pair is not the protocol's pair.
    ra=$(($(wc -l < "$a") - 1)); rb=$(($(wc -l < "$b") - 1))
    if [ "$ra" -ne 97 ] || [ "$rb" -ne 97 ]; then
      echo "SKIP ${cell} ${head}: rows ${ra}/${rb}, expected 97/97"
      continue
    fi
    python3 "$HERE/paired_bootstrap.py" --k0 "$a" --k3 "$b" \
      --label "${cell}_stop200v100_${head}" --out "$OUT" || continue
    n=$((n + 1))
  done
done
echo "stop_bootstrap: ${n} contrast(s) -> $OUT"
