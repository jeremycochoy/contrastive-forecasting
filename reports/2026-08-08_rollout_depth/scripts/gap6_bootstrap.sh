#!/usr/bin/env bash
# #373 review gap 6 — the two head draws on A3's bb200k student, with
# intervals.
#
# Two contrasts, and they answer different questions:
#
#   headseed_draw2v1   the SAME backbone under two head seeds. Nothing but
#                      the head seed moves, so this is a direct measurement
#                      of the head-seed spread on this cell, against the
#                      ±0.0384 band the whole report thresholds on.
#   stop200v100_draw2  bb100k (head seed 20260722) against bb200k under the
#                      second draw. This one is NOT seed-matched: it carries
#                      the stop and the head seed together. It is here to
#                      show how far the ladder's A3 row moves when the
#                      second draw replaces the first, and for nothing else.
#
# `--k0` is the first-named arm and `--k3` the second, so a negative delta
# means the second arm scored better.
#
# Usage: bash gap6_bootstrap.sh [out.csv]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
EVAL="${EVAL:-$RES/eval}"
ROOT_EVAL="${ROOT_EVAL:-/home/jupyter/checkpoints_backup/cf-373/eval}"
OUT="${1:-$RES/gap6_bootstrap.csv}"

csv_for(){ # <tag>
  local tag="$1" p
  for p in "$EVAL/$tag/all_results.csv" \
           "$ROOT_EVAL/$tag/gift/all_results.csv" \
           "$ROOT_EVAL/$tag/all_results.csv"; do
    [ -s "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

rows(){ echo $(( $(wc -l < "$1") - 1 )); }

D1="$(csv_for A3_k3_bb200k_student || true)"
D2="$(csv_for A3_k3_bb200k_student_s20260723 || true)"
S1="$(csv_for A3_k3_bb100k_student || true)"

for f in "$D1" "$D2" "$S1"; do
  [ -n "$f" ] || { echo "ABORT: a 97-config CSV is missing"; exit 3; }
  [ "$(rows "$f")" -eq 97 ] || { echo "ABORT: $f has $(rows "$f") rows, want 97"; exit 4; }
done

rm -f "$OUT"
python3 "$HERE/paired_bootstrap.py" --k0 "$D1" --k3 "$D2" \
  --label "A3_bb200k_student_headseed_draw2v1" --out "$OUT"
python3 "$HERE/paired_bootstrap.py" --k0 "$S1" --k3 "$D2" \
  --label "A3_stop200v100_student_draw2" --out "$OUT"
echo "gap6_bootstrap -> $OUT"
