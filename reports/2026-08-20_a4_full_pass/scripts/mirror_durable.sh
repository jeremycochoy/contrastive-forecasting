#!/bin/bash
# #407 review gap 5 — keep this study's numbers off `/tmp`.
#
# The launch used `WT=/tmp/contrastive-forecasting-407`. Three things live
# under that path for 47 hours:
#
#   score_<tag>.txt                  the one number per (stop, head), which
#                                    `stop_k.sh` writes into #373's results.
#   reports/.../a4_full_pass/results this study's evidence.
#   leg_<cell>.log, run_<run>.log    the two logs the continuity gates read.
#
# A `/tmp` clean removes all three. The checkpoints and the GIFT-Eval CSVs
# sit under `/home/jupyter` and survive, so a score stays recomputable, but
# `check_leg_done` reads a log that would be gone and the gate then fails
# closed and stops the study.
#
# This copies all three to a durable root beside the checkpoints. Atomic:
# every file lands as `.tmp` and is moved over the old copy, so a copy that
# dies half way leaves the prior good file. Idempotent, so the watchdog can
# call it every hour and the driver can call it at every stop.
#
# The mirror is a COPY, not the working path. Nothing reads it while the
# study runs. It exists so that a wiped `/tmp` costs a restore rather than
# the study.
#
# Usage: [WT=<checkout>] [CF407_DURABLE=<root>] mirror_durable.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
GIT_ROOT="$(dirname "$(dirname "$STUDY")")"

WT="${WT:-$GIT_ROOT}"
DUR="${CF407_DURABLE:-/home/jupyter/cf407_durable}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"
STUDY_RES="${CF407_RESULTS:-$STUDY/results}"

mkdir -p "$DUR/parent_results" "$DUR/study_results" || exit 2

# Copy one file, and leave the old copy in place if the copy fails.
put(){ # <src> <dst>
  local src="$1" dst="$2"
  [ -f "$src" ] || return 0
  mkdir -p "$(dirname "$dst")" || return 1
  cp -f "$src" "$dst.tmp" || { rm -f "$dst.tmp"; return 1; }
  # A truncated copy is worse than an old one, so an empty copy of a file
  # that has content is thrown away. An empty SOURCE is legal: `.gitkeep`
  # is empty on purpose, and rejecting it would report a failure forever.
  if [ -s "$src" ] && [ ! -s "$dst.tmp" ]; then
    rm -f "$dst.tmp"; return 1
  fi
  mv -f "$dst.tmp" "$dst"
}

n=0 fail=0
# The scores #373's stop script writes, this card's and the replicates'.
for f in "$PARENT_RES"/score_A4_k3_bb*.txt; do
  [ -f "$f" ] || continue
  put "$f" "$DUR/parent_results/$(basename "$f")" && n=$(( n + 1 )) || fail=1
done
# The two logs the continuity gates read.
for f in "$PARENT_RES"/leg_arm6_v2_combab_alignS.log \
         "$PARENT_RES"/run_cf393_arm6_v2_combab_alignS_cf373k3.log \
         "$PARENT_RES"/stops.log; do
  [ -f "$f" ] || continue
  put "$f" "$DUR/parent_results/$(basename "$f")" && n=$(( n + 1 )) || fail=1
done
# This study's evidence, whole. It is a few MB: logs, scores, the 97-row
# CSVs and the downsampled curves.
while IFS= read -r f; do
  rel="${f#$STUDY_RES/}"
  case "$rel" in *.tmp) continue;; esac
  put "$f" "$DUR/study_results/$rel" && n=$(( n + 1 )) || fail=1
done < <(find "$STUDY_RES" -type f 2>/dev/null)

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-mirror] $n files -> $DUR (fail=$fail)"
exit "$fail"
