#!/bin/bash
# #412 — the head and the 97 GIFT-Eval configs of every checkpoint that has
# neither.
#
# WHY THIS SCRIPT EXISTS. `phase1.sh` runs the head of a stop only when
# `run_arm.sh` returns 0, which is right: a leg that died must not be scored.
# But `run_arm.sh` can return non-zero AFTER a good leg, and then a trained
# checkpoint sits on disk with no score.
#
# That happened on 2026-09-06. `run_arm.sh` was edited in place while lane C
# ran it. Bash reads a script by byte offset and re-reads the file after each
# command, so the running shell resumed inside the NEW text and died on a
# syntax error. The leg was already finished: the 100,000-step checkpoint of
# `k3_r100_09` was on disk, and its head was skipped.
#
# This script closes that gap. It is a sweeper, not a driver: it starts the
# head of a checkpoint that has one neither running nor written, and nothing
# else. Every stage under it is idempotent.
#
# WHAT IT WILL NOT DO. It never starts a second head for one checkpoint. A
# head and an evaluation carry the backbone path on their command line, so a
# stage already running is visible, and the `head_eval.sh` driver above them
# is visible too.
#
# The head waits for `CF412_HEAD_VRAM_MIB` of free memory, so two sweeps on
# one card queue rather than collide.
#
# Usage:  missing_heads.sh                 # start what is missing
#         CF412_DRY_RUN=1 missing_heads.sh # print what it would start
#         BB_GPU=1 missing_heads.sh
#         while :; do missing_heads.sh; sleep 900; done   # the loop
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-0}"
CF412_SWEEP_AGE_MIN="${CF412_SWEEP_AGE_MIN:-5}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 sweep] $*" \
  | tee -a "$CF412_RESULTS/phase1.log"; }

# A process whose command line holds this string, other than this script's own
# shell. The bracket keeps the pattern from matching the pattern.
running(){  # <string>
  local pat="$1"
  pgrep -f "[${pat:0:1}]${pat:1}" >/dev/null 2>&1
}

started=0
for arm in $CF412_ARMS; do
  for stop in $CF412_STOPS; do
    bb="$(cf412_bb_ckpt "$arm" "$stop")"
    [ -n "$bb" ] && [ -f "$bb" ] || continue
    [ -s "$(cf412_score_file "$arm" "$stop")" ] && continue
    # A checkpoint the lane wrote seconds ago is one the lane is about to
    # score itself. `phase1.sh` calls `head_eval.sh` right after a leg
    # returns, and a sweep in that window would start a second head on one
    # checkpoint.
    if [ -n "$(find "$bb" -mmin -"$CF412_SWEEP_AGE_MIN" 2>/dev/null)" ]; then
      log "$arm at $stop — its checkpoint is under $CF412_SWEEP_AGE_MIN" \
          "minute(s) old. Its lane has first claim."
      continue
    fi
    if running "head_eval.sh $arm $stop"; then
      log "$arm at $stop — a driver already runs it"; continue
    fi
    if running "$(basename "$bb")"; then
      log "$arm at $stop — its head or its evaluation already runs"; continue
    fi
    if [ -n "${CF412_DRY_RUN:-}" ]; then
      log "$arm at $stop — WOULD start the head on gpu $BB_GPU"; continue
    fi
    log "$arm at $stop — no score and no head. Starting one on gpu $BB_GPU."
    BB_GPU="$BB_GPU" bash "$HERE/head_eval.sh" "$arm" "$stop" \
      >>"$CF412_RESULTS/sweep.log" 2>&1
    log "$arm at $stop — head rc=$?"
    started=$(( started + 1 ))
  done
done
log "swept $started head(s)"
exit 0
