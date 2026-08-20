#!/bin/bash
# #407 — bring this study's evidence into the git checkout.
#
# The card reuses #373's launcher and #373's stop script, and both write
# where #373 taught them to write:
#
#   <checkout>/reports/2026-08-08_rollout_depth/results/   scores, run logs
#   <durable root>/eval/<tag>/                             heads, GIFT-Eval
#   <durable root>/<cell>/leg_<N>k/                        checkpoints, curves
#
# None of those is this study's directory. This script copies the parts a
# reader of the report needs into `reports/2026-08-20_a4_full_pass/`, so
# #407's numbers do not live inside #373's report.
#
# What comes across, and what does not:
#
#   score_<tag>.txt        the one number per (stop, head).
#   eval/<tag>/            the 97 per-config rows, the summary the score is
#                          read off, and the log that names the backbone
#                          and the head that produced it.
#   results/*.log          what trained, when, and on which card.
#   results/curves/        the losses CSV, downsampled. The raw file is
#                          about 30 MB per leg.
#
# Checkpoints stay out. They are 5 MB each and a reader does not need them.
#
# Idempotent, and safe to run while the driver is still training. An eval
# that is short of 97 configs is skipped rather than copied, so no figure
# can average over fewer configs and read as if it were the study's metric.
#
# Usage: [WT=<checkout>] [RUNS=<durable root>] bash collect.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
GIT_ROOT="$(dirname "$(dirname "$STUDY")")"
# Same knob as run_pass.sh: on a rented box the evidence belongs on the
# durable root, not in a checkout that goes away with the instance.
RES="${CF407_RESULTS:-$STUDY/results}"

WT="${WT:-$GIT_ROOT}"
RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"
DOWNSAMPLE="$GIT_ROOT/scripts/downsample_curve.py"

CELL="arm6_v2_combab_alignS"
RUN_NAME="cf393_${CELL}_cf373k3"
STOPS="${STOPS:-300 450 665}"

mkdir -p "$RES/eval" "$RES/curves"
say(){ echo "[cf407-collect] $*"; }
n_score=0 n_eval=0 n_log=0 n_curve=0 n_skip=0

# ---- the scores, one number per (stop, head) ------------------------------
for stop_k in $STOPS; do
  for head in student teacher; do
    src="$PARENT_RES/score_A4_k3_bb${stop_k}k_${head}.txt"
    [ -s "$src" ] || continue
    cp -f "$src" "$RES/" && n_score=$(( n_score + 1 ))
  done
done

# ---- the evals, one directory per (stop, head) ----------------------------
for stop_k in $STOPS; do
  for head in student teacher; do
    tag="A4_k3_bb${stop_k}k_${head}"
    d="$RUNS/eval/$tag"
    [ -f "$d/gift/all_results.csv" ] || continue
    rows=$(( $(wc -l <"$d/gift/all_results.csv") - 1 ))
    if [ "$rows" -ne 97 ]; then
      say "skip $tag: $rows configs, want 97"; n_skip=$(( n_skip + 1 )); continue
    fi
    mkdir -p "$RES/eval/$tag"
    for f in gift/all_results.csv gift/summary.txt eval_local.log stop.log; do
      [ -f "$d/$f" ] && cp -f "$d/$f" "$RES/eval/$tag/$(basename "$f")"
    done
    n_eval=$(( n_eval + 1 ))
  done
done

# ---- the logs #373's scripts wrote into #373's directory ------------------
for f in "$PARENT_RES/run_${RUN_NAME}.log" "$PARENT_RES/leg_${CELL}.log"; do
  [ -f "$f" ] || continue
  cp -f "$f" "$RES/$(basename "$f")" && n_log=$(( n_log + 1 ))
done

# ---- the training curves, downsampled ------------------------------------
# The leg directory name carries the stop, so the copy keeps it: two legs
# write the same `<run>_losses.csv` name into two directories. The glob
# tolerates train.py's `_rN` infix, which a re-fired leg picks up.
for stop_k in $STOPS; do
  for src in "$RUNS/$CELL/leg_${stop_k}k/${RUN_NAME}"*_losses.csv; do
    [ -f "$src" ] || continue
    dst="$RES/curves/leg_${stop_k}k__$(basename "$src")"
    if [ -f "$DOWNSAMPLE" ]; then
      python3 "$DOWNSAMPLE" "$src" "$dst" --stride 200 --dense-until 0 \
        >/dev/null 2>&1 || { say "downsample failed for $src"; continue; }
    else
      cp -f "$src" "$dst" || continue
    fi
    n_curve=$(( n_curve + 1 ))
  done
done

say "scores $n_score  evals $n_eval  logs $n_log  curves $n_curve  skipped $n_skip"
