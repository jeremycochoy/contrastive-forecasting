#!/bin/bash
# #373 — gather the study's artefacts into the git checkout for committing.
#
# Three places hold them and none of them is the checkout:
#
#   $RUN_WT/reports/.../results   the run worktree — launcher logs, verify
#                                 CSVs, step-time tables, score files. The
#                                 launchers refuse a WT under /tmp, so the
#                                 training checkout is not the git one.
#   ~/cf373_sync/<box>/sync       what the sync loops pulled off each box —
#                                 trainer logs and losses CSVs.
#   $CF373_ROOT                   the durable root — checkpoints, heads,
#                                 GIFT-Eval outputs.
#
# Checkpoints stay where they are: an 80 MB backbone does not belong in git.
# Everything else that is small and is evidence comes across.
#
# Usage: bash collect.sh [git checkout root]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="${1:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
RUN_WT="${RUN_WT:-/home/jupyter/wt-cf-373-train}"
SYNC_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
mkdir -p "$DST/results" "$DST/plots"

say(){ echo "[collect] $*"; }

# 1. The run worktree's results.
if [ -d "$RUN_WT/reports/2026-08-08_rollout_depth/results" ]; then
  rsync -a --exclude='*.pth' \
    "$RUN_WT/reports/2026-08-08_rollout_depth/results/" "$DST/results/"
  say "run worktree results"
fi

# 2. Per-box: the trainer log and the losses CSV of every run that box
#    carried. Named by box so two boxes cannot collide on `run_<name>.log`.
for d in "$SYNC_BASE"/*; do
  [ -d "$d" ] || continue
  lbl="$(basename "$d")"
  mkdir -p "$DST/sync/$lbl"
  find "$d" \( -name '*_losses.csv' -o -name '*.log' -o -name '*_latent_drift.csv' \
               -o -name '*_attn_amplitude.csv' \) -type f 2>/dev/null \
  | while read -r f; do
      cp -f "$f" "$DST/sync/$lbl/$(basename "$f")"
    done
  n=$(ls -1 "$DST/sync/$lbl" 2>/dev/null | wc -l)
  say "box $lbl: $n file(s)"
done

# 3. GIFT-Eval outputs: the per-config CSV and the summary, per stop. These
#    are the study's numbers; the head checkpoints beside them are not.
if [ -d "$CF373_ROOT/eval" ]; then
  for d in "$CF373_ROOT"/eval/*/; do
    [ -d "$d" ] || continue
    tag="$(basename "$d")"
    mkdir -p "$DST/results/eval/$tag"
    for f in gift/all_results.csv gift/summary.txt stop.log eval_local.log; do
      [ -f "$d/$f" ] && cp -f "$d/$f" "$DST/results/eval/$tag/$(basename "$f")"
    done
  done
  say "eval outputs: $(ls -1 "$DST/results/eval" 2>/dev/null | wc -l) stop(s)"
fi

say "-> $DST"
