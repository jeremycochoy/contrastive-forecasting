#!/bin/bash
# #401 diagnosis — is the head-and-eval path at fault, or the backbone?
#
# The three phase-1 scores (2.04, 4.53, 7.93) sit 2x to 7x outside the whole
# range #373 published on this same pipeline (1.0660 to 1.3998). Two parts of
# the path differ from #373: the head budget, 30,000 steps here against
# #373's 15,000 default, and the `ARCH` list `eval_local.sh` passes.
#
# This script puts #401's path, at #401's head budget, on backbones that are
# KNOWN GOOD. If a known-good backbone reproduces its published score, the
# path is clean and the phase-1 numbers measure the backbones they were given.
#
#   C1 pathbound   #373's own G1 subject: the published #379 B5 parent
#                  checkpoint at bb40k. #373 ran it through THIS runner at
#                  15,000 head steps and published 1.2751. Here it runs at
#                  30,000. The delta is the head budget, and nothing else.
#
#   C2 k0anchor    the study's own configuration at k = 0: #393's
#                  `arm6_v2_combab_alignS` parent at bb40k, the checkpoint
#                  cell A4 was built from. It gives the depth ladder a low-k
#                  point measured by #401's path, not read from a report.
#
# Both write to their own root and their own results directory, so no score
# file of this study changes and `collect.sh` reads none of them.
#
# Usage:  BB_GPU=0 bash diag_path.sh            # both controls, in order
#         CF401_DRY_RUN=1 bash diag_path.sh     # print, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

# Not CF401_ROOT and not CF401_RESULTS. A control is not a cell.
DIAG_ROOT="${DIAG_ROOT:-${CF401_ROOT%/}-diag}"
DIAG_RES="${DIAG_RES:-$CF401_RESULTS/diag}"
RUNNER="$CF401_PARENT/scripts/head_eval_bb.sh"
BB_GPU="${BB_GPU:-0}"
HEAD_STEPS="${HEAD_STEPS:-$CF401_HEAD_STEPS_P1}"

# tag <TAB> backbone <TAB> published GM-Relative MASE at 15,000 head steps
CONTROLS=(
  "c1_pathbound_b5pub_bb40k_h30k_student	/home/jupyter/checkpoints_backup/cf-379/runs/bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_40k.pth	1.2751"
  "c2_k0anchor_a4parent_bb40k_h30k_student	/home/jupyter/checkpoints_backup/cf-393/arm6_v2_combab_alignS/leg_40k/cf393_arm6_v2_combab_alignS_40k.pth	-"
)

[ -f "$RUNNER" ] || { echo "ABORT: no head script at $RUNNER" >&2; exit 2; }
mkdir -p "$DIAG_RES"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 diag] $*" \
  | tee -a "$DIAG_RES/diag_path.log"; }

rc_all=0
for row in "${CONTROLS[@]}"; do
  IFS=$'\t' read -r tag bb published <<<"$row"

  if [ -n "${CF401_DRY_RUN:-}" ]; then
    echo "control $tag"
    echo "  bb=$bb"
    echo "  exists=$( [ -f "$bb" ] && echo yes || echo NO )"
    echo "  head steps=$HEAD_STEPS  published@15k=$published"
    echo "  score=$DIAG_RES/score_${tag}.txt"
    echo "  eval=$DIAG_ROOT/eval/$tag"
    continue
  fi

  [ -f "$bb" ] || { log "ABORT: no backbone at $bb"; rc_all=3; continue; }

  log "start $tag steps=$HEAD_STEPS published@15k=$published bb=$(basename "$bb")"
  CF373_ROOT="$DIAG_ROOT" CF_STUDY_DIR="$CF401_STUDY" WT="$CF401_WT" \
    CF_RESULTS="$DIAG_RES" CF_STOP_K=40 BB_GPU="$BB_GPU" \
    bash "$RUNNER" "$tag" "$bb" "$CF401_ENC" "$HEAD_STEPS"
  rc=$?
  log "done $tag rc=$rc score=$(cat "$DIAG_RES/score_${tag}.txt" 2>/dev/null || echo none) published@15k=$published"
  [ $rc -eq 0 ] || rc_all=$rc
done

[ -n "${CF401_DRY_RUN:-}" ] && exit 0
log "controls drained rc=$rc_all"
exit $rc_all
