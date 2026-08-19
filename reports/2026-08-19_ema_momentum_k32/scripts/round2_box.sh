#!/bin/bash
# #404 round 2 — one added arm, from a bare box to its artefacts on elisa.
#
# The review of PR #405 asks for three more arms. Each one is independent, so
# each one takes its own single-card box and the three run at the same time.
#
# Round 1 rented one box with four RTX 5090s at $1.7180/h. The offers that
# carry the CPU this cell needs are single-card boxes at $0.3356/h, so three of
# them cost $1.0068/h against $1.7180/h for the same three lanes. The CPU sets
# the step rate here: the backbone is d_model = 64 at batch 64, and #373
# measured 5.6 to 6.7 steps/s on a Zen 4 desktop part against 1.1 steps/s on an
# EPYC 7452.
#
# The stages, in order:
#
#   1. provision one box, and record its address
#   2. bootstrap it, which is #373's payload plus this study's directory
#   3. start the sync loop for this box (CLAUDE.md: every remote run has one)
#   4. train the backbone to 40,000 steps, detached on the box
#   5. train the 30,000-step student head, detached on the box
#   6. pull the artefacts of this arm into the canonical tree on elisa
#
# It does NOT destroy the box. The review's gap 6 says the box must live until
# every score exists, and the scores come from the GIFT-Evals on elisa.
# `round2.sh` destroys the three boxes after that.
#
# Everything is idempotent. A stage whose output is on disk is a no-op, so a
# re-fired driver after a dead session costs only what did not finish.
#
# Usage:
#   nohup setsid bash scripts/round2_box.sh box_b a085 \
#     > results/round2_box_b.out 2>&1 &
set -uo pipefail

LABEL="${1:?usage: round2_box.sh <box label> <arm>}"
ARM="${2:?usage: round2_box.sh <box label> <arm>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CF404_BOX_LABEL="$LABEL"
. "$HERE/study.sh"
cf404_require_arm "$ARM" || exit $?

STOP="${STOP:-$CF404_STOPS}"
cf404_require_stop "$STOP" || exit $?

# The canonical tree. Round 1's four arms are already here, and the eval, the
# figures and `report_assets.sh` all read this one root. The per-box sync loop
# lands its own copy under $CF404_SYNC_DIR beside it, as the safety net.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"

VAST_LABEL="cf404-${LABEL//_/-}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
ENVF="$CF404_RESULTS/round2_${LABEL}.env"
DONE="$CF404_RESULTS/round2_${LABEL}.done"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
POLL="${POLL:-300}"
BB_TIMEOUT="${BB_TIMEOUT:-32400}"        # 9 h ceiling on the backbone
HEAD_TIMEOUT="${HEAD_TIMEOUT:-14400}"    # 4 h ceiling on the head
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS"
LOG="$CF404_RESULTS/round2_${LABEL}.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 $LABEL/$ARM] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

NAME="$(cf404_run_name "$ARM")"
TAG="$(cf404_tag "$ARM" "$STOP" "$CF404_HEAD_STEPS")"
KK=$(( STOP / 1000 ))
BOX_LEG="$CF404_BOX_RUNS/$ARM/$CF404_CELL/leg_${KK}k"
BOX_HEAD="$CF404_BOX_RUNS/$ARM/eval/$TAG/qhead_${TAG}_s${HEAD_SEED:-20260722}_final.pth"

if [ -f "$DONE" ]; then
  say "already finished — $(cat "$DONE")"
  exit 0
fi

# ---- 1: the box -------------------------------------------------------------
#
# An address on disk is reused when the box still answers. A box that does not
# answer is NOT destroyed here: this driver only ever destroys an instance it
# provisioned itself, and `round2.sh` owns the teardown (CLAUDE.md, the shared
# vast.ai account).
INSTANCE=""; HOST=""; PORT=""
if [ -s "$ENVF" ]; then
  # shellcheck disable=SC1090
  . "$ENVF"
  if [ -n "${HOST:-}" ] && rsh true; then
    say "reusing instance $INSTANCE at $HOST:$PORT"
  else
    say "the box in $ENVF does not answer — provisioning another"
    HOST=""
  fi
fi

if [ -z "${HOST:-}" ]; then
  say "provisioning"
  out="$(VAST_SEARCH_ARGS="${VAST_SEARCH_ARGS:---max-bid 0.45}" \
        VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|7950X|9950X|9800X3D}" \
        bash "$CF404_PARENT/scripts/provision_box.sh" "$VAST_LABEL" \
          "${PROVISION_TRIES:-8}" 2>>"$LOG")"
  read -r INSTANCE HOST PORT <<<"$(printf '%s\n' "$out" | tail -1)"
  [ -n "${PORT:-}" ] || { say "ABORT: no box"; exit 2; }
  printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\n' "$INSTANCE" "$HOST" "$PORT" >"$ENVF"
  say "instance $INSTANCE at $HOST:$PORT"
fi

# ---- 2: the payload ---------------------------------------------------------
if rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh"; then
  say "the box already carries the study"
else
  say "bootstrap"
  WT="$CF404_REPO" bash "$HERE/bootstrap_box.sh" "$HOST" "$PORT" >>"$LOG" 2>&1 \
    || { say "ABORT: bootstrap failed, see $LOG"; exit 3; }
  say "bootstrap OK"
fi

# The arms table on the box has to hold this arm. The bootstrap ships the
# directory of the local checkout, so a stale copy is the one way this can
# fail, and it would train nothing.
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARM' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  >>"$LOG" 2>&1 || { say "ABORT: the box refuses arm $ARM"; exit 3; }

# ---- 3: the sync loop -------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop -> $CF404_SYNC_DIR ($(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here)"

# ---- 4: the backbone --------------------------------------------------------
box_bb(){ rsh "ls -1 $BOX_LEG/${NAME}_${KK}k.pth 2>/dev/null | head -1"; }

if [ -n "$(box_bb)" ]; then
  say "backbone already on the box"
else
  if rsh "pgrep -f 'run_leg_k.sh $CF404_CELL' >/dev/null"; then
    say "a trainer already runs on the box"
  else
    say "starting the backbone"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$ARM' GPUS=0 \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi

  # The momentum and the seed have to reach the trainer. `run_arm.sh` reads
  # both back off the trainer's own command line and stops the leg when either
  # is wrong, so this waits for its verdict rather than for the checkpoint.
  waited=0
  while [ "$waited" -lt 1800 ]; do
    verdict="$(rsh "grep -h 'arm $ARM ' /root/cf/$STUDY_REL/results/arms.log 2>/dev/null | tail -2")"
    case "$verdict" in
      *STOPPED*) say "ABORT: the box stopped the leg — $verdict"; exit 4 ;;
      *"reached the trainer"*) say "guard OK — $verdict"; break ;;
    esac
    sleep 30; waited=$(( waited + 30 ))
  done
  [ "$waited" -lt 1800 ] || say "WARNING: no guard line in 30 min, going on"

  say "waiting for ${STOP} steps"
  waited=0
  while [ -z "$(box_bb)" ]; do
    if [ "$waited" -ge "$BB_TIMEOUT" ]; then
      say "ABORT: no backbone after ${waited}s"; exit 5
    fi
    if [ $(( waited % 1800 )) -eq 0 ]; then
      say "  step $(rsh "grep -h '^step ' /root/cf/$STUDY_REL/results/run_${NAME}.log 2>/dev/null | tail -1" | cut -c1-90)"
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "backbone done"
fi

# ---- 5: the head ------------------------------------------------------------
box_head(){ rsh "test -f $BOX_HEAD && wc -c <$BOX_HEAD"; }

if [ "$(box_head || echo 0)" -gt 200000 ] 2>/dev/null; then
  say "head already on the box"
else
  if rsh "pgrep -f train_forecasting_head >/dev/null"; then
    say "a head trainer already runs on the box"
  else
    say "starting the head"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$ARM' GPUS=0 \
         nohup setsid bash scripts/heads_box.sh \
           > results/heads_${LABEL}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi
  waited=0
  while :; do
    sz="$(box_head || echo 0)"
    [ "${sz:-0}" -gt 200000 ] 2>/dev/null && { say "head done, $sz B"; break; }
    if [ "$waited" -ge "$HEAD_TIMEOUT" ]; then
      say "ABORT: no head after ${waited}s"; exit 6
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
fi

# ---- 6: the artefacts, into the canonical tree ------------------------------
#
# The sync loop walks the whole tree every 15 minutes. These pulls take the
# files the eval and the figures block on, straight into the root round 1
# wrote, so no merge step stands between the box and the score.
pull(){  # <remote> <local> <floor>
  local dst="$2"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$3" ] && { say "  have $(basename "$dst")"; return 0; }
  mkdir -p "$(dirname "$dst")"
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$dst" "$3" >>"$LOG" 2>&1
  [ -f "$dst" ] || { say "  MISSING $(basename "$dst")"; return 1; }
  say "  $(basename "$dst") $(wc -c <"$dst") B"
}

LEG_LOCAL="$MAIN_ROOT/$ARM/$CF404_CELL/leg_${KK}k"
say "pulling into $MAIN_ROOT"
missing=0
pull "$BOX_LEG/${NAME}_${KK}k.pth" "$LEG_LOCAL/${NAME}_${KK}k.pth" 3000000 || missing=1
pull "$BOX_LEG/${NAME}_${KK}k_optimizer.pth" "$LEG_LOCAL/${NAME}_${KK}k_optimizer.pth" 4000000 || missing=1
pull "$BOX_LEG/${NAME}_losses.csv" "$LEG_LOCAL/${NAME}_losses.csv" 1000000 || missing=1
pull "$BOX_LEG/${NAME}_attn_amplitude.csv" "$LEG_LOCAL/${NAME}_attn_amplitude.csv" 1000 || missing=1
pull "$BOX_LEG/${NAME}_latent_drift.csv" "$LEG_LOCAL/${NAME}_latent_drift.csv" 100 || missing=1
pull "$BOX_HEAD" "$MAIN_ROOT/$ARM/eval/$TAG/$(basename "$BOX_HEAD")" 200000 || missing=1
pull "/root/cf/$STUDY_REL/results/run_${NAME}.log" "$MAIN_DIR/results/run_${NAME}.log" 1000 || missing=1

if [ "$missing" -ne 0 ]; then
  say "ABORT: an artefact did not land"
  exit 7
fi

printf 'instance=%s host=%s port=%s at %s\n' "$INSTANCE" "$HOST" "$PORT" \
  "$(date '+%F %T')" >"$DONE"
say "DONE — the box stays up until every score exists"
