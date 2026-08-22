#!/bin/bash
# #401 — hand ONE arm from elisa to the box, and prove the move before
# anything on elisa stops.
#
# The two arms started on elisa GPU 0 while the box was rented. The box is the
# primary compute and it takes both, one arm per card. The steps elisa already
# computed stay.
#
# This script does the four parts that must not race, in order, and then it
# STOPS. It never kills the elisa leg. The operator reads the report it prints
# and stops the elisa leg by hand, because "the box resumed at the right step"
# is a judgement over two numbers and a curve, not a file test.
#
#   1. WAIT for the arm's first periodic checkpoint. The legs save every
#      20,000 steps. A move before that save throws away every step the arm
#      holds, so this waits instead of moving.
#   2. MOVE the resume bundle (scripts/move_arm_to_box.sh), which gates on
#      every required file class before one byte leaves.
#   3. START the arm's phase-1 backbone ladder on the box, on its own card,
#      with no head — the box has no GIFT-Eval data.
#   4. VERIFY. The box's trainer must say which step it resumed at, and it
#      must reach a step ABOVE the moved checkpoint. A trainer that silently
#      restarted at 0 (a missing optimizer companion does exactly that) is
#      caught here, while elisa still holds the arm.
#
# Usage:
#   bash scripts/handover_arm.sh <k> <ssh host> <ssh port> <box gpu>
#   bash scripts/handover_arm.sh 8 ssh2.vast.ai 16048 0
set -uo pipefail

K="${1:?usage: handover_arm.sh <k> <ssh host> <ssh port> <box gpu>}"
HOST="${2:?usage: handover_arm.sh <k> <ssh host> <ssh port> <box gpu>}"
PORT="${3:?usage: handover_arm.sh <k> <ssh host> <ssh port> <box gpu>}"
GPU="${4:?usage: handover_arm.sh <k> <ssh host> <ssh port> <box gpu>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
cf401_require_depth "$K" || exit $?

NAME="$(cf401_run_name "$K")"
SRC_LEG="$(cf401_leg_dir "$K" 40000)"
STUDY_REL="reports/$(basename "$CF401_STUDY")"
BOX_CF="${BOX_CF:-/root/cf}"
BOX_STUDY="$BOX_CF/$STUDY_REL"
BOX_RES="$BOX_STUDY/results/$CF401_REDUCE"
WAIT_MAX="${WAIT_MAX:-21600}"     # 6 h. k = 32 needs about 2.5 h from 5,000.
VERIFY_MAX="${VERIFY_MAX:-2400}"  # 40 min. A cold HF reader takes minutes.

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20)
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

OUT="$CF401_RESULTS/handover_k${K}.log"
mkdir -p "$CF401_RESULTS"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [handover k=$K] $*" | tee -a "$OUT"; }

say "START host=$HOST:$PORT box gpu=$GPU"

# ---- 1. Wait for the first periodic save -------------------------------------
# Both the weights and the optimizer companion, because the companion is what
# carries the step counter. The size check is against a re-read a few seconds
# later: torch.save writes in place, so a file that exists is not yet a file
# that is finished.
say "waiting for a periodic checkpoint under $SRC_LEG"
waited=0
while :; do
  pth="$(ls "$SRC_LEG/$NAME"_[0-9]*k.pth 2>/dev/null | grep -v optimizer \
        | sed -E 's|.*_([0-9]+)k\.pth$|\1 &|' | sort -k1,1n | tail -1 \
        | cut -d' ' -f2-)"
  if [ -n "$pth" ] && [ -f "${pth%.pth}_optimizer.pth" ]; then
    a="$(stat -c%s "$pth")$(stat -c%s "${pth%.pth}_optimizer.pth")"
    sleep 20
    b="$(stat -c%s "$pth")$(stat -c%s "${pth%.pth}_optimizer.pth")"
    [ "$a" = "$b" ] && { say "checkpoint ready: $(basename "$pth")"; break; }
    say "checkpoint still growing, waiting"
  fi
  if [ "$waited" -ge "$WAIT_MAX" ]; then
    say "ABORT: no checkpoint in ${WAIT_MAX}s"; exit 3; fi
  sleep 60; waited=$(( waited + 60 ))
done
STEP_K="$(printf '%s\n' "$pth" | sed -E 's|.*_([0-9]+)k\.pth$|\1|')"

# The elisa leg keeps training while the bundle copies, so it may pass the
# moved step. That is not lost work: the box resumes at the moved step and
# elisa's extra steps are a few minutes of a card that is about to take the
# heads.
say "elisa is at step $(tail -1 "$SRC_LEG/${NAME}_losses.csv" | cut -d, -f1)"

# ---- 2. Move ------------------------------------------------------------------
say "moving the resume bundle"
bash "$HERE/move_arm_to_box.sh" "$K" "$HOST" "$PORT" 2>&1 | tee -a "$OUT"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "ABORT: the move failed"; exit 4; }

# ---- 3. Start the arm on the box ---------------------------------------------
# phase1.sh climbs this arm's three stops, resuming each from the one below.
# CF401_HEADS=0: the box has no GIFT-Eval data and no gift_eval package, so a
# head trained here would still wait for elisa. elisa runs heads_watch.sh.
say "starting k=$K on the box, gpu $GPU, backbones only"
rsh "cd '$BOX_STUDY' && CF401_ROOT='$CF401_BOX_RUNS' WT='$BOX_CF' \
     DEPTHS='$K' BB_GPU='$GPU' CF401_HEADS=0 \
     setsid nohup bash scripts/phase1.sh >> '$BOX_RES/phase1_k${K}.out' 2>&1 \
     < /dev/null & disown; exit 0" </dev/null >/dev/null 2>&1
sleep 10
rsh "ps -eo pid,args | grep -c '[p]hase1.sh'" </dev/null | tr -dc '0-9' \
  | grep -qv '^0$' || { say "ABORT: phase1.sh did not start on the box"; exit 5; }
say "phase1.sh runs on the box"

# ---- 4. Verify the step and the curve continue --------------------------------
# Two facts, both out of the box's own trainer log. `Resumed from ... at step N`
# proves the optimizer companion was read — without it `load_training_state`
# falls back and the run starts at 0. A step line ABOVE the moved checkpoint
# proves the run is climbing from there and not from 0.
BOX_LOG="$BOX_RES/run_$NAME.log"
# The step comes from the box's own losses CSV, not from the progress line.
# The progress line prints the step in brackets — `[  13200] loss=...` — and
# carries no `step` word, so a grep for one matches only the resume line and
# would report a run that never moved as a run that climbed.
BOX_CSV="$CF401_BOX_RUNS/k$K/$CF401_CELL/leg_40k/${NAME}_losses.csv"
want=$(( STEP_K * 1000 ))
say "verifying: the box must resume at step $want and climb"
waited=0; resumed=""; cur=""
while [ "$waited" -lt "$VERIFY_MAX" ]; do
  resumed="$(rsh "grep -a 'Resumed from' '$BOX_LOG' 2>/dev/null | tail -1" </dev/null)"
  cur="$(rsh "tail -1 '$BOX_CSV' 2>/dev/null | cut -d, -f1" </dev/null | tr -dc '0-9')"
  if [ -n "$resumed" ] && [ -n "$cur" ] && [ "$cur" -gt "$want" ] 2>/dev/null; then
    break
  fi
  sleep 30; waited=$(( waited + 30 ))
done

echo | tee -a "$OUT"
say "box says: ${resumed:-(no 'Resumed from' line)}"
say "box step: ${cur:-(none)} (elisa handed over step $want)"
say "box loss: $(rsh "tail -1 '$BOX_CSV' 2>/dev/null | cut -d, -f2" </dev/null)"
say "elisa loss at handover: $(awk -F, -v s="$want" '$1 == s {print $2}' \
      "$SRC_LEG/${NAME}_losses.csv" 2>/dev/null | head -1)"

if [ -z "$resumed" ]; then
  say "STOP: the box printed no 'Resumed from' line in ${VERIFY_MAX}s."
  say "  Do NOT stop the elisa leg. Read $BOX_LOG on the box."
  exit 6
fi
case "$resumed" in
  *"at step $want"*) say "OK — the box resumed at step $want" ;;
  *) say "STOP: the box resumed at a step that is not $want."
     say "  Do NOT stop the elisa leg. Read $BOX_LOG on the box."
     exit 6 ;;
esac

say "DONE — k=$K trains on the box from step $want"
say "  The elisa leg for k=$K is STILL RUNNING and holds the card."
say "  Read the two curves, then stop it by hand."
