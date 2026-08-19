#!/bin/bash
# #404 — the rest of the study, in one chain, from elisa.
#
# The four backbones are on the box and the four heads train there now. This
# drives what is left, so no stage waits on a session that is not watching:
#
#   1. wait for the four heads on the box
#   2. pull each head and its bb40k backbone, as soon as it exists
#   3. run the four 97-config GIFT-Evals here, together, on the CPU
#   4. collect the scores and draw the card's four deliverables
#
# Step 2 is a TARGETED pull beside the 15-minute sync loop, not instead of it.
# The loop still walks the whole tree every tick. This one takes the two files
# an eval blocks on, the moment they exist, so a head does not sit on the box
# for a quarter of an hour after it is written.
#
# Usage:
#   nohup setsid bash scripts/drive.sh > results/drive.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"

HOST="${HOST:-ssh1.vast.ai}"
PORT="${PORT:-29998}"
POLL="${POLL:-180}"
BOX_RUNS="${BOX_RUNS:-$CF404_BOX_RUNS}"
ARMS="${ARMS:-$CF404_ARMS}"
STOP="${STOP:-$CF404_STOPS}"
TIMEOUT="${TIMEOUT:-14400}"      # 4 h ceiling on the head wait
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
SYNC_ROOT="$CF404_SYNC_ROOT"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 drive] $*"; }
rsh(){ timeout 120 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

read -r -a arm_list <<<"$ARMS"
tag_of(){ cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS"; }
head_rel(){  # <arm> -> path under the runs root, box and elisa share it
  local t; t="$(tag_of "$1")"
  printf '%s/eval/%s/qhead_%s_s20260722_final.pth\n' "$1" "$t" "$t"
}
bb_rel(){  # <arm>
  printf '%s/%s/leg_%dk/%s_%dk.pth\n' "$1" "$CF404_CELL" $(( STOP / 1000 )) \
    "$(cf404_run_name "$1")" $(( STOP / 1000 ))
}

pull(){  # <relative path> <floor bytes>
  local rel="$1" floor="$2" dst="$SYNC_ROOT/$1"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$floor" ] && return 0
  mkdir -p "$(dirname "$dst")"
  bash "$SAFE_PULL" "$HOST" "$PORT" "$BOX_RUNS/$rel" "$dst" "$floor" >/dev/null 2>&1
  [ -f "$dst" ]
}

# ---- 1 and 2: wait for each head, and take it the moment it lands ------------
say "waiting for the four heads on the box"
waited=0
declare -A have=()
while :; do
  listing="$(rsh "for a in $ARMS; do
      t=\${a}_bb40k_h30k_student
      f=$BOX_RUNS/\$a/eval/\$t/qhead_\${t}_s20260722_final.pth
      [ -f \"\$f\" ] && echo \"\$a \$(wc -c <\"\$f\")\" || echo \"\$a 0\"
    done; echo \"LIVE \$(pgrep -c -f '[t]rain_forecasting_head' || echo 0)\"")"
  live="$(printf '%s\n' "$listing" | awk '/^LIVE /{print $2}')"
  n=0
  while read -r arm size; do
    [ "$arm" = "LIVE" ] && continue
    [ "${size:-0}" -gt 200000 ] || continue
    n=$(( n + 1 ))
    [ -n "${have[$arm]:-}" ] && continue
    say "head $arm on the box, $size B — pulling it and its bb40k"
    pull "$(head_rel "$arm")" 200000 && pull "$(bb_rel "$arm")" 3000000 \
      && { have[$arm]=1; say "head $arm here"; } \
      || say "head $arm did not land yet, retrying next tick"
  done < <(printf '%s\n' "$listing" | grep -v '^LIVE ')

  [ "$n" -ge "${#arm_list[@]}" ] && [ "${#have[@]}" -ge "${#arm_list[@]}" ] && \
    { say "all ${#have[@]} head(s) here"; break; }
  if [ -n "${live:-}" ] && [ "$live" = "0" ] && [ "$waited" -gt 600 ]; then
    say "no head trainer alive and $n of ${#arm_list[@]} head(s) on the box"
    break
  fi
  if [ "$waited" -ge "$TIMEOUT" ]; then
    say "TIMEOUT after ${waited}s with $n of ${#arm_list[@]} head(s)"
    break
  fi
  sleep "$POLL"; waited=$(( waited + POLL ))
done

# ---- 3: the four GIFT-Evals, here, on the CPU --------------------------------
say "starting the GIFT-Evals"
bash "$HERE/evals_elisa.sh"
say "evals rc=$?"

# ---- 4: the deliverables -----------------------------------------------------
say "collect and draw"
CF404_ROOT="$SYNC_ROOT" bash "$HERE/make_plots.sh"
say "DRIVE DONE"
