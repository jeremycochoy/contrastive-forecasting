#!/bin/bash
# #412 pass 5 — the phase-1 re-run at the winning rate, queued behind pass 4.
#
# WHY THIS PASS EXISTS. The card pre-registered one rule before the rate sweep
# ran: a rate distance D above the seed band voids every 1e-3 number, and
# phase 1 runs again at the winning rate. D came in at 0.1675, which is three
# bands, so the rule fired. Three of the six configurations never ran again.
# This lane runs those three at 5.6e-4:
#
#   k32_r200_08_lr56       configuration 3. k = 32, sum, EMA 0.8 to 1.0 with
#                          the ramp at 40,000, so the ramp COMPLETES inside
#                          the stop. Its 1e-3 twin used the mean and a
#                          200,000-step ramp, and it lost the task at step
#                          28,152 with the ramp only at 0.8285.
#   k32_r100_09_dec_lr56   configuration 5. k = 32, sum, L_rep to zero by step
#                          2,000. Its 1e-3 twin used the mean and lost the
#                          task at step 18,634.
#   k8_r100_09_lr56        configuration 6. k = 8, sum. Its 1e-3 twin used the
#                          mean and scored 1.4537.
#
# ALL THREE MOVE THE REDUCTION TO SUM. The mean arms of this card score 1.4404
# to 1.4629 and two of them lost the task. Every sum arm scored 1.18 to 1.35
# and none lost the task. So the sum is the setting under which these three
# configurations can answer the card's question.
#
# WHY NOT `pass2_lane.sh`. That lane pins a list of legs to ONE card and waits
# for free memory only. Two facts break it here. First, pass 4 holds both
# cards, and free memory alone would start a k = 32 leg BESIDE a pass-4 k = 3
# leg on the same card. Second, the box carries other projects, and today one
# card holds 11.7 GB of another project, which no k = 32 leg fits behind.
#
# WHAT THIS LANE DOES.
#   1. It waits until a card holds NO #412 backbone trainer. One #412 leg for
#      each card, so no leg of this pass slows a leg of pass 4.
#   2. On such a card it starts the FIRST pending arm that fits the free
#      memory of that card. A k = 8 arm fits a card that a k = 32 arm does
#      not, so a card never idles while the queue holds work for it.
#   3. It waits for that arm's trainer to appear on the card before it looks
#      at the next card. The gate reads the process table, and a card with a
#      leg that has not started yet reads as free.
#
# It starts no head. `head_sweep.sh` on the other card trains every head and
# runs the 97-config evaluation.
#
# Usage:  nohup bash scripts/pass5_lane.sh >>results/pass5_lane.out 2>&1 &
#         CF412_QUEUE="k8_r100_09_lr56:40000" bash scripts/pass5_lane.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

# The three arms, in the order of the card, each to the 40,000-step stop.
QUEUE="${CF412_QUEUE:-k32_r200_08_lr56:40000 k32_r100_09_dec_lr56:40000 k8_r100_09_lr56:40000}"
CARDS="${CF412_CARDS:-0 1}"
LANE="${CF412_LANE:-pass5}"
# How often the lane reads the cards. A leg is many hours, so a two-minute
# poll costs nothing and a card never sits free for long.
POLL="${CF412_QUEUE_POLL:-120}"
# How long the lane waits for a started leg's trainer to reach the process
# table. `run_arm.sh` waits up to 1,800 s for the trainer's command line, so
# this is that wait plus a margin.
START_TIMEOUT="${CF412_START_TIMEOUT:-2400}"
# How long the whole queue waits for a card. Pass 4 climbs to 200,000 steps,
# which is above one day at this width.
QUEUE_TIMEOUT="${CF412_QUEUE_TIMEOUT:-345600}"

mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 $LANE] $*" \
  | tee -a "$CF412_RESULTS/${LANE}.log"; }

# The card one process runs on, from its own environment.
#
# `run_leg_k.sh` sets CUDA_VISIBLE_DEVICES to BB_GPU, so a TRAINER names its
# card there. A LANE carries BB_GPU only, so that is the second read. A
# process that names neither gives an empty answer and holds no card.
p5_gpu_of_pid(){  # <pid>
  local env
  env="$(tr '\0' '\n' <"/proc/${1:?pid}/environ" 2>/dev/null)"
  printf '%s\n' "$env" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1 \
    | grep -q . && { printf '%s\n' "$env" \
        | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1; return 0; }
  printf '%s\n' "$env" | sed -n 's/^BB_GPU=//p' | head -1
}

# Everything that holds a card for #412, as `<pid> <card>` lines. Two kinds.
#
# A TRAINER. The match is on the trainer's `--run-name`, which every leg of
# this card carries, AND on python as the executable. `pgrep -f` alone matches
# a watcher shell that merely NAMES an arm, and three sessions watch this
# card.
#
# A LANE. Pass 4 runs under `phase1.sh`, which trains a head and runs its
# 97-config evaluation INLINE between two legs of one arm. That window is
# about 4.6 hours, and through it the card holds no backbone trainer. A queue
# that read the trainers alone would start a leg there, and the pass-4 lane
# would then wait for memory that does not come back. The match is on the
# FIRST argument of the process, so a shell that merely names the script does
# not count.
p5_bb_trainers(){
  local pid
  for pid in $(ps -eo pid,args --no-headers 2>/dev/null \
                 | awk '($2 ~ /python/ && /--run-name cf393_.*_cf412_/) ||
                        ($3 ~ /(phase1|pass2_lane)\.sh$/) { print $1 }'); do
    printf '%s %s\n' "$pid" "$(p5_gpu_of_pid "$pid")"
  done
}

# Exit 0 when a #412 leg or lane holds this card.
p5_card_busy(){  # <card>
  p5_bb_trainers | awk -v g="${1:?card}" '$2 == g { found = 1 } END { exit !found }'
}

p5_free_mib(){  # <card>
  nvidia-smi --id="${1:?card}" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

# Wait until this arm's trainer runs on this card. Exit 0 when it does.
p5_wait_for_trainer(){  # <arm> <card>
  local arm="${1:?arm}" card="${2:?card}" name waited=0
  name="$(cf412_run_name "$arm")"
  while [ "$waited" -lt "$START_TIMEOUT" ]; do
    if ps -eo pid,args --no-headers 2>/dev/null \
         | awk -v n="--run-name $name " '$2 ~ /python/ && index($0 " ", n) { exit 0 } END { exit 1 }'
    then
      p5_card_busy "$card" && return 0
    fi
    sleep 15; waited=$(( waited + 15 ))
  done
  return 1
}

# `CF412_QUEUE_CHECK=1` prints what the gate reads, and starts nothing. A
# session proves the gate with it before it launches the queue, and reads the
# state of the cards with it while the queue runs.
if [ -n "${CF412_QUEUE_CHECK:-}" ]; then
  echo "holders — pid card:"
  p5_bb_trainers | sed 's/^/  /'
  for card in $CARDS; do
    printf 'gpu %s  %s MiB free  %s\n' "$card" "$(p5_free_mib "$card")" \
      "$(p5_card_busy "$card" && echo HELD || echo free)"
  done
  exit 0
fi

for leg in $QUEUE; do
  cf412_require_arm "${leg%%:*}" || exit $?
  cf412_require_stop "${leg##*:}" || exit $?
done
log "cards: $CARDS  queue: $QUEUE"

pending=($QUEUE)
pids=(); names=()
waited=0
while [ "${#pending[@]}" -gt 0 ]; do
  for card in $CARDS; do
    [ "${#pending[@]}" -gt 0 ] || break
    # One #412 leg for each card. This is what keeps pass 5 behind pass 4.
    p5_card_busy "$card" && continue
    free="$(p5_free_mib "$card")"
    [ -n "$free" ] || continue
    # The first pending leg this card holds the memory for.
    pick=-1
    for i in "${!pending[@]}"; do
      arm="${pending[$i]%%:*}"
      [ "$free" -ge "$(cf412_leg_vram_mib "$arm")" ] && { pick="$i"; break; }
    done
    [ "$pick" -ge 0 ] || continue
    leg="${pending[$pick]}"; arm="${leg%%:*}"; stop="${leg##*:}"
    unset 'pending[pick]'; pending=(${pending[@]+"${pending[@]}"})
    # A peer session runs this card too. Its trainer would write this arm's
    # losses CSV and this arm's checkpoints beside mine.
    if bash "$HERE/arm_busy.sh" "$arm" >/dev/null 2>&1; then
      log "$arm SKIPPED — a trainer already runs it"
      continue
    fi
    if [ -n "$(cf412_bb_ckpt "$arm" "$stop")" ]; then
      log "$arm at $stop already on disk — the head sweep scores it"
      continue
    fi
    log "$arm -> $stop steps on gpu $card (${free} MiB free," \
        "needs $(cf412_leg_vram_mib "$arm"))"
    BB_GPU="$card" nohup bash "$HERE/run_arm.sh" "$arm" "$stop" \
      >>"$CF412_RESULTS/${LANE}_${arm}_${stop}.log" 2>&1 &
    pids+=("$!"); names+=("$arm:$stop:$card")
    if p5_wait_for_trainer "$arm" "$card"; then
      log "$arm trainer is up on gpu $card"
    else
      log "$arm WARNING — no trainer on gpu $card after ${START_TIMEOUT}s." \
          "See $CF412_RESULTS/${LANE}_${arm}_${stop}.log"
    fi
    waited=0
  done
  [ "${#pending[@]}" -gt 0 ] || break
  if [ "$waited" -ge "$QUEUE_TIMEOUT" ]; then
    log "TIMEOUT after ${waited}s — no card freed for ${pending[*]}"
    break
  fi
  [ $(( waited % 1800 )) -eq 0 ] && [ "$waited" -gt 0 ] && \
    log "waits for a card — pending ${pending[*]}"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  arm="${names[$i]%%:*}"
  if [ "$rc" -eq "$CF412_RC_COLLAPSED" ]; then
    log "${names[$i]} LOST the contrastive task — see $(cf412_collapse_file "$arm")"
  elif [ "$rc" -eq 0 ]; then
    log "${names[$i]} DONE — $(cf412_bb_ckpt "$arm" "${names[$i]#*:}" 2>/dev/null)"
  else
    log "${names[$i]} FAILED rc=$rc"
    failed=$(( failed + 1 ))
  fi
done
log "lane done — $failed failure(s)"
[ "$failed" -eq 0 ] || exit 1
