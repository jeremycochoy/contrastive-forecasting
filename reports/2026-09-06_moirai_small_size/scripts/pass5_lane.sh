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
# The room a leg leaves for a NEIGHBOUR THAT GROWS, in MiB.
#
# `cf412_leg_vram_mib` reads the free memory one time, at the start of a leg.
# A neighbour that grows after that start can kill the leg, or the leg can
# kill the neighbour. The rnd-483 session shares both cards of this box and
# measured its own worker at 3,212 MiB today and at 6,880 MiB on the largest
# point of its sample. That is 3,668 MiB of growth under a leg that has
# already started, so a leg starts only when the card holds that much again.
#
# TODAY THIS RULE REFUSES GPU 0 FOR EVERY ARM OF PASS 5. That card holds
# another user's Jupyter kernel at 11,692 MiB, so it frees to about 9,292 MiB,
# and the smallest arm of this pass needs 8,400 plus this headroom. The
# alternative is a leg with 2,132 MiB of slack against a neighbour that grows
# by 3,668, which makes one of the two jobs fail. One of them is another
# project's.
NEIGHBOUR_MIB="${CF412_NEIGHBOUR_MIB:-3700}"
# The size the rnd-483 worker reaches on its largest point, in MiB. Measured
# by that session over its own 46-point sample.
#
# THE RESERVE IS THE GROWTH THAT IS LEFT, not a fixed number. A neighbour at
# 3,332 MiB can still take 3,548. A neighbour BETWEEN points holds nothing and
# can take the whole 6,880, which is the case the fixed number priced wrong:
# the k = 8 opening of 12:33 came while that worker sat at 744 MiB.
#
# It also stops the queue refusing a card it should take. With a head of 6,252
# MiB on GPU 1 and the worker at 3,332, the card offers 14,980 and the fixed
# rule wants 15,000. The peak sum is 10,418 + 6,252 + 6,880 = 23,550 of
# 24,564, so the leg fits with 1,014 to spare and only the arithmetic said no.
CF412_NEIGHBOUR_PEAK="${CF412_NEIGHBOUR_PEAK:-6880}"

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
# A LANE HOLDS A CARD ONLY WHILE IT STILL HAS A LEG TO RUN THERE.
#
# `phase1.sh` carries ARMS and STOPS in its own environment. A lane whose
# every (arm, stop) already has a checkpoint has no leg left: it is training a
# head or running an evaluation, and it EXITS rather than ask for the card
# again. Counting such a lane idles a card for the 4.6 hours of a head and an
# evaluation. On 2026-09-11 that was GPU 1 with 14,978 MiB free, against a
# k = 32 leg that needs 14,848 and fits beside the head at every peak.
#
# A lane that names NEITHER variable runs the whole card, so it counts. The
# conservative answer is the default.
p5_lane_has_leg_left(){  # <pid>
  local env arms stops a s
  env="$(tr '\0' '\n' <"/proc/${1:?pid}/environ" 2>/dev/null)"
  arms="$(printf '%s' "$env" | sed -n 's/^ARMS=//p' | head -1)"
  stops="$(printf '%s' "$env" | sed -n 's/^STOPS=//p' | head -1)"
  [ -n "$arms" ] && [ -n "$stops" ] || return 0
  for a in $arms; do
    for s in $stops; do
      [ -n "$(cf412_bb_ckpt "$a" "$s")" ] || return 0
    done
  done
  return 1
}

p5_bb_trainers(){
  local pid args
  for pid in $(ps -eo pid,args --no-headers 2>/dev/null \
                 | awk '($2 ~ /python/ && /--run-name cf393_.*_cf412_/) ||
                        ($3 ~ /(phase1|pass2_lane)\.sh$/) { print $1 }'); do
    args="$(ps -o args= -p "$pid" 2>/dev/null)"
    case "$args" in
      *phase1.sh*|*pass2_lane.sh*)
        p5_lane_has_leg_left "$pid" || continue ;;
    esac
    printf '%s %s\n' "$pid" "$(p5_gpu_of_pid "$pid")"
  done
}

# A card another pass CLAIMS, by a file rather than by a process.
#
# WHY A FILE. Pass 5 queues BEHIND pass 4, and a process gate cannot express
# that. Pass 4 held GPU 1 through its lane's `BB_GPU` until 02:30 on
# 2026-09-11, when that lane turned out to be an orphan and its owner killed
# it. After that no process of pass 4 named GPU 1, so the gate read the card
# free, and the next opening would have gone to pass 5 instead of to pass 4's
# `dec10k` arm. The two do not fit together: GPU 1 gives about 19,074 MiB
# beside the rnd-483 worker, a pass-5 k = 32 leg takes 15,000 with the
# headroom, and 4,074 locks out a 7,700 MiB leg.
#
# WHO RELEASES IT. The claiming session, with `rm`. No message to this session
# is needed and no restart, because the queue reads the file on every poll.
p5_card_deferred(){  # <card>
  [ -f "$CF412_RESULTS/pass5_defer_gpu${1:?card}.txt" ]
}

# Exit 0 when a #412 leg or lane holds this card, or another pass claims it.
p5_card_busy(){  # <card>
  p5_card_deferred "${1:?card}" && return 0
  p5_bb_trainers | awk -v g="$1" '$2 == g { found = 1 } END { exit !found }'
}

p5_free_mib(){  # <card>
  nvidia-smi --id="${1:?card}" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

# How much room a growing neighbour still needs on this card, in MiB.
#
# It reads the neighbour's CURRENT size and subtracts it from its measured
# peak. A card with no neighbour reserves the whole peak, because the next
# point can start at any moment.
p5_neighbour_reserve(){  # <card>
  local uuid pid u mem used=0 r
  uuid="$(nvidia-smi --id="${1:?card}" --query-gpu=uuid --format=csv,noheader \
          2>/dev/null | tr -d ' ')"
  [ -n "$uuid" ] || { printf '%s\n' "$NEIGHBOUR_MIB"; return 0; }
  while IFS=, read -r pid u mem; do
    pid="$(printf '%s' "$pid" | tr -dc '0-9')"
    [ "$(printf '%s' "$u" | tr -d ' ')" = "$uuid" ] || continue
    ps -o args= -p "$pid" 2>/dev/null | grep -q 'trainline/train\.py' || continue
    used=$(( used + $(printf '%s' "$mem" | tr -dc '0-9') ))
  done <<EOF
$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null)
EOF
  r=$(( CF412_NEIGHBOUR_PEAK - used ))
  [ "$r" -lt 0 ] && r=0
  printf '%s\n' "$r"
}

# Wait until this arm's trainer runs on this card. Exit 0 when it does.
p5_wait_for_trainer(){  # <arm> <card>
  local arm="${1:?arm}" card="${2:?card}" name waited=0
  name="$(cf412_run_name "$arm")"
  while [ "$waited" -lt "$START_TIMEOUT" ]; do
    # `exit 0` inside a rule jumps to END, and END's own `exit` overrides it,
    # so the rule-exit form ALWAYS returned failure. Every arm of pass 5 then
    # logged "no trainer after 2400s" for a trainer that was running, and the
    # queue waited the full timeout after each start before reading the next
    # card. The match sets a flag and END alone decides.
    if ps -eo pid,args --no-headers 2>/dev/null \
         | awk -v n="--run-name $name " \
             '$2 ~ /python/ && index($0 " ", n) { found = 1 } END { exit !found }'
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
    free="$(p5_free_mib "$card")"
    fits=""
    for leg in $QUEUE; do
      arm="${leg%%:*}"
      [ "${free:-0}" -ge \
        $(( $(cf412_leg_vram_mib "$arm") + $(p5_neighbour_reserve "$card") )) ] \
        && fits="$fits $arm"
    done
    printf 'gpu %s  %s MiB free  %s%s  fits:%s\n' "$card" "$free" \
      "$(p5_card_busy "$card" && echo HELD || echo free)" \
      "$(p5_card_deferred "$card" && echo ' (claimed by another pass)')" \
      "${fits:- none}"
  done
  exit 0
fi

for leg in $QUEUE; do
  cf412_require_arm "${leg%%:*}" || exit $?
  cf412_require_stop "${leg##*:}" || exit $?
done
log "cards: $CARDS  queue: $QUEUE"

# ---- A dead leg goes back in the queue ---------------------------------------
#
# WHY. A leg of this pass is 9 to 13 hours, and it shares a card with another
# project whose worker grows by up to 3,668 MiB. The headroom above makes a
# death unlikely, not impossible, and a death costs the whole arm: the pass
# rests on each arm's score at the 40,000-step stop, and an arm with no
# 40,000-step checkpoint carries no score.
#
# A RE-FIRE IS CHEAP. `run_leg_k.sh` resumes the arm's furthest checkpoint
# with its optimizer state, and the trainer saves every 20,000 steps. So a
# death at step 30,000 costs the 10,000 steps since the last save, not the
# leg.
#
# TWO EXITS ARE RESULTS AND NEVER RETRY.
#   4  the AUC gate stopped the arm. That is a measurement, and the same arm
#      would lose the task again.
#   3  the trainer took an objective this arm does not carry, or it named
#      none. That is a wiring defect, and it repeats.
ATTEMPTS="${CF412_ATTEMPTS:-3}"

retry=($QUEUE)
attempt=1
gave_up=0
while [ "${#retry[@]}" -gt 0 ] && [ "$attempt" -le "$ATTEMPTS" ]; do
[ "$attempt" -gt 1 ] && log "attempt $attempt of $ATTEMPTS for ${retry[*]}"
pending=(${retry[@]+"${retry[@]}"})
retry=()
pids=(); names=()
waited=0
while [ "${#pending[@]}" -gt 0 ]; do
  # An arm whose checkpoint is on disk is done, WHOEVER trained it. This runs
  # before the cards are read, so an arm another session finished leaves this
  # queue at once and the queue can end.
  for i in "${!pending[@]}"; do
    arm="${pending[$i]%%:*}"; stop="${pending[$i]##*:}"
    if [ -n "$(cf412_bb_ckpt "$arm" "$stop")" ]; then
      log "$arm at $stop is on disk — the head sweep scores it"
      unset 'pending[i]'
    fi
  done
  pending=(${pending[@]+"${pending[@]}"})
  [ "${#pending[@]}" -gt 0 ] || break

  for card in $CARDS; do
    [ "${#pending[@]}" -gt 0 ] || break
    # One #412 leg for each card. This is what keeps pass 5 behind pass 4.
    p5_card_busy "$card" && continue
    free="$(p5_free_mib "$card")"
    [ -n "$free" ] || continue
    # The first pending leg this card holds the memory for, with the room a
    # growing neighbour needs on top of it.
    pick=-1
    for i in "${!pending[@]}"; do
      arm="${pending[$i]%%:*}"
      # AN ARM ANOTHER SESSION TRAINS STAYS PENDING. The orchestrator
      # hand-started `k8_r100_09_lr56` under its own lane at 12:33. Skipping
      # the arm and DROPPING it would leave nothing to re-fire if that run
      # died, because this queue only waits on legs it started itself. Held
      # here instead, the arm returns to this queue the moment that run ends
      # with no checkpoint, and the prune above removes it when it succeeds.
      bash "$HERE/arm_busy.sh" "$arm" >/dev/null 2>&1 && continue
      [ "$free" -ge \
        $(( $(cf412_leg_vram_mib "$arm") + $(p5_neighbour_reserve "$card") )) ] \
        && { pick="$i"; break; }
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
    log "$arm -> $stop steps on gpu $card (${free} MiB free, needs" \
        "$(cf412_leg_vram_mib "$arm") plus $(p5_neighbour_reserve "$card")" \
        "for a neighbour that grows)"
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

# A leg still pending here never found a card inside the queue timeout, which
# is four days. That is not a case a re-fire helps.
for leg in ${pending[@]+"${pending[@]}"}; do
  log "$leg NOT STARTED — no card cleared inside ${QUEUE_TIMEOUT}s"
  gave_up=$(( gave_up + 1 ))
done
pending=()

for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  arm="${names[$i]%%:*}"; rest="${names[$i]#*:}"; stop="${rest%%:*}"
  ckpt="$(cf412_bb_ckpt "$arm" "$stop")"
  if [ "$rc" -eq "$CF412_RC_COLLAPSED" ]; then
    log "$arm at $stop LOST the contrastive task — see" \
        "$(cf412_collapse_file "$arm")"
  elif [ -n "$ckpt" ]; then
    log "$arm at $stop DONE — $ckpt"
  elif [ "$rc" -eq 3 ]; then
    log "$arm at $stop STOPPED rc=3 — the leg named the wrong objective or" \
        "none. No re-fire: it repeats. See $(cf412_leg_log "$arm")"
    gave_up=$(( gave_up + 1 ))
  else
    log "$arm at $stop FAILED rc=$rc with no checkpoint — back in the queue"
    retry+=("$arm:$stop")
  fi
done
attempt=$(( attempt + 1 ))
done

for leg in ${retry[@]+"${retry[@]}"}; do
  log "$leg GAVE UP after $ATTEMPTS attempt(s)"
  gave_up=$(( gave_up + 1 ))
done
log "lane done — $gave_up arm(s) with no checkpoint"
[ "$gave_up" -eq 0 ] || exit 1
