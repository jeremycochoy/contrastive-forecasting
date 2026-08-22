#!/bin/bash
# #404 round 7 — the winning direction, extended past a momentum of 0.940.
#
# WHY THIS ROUND EXISTS. The card wants a lower GM-Relative MASE. Nine arms
# have scored, and ONE of them goes below the k = 0 parent of this cell:
#
#   r100_09   0.9 to 1.0 over 100,000 steps, holds 0.940 at the 40,000-step
#             stop, scores 1.1507 against the parent's 1.1600.
#
# No arm of rounds 1 to 6 holds more than 0.950 at that stop. Two readings of
# the nine fit that result, and one round separates them:
#
#   - the RAMP LENGTH sets the score. Two arms start at 0.9 and the faster
#     ramp wins by 0.0277. But two arms start at 0.8 and the faster ramp
#     LOSES by 0.0453, so a fast ramp helps from a HIGH start only.
#   - the momentum AT THE STOP sets the score, and 0.940 is not the top of
#     that curve.
#
# THE TWO ARMS. Both go past 0.940 and neither reaches 1.0. A momentum of 1.0
# freezes the teacher, and reports/2026-08-04_ema_sched_ladder/ measured that
# as bad.
#
#   r60_09    --ema-tau 0.9  --ema-tau-end 1.0 --ema-tau-ramp-steps 60000
#             holds 0.967 at the stop
#   r100_095  --ema-tau 0.95 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000
#             holds 0.970 at the stop
#
# The pair differs in the START value at one similar END value, 0.967 against
# 0.970. So the pair says whether the start or the end sets the score.
#
# Everything else is round 1's: k = 32, the mean reduction, the align target
# teacher, align weight 1.0, 40,000 backbone steps, 30,000 head steps, head
# seed 20260722, backbone seed 20260520, the 97-config GIFT-Eval.
#
# ---- THE MACHINE -------------------------------------------------------------
#
# ONE box with ONE card. A datacenter host, reliability at or above 0.99, and
# a desktop-class CPU. TWO lanes share that card: round 6 held three lanes at
# 2.3 steps each second on one 5090, so two lanes are comfortable. This script
# rents no second box.
#
# ---- THE RULE THAT DESTROYS THE BOX ------------------------------------------
#
# Round 6 destroyed a box on a MISSING head checkpoint that it logged as a
# warning. `destroy_box` here asks ONE question: is EVERY head of this round on
# elisa's disk, by name and above MIN_HEAD_BYTES? It returns without acting
# when the answer is no. Every other exit path calls `leave_box_alive`, which
# names the instance and prints the command that destroys it. A box that stays
# alive costs money and a person can see it. A box destroyed early costs the
# run. This is `recover_w3_head.sh`'s rule, over two arms instead of one.
#
# ---- THE BUDGET --------------------------------------------------------------
#
# The credit is $6.58 and the limit for this round is $4. Two lanes to 40,000
# steps take about 4.5 h at round 6's measured rate, and the heads add about
# 45 min on top. At MAX_BID that is about $2.50. MAX_SPEND is $3.20, which
# leaves 7 h of runway before the cap and $0.80 of margin under the limit.
#
# WHAT THE CAP DOES. It PULLS every artefact first, then applies the head
# rule. A cap is a budget event and not a data event: whatever the box holds
# at that moment is still worth a head on elisa.
#
# ---- WHAT IT DOES, in order --------------------------------------------------
#
#   1. gates      — neither arm is scored, and the study reads its own table
#   2. box        — one card, datacenter, a desktop CPU, under MAX_BID
#   3. payload    — #373's bootstrap, then this study, then the arm gates
#   4. sync loop  — 15 min ticks for the whole run (CLAUDE.md)
#   5. lanes      — two backbones on card 0, verified by process AND by file
#   6. heads      — each arm's head starts the moment its own backbone lands
#   7. artefacts  — into the canonical tree under $HOME/cf404_sync/box_a
#   8. teardown   — every head is here FIRST, and only then does the box go
#   9. evals      — the two 97-config GIFT-Evals on elisa's CPUs
#  10. scores     — the two score files, then collect.sh and the figures
#
# Usage:
#   nohup setsid bash scripts/round7.sh > results/round7.out 2>&1 &
#   CF404_DRY_RUN=1 bash scripts/round7.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r7}"
. "$HERE/study.sh"

ARMS="${ARMS:-r60_09 r100_095}"
LABEL="$CF404_BOX_LABEL"
VAST_LABEL="${VAST_LABEL:-cf404-box-r7}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
PR="${PR:-405}"
HEAD_SEED="${HEAD_SEED:-20260722}"

# The canonical tree. Every arm of this card lands here, and the eval, the
# figures and `collect.sh` all read this one root. The 15-minute sync loop
# writes under the round's OWN label, so the two never race on one name.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

POLL="${POLL:-300}"
BB_TIMEOUT="${BB_TIMEOUT:-25200}"      # 7 h on two lanes, against 4.5 h measured
EVAL_TIMEOUT="${EVAL_TIMEOUT:-28800}"  # 8 h on two 97-config evals
DEADLINE_HOURS="${DEADLINE_HOURS:-8}"
MAX_SPEND="${MAX_SPEND:-3.20}"
MIN_HEAD_BYTES="${MIN_HEAD_BYTES:-400000}"
MIN_BB_BYTES="${MIN_BB_BYTES:-3000000}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-20000}"  # two lanes held 5,674 MiB each in round 6
MIN_RELIABILITY="${MIN_RELIABILITY:-0.99}"
MAX_BID="${MAX_BID:-0.45}"
VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|9800X3D|7950X3D|9950X3D|7950X|9950X|7900X|9900X|7700X|9700X}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round7.log"
ENVF="${ENVF:-$CF404_RESULTS/round7.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round7] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

read -r -a arm_list <<<"$ARMS"
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done
cf404_require_stop "$STOP" || exit $?
NARM="${#arm_list[@]}"

KK=$(( STOP / 1000 ))
box_leg(){ printf '%s/%s/%s/leg_%dk\n' "$CF404_BOX_RUNS" "$1" "$CF404_CELL" "$KK"; }
box_bb(){    # <arm> — the backbone checkpoint on the box, or nothing
  rsh "ls -1 $(box_leg "$1")/$(cf404_run_name "$1")_${KK}k.pth 2>/dev/null | head -1"
}
box_head(){  # <arm> — the head checkpoint size on the box, or 0
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "wc -c <$CF404_BOX_RUNS/$1/eval/$tag/qhead_${tag}_s${HEAD_SEED}_final.pth 2>/dev/null" \
    | tr -d ' ' | grep -E '^[0-9]+$' || echo 0
}
box_log(){ printf '/root/cf/%s/results/run_%s.log\n' "$STUDY_REL" "$(cf404_run_name "$1")"; }

# Does a trainer for THIS arm run on the box? The run name is on the trainer's
# own command line, through --run-name and --save-dir, and it carries the arm.
# A box-wide question would answer "yes" for the lane already up, and the
# second lane would never start.
box_arm_running(){  # <arm>
  rsh "pgrep -f '$(cf404_pgrep_pattern "$(cf404_run_name "$1")")' >/dev/null && echo yes"
}
box_head_running(){  # <arm>
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "pgrep -f '$(cf404_pgrep_pattern "qhead_${tag}_s${HEAD_SEED}")' >/dev/null && echo yes"
}

# Where each head must land on elisa before the box can go.
head_here_path(){  # <arm>
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  printf '%s/%s/eval/%s/qhead_%s_s%s_final.pth\n' \
    "$MAIN_ROOT" "$1" "$tag" "$tag" "$HEAD_SEED"
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "round7 arms='$ARMS' box=$VAST_LABEL stop=$STOP head=$CF404_HEAD_STEPS"
  for arm in "${arm_list[@]}"; do
    echo "  $arm alpha=$(cf404_alpha "$arm") $(cf404_schedule "$arm")" \
         "ramp=$(cf404_ramp "$arm") seed=$(cf404_seed "$arm")" \
         "align_w=$(cf404_align_weight "$arm")" \
         "holds $(cf404_momentum_at "$arm" "$STOP") at $STOP"
    echo "    ema  = $(cf404_ema_args "$arm")"
    echo "    head = $(head_here_path "$arm")"
    echo "    score= $(cf404_score_file "$arm" "$STOP")"
  done
  echo "  head seed=$HEAD_SEED canonical root=$MAIN_ROOT sync=$CF404_SYNC_DIR"
  echo "  box: 1 card, datacenter, reliability >= $MIN_RELIABILITY, <= \$$MAX_BID/h,"
  echo "       >= $MIN_VRAM_MIB MiB, Default compute mode, desktop CPU"
  echo "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND pr=#$PR"
  exit 0
fi

echo "$$" >"$CF404_RESULTS/round7.pid"
say "START arms='$ARMS' box=$VAST_LABEL deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 1: the gates ------------------------------------------------------------
todo=0
for arm in "${arm_list[@]}"; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then say "$arm is already scored $(tr -d ' \t\r\n' <"$f")"
  else todo=$(( todo + 1 )); fi
done
[ "$todo" -gt 0 ] || { say "every arm of this round is scored — nothing to rent"; exit 0; }
say "$todo of $NARM arm(s) need a score"

# ---- the teardown, which ONE condition opens ---------------------------------
#
# EVERY head of this round has to be on elisa's disk, by name and by size,
# before the box can go. Round 6 destroyed a box on a missing checkpoint. This
# function cannot.
TORN=0
heads_are_here(){
  local arm p n
  for arm in "${arm_list[@]}"; do
    p="$(head_here_path "$arm")"
    [ -f "$p" ] || return 1
    n="$(wc -c <"$p")"
    [ "${n:-0}" -ge "$MIN_HEAD_BYTES" ] || return 1
  done
}
heads_here_report(){
  local arm p
  for arm in "${arm_list[@]}"; do
    p="$(head_here_path "$arm")"
    printf '%s=%s ' "$arm" "$([ -f "$p" ] && wc -c <"$p" || echo MISSING)"
  done
}
destroy_box(){
  local inst
  [ "$TORN" -eq 1 ] && return 0
  if ! heads_are_here; then
    say "REFUSING to destroy: heads here are $(heads_here_report)"
    return 1
  fi
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  [ -n "$inst" ] || { say "teardown: no instance id in $ENVF"; return 1; }
  TORN=1
  say "teardown: every head is here ($(heads_here_report)) — destroying $inst ($VAST_LABEL)"
  # By pid, from the working directory. NEVER a pattern: on 2026-08-19 a
  # pattern for this loop also matched four running eval shards, and elisa
  # carries other sessions' work.
  say "teardown: stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
  timeout 300 vastrun-destroy "$inst" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}
# The box stays up. This is the loud path, and it names the instance so a
# person can act on one line.
leave_box_alive(){  # <why>
  local inst; inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  say "STOP: $1"
  say "STOP: heads here are $(heads_here_report)"
  say "STOP: the box STAYS ALIVE. Instance ${inst:-?} at ${HOST:-?}:${PORT:-?}"
  say "STOP: destroy it by hand with: vastrun-destroy ${inst:-?} $VAST_LABEL"
}

box_spent(){
  local inst
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  [ -n "$inst" ] || return 0
  timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="$inst" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print substr(v, 2) }'
}

# ---- the pull, defined before the watchdog that calls it --------------------
#
# A targeted pull beside the 15-minute sync loop, not instead of it.
pull(){  # <remote> <local> <floor>
  local dst="$2"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$3" ] && { say "  have $(basename "$dst")"; return 0; }
  mkdir -p "$(dirname "$dst")"
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$dst" "$3" >>"$LOG" 2>&1
  [ -f "$dst" ] || { say "  MISSING $(basename "$dst")"; return 1; }
  say "  $(basename "$dst") $(wc -c <"$dst") B"
}

pull_arm(){  # <arm>
  local arm="$1" NAME TAG RL LL missing=0
  NAME="$(cf404_run_name "$arm")"
  TAG="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
  RL="$(box_leg "$arm")"; LL="$MAIN_ROOT/$arm/$CF404_CELL/leg_${KK}k"
  say "$arm: pulling into $MAIN_ROOT"
  pull "$RL/${NAME}_${KK}k.pth"           "$LL/${NAME}_${KK}k.pth"           "$MIN_BB_BYTES" || missing=1
  pull "$RL/${NAME}_${KK}k_optimizer.pth" "$LL/${NAME}_${KK}k_optimizer.pth" 4000000 || missing=1
  pull "$RL/${NAME}_losses.csv"           "$LL/${NAME}_losses.csv"           1000000 || missing=1
  pull "$RL/${NAME}_attn_amplitude.csv"   "$LL/${NAME}_attn_amplitude.csv"   1000     || missing=1
  pull "$RL/${NAME}_latent_drift.csv"     "$LL/${NAME}_latent_drift.csv"     100      || missing=1
  pull "$CF404_BOX_RUNS/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth" \
       "$MAIN_ROOT/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth" "$MIN_HEAD_BYTES" || missing=1
  # The head's own losses CSV and its stop log. Small files, so a miss on them
  # is not a stop.
  for f in "qhead_${TAG}_s${HEAD_SEED}_losses.csv" "stop.log"; do
    SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" \
      "$CF404_BOX_RUNS/$arm/eval/$TAG/$f" "$MAIN_ROOT/$arm/eval/$TAG/$f" 200 \
      >>"$LOG" 2>&1
  done
  pull "/root/cf/$STUDY_REL/results/run_${NAME}.log" \
       "$CF404_RESULTS/run_${NAME}.log" 1000 || missing=1
  [ "$missing" -eq 0 ]
}

pull_all(){
  local arm
  for arm in "${arm_list[@]}"; do
    pull_arm "$arm" || say "$arm: WARNING — an artefact did not land"
  done
}

# The watchdog holds no other state, so it survives every failure of the stages
# below. A stage that hangs on a dead box would otherwise bill until a person
# looks. It is forked before stage 2, so it carries no HOST and no PORT — it
# reads them back out of the same `.env` file the teardown names the instance
# from.
watchdog(){
  local secs waited=0 spent why=""
  secs="$(awk -v h="$DEADLINE_HOURS" 'BEGIN{printf "%d", h*3600}')"
  while [ "$waited" -lt "$secs" ]; do
    sleep 600; waited=$(( waited + 600 ))
    [ -s "$ENVF" ] || continue
    spent="$(box_spent)"
    [ -n "$spent" ] || continue
    if awk -v s="$spent" -v m="$MAX_SPEND" 'BEGIN{exit !(s+0 >= m+0)}'; then
      why="the box has spent \$$spent of \$$MAX_SPEND"
      break
    fi
  done
  [ -n "$why" ] || why="${DEADLINE_HOURS} h reached"
  say "WATCHDOG: $why — pulling every artefact, then applying the head rule"
  # shellcheck disable=SC1090
  . "$ENVF"
  pull_all
  destroy_box || leave_box_alive "$why, and a head is not here"
}
watchdog & WATCHDOG=$!
stop_watchdog(){ kill -TERM "$WATCHDOG" 2>/dev/null; }
# A failure before the heads land NEVER destroys the box: `destroy_box` holds
# the rule and this path only reports.
die(){ leave_box_alive "$*"; stop_watchdog; exit 1; }

# The ONE exception, and it holds no data. A box this invocation rented seconds
# ago, whose card the study cannot use, carries no checkpoint and no head. It
# goes back, and the round says why. A box read back out of `$ENVF` is NOT
# this case: it may hold a lane from an earlier invocation.
FRESH=0
discard_box(){  # <why>
  local inst
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  if [ "$FRESH" -ne 1 ]; then leave_box_alive "$1"; stop_watchdog; exit 2; fi
  say "DISCARD: $1"
  say "DISCARD: this round rented $inst a moment ago and it holds no artefact"
  timeout 300 vastrun-destroy "$inst" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  rm -f "$ENVF"
  stop_watchdog
  exit 2
}

# ---- 2: the box -------------------------------------------------------------
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
  say "searching — 1 card, datacenter, reliability >= $MIN_RELIABILITY, <= \$$MAX_BID/h"
  out="$(VAST_SEARCH_ARGS="--num-gpus 1 --min-reliability $MIN_RELIABILITY --max-bid $MAX_BID" \
        VAST_SEARCH_LIMIT=40 VAST_CPU_RE="$VAST_CPU_RE" \
        bash "$CF404_PARENT/scripts/provision_box.sh" "$VAST_LABEL" 8 2>>"$LOG")"
  read -r INSTANCE HOST PORT <<<"$(printf '%s\n' "$out" | tail -1)"
  [ -n "${PORT:-}" ] || { say "ABORT: no box"; stop_watchdog; exit 2; }
  printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\n' "$INSTANCE" "$HOST" "$PORT" >"$ENVF"
  FRESH=1
  say "instance $INSTANCE at $HOST:$PORT"
fi

# The card this box actually carries. TWO lanes share ONE card, so the card
# has to be in Default compute mode and it has to hold both lanes.
CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
NCARD="$(printf '%s\n' "$CARD" | grep -c .)"
say "card(s): $(printf '%s' "$CARD" | tr '\n' '|')  count=$NCARD"
[ "${NCARD:-0}" -ge 1 ] || discard_box "no card on the box"
VRAM="$(printf '%s\n' "$CARD" | head -1 | awk -F', ' '{gsub(/[^0-9]/,"",$3); print $3}')"
[ "${VRAM:-0}" -ge "$MIN_VRAM_MIB" ] \
  || discard_box "the card holds ${VRAM:-0} MiB, and $NARM lanes want $MIN_VRAM_MIB MiB"
case "$CARD" in
  *Default*) ;;
  *) discard_box "the card is not in Default compute mode — $NARM lanes cannot share it" ;;
esac
GPUS_BB="$(printf '0 %.0s' "${arm_list[@]}")"; GPUS_BB="${GPUS_BB% }"
say "$NARM arms on card 0, lanes '$GPUS_BB' (Default compute mode, $VRAM MiB)"

# ---- 3: the payload ---------------------------------------------------------
if rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh"; then
  say "the box already carries the study"
else
  say "bootstrap"
  WT="$CF404_REPO" bash "$HERE/bootstrap_box.sh" "$HOST" "$PORT" >>"$LOG" 2>&1 \
    || die "bootstrap failed, see $LOG"
  say "bootstrap OK"
fi
say "the box's arms table holds: $(rsh "awk -F'\t' '!/^#/ && NF>=4 {printf \"%s \", \$1}' /root/cf/$STUDY_REL/scripts/arms.tsv")"

# The box has to hold THIS round's arms table, or it would refuse the new arms.
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  >>"$LOG" 2>&1 || die "the box refuses one of '$ARMS' — its arms table is stale"

# And it has to build each arm's command line with the RAMP on it. This reads
# the box's OWN copy. The two arms of this round differ from the nine that
# scored in the ramp length alone, and `r100_095` differs from `r100_09` in the
# start value alone, so a table that shipped stale would train a duplicate of
# an arm that already has a number.
for arm in "${arm_list[@]}"; do
  got="$(rsh "cd /root/cf/$STUDY_REL && CF404_DRY_RUN=1 bash scripts/run_arm.sh $arm $STOP" \
         | sed -n 's/^  ema=//p')"
  want="$(cf404_ema_args "$arm")"
  [ "$got" = "$want" ] || die "the box builds '$got' for $arm, this table says '$want'"
  say "the box builds $arm at '$got', seed $(cf404_seed "$arm")," \
      "align_w $(cf404_align_weight "$arm"), k=$CF404_K, reduce=$CF404_REDUCE"
done

# ---- 4: the sync loop -------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR," \
    "$(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here"
[ "$(cf404_sync_loops "$CF404_SYNC_DIR")" -ge 1 ] \
  || die "no sync loop for $CF404_SYNC_DIR — CLAUDE.md wants one for the whole run"

# ---- 5: the two lanes, and the proof that both run --------------------------
bb_left(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ -n "$(box_bb "$arm")" ] || n=$(( n + 1 )); done
  echo "$n"
}

started=""
for arm in "${arm_list[@]}"; do
  if [ -n "$(box_bb "$arm")" ]; then
    say "$arm: the ${KK}k backbone is already on the box"
  elif [ -n "$(box_arm_running "$arm")" ]; then
    say "$arm: a trainer already runs on the box"
  else
    say "$arm: starting a lane on card 0"
    rsh "cd /root/cf/$STUDY_REL && mkdir -p results && ARMS='$arm' GPUS='0' \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}_${arm}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
    started="$started $arm"
    sleep 180   # two cold HuggingFace readers on one connection, staggered
  fi
done
say "lanes started this round:${started:- none}"

# The guard line of EVERY new arm, off that arm's own trainer command line. It
# reads the momentum, the reduction, the seed and the align weight back.
if [ -n "$started" ]; then
  say "waiting for the guard line of each new arm"
  waited=0; ok_arms=0; verdict=""; n_new=$(echo $started | wc -w)
  while [ "$waited" -lt 3000 ]; do
    verdict="$(rsh "grep -h 'reached the trainer' /root/cf/$STUDY_REL/results/arms.log 2>/dev/null")"
    stopped="$(rsh "grep -h 'STOPPED' /root/cf/$STUDY_REL/results/arms.log 2>/dev/null | tail -2")"
    [ -n "$stopped" ] && die "the box stopped a leg — $stopped"
    ok_arms=0
    for arm in $started; do
      printf '%s\n' "$verdict" | grep -q "arm $arm " && ok_arms=$(( ok_arms + 1 ))
    done
    [ "$ok_arms" -ge "$n_new" ] && break
    sleep 30; waited=$(( waited + 30 ))
  done
  printf '%s\n' "$verdict" | sed 's/^/  /' | tee -a "$LOG"
  [ "$ok_arms" -ge "$n_new" ] \
    || die "only $ok_arms of $n_new new arm(s) reached a trainer in ${waited}s"
fi

say "waiting for the first rows of each losses CSV"
waited=0
while [ "$waited" -lt 2400 ]; do
  rows_ok=0
  for arm in "${arm_list[@]}"; do
    r="$(rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
              [ -n \"\$csv\" ] && grep -c '^' \"\$csv\" || echo 0")"
    case "$r" in ''|*[!0-9]*) r=0 ;; esac
    [ "$r" -ge 2 ] && rows_ok=$(( rows_ok + 1 ))
  done
  [ "$rows_ok" -ge "$NARM" ] && break
  sleep 30; waited=$(( waited + 30 ))
done

{
  rsh "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
  rsh "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  for arm in "${arm_list[@]}"; do
    echo "--- $arm seed $(cf404_seed "$arm") align_w $(cf404_align_weight "$arm") ---"
    rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
         if [ -n \"\$csv\" ]; then \
           echo \"csv \$csv\"; \
           echo \"depth_cols \$(head -1 \"\$csv\" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*\$')\"; \
           echo \"rows \$(grep -c '^' \"\$csv\")\"; head -3 \"\$csv\" | cut -c1-160; \
         else echo 'csv MISSING'; fi"
    line="$(rsh "grep -h '^Command line:' $(box_log "$arm") 2>/dev/null | tail -1")"
    echo "cmdline ema $(printf '%s' "$line" | cf404_ema_of_cmdline)"
    echo "cmdline seed $(printf '%s' "$line" | cf404_seed_of_cmdline)"
    echo "cmdline align_w $(printf '%s' "$line" | cf404_align_of_cmdline)"
  done
} >"$CF404_RESULTS/round7_verify.txt" 2>&1
sed 's/^/  /' "$CF404_RESULTS/round7_verify.txt" | tee -a "$LOG"

used="$(awk -F', ' 'NR>1{gsub(/[^0-9]/,"",$2); s += $2} END{print s+0}' \
        <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
say "GPU memory in use ${used:-0} MiB, $apps compute app(s), $NARM arm(s) wanted"
[ "${used:-0}" -ge 500 ] || die "the card holds ${used:-0} MiB — no trainer is on it"
[ "$apps" -ge "$NARM" ] || die "$apps compute app(s) for $NARM arm(s)"
for arm in "${arm_list[@]}"; do
  # An exact prefix, never a regex: `--- r100_09 seed` is a prefix of
  # `--- r100_095 seed`, and a loose match would read one arm's block as
  # another's.
  blk="$(awk -v m="--- $arm seed " 'index($0, m) == 1 {f=1} f' "$CF404_RESULTS/round7_verify.txt")"
  cols="$(printf '%s\n' "$blk" | awk '/^depth_cols /{print $2; exit}')"
  [ "${cols:-0}" -eq $(( CF404_K + 1 )) ] \
    || die "arm $arm writes ${cols:-0} depth columns, k=$CF404_K wants $(( CF404_K + 1 ))"
  # The momentum, the seed and the weight the TRAINER runs, not the ones the
  # table wants. The two are compared here for every lane.
  got_ema="$(printf '%s\n' "$blk" | sed -n 's/^cmdline ema //p' | head -1)"
  got_seed="$(printf '%s\n' "$blk" | awk '/^cmdline seed /{print $3; exit}')"
  got_alw="$(printf '%s\n' "$blk" | awk '/^cmdline align_w /{print $3; exit}')"
  [ "$got_ema" = "$(cf404_ema_sig "$arm")" ] \
    || die "arm $arm trains ema '$got_ema', the table says '$(cf404_ema_sig "$arm")'"
  [ "$got_seed" = "$(cf404_seed "$arm")" ] \
    || die "arm $arm trains seed '$got_seed', the table says '$(cf404_seed "$arm")'"
  cf404_num_eq "$got_alw" "$(cf404_align_weight "$arm")" \
    || die "arm $arm trains align_w '$got_alw', the table says '$(cf404_align_weight "$arm")'"
done
say "VERIFIED — $NARM trainer(s) up, each at $(( CF404_K + 1 )) depth columns,"
say "  each at the momentum, the seed and the align weight its row names"
for arm in "${arm_list[@]}"; do
  say "  $arm STEP RATE $(rsh "grep -hoE '[0-9.]+ sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
done

# ---- 6: the heads, each one started as its own backbone lands ---------------
#
# A head reports 0 % GPU utilization on this card, so it costs the lane beside
# it almost nothing, and the first arm is scored while the second still trains.
head_left(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ "$(box_head "$arm")" -gt "$MIN_HEAD_BYTES" ] || n=$(( n + 1 )); done
  echo "$n"
}

say "waiting for ${STOP} steps and a head on $NARM arm(s)"
waited=0
while [ "$(bb_left)" -gt 0 ] || [ "$(head_left)" -gt 0 ]; do
  [ "$waited" -ge "$BB_TIMEOUT" ] && { say "TIMEOUT after ${waited}s — $(bb_left) backbone(s) and $(head_left) head(s) missing"; break; }
  for arm in "${arm_list[@]}"; do
    [ -n "$(box_bb "$arm")" ] || continue
    [ "$(box_head "$arm")" -gt "$MIN_HEAD_BYTES" ] && continue
    [ -n "$(box_head_running "$arm")" ] && continue
    say "$arm: the backbone landed — starting its head, seed $HEAD_SEED"
    rsh "cd /root/cf/$STUDY_REL && mkdir -p results && ARMS='$arm' GPUS='0' \
         nohup setsid bash scripts/heads_box.sh \
           > results/heads_${LABEL}_${arm}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
    sleep 60
  done
  if [ $(( waited % 1800 )) -eq 0 ]; then
    for arm in "${arm_list[@]}"; do
      say "  $arm bb=$(rsh "grep -hoE '^\[ *[0-9]+\].*sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1") head=$(box_head "$arm")"
    done
    say "  spend \$$(box_spent) of \$$MAX_SPEND"
  fi
  sleep "$POLL"; waited=$(( waited + POLL ))
done
say "backbones left $(bb_left), heads left $(head_left)"
say "heads on the box: $(for a in "${arm_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"

# ---- 7: the artefacts, into the canonical tree ------------------------------
#
# The sync loop is stopped FIRST, by pid, so that two writers cannot land on
# one `.tmp` name.
say "stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
for try in 1 2 3; do
  pull_all
  heads_are_here && break
  say "pull pass $try did not land every head — $(heads_here_report)"
  sleep 60
done

# ---- 8: every artefact READS, and only then does the box go -----------------
#
# The 97-config GIFT-Eval runs on elisa's CPUs, so the box does no work during
# it. A box that is gone cannot be pulled from again, so nothing may be wrong
# with what landed. A size floor does not prove that: a half-written checkpoint
# is large. `torch.load` does.
say "checking that every pulled artefact reads, before the box goes"
readable=1
for arm in "${arm_list[@]}"; do
  NAME="$(cf404_run_name "$arm")"
  BB="$MAIN_ROOT/$arm/$CF404_CELL/leg_${KK}k/${NAME}_${KK}k.pth"
  HD="$(head_here_path "$arm")"
  out="$(cd "$CF404_REPO" && python3 - "$BB" "$HD" <<'PYCK' 2>&1
import sys, torch
for path in sys.argv[1:]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        n = len(sd) if hasattr(sd, "__len__") else -1
        print(f"OK {path.split('/')[-1]} {n} tensors")
    except Exception as exc:
        print(f"BAD {path.split('/')[-1]} {type(exc).__name__}: {exc}")
PYCK
)"
  printf '%s\n' "$out" | sed 's/^/  /' | tee -a "$LOG"
  printf '%s\n' "$out" | grep -q '^BAD ' && readable=0
done

if [ "$readable" -eq 1 ] && heads_are_here; then
  say "every artefact reads and every head is here — tearing the box down"
  destroy_box || leave_box_alive "the head rule refused the teardown"
  stop_watchdog
  say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"
else
  leave_box_alive "an artefact did not read, or a head is not here"
  say "  (the watchdog still holds \$$MAX_SPEND and ${DEADLINE_HOURS} h)"
fi

# ---- 9: the two 97-config GIFT-Evals, on elisa ------------------------------
#
# Detached. `head_eval.sh` trains a head that is not on disk and then evals, so
# an arm whose head the box did not finish is covered here too.
say "starting the 97-config GIFT-Evals for '$ARMS' on elisa's CPUs, detached"
ARMS="$ARMS" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  nohup setsid bash "$HERE/evals_elisa.sh" \
    >"$CF404_RESULTS/evals_round7.out" 2>&1 < /dev/null &
sleep 30
say "evals launched, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern head_eval)" | grep -c .) eval process(es) on elisa"

# ---- 10: the scores ---------------------------------------------------------
scored_now(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ -s "$(cf404_score_file "$arm" "$STOP")" ] && n=$(( n + 1 )); done
  echo "$n"
}

say "waiting for $NARM score(s)"
waited=0
while [ "$(scored_now)" -lt "$NARM" ]; do
  [ "$waited" -ge "$EVAL_TIMEOUT" ] && { say "only $(scored_now) score(s) after ${waited}s"; break; }
  [ $(( waited % 1800 )) -eq 0 ] && \
    say "  $(scored_now) of $NARM scored, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern head_eval)" | grep -c .) eval process(es)"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

scored=0
for arm in "${arm_list[@]}"; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then
    say "SCORE $arm $(tr -d ' \t\r\n' <"$f") (holds $(cf404_momentum_at "$arm" "$STOP") at $STOP, align_w $(cf404_align_weight "$arm"))"
    scored=$(( scored + 1 ))
  else say "SCORE $arm MISSING"; fi
done

# ---- the tables and the figures ---------------------------------------------
say "collect.sh, then every figure"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  bash "$HERE/make_plots.sh" >"$CF404_RESULTS/make_plots_round7.out" 2>&1
sed 's/^/  /' "$CF404_RESULTS/make_plots_round7.out" | tail -30 | tee -a "$LOG"

# ---- the box, if stage 8 left it up -----------------------------------------
if [ "$TORN" -eq 0 ]; then
  destroy_box || leave_box_alive "the round ended and a head is still not here"
fi
stop_watchdog
say "ROUND 7 DONE — $scored of $NARM arm(s) scored"
