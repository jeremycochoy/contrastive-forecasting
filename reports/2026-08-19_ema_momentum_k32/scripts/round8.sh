#!/bin/bash
# #404 round 8 — the repeat spread this card measures itself, and the winner
# at a second seed.
#
# WHY THIS ROUND EXISTS. Eleven arms have scored. Ten of the eleven carry ONE
# backbone seed, 20260520. The one repeat, `s08b` at seed 20260521, COLLAPSED:
# its contrastive AUC fell to 0.5745 and it scored 1.5459. So the card ranks
# eleven arms and measures NO spread between two stable runs. Every gap it
# reads as a result stands against 0.007 to 0.015, and that band comes from
# #373's cell at k = 3 against the STUDENT target.
#
# The winner is where this matters most. `r100_09` scores 1.1507 and the k = 0
# parent scores 1.1600. The margin is 0.0093, and 0.0093 sits INSIDE the
# borrowed band. So the card cannot yet say the winner beats the parent.
#
# THE THREE THINGS THIS ROUND RUNS.
#
#   s08c       its backbone is on disk at seed 20260522, AUC 0.9776. Its head
#              is on disk too: round 4 trained it to 30,000 steps at head seed
#              20260722 and its trainer returned rc=0. This round evaluates
#              it. It costs NO GPU.
#   s08d       its backbone is on disk at seed 20260523, AUC 0.9746. Round 4
#              stopped its head at about 25,500 steps of 30,000, so the head
#              trains again. The backbone goes UP to the box.
#   r100_09b   `r100_09` again at backbone seed 20260524. Backbone AND head.
#
# `s08`, `s08c` and `s08d` are then three STABLE seeds of one arm. Their range
# is this cell's OWN repeat spread, measured and not borrowed. `r100_09` and
# `r100_09b` then say whether the winner holds at a second seed.
#
# ---- THE MACHINE -------------------------------------------------------------
#
# ONE box with ONE card. A datacenter host, reliability at or above 0.99, and
# a desktop-class CPU. The head goes FIRST and the backbone follows it. Both
# share card 0: a head reports about 0 % GPU utilization beside a trainer, so
# the two lanes cost each other almost nothing. This script rents no second
# box.
#
# ---- THE RULE THAT DESTROYS THE BOX ------------------------------------------
#
# `destroy_box` asks ONE question: is EVERY head this round trains on elisa's
# disk, by name and above MIN_HEAD_BYTES? It returns without acting when the
# answer is no. Every other exit path calls `leave_box_alive`, which names the
# instance and prints the command that destroys it. This is
# `recover_w3_head.sh`'s rule.
#
# ---- THE BUDGET --------------------------------------------------------------
#
# The credit is $4.20 and the limit for this round is $3. The `r100_09b`
# backbone takes about 4 h at round 7's measured rate, and the two heads add
# about 1 h. At MAX_BID that is about $2.25. MAX_SPEND is $2.60, which leaves
# $0.40 of margin under the limit.
#
# ---- WHAT IT DOES, in order --------------------------------------------------
#
#   1. gates      — which arms need a score, and the study reads its own table
#   2. box        — one card, datacenter, a desktop CPU, under MAX_BID
#   3. payload    — #373's bootstrap, this study, the arm gates, and the
#                   `s08d` backbone, which goes UP
#   4. sync loop  — 15 min ticks for the whole run (CLAUDE.md)
#   5. head 1     — `s08d` on card 0, verified by process AND by file
#   6. lane       — the `r100_09b` backbone on card 0, verified the same way
#   7. head 2     — `r100_09b`'s head, the moment its backbone lands
#   8. artefacts  — into the canonical tree under $HOME/cf404_sync/box_a
#   9. teardown   — every head is here FIRST, and only then does the box go
#  10. evals      — the 97-config GIFT-Evals on elisa's CPUs
#  11. scores     — the score files, then collect.sh and the figures
#
# Usage:
#   nohup setsid bash scripts/round8.sh > results/round8.out 2>&1 &
#   CF404_DRY_RUN=1 bash scripts/round8.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r8}"
. "$HERE/study.sh"

# Every arm this round must score. `s08c` is here because its eval runs, and
# it is NOT in HEAD_ARMS or BB_ARMS because it needs no GPU at all.
ARMS="${ARMS:-s08c s08d r100_09b}"
# The arms whose head this round trains on the box. The teardown gate reads
# THIS list, not ARMS: `s08c` already has a head and the box never makes one.
HEAD_ARMS="${HEAD_ARMS:-s08d r100_09b}"
# The arms whose BACKBONE the box trains. `s08d`'s is on elisa and goes up.
BB_ARMS="${BB_ARMS:-r100_09b}"
# The arms whose backbone this machine pushes to the box.
PUSH_ARMS="${PUSH_ARMS:-s08d}"

LABEL="$CF404_BOX_LABEL"
VAST_LABEL="${VAST_LABEL:-cf404-box-r8}"
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
BB_TIMEOUT="${BB_TIMEOUT:-21600}"      # 6 h against 4.5 h measured
EVAL_TIMEOUT="${EVAL_TIMEOUT:-28800}"  # 8 h on three 97-config evals
DEADLINE_HOURS="${DEADLINE_HOURS:-7}"
MAX_SPEND="${MAX_SPEND:-2.60}"
MIN_HEAD_BYTES="${MIN_HEAD_BYTES:-400000}"
MIN_BB_BYTES="${MIN_BB_BYTES:-3000000}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-16000}"  # one trainer held 5,674 MiB in round 7
MIN_RELIABILITY="${MIN_RELIABILITY:-0.99}"
MAX_BID="${MAX_BID:-0.45}"
VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|9800X3D|7950X3D|9950X3D|7950X|9950X|7900X|9900X|7700X|9700X}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round8.log"
ENVF="${ENVF:-$CF404_RESULTS/round8.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round8] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

read -r -a arm_list  <<<"$ARMS"
read -r -a head_list <<<"$HEAD_ARMS"
read -r -a bb_list   <<<"$BB_ARMS"
read -r -a push_list <<<"$PUSH_ARMS"
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done
cf404_require_stop "$STOP" || exit $?
NARM="${#arm_list[@]}"
NHEAD="${#head_list[@]}"

KK=$(( STOP / 1000 ))
box_leg(){ printf '%s/%s/%s/leg_%dk\n' "$CF404_BOX_RUNS" "$1" "$CF404_CELL" "$KK"; }
box_bb_path(){ printf '%s/%s_%dk.pth\n' "$(box_leg "$1")" "$(cf404_run_name "$1")" "$KK"; }
box_bb(){    # <arm> — the backbone checkpoint on the box, or nothing
  rsh "ls -1 $(box_bb_path "$1") 2>/dev/null | head -1"
}
box_head(){  # <arm> — the head checkpoint size on the box, or 0
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "wc -c <$CF404_BOX_RUNS/$1/eval/$tag/qhead_${tag}_s${HEAD_SEED}_final.pth 2>/dev/null" \
    | tr -d ' ' | grep -E '^[0-9]+$' || echo 0
}
box_log(){ printf '/root/cf/%s/results/run_%s.log\n' "$STUDY_REL" "$(cf404_run_name "$1")"; }
box_head_log(){  # <arm> — #373's head script writes its stop log here
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  printf '%s/%s/eval/%s/stop.log\n' "$CF404_BOX_RUNS" "$1" "$tag"
}

# Does a trainer for THIS arm run on the box? The run name is on the trainer's
# own command line, through --run-name and --save-dir, and it carries the arm.
box_arm_running(){  # <arm>
  rsh "pgrep -f '$(cf404_pgrep_pattern "$(cf404_run_name "$1")")' >/dev/null && echo yes"
}

# Does a head for THIS arm run OR WAIT on the box?
#
# ROUND 7 GOT THIS WRONG. It matched `qhead_<tag>_s<seed>`, which is the
# PYTHON TRAINER's command line alone. #373's `head_eval_bb.sh` takes a
# per-card lock before it trains, `flock -w 86400` on /tmp/cf373_head_gpu0.lock,
# so ONE head trains per card at a time. A head QUEUED behind that lock carries
# the tag WITHOUT the `qhead_` prefix, so the old pattern missed it and the
# driver read a waiting head as a dead one. It then started another every poll.
#
# The TAG matches both: `head_eval_bb.sh <tag> ...` on the queued shell, and
# `--save-path .../qhead_<tag>_s<seed>_...` on the trainer. The two tags of this
# round differ before `_bb`, so neither is a prefix of the other.
box_head_running(){  # <arm>
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "pgrep -f '$(cf404_pgrep_pattern "$tag")' >/dev/null && echo yes"
}

# Where each head must land on elisa before the box can go.
head_here_path(){  # <arm>
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  printf '%s/%s/eval/%s/qhead_%s_s%s_final.pth\n' \
    "$MAIN_ROOT" "$1" "$tag" "$tag" "$HEAD_SEED"
}
# The backbone this machine holds for an arm the box does not train.
bb_here_path(){  # <arm>
  printf '%s/%s/%s/leg_%dk/%s_%dk.pth\n' \
    "$MAIN_ROOT" "$1" "$CF404_CELL" "$KK" "$(cf404_run_name "$1")" "$KK"
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "round8 arms='$ARMS' heads='$HEAD_ARMS' backbones='$BB_ARMS' push='$PUSH_ARMS'"
  echo "  box=$VAST_LABEL stop=$STOP head=$CF404_HEAD_STEPS seed=$HEAD_SEED"
  for arm in "${arm_list[@]}"; do
    echo "  $arm alpha=$(cf404_alpha "$arm") $(cf404_schedule "$arm")" \
         "ramp=$(cf404_ramp "$arm") seed=$(cf404_seed "$arm")" \
         "align_w=$(cf404_align_weight "$arm")" \
         "holds $(cf404_momentum_at "$arm" "$STOP") at $STOP"
    echo "    ema  = $(cf404_ema_args "$arm")"
    echo "    bb   = $(bb_here_path "$arm") $([ -f "$(bb_here_path "$arm")" ] && wc -c <"$(bb_here_path "$arm")" || echo MISSING)"
    echo "    head = $(head_here_path "$arm") $([ -f "$(head_here_path "$arm")" ] && wc -c <"$(head_here_path "$arm")" || echo MISSING)"
    echo "    score= $(cf404_score_file "$arm" "$STOP")"
  done
  echo "  canonical root=$MAIN_ROOT sync=$CF404_SYNC_DIR"
  echo "  box: 1 card, datacenter, reliability >= $MIN_RELIABILITY, <= \$$MAX_BID/h,"
  echo "       >= $MIN_VRAM_MIB MiB, Default compute mode, desktop CPU"
  echo "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND pr=#$PR"
  exit 0
fi

echo "$$" >"$CF404_RESULTS/round8.pid"
say "START arms='$ARMS' heads='$HEAD_ARMS' box=$VAST_LABEL deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 1: the gates ------------------------------------------------------------
todo=0
for arm in "${arm_list[@]}"; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then say "$arm is already scored $(tr -d ' \t\r\n' <"$f")"
  else todo=$(( todo + 1 )); fi
done
say "$todo of $NARM arm(s) need a score"

# Which of them still needs GPU time. An arm whose head is already here needs
# none, and a round with no GPU work rents nothing.
gpu_todo=0
for arm in "${head_list[@]}"; do
  p="$(head_here_path "$arm")"
  if [ -f "$p" ] && [ "$(wc -c <"$p")" -ge "$MIN_HEAD_BYTES" ]; then
    say "$arm: its head is already here, $(wc -c <"$p") B"
  else gpu_todo=$(( gpu_todo + 1 )); fi
done
[ "$gpu_todo" -gt 0 ] || { say "every head of this round is here — nothing to rent"; }

# Every backbone this machine must push has to BE here first.
for arm in "${push_list[@]}"; do
  p="$(bb_here_path "$arm")"
  [ -f "$p" ] && [ "$(wc -c <"$p")" -ge "$MIN_BB_BYTES" ] \
    || { say "ABORT: $arm has no ${KK}k backbone here at $p"; exit 2; }
  say "$arm: the ${KK}k backbone is here, $(wc -c <"$p") B — it goes UP"
done

# ---- the teardown, which ONE condition opens ---------------------------------
#
# EVERY head this round TRAINS has to be on elisa's disk, by name and by size,
# before the box can go. `s08c` is not in this list: its head is already here
# and the box never makes one.
TORN=0
heads_are_here(){
  local arm p n
  for arm in "${head_list[@]}"; do
    p="$(head_here_path "$arm")"
    [ -f "$p" ] || return 1
    n="$(wc -c <"$p")"
    [ "${n:-0}" -ge "$MIN_HEAD_BYTES" ] || return 1
  done
}
heads_here_report(){
  local arm p
  for arm in "${head_list[@]}"; do
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
pull(){  # <remote> <local> <floor>
  local dst="$2"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$3" ] && { say "  have $(basename "$dst")"; return 0; }
  mkdir -p "$(dirname "$dst")"
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$dst" "$3" >>"$LOG" 2>&1
  [ -f "$dst" ] || { say "  MISSING $(basename "$dst")"; return 1; }
  say "  $(basename "$dst") $(wc -c <"$dst") B"
}

# An arm the box TRAINED: backbone, optimizer, curves, log and head.
pull_bb_arm(){  # <arm>
  local arm="$1" NAME RL LL missing=0
  NAME="$(cf404_run_name "$arm")"
  RL="$(box_leg "$arm")"; LL="$MAIN_ROOT/$arm/$CF404_CELL/leg_${KK}k"
  say "$arm: pulling the backbone side into $MAIN_ROOT"
  pull "$RL/${NAME}_${KK}k.pth"           "$LL/${NAME}_${KK}k.pth"           "$MIN_BB_BYTES" || missing=1
  pull "$RL/${NAME}_${KK}k_optimizer.pth" "$LL/${NAME}_${KK}k_optimizer.pth" 4000000 || missing=1
  pull "$RL/${NAME}_losses.csv"           "$LL/${NAME}_losses.csv"           1000000 || missing=1
  pull "$RL/${NAME}_attn_amplitude.csv"   "$LL/${NAME}_attn_amplitude.csv"   1000     || missing=1
  pull "$RL/${NAME}_latent_drift.csv"     "$LL/${NAME}_latent_drift.csv"     100      || missing=1
  pull "/root/cf/$STUDY_REL/results/run_${NAME}.log" \
       "$CF404_RESULTS/run_${NAME}.log" 1000 || missing=1
  [ "$missing" -eq 0 ]
}

# An arm the box gave a HEAD: the head, its losses CSV and its stop log.
pull_head_arm(){  # <arm>
  local arm="$1" TAG missing=0
  TAG="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
  say "$arm: pulling the head into $MAIN_ROOT"
  pull "$CF404_BOX_RUNS/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth" \
       "$MAIN_ROOT/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth" "$MIN_HEAD_BYTES" || missing=1
  # Small files, so a miss on them is not a stop.
  for f in "qhead_${TAG}_s${HEAD_SEED}_final_optimizer.pth" \
           "qhead_${TAG}_s${HEAD_SEED}_losses.csv" "stop.log"; do
    SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" \
      "$CF404_BOX_RUNS/$arm/eval/$TAG/$f" "$MAIN_ROOT/$arm/eval/$TAG/$f" 200 \
      >>"$LOG" 2>&1
  done
  [ "$missing" -eq 0 ]
}

pull_all(){
  local arm
  for arm in "${bb_list[@]}"; do
    pull_bb_arm "$arm" || say "$arm: WARNING — a backbone artefact did not land"
  done
  for arm in "${head_list[@]}"; do
    pull_head_arm "$arm" || say "$arm: WARNING — the head did not land"
  done
}

# The watchdog holds no other state, so it survives every failure of the stages
# below. It is forked before stage 2, so it carries no HOST and no PORT — it
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
# A failure before the heads land NEVER destroys the box.
die(){ leave_box_alive "$*"; stop_watchdog; exit 1; }

# The ONE exception, and it holds no data. A box this invocation rented seconds
# ago, whose card the study cannot use, carries no checkpoint and no head.
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

# The card this box actually carries. A head and a trainer share ONE card, so
# the card has to be in Default compute mode.
CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
NCARD="$(printf '%s\n' "$CARD" | grep -c .)"
say "card(s): $(printf '%s' "$CARD" | tr '\n' '|')  count=$NCARD"
[ "${NCARD:-0}" -ge 1 ] || discard_box "no card on the box"
VRAM="$(printf '%s\n' "$CARD" | head -1 | awk -F', ' '{gsub(/[^0-9]/,"",$3); print $3}')"
[ "${VRAM:-0}" -ge "$MIN_VRAM_MIB" ] \
  || discard_box "the card holds ${VRAM:-0} MiB, and this round wants $MIN_VRAM_MIB MiB"
case "$CARD" in
  *Default*) ;;
  *) discard_box "the card is not in Default compute mode — a head and a trainer cannot share it" ;;
esac
say "the head lane and the backbone lane share card 0 (Default compute mode, $VRAM MiB)"

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

# The box has to hold THIS round's arms table, or it would refuse the new arm.
rsh "cd /root/cf/$STUDY_REL && ARMS='$BB_ARMS' GPUS='0' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  >>"$LOG" 2>&1 || die "the box refuses '$BB_ARMS' — its arms table is stale"
rsh "cd /root/cf/$STUDY_REL && ARMS='$HEAD_ARMS' GPUS='0' CF404_DRY_RUN=1 bash scripts/heads_box.sh" \
  >>"$LOG" 2>&1 || die "the box refuses a head of '$HEAD_ARMS' — its arms table is stale"

# And it has to build the backbone arm's command line with the RAMP and the
# SEED on it. `r100_09b` differs from `r100_09` in the SEED alone, so a table
# that shipped stale would train a duplicate of an arm that already has a
# number, under a name that says otherwise. This reads the box's OWN copy.
for arm in "${bb_list[@]}"; do
  got="$(rsh "cd /root/cf/$STUDY_REL && CF404_DRY_RUN=1 bash scripts/run_arm.sh $arm $STOP" \
         | sed -n 's/^  ema=//p')"
  want="$(cf404_ema_args "$arm")"
  [ "$got" = "$want" ] || die "the box builds '$got' for $arm, this table says '$want'"
  got_seed="$(rsh "cd /root/cf/$STUDY_REL && CF404_DRY_RUN=1 bash scripts/run_arm.sh $arm $STOP" \
              | sed -n 's/^  seed=//p')"
  say "the box builds $arm at '$got', seed '${got_seed:-<not printed>}'," \
      "align_w $(cf404_align_weight "$arm"), k=$CF404_K, reduce=$CF404_REDUCE"
done

# The backbone of every push arm goes UP. It trained on a box that is gone.
for arm in "${push_list[@]}"; do
  src="$(bb_here_path "$arm")"; dst="$(box_bb_path "$arm")"
  want="$(wc -c <"$src")"
  if [ "$(rsh "wc -c <$dst 2>/dev/null" | tr -d ' ')" = "$want" ]; then
    say "$arm: the backbone is already on the box, $want B"
  else
    say "$arm: pushing the backbone, $want B"
    rsh "mkdir -p $(dirname "$dst")" || die "cannot make $arm's leg directory on the box"
    timeout 600 scp -q "${SSH_OPTS[@]}" -P "$PORT" "$src" "root@$HOST:$dst.tmp" \
      || die "$arm's backbone did not go up"
    rsh "mv $dst.tmp $dst"
    got="$(rsh "wc -c <$dst 2>/dev/null" | tr -d ' ')"
    [ "$got" = "$want" ] || die "$arm's backbone on the box is $got B, not $want B"
    say "$arm: the backbone is on the box, $got B"
  fi
done

# ---- 4: the sync loop -------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR," \
    "$(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here"
[ "$(cf404_sync_loops "$CF404_SYNC_DIR")" -ge 1 ] \
  || die "no sync loop for $CF404_SYNC_DIR — CLAUDE.md wants one for the whole run"

# ---- 5: the head of every arm whose backbone is ALREADY on the box ----------
#
# The card puts the head first. It costs 31 minutes and it de-risks the round:
# an arm whose head lands early is an arm the teardown gate can already pass.
start_head(){  # <arm>
  say "$arm: starting its head, $CF404_HEAD_STEPS steps, seed $HEAD_SEED, card 0"
  rsh "cd /root/cf/$STUDY_REL && mkdir -p results && ARMS='$1' GPUS='0' \
       nohup setsid bash scripts/heads_box.sh \
         > results/heads_${LABEL}_${1}.out 2>&1 < /dev/null & echo started" \
    >>"$LOG" 2>&1
}

first_heads=""
for arm in "${head_list[@]}"; do
  cf404_is_in "$arm" "$BB_ARMS" && continue      # its backbone is not there yet
  if [ "$(box_head "$arm")" -gt "$MIN_HEAD_BYTES" ]; then
    say "$arm: its head is already on the box"
  elif [ -n "$(box_head_running "$arm")" ]; then
    say "$arm: a head already runs or waits on the box"
  else
    start_head "$arm"; first_heads="$first_heads $arm"
    sleep 90
  fi
done
say "head lane(s) started first:${first_heads:- none}"

# By process AND by file. A head that took the lock and waits writes no file,
# so both answers go into the log and only the pair is a launch.
for arm in $first_heads; do
  waited=0
  while [ "$waited" -lt 900 ]; do
    steps="$(rsh "grep -aoE '^\[ *[0-9]+\]' $(box_head_log "$arm") 2>/dev/null | tail -1")"
    [ -n "$steps" ] && break
    sleep 30; waited=$(( waited + 30 ))
  done
  say "$arm head: process=$(box_head_running "$arm" | tr -d '\n')" \
      "file=${steps:-<no step line yet>} after ${waited}s"
done

# ---- 6: the backbone lane ---------------------------------------------------
started=""
for arm in "${bb_list[@]}"; do
  if [ -n "$(box_bb "$arm")" ]; then
    say "$arm: the ${KK}k backbone is already on the box"
  elif [ -n "$(box_arm_running "$arm")" ]; then
    say "$arm: a trainer already runs on the box"
  else
    say "$arm: starting a backbone lane on card 0"
    rsh "cd /root/cf/$STUDY_REL && mkdir -p results && ARMS='$arm' GPUS='0' \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}_${arm}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
    started="$started $arm"
    sleep 120   # two cold HuggingFace readers on one connection, staggered
  fi
done
say "backbone lanes started this round:${started:- none}"

# The guard line of every new lane, off that lane's OWN trainer command line.
if [ -n "$started" ]; then
  say "waiting for the guard line of each new lane"
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
    || die "only $ok_arms of $n_new new lane(s) reached a trainer in ${waited}s"
fi

say "waiting for the first rows of each losses CSV"
waited=0
while [ "$waited" -lt 2400 ]; do
  rows_ok=0
  for arm in "${bb_list[@]}"; do
    r="$(rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
              [ -n \"\$csv\" ] && grep -c '^' \"\$csv\" || echo 0")"
    case "$r" in ''|*[!0-9]*) r=0 ;; esac
    [ "$r" -ge 2 ] && rows_ok=$(( rows_ok + 1 ))
  done
  [ "$rows_ok" -ge "${#bb_list[@]}" ] && break
  sleep 30; waited=$(( waited + 30 ))
done

{
  rsh "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
  rsh "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  for arm in "${bb_list[@]}"; do
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
    echo "cmdline reduce $(printf '%s' "$line" | cf404_reduce_of_cmdline)"
  done
} >"$CF404_RESULTS/round8_verify.txt" 2>&1
sed 's/^/  /' "$CF404_RESULTS/round8_verify.txt" | tee -a "$LOG"

used="$(awk -F', ' 'NR>1{gsub(/[^0-9]/,"",$2); s += $2} END{print s+0}' \
        <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
say "GPU memory in use ${used:-0} MiB, $apps compute app(s)"
[ "${used:-0}" -ge 500 ] || die "the card holds ${used:-0} MiB — no trainer is on it"
for arm in "${bb_list[@]}"; do
  # An exact prefix, never a regex: `--- r100_09 seed` is a prefix of
  # `--- r100_09b seed`, and a loose match would read one arm's block as
  # another's.
  blk="$(awk -v m="--- $arm seed " 'index($0, m) == 1 {f=1} f' "$CF404_RESULTS/round8_verify.txt")"
  cols="$(printf '%s\n' "$blk" | awk '/^depth_cols /{print $2; exit}')"
  [ "${cols:-0}" -eq $(( CF404_K + 1 )) ] \
    || die "arm $arm writes ${cols:-0} depth columns, k=$CF404_K wants $(( CF404_K + 1 ))"
  got_ema="$(printf '%s\n' "$blk" | sed -n 's/^cmdline ema //p' | head -1)"
  got_seed="$(printf '%s\n' "$blk" | awk '/^cmdline seed /{print $3; exit}')"
  got_alw="$(printf '%s\n' "$blk" | awk '/^cmdline align_w /{print $3; exit}')"
  got_red="$(printf '%s\n' "$blk" | awk '/^cmdline reduce /{print $3; exit}')"
  [ "$got_ema" = "$(cf404_ema_sig "$arm")" ] \
    || die "arm $arm trains ema '$got_ema', the table says '$(cf404_ema_sig "$arm")'"
  [ "$got_seed" = "$(cf404_seed "$arm")" ] \
    || die "arm $arm trains seed '$got_seed', the table says '$(cf404_seed "$arm")'"
  cf404_num_eq "$got_alw" "$(cf404_align_weight "$arm")" \
    || die "arm $arm trains align_w '$got_alw', the table says '$(cf404_align_weight "$arm")'"
  [ "$got_red" = "$CF404_REDUCE" ] \
    || die "arm $arm trains reduce '$got_red', this cell wants '$CF404_REDUCE'"
done
say "VERIFIED — every lane at $(( CF404_K + 1 )) depth columns, and at the"
say "  momentum, the SEED, the reduction and the align weight its row names"
for arm in "${bb_list[@]}"; do
  say "  $arm STEP RATE $(rsh "grep -hoE '[0-9.]+ sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
done

# ---- 7: the second head, the moment its backbone lands ----------------------
bb_left(){
  local arm n=0
  for arm in "${bb_list[@]}"; do [ -n "$(box_bb "$arm")" ] || n=$(( n + 1 )); done
  echo "$n"
}
head_left(){
  local arm n=0
  for arm in "${head_list[@]}"; do [ "$(box_head "$arm")" -gt "$MIN_HEAD_BYTES" ] || n=$(( n + 1 )); done
  echo "$n"
}

say "waiting for ${STOP} steps on ${#bb_list[@]} lane(s) and $NHEAD head(s)"
waited=0
while [ "$(bb_left)" -gt 0 ] || [ "$(head_left)" -gt 0 ]; do
  [ "$waited" -ge "$BB_TIMEOUT" ] && { say "TIMEOUT after ${waited}s — $(bb_left) backbone(s) and $(head_left) head(s) missing"; break; }
  for arm in "${head_list[@]}"; do
    # An arm whose backbone the box TRAINS needs that backbone first. An arm
    # whose backbone went UP already has one, so it never waits here.
    if cf404_is_in "$arm" "$BB_ARMS"; then
      [ -n "$(box_bb "$arm")" ] || continue
    fi
    [ "$(box_head "$arm")" -gt "$MIN_HEAD_BYTES" ] && continue
    [ -n "$(box_head_running "$arm")" ] && continue
    start_head "$arm"
    sleep 90
  done
  if [ $(( waited % 1800 )) -eq 0 ]; then
    for arm in "${bb_list[@]}"; do
      say "  $arm bb=$(rsh "grep -hoE '^\[ *[0-9]+\].*sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
    done
    for arm in "${head_list[@]}"; do
      say "  $arm head=$(box_head "$arm") B $(rsh "grep -aoE '^\[ *[0-9]+\]' $(box_head_log "$arm") 2>/dev/null | tail -1")"
    done
    say "  spend \$$(box_spent) of \$$MAX_SPEND"
  fi
  sleep "$POLL"; waited=$(( waited + POLL ))
done
say "backbones left $(bb_left), heads left $(head_left)"
say "heads on the box: $(for a in "${head_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"

# ---- 8: the artefacts, into the canonical tree ------------------------------
say "stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
for try in 1 2 3; do
  pull_all
  heads_are_here && break
  say "pull pass $try did not land every head — $(heads_here_report)"
  sleep 60
done

# ---- 9: every artefact READS, and only then does the box go -----------------
say "checking that every pulled artefact reads, before the box goes"
readable=1
for arm in "${arm_list[@]}"; do
  BB="$(bb_here_path "$arm")"
  HD="$(head_here_path "$arm")"
  [ -f "$BB" ] || { say "  $arm: no backbone at $BB"; readable=0; continue; }
  [ -f "$HD" ] || { say "  $arm: no head at $HD"; readable=0; continue; }
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

# ---- 10: the 97-config GIFT-Evals, on elisa ---------------------------------
#
# An arm whose eval ALREADY runs is skipped. `s08c` needs no GPU, so this
# round started its eval hours before this stage. A second eval of one arm
# would take four more CPU shards from a machine other sessions share.
# TWO conditions, and both are read off /proc, never off the pattern alone.
# The process has to BE an eval — its command line names
# `eval_gift_eval_official.py` — and it has to carry THIS arm's backbone. A
# pattern alone also matches the shell that carries it, and elisa runs other
# sessions' evals.
eval_running(){  # <arm> — is a 97-config eval of this arm's backbone up?
  local bb p cl; bb="$(basename "$(bb_here_path "$1")")"
  for p in $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern "$bb")" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    cl="$(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null)"
    case "$cl" in *eval_gift_eval_official.py*) ;; *) continue ;; esac
    case "$cl" in *"$bb"*) echo yes; return 0 ;; esac
  done
  return 1
}

EVAL_ARMS=""
for arm in "${arm_list[@]}"; do
  [ -s "$(cf404_score_file "$arm" "$STOP")" ] && continue
  [ -n "$(eval_running "$arm")" ] && { say "$arm: an eval already runs — not starting a second"; continue; }
  EVAL_ARMS="$EVAL_ARMS $arm"
done
EVAL_ARMS="${EVAL_ARMS# }"
if [ -n "$EVAL_ARMS" ]; then
  say "starting the 97-config GIFT-Evals for '$EVAL_ARMS' on elisa's CPUs, detached"
  ARMS="$EVAL_ARMS" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
    nohup setsid bash "$HERE/evals_elisa.sh" \
      >"$CF404_RESULTS/evals_round8.out" 2>&1 < /dev/null &
  sleep 60
  say "evals launched, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern head_eval)" | grep -c .) eval process(es) on elisa"
else
  say "every arm of this round is scored — no eval to start"
fi

# ---- 11: the scores ---------------------------------------------------------
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
    say "  $(scored_now) of $NARM scored"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

scored=0
for arm in "${arm_list[@]}"; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then
    say "SCORE $arm $(tr -d ' \t\r\n' <"$f") (seed $(cf404_seed "$arm"), holds $(cf404_momentum_at "$arm" "$STOP") at $STOP)"
    scored=$(( scored + 1 ))
  else say "SCORE $arm MISSING"; fi
done

# ---- the tables and the figures ---------------------------------------------
say "collect.sh, then every figure"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  bash "$HERE/make_plots.sh" >"$CF404_RESULTS/make_plots_round8.out" 2>&1
sed 's/^/  /' "$CF404_RESULTS/make_plots_round8.out" | tail -30 | tee -a "$LOG"

# ---- the box, if stage 9 left it up -----------------------------------------
if [ "$TORN" -eq 0 ]; then
  destroy_box || leave_box_alive "the round ended and a head is still not here"
fi
stop_watchdog
say "ROUND 8 DONE — $scored of $NARM arm(s) scored"
