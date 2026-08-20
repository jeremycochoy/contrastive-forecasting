#!/bin/bash
# #404 round 4 — two more backbone seeds of the s08 arm, side by side.
#
# WHY THIS ROUND EXISTS. `s08b` was meant to measure this cell's run-to-run
# spread. It measured a COLLAPSE instead: its contrastive AUC went 0.91 at
# 10,000 steps to 0.57 at 40,000, its top1 went 0.273 to 0.009, and it scored
# 1.5459 against s08's 1.1782. Every other arm of this card holds AUC 0.93 to
# 0.98 at 40,000 steps, and all five carry backbone seed 20260520. One seed
# collapsed. Two readings fit that, and the card cannot tell them apart:
#
#   - 20260521 was unlucky, and a collapse here is rare, or
#   - 20260520 was lucky, and this cell is unstable.
#
# Under the second reading every ranking of this card rests on ONE seed. So
# this round trains the SAME arm at two more seeds:
#
#   s08c  backbone seed 20260522
#   s08d  backbone seed 20260523
#
# Everything else is s08's: alpha 0.8 rising to 1.0 at 200,000, k = 32, the
# mean reduction, the align target teacher, 40,000 backbone steps, 30,000 head
# steps, head seed 20260722, the 97-config GIFT-Eval.
#
# ---- THE MACHINE, and why it carries one card --------------------------------
#
# The card asks for ONE box with TWO cards, a datacenter host at reliability
# 0.99 or better, and a DESKTOP-class CPU. The step rate of this cell is set by
# the CPU and not by the card: #373 measured 5.6 to 6.7 steps/s on a Zen 4
# desktop part against 1.1 steps/s on an EPYC 7452, six times.
#
# On 2026-08-20 the whole datacenter multi-GPU pool carries SERVER CPUs. Every
# one of the 23 two-card offers at reliability 0.99 or better runs an EPYC or a
# Xeon: 7763, 7V13, 7452, 9274F, Xeon Gold 6133, Xeon 6767P. The cheapest is
# $1.04/h. At six times the step time that box needs about 22 hours per arm and
# about $23, against a limit of $7. A two-card server box is SLOWER and DEARER
# than one desktop box, and it breaks the budget.
#
# So `provision` asks for the card's own shape FIRST — two cards, datacenter,
# reliability 0.99, a desktop CPU — and takes it when the pool holds one. When
# the pool holds none it takes one card of the same class, and the two arms
# share that card. They fit: one leg of this cell holds 5.7 GB of a 32 GB RTX
# 5090 and leaves the card at 27 to 34 % utilization, so the second leg takes
# idle silicon. `gpu_gate` returns at once on a `Default`-mode card, and stage
# 2 refuses an `Exclusive_Process` card, where the second leg would die inside
# `.to(device)`.
#
# NO SECOND BOX, at either shape. One box, two lanes.
#
# ---- THE VERIFICATION --------------------------------------------------------
#
# A box at 0 % GPU with no run directory is a failed launch, not a slow start.
# So stage 5 proves both trainers RUN before this driver leaves the box alone:
# the guard line that reads alpha AND THE SEED back off each trainer's own
# command line, the GPU memory in use, one compute app per lane, and the first
# rows of each losses CSV with its 33 depth columns.
#
# The SEED is the value this round turns on. Two arms that differ in the seed
# alone write the same file names, the same CSV columns and the same log lines,
# so a seed that did not reach the trainer gives one run twice under two names.
# `run_arm.sh` stops such a leg in its first minute.
#
# ---- THE BUDGET --------------------------------------------------------------
#
# Credit is $13.95 and this round may spend $7. The watchdog tears the box down
# at MAX_SPEND dollars or DEADLINE_HOURS hours, whichever comes first, whatever
# stage is running. At $0.40/h the whole round is about $3.
#
# THE TEARDOWN COMES LAST. The box lives until both scores exist.
#
# Usage:
#   nohup setsid bash scripts/round4.sh > results/round4.out 2>&1 &
#   CF404_DRY_RUN=1 bash scripts/round4.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r4}"
. "$HERE/study.sh"

ARMS="${ARMS:-s08c s08d}"
LABEL="$CF404_BOX_LABEL"
VAST_LABEL="cf404-${LABEL//_/-}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
PR="${PR:-405}"
AGENT="${AGENT:-ExperimentRunner claude-opus-5}"
HEAD_SEED="${HEAD_SEED:-20260722}"

# The canonical tree. Every arm of this card is here, and the eval, the figures
# and `collect.sh` all read this one root. The box_r4 sync loop keeps its own
# tree as the safety net and never writes this one.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

POLL="${POLL:-300}"
BB_TIMEOUT="${BB_TIMEOUT:-43200}"      # 12 h on two backbones, against 4 h measured for one
HEAD_TIMEOUT="${HEAD_TIMEOUT:-14400}"  # 4 h on two heads, against 1.1 h measured for one
EVAL_TIMEOUT="${EVAL_TIMEOUT:-25200}"  # 7 h on two 97-config evals, against 1.9 h for one
DEADLINE_HOURS="${DEADLINE_HOURS:-14}"
MAX_SPEND="${MAX_SPEND:-6.00}"
# The offer gate. Datacenter is the default of `vastrun-search` — it refuses a
# non-datacenter host unless it is given `--prosumer`, which this script never
# passes. The CPU regex is the desktop-class filter `provision_box.sh` applies
# to the hardware columns.
MIN_RELIABILITY="${MIN_RELIABILITY:-0.99}"
MAX_BID="${MAX_BID:-0.45}"
VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|9800X3D|7950X3D|9950X3D|7950X|9950X|7900X|9900X|7700X|9700X}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round4.log"
ENVF="${ENVF:-$CF404_RESULTS/round4.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round4] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

read -r -a arm_list <<<"$ARMS"
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done
cf404_require_stop "$STOP" || exit $?

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

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "round4 arms='$ARMS' box=$VAST_LABEL stop=$STOP head=$CF404_HEAD_STEPS"
  for arm in "${arm_list[@]}"; do
    echo "  $arm alpha=$(cf404_alpha "$arm") $(cf404_schedule "$arm")" \
         "ramp=$(cf404_ramp "$arm") bb seed=$(cf404_seed "$arm")"
    echo "    score=$(cf404_score_file "$arm" "$STOP")"
  done
  echo "  head seed=$HEAD_SEED canonical root=$MAIN_ROOT"
  echo "  offer: datacenter, reliability>=$MIN_RELIABILITY, max bid \$$MAX_BID/h,"
  echo "         CPU '$VAST_CPU_RE', two cards first then one"
  echo "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND pr=#$PR"
  exit 0
fi

echo "$$" >"$CF404_RESULTS/round4.pid"
say "START arms='$ARMS' box=$VAST_LABEL deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- the teardown, which every exit path runs -------------------------------
#
# Only the instance THIS round provisioned is destroyed, and only by the id its
# own `.env` file records. `vastrun-destroy` takes the id and the label together
# as a confirmation token. The vast.ai account is shared with other sessions.
TORN=0
teardown(){
  local inst
  [ "$TORN" -eq 1 ] && return 0
  [ -s "$ENVF" ] || { say "teardown: no address on file"; return 0; }
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF")"
  [ -n "$inst" ] || { say "teardown: no instance id in $ENVF"; return 0; }
  TORN=1
  say "teardown: destroying $inst ($VAST_LABEL)"
  timeout 300 vastrun-destroy "$inst" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  # By pid, from the working directory. NEVER a pattern: on 2026-08-19 a
  # pattern for this loop also matched four running eval shards.
  say "teardown: stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}

box_spent(){
  local inst
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF" 2>/dev/null)"
  [ -n "$inst" ] || return 0
  timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="$inst" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print substr(v, 2) }'
}

# The watchdog holds no other state, so it survives every failure of the stages
# below. A stage that hangs on a dead box would otherwise bill until a person
# looks.
watchdog(){
  local secs waited=0 spent
  secs="$(awk -v h="$DEADLINE_HOURS" 'BEGIN{printf "%d", h*3600}')"
  while [ "$waited" -lt "$secs" ]; do
    sleep 600; waited=$(( waited + 600 ))
    spent="$(box_spent)"
    [ -n "$spent" ] || continue
    if awk -v s="$spent" -v m="$MAX_SPEND" 'BEGIN{exit !(s+0 >= m+0)}'; then
      say "WATCHDOG: the box has spent \$$spent of \$$MAX_SPEND — tearing it down"
      teardown
      return 0
    fi
  done
  say "WATCHDOG: ${DEADLINE_HOURS} h reached — tearing the box down"
  teardown
}
watchdog & WATCHDOG=$!
stop_watchdog(){ kill -TERM "$WATCHDOG" 2>/dev/null; }
die(){ say "ABORT: $*"; teardown; stop_watchdog; exit 1; }

# ---- 1: the box ------------------------------------------------------------
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

# One search spec, one call. `provision_box.sh` walks every offer of one search
# before it searches again, and it destroys any instance it created and could
# not reach.
provision(){  # <num gpus> <tries>
  local n="$1" tries="$2" out
  say "searching — $n card(s), datacenter, reliability >= $MIN_RELIABILITY," \
      "max bid \$$MAX_BID/h, desktop CPU"
  timeout 200 vastrun-search --num-gpus "$n" --min-reliability "$MIN_RELIABILITY" \
    --max-bid "$MAX_BID" --hardware --limit 40 \
    >"$CF404_RESULTS/round4_offers_${n}gpu.txt" 2>&1
  say "  $(awk -v re="$VAST_CPU_RE" 'NR>1 && $1 ~ /^[0-9]+$/ && $0 ~ re' \
       "$CF404_RESULTS/round4_offers_${n}gpu.txt" | grep -c .) offer(s) with a desktop CPU," \
      "$(awk 'NR>1 && $1 ~ /^[0-9]+$/' "$CF404_RESULTS/round4_offers_${n}gpu.txt" | grep -c .) in all"
  out="$(VAST_SEARCH_ARGS="--num-gpus $n --min-reliability $MIN_RELIABILITY --max-bid $MAX_BID" \
        VAST_SEARCH_LIMIT=40 VAST_CPU_RE="$VAST_CPU_RE" \
        bash "$CF404_PARENT/scripts/provision_box.sh" "$VAST_LABEL" "$tries" 2>>"$LOG")"
  read -r INSTANCE HOST PORT <<<"$(printf '%s\n' "$out" | tail -1)"
  [ -n "${PORT:-}" ]
}

if [ -z "${HOST:-}" ]; then
  # The card's own shape first. `PROVISION_TRIES=1` on the two-card pass: when
  # no two-card offer carries a desktop CPU, the search matches nothing and a
  # second attempt would only wait 20 s and match nothing again.
  provision 2 "${PROVISION_TRIES_2:-1}" || {
    say "no two-card offer carries a desktop CPU — taking one card of the same class"
    say "  (a two-card server box runs this cell about six times slower per step)"
    provision 1 "${PROVISION_TRIES:-8}"; }
  [ -n "${PORT:-}" ] || { say "ABORT: no box"; stop_watchdog; exit 2; }
  printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\n' "$INSTANCE" "$HOST" "$PORT" >"$ENVF"
  say "instance $INSTANCE at $HOST:$PORT"
fi

# ---- 2: the cards the box actually carries ----------------------------------
#
# The lanes are laid over the cards this box HAS, read off its own driver. Two
# cards take one arm each. One card takes both, which `gpu_gate` allows only in
# `Default` compute mode: an `Exclusive_Process` card takes ONE CUDA context and
# the second lane would die inside `.to(device)`.
CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
NCARD="$(printf '%s\n' "$CARD" | grep -c .)"
say "card(s): $(printf '%s' "$CARD" | tr '\n' '|')  count=$NCARD"
[ "${NCARD:-0}" -ge 1 ] || die "no card on the box"
say "cpu: $(rsh "lscpu | sed -nE 's/^Model name: +//p'; nproc") "
say "ram: $(rsh "free -g | awk '/^Mem:/{print \$2\" GB total, \"\$7\" GB available\"}'")"

if [ "$NCARD" -ge "${#arm_list[@]}" ]; then
  GPUS_BB="$(seq -s' ' 0 $(( ${#arm_list[@]} - 1 )))"
  say "one arm per card, lanes '$GPUS_BB'"
else
  case "$CARD" in
    *Default*) ;;
    *) die "the box carries $NCARD card(s) for ${#arm_list[@]} arm(s) and the card is not in Default compute mode — two lanes cannot share it" ;;
  esac
  GPUS_BB="$(printf '0 %.0s' "${arm_list[@]}")"; GPUS_BB="${GPUS_BB% }"
  say "${#arm_list[@]} arms on card 0, lanes '$GPUS_BB' (Default compute mode)"
fi

# ---- 3: the payload ---------------------------------------------------------
if rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh"; then
  say "the box already carries the study"
else
  say "bootstrap"
  WT="$CF404_REPO" bash "$HERE/bootstrap_box.sh" "$HOST" "$PORT" >>"$LOG" 2>&1 \
    || die "bootstrap failed, see $LOG"
  say "bootstrap OK"
fi
# The box has to hold THIS round's arms table, or it would refuse both arms.
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  >>"$LOG" 2>&1 || die "the box refuses one of '$ARMS' — its arms table is stale"
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  | sed 's/^/  /' | tee -a "$LOG"

# ---- 4: the sync loop -------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR," \
    "$(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here"

# ---- 5: the two backbones, and the proof that both run ----------------------
bb_left(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ -n "$(box_bb "$arm")" ] || n=$(( n + 1 )); done
  echo "$n"
}

if [ "$(bb_left)" -eq 0 ]; then
  say "every backbone is already on the box"
else
  # The check that cost round 2 three boxes. `cf404_pgrep_pattern` keeps the
  # pattern from matching the SSH shell that carries it.
  if rsh "pgrep -f '$(cf404_pgrep_pattern "run_leg_k.sh $CF404_CELL")' >/dev/null"; then
    say "a trainer already runs on the box"
  else
    say "starting ${#arm_list[@]} backbone(s), lanes '$GPUS_BB'"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi

  say "waiting for the guard line of each arm (it reads alpha AND THE SEED"
  say "  back off the trainer's own command line)"
  waited=0; ok_arms=0; verdict=""
  while [ "$waited" -lt 3000 ]; do
    verdict="$(rsh "grep -h 'reached the trainer' /root/cf/$STUDY_REL/results/arms.log 2>/dev/null")"
    stopped="$(rsh "grep -h 'STOPPED' /root/cf/$STUDY_REL/results/arms.log 2>/dev/null | tail -2")"
    [ -n "$stopped" ] && die "the box stopped a leg — $stopped"
    ok_arms=0
    for arm in "${arm_list[@]}"; do
      printf '%s\n' "$verdict" | grep -q "arm $arm " && ok_arms=$(( ok_arms + 1 ))
    done
    [ "$ok_arms" -ge "${#arm_list[@]}" ] && break
    sleep 30; waited=$(( waited + 30 ))
  done
  printf '%s\n' "$verdict" | sed 's/^/  /' | tee -a "$LOG"
  [ "$ok_arms" -ge "${#arm_list[@]}" ] \
    || die "only $ok_arms of ${#arm_list[@]} arm(s) reached a trainer in ${waited}s"

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
    [ "$rows_ok" -ge "${#arm_list[@]}" ] && break
    sleep 30; waited=$(( waited + 30 ))
  done

  {
    rsh "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
    rsh "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    for arm in "${arm_list[@]}"; do
      echo "--- $arm seed $(cf404_seed "$arm") ---"
      rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
           if [ -n \"\$csv\" ]; then \
             echo \"csv \$csv\"; \
             echo \"depth_cols \$(head -1 \"\$csv\" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*\$')\"; \
             echo \"rows \$(grep -c '^' \"\$csv\")\"; head -3 \"\$csv\" | cut -c1-160; \
           else echo 'csv MISSING'; fi"
    done
  } >"$CF404_RESULTS/round4_verify.txt" 2>&1
  sed 's/^/  /' "$CF404_RESULTS/round4_verify.txt" | tee -a "$LOG"

  used="$(awk -F', ' 'NR>1{gsub(/[^0-9]/,"",$2); s += $2} END{print s+0}' \
          <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
  apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
  say "GPU memory in use ${used:-0} MiB over $NCARD card(s), $apps compute app(s)," \
      "${#arm_list[@]} arm(s) wanted"
  [ "${used:-0}" -ge 500 ] || die "the card(s) hold ${used:-0} MiB — no trainer is on them"
  [ "$apps" -ge "${#arm_list[@]}" ] || die "$apps compute app(s) for ${#arm_list[@]} arm(s)"
  for arm in "${arm_list[@]}"; do
    cols="$(grep -A2 -- "--- $arm seed" "$CF404_RESULTS/round4_verify.txt" \
            | awk '/^depth_cols /{print $2; exit}')"
    [ "${cols:-0}" -eq $(( CF404_K + 1 )) ] \
      || die "arm $arm writes ${cols:-0} depth columns, k=$CF404_K wants $(( CF404_K + 1 ))"
  done
  say "VERIFIED — ${#arm_list[@]} trainer(s) up, each at $(( CF404_K + 1 )) depth columns"
  for arm in "${arm_list[@]}"; do
    say "  $arm STEP RATE $(rsh "grep -hoE '[0-9.]+ sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
  done

  say "waiting for ${STOP} steps on ${#arm_list[@]} arm(s)"
  waited=0
  while [ "$(bb_left)" -gt 0 ]; do
    [ "$waited" -ge "$BB_TIMEOUT" ] && die "no backbone after ${waited}s"
    if [ $(( waited % 1800 )) -eq 0 ]; then
      for arm in "${arm_list[@]}"; do
        say "  $arm $(rsh "grep -hoE '^\[ *[0-9]+\].*sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
      done
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "both backbones are done"
fi

# ---- 6: the two heads -------------------------------------------------------
head_left(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ "$(box_head "$arm")" -gt 200000 ] || n=$(( n + 1 )); done
  echo "$n"
}

if [ "$(head_left)" -eq 0 ]; then
  say "every head is already on the box"
else
  if rsh "pgrep -f '$(cf404_pgrep_pattern train_forecasting_head)' >/dev/null"; then
    say "a head trainer already runs on the box"
  else
    say "starting ${#arm_list[@]} head(s), lanes '$GPUS_BB', seed $HEAD_SEED"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' \
         nohup setsid bash scripts/heads_box.sh \
           > results/heads_${LABEL}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi
  waited=0
  while [ "$(head_left)" -gt 0 ]; do
    [ "$waited" -ge "$HEAD_TIMEOUT" ] && {
      say "TIMEOUT: $(head_left) head(s) missing after ${waited}s — going on"; break; }
    [ $(( waited % 1800 )) -eq 0 ] && \
      say "  heads left $(head_left), sizes: $(for a in "${arm_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "heads: $(for a in "${arm_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"
fi

# ---- 7: the artefacts, into the canonical tree ------------------------------
#
# A targeted pull beside the 15-minute sync loop, not instead of it. It takes
# the files an eval blocks on, straight into the root the other arms wrote.
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
  pull "$RL/${NAME}_${KK}k.pth"           "$LL/${NAME}_${KK}k.pth"           3000000 || missing=1
  pull "$RL/${NAME}_${KK}k_optimizer.pth" "$LL/${NAME}_${KK}k_optimizer.pth" 4000000 || missing=1
  pull "$RL/${NAME}_losses.csv"           "$LL/${NAME}_losses.csv"           1000000 || missing=1
  pull "$RL/${NAME}_attn_amplitude.csv"   "$LL/${NAME}_attn_amplitude.csv"   1000     || missing=1
  pull "$RL/${NAME}_latent_drift.csv"     "$LL/${NAME}_latent_drift.csv"     100      || missing=1
  pull "$CF404_BOX_RUNS/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth" \
       "$MAIN_ROOT/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth" 200000 || missing=1
  pull "/root/cf/$STUDY_REL/results/run_${NAME}.log" \
       "$CF404_RESULTS/run_${NAME}.log" 1000 || missing=1
  [ "$missing" -eq 0 ]
}

for arm in "${arm_list[@]}"; do
  pull_arm "$arm" || say "$arm: WARNING — an artefact did not land"
done

# ---- 8: the two 97-config GIFT-Evals, on elisa ------------------------------
#
# Detached, so the box can go the minute both scores exist. Both evals run at
# the same time: #393's counting semaphore caps elisa's cores.
say "starting the 97-config GIFT-Evals for '$ARMS' on elisa CPUs, detached"
ARMS="$ARMS" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  nohup setsid bash "$HERE/evals_elisa.sh" \
    >"$CF404_RESULTS/evals_round4.out" 2>&1 < /dev/null &
sleep 30
say "evals launched, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern head_eval)" | grep -c .) eval process(es) on elisa"

scored_now(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ -s "$(cf404_score_file "$arm" "$STOP")" ] && n=$(( n + 1 )); done
  echo "$n"
}

say "waiting for ${#arm_list[@]} score(s)"
waited=0
while [ "$(scored_now)" -lt "${#arm_list[@]}" ]; do
  [ "$waited" -ge "$EVAL_TIMEOUT" ] && { say "only $(scored_now) score(s) after ${waited}s"; break; }
  [ $(( waited % 1800 )) -eq 0 ] && \
    say "  $(scored_now) of ${#arm_list[@]} scored, $(pgrep -u "$(id -u)" -f "$(cf404_pgrep_pattern head_eval)" | grep -c .) eval process(es)"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

scored=0
for arm in "${arm_list[@]}"; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then say "SCORE $arm $(tr -d ' \t\r\n' <"$f")"; scored=$(( scored + 1 ))
  else say "SCORE $arm MISSING"; fi
done

# ---- 9: the teardown --------------------------------------------------------
#
# Every score that exists, exists now. The box outlived the scores.
say "$scored of ${#arm_list[@]} arm(s) scored — tearing the box down"
teardown
stop_watchdog
say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"
say "ROUND 4 DONE — $scored of ${#arm_list[@]} arm(s) scored"
