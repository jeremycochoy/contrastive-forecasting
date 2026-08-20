#!/bin/bash
# #404 round 5 — the ramp LENGTH, at the two momenta the ladder brackets.
#
# WHY THIS ROUND EXISTS. The card wants a lower GM-Relative MASE. Round 4
# measured the collapse rate instead, and the card still has no arm that beats
# its own parent: the k = 0 parent of this cell scores 1.1600 at 40,000 steps
# and the best arm here scores 1.1782. k = 3 scores 1.0862 at the same stop.
# The card names 1.1637 and 1.0660 as its targets.
#
# The RAMP LENGTH is the one axis this card never moved. Every ramp arm of
# rounds 1 to 4 runs 200,000 steps. The EMA schedule ladder,
# reports/2026-08-04_ema_sched_ladder/, trained ten runs on that axis, and a
# momentum that reaches 1.0 at step 100,000 scored 0.0259 BELOW the fixed 0.9
# reference at the 40,000-step stop, where the momentum held 0.94. The same run
# scored 0.0251 ABOVE at the 100,000-step stop, where the momentum held 1.0.
# The ladder's latent table gives the reason: past step 100,000 the teacher
# latent moves 0.019 or less per 20,000 steps, so a momentum at 1.0 freezes the
# teacher. A 100,000-step ramp read at 40,000 steps is the good half of that
# curve, and this round stops there.
#
# Two arms, both at a 100,000-step ramp:
#
#   r100_09  --ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000
#   r100_08  --ema-tau 0.8 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000
#
# The momentum each arm HOLDS at the 40,000-step stop, from
# `src.models.ema_tau_at_step`, which is linear and clamped:
#
#   s08, a 200,000-step ramp     0.840
#   r100_08                      0.880
#   s09, a 200,000-step ramp     0.920
#   r100_09                      0.940
#
# So r100_09 lands on the value the ladder measured as its best, and r100_08
# brackets it from below. Neither reaches 1.0 at the stop, so neither freezes
# its teacher there.
#
# Everything else is round 1's: k = 32, the mean reduction, the align target
# teacher, 40,000 backbone steps, 30,000 head steps, head seed 20260722,
# backbone seed 20260520, the 97-config GIFT-Eval.
#
# ---- WHAT THIS ROUND DOES NOT RUN -------------------------------------------
#
# `s08c` and `s08d` reached 40,000 backbone steps at 14:18 and their
# checkpoints are on disk. Their heads and their evals are DROPPED: they only
# measure the collapse rate, and this round spends the card on the score.
#
# ---- THE MACHINE -------------------------------------------------------------
#
# ONE box, and the SAME box: instance 48192413, label `cf404-box-r4`, one RTX
# 5090 in `Default` compute mode. Round 4 put two lanes on that one card, took
# it to 87 % utilization, and both lanes finished in about the time one lane
# took. This round does the same. It NEVER provisions: when the box does not
# answer, it stops, because a second box is not allowed.
#
# ---- THE BUDGET --------------------------------------------------------------
#
# Credit is $11.17 and the box has spent $2.40. The limit is $6 of TOTAL box
# spend, which is what `vastrun-status` reports and what the watchdog reads. So
# MAX_SPEND is $5.50 and it leaves the rest as margin.
#
# THE TEARDOWN COMES BEFORE THE EVALS, and this is the one place round 5
# differs from round 4. The 97-config GIFT-Eval runs on elisa CPUs, not on the
# box, so the box is idle for the 2.5 hours it takes. Round 4 held the box
# through it and that costs about $1.07, which does not fit under $5.50. So
# stage 8 pulls every artefact, LOADS each one to prove it is readable, and
# only then destroys the box. Stage 9 scores what is already on local disk.
#
# Usage:
#   nohup setsid bash scripts/round5.sh > results/round5.out 2>&1 &
#   CF404_DRY_RUN=1 bash scripts/round5.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r4}"
. "$HERE/study.sh"

ARMS="${ARMS:-r100_09 r100_08}"
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
DEADLINE_HOURS="${DEADLINE_HOURS:-8}"
MAX_SPEND="${MAX_SPEND:-5.50}"
# The offer gate of round 4, kept only so a later round can search again. THIS
# round never calls `vastrun-search`: it reuses one box and rents none.
MIN_RELIABILITY="${MIN_RELIABILITY:-0.99}"
MAX_BID="${MAX_BID:-0.45}"
VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|9800X3D|7950X3D|9950X3D|7950X|9950X|7900X|9900X|7700X|9700X}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round5.log"
ENVF="${ENVF:-$CF404_RESULTS/round5.env}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round5] $*" | tee -a "$LOG"; }
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
  echo "round5 arms='$ARMS' box=$VAST_LABEL stop=$STOP head=$CF404_HEAD_STEPS"
  for arm in "${arm_list[@]}"; do
    echo "  $arm alpha=$(cf404_alpha "$arm") $(cf404_schedule "$arm")" \
         "ramp=$(cf404_ramp "$arm") bb seed=$(cf404_seed "$arm")"
    echo "    score=$(cf404_score_file "$arm" "$STOP")"
  done
  echo "  head seed=$HEAD_SEED canonical root=$MAIN_ROOT"
  echo "  momentum at the ${STOP}-step stop:"
  for arm in "${arm_list[@]}"; do
    echo "    $arm $(cf404_momentum_at "$arm" "$STOP")"
  done
  echo "  box: REUSED from $ENVF, never rented — $(sed -n 's/^INSTANCE=/instance /p' "$ENVF" 2>/dev/null)"
  echo "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND pr=#$PR"
  exit 0
fi

echo "$$" >"$CF404_RESULTS/round5.pid"
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
    say "the box in $ENVF does not answer"
    HOST=""
  fi
fi

# THIS ROUND NEVER PROVISIONS. Round 4 rented instance 48192413 and it still
# runs, so round 5 takes that card and no other. A second box is not allowed,
# and a box this driver did not rent is not this driver's to destroy: the
# vast.ai account is shared with other agent sessions.
if [ -z "${HOST:-}" ]; then
  say "ABORT: the box in $ENVF does not answer, and this round may not rent another"
  stop_watchdog
  exit 2
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
# The box carries round 4's scripts, and round 4's arms table has no row for
# either arm of this round. So SHIP the scripts directory again before the
# gate below reads it. The tar holds no results and no plots, and it overwrites
# in place, so a box that already runs keeps its checkpoints.
say "shipping scripts/ to the box (its arms table predates $ARMS)"
TGZ="/tmp/cf404_r5_scripts.$$.tgz"
tar czf "$TGZ" -C "$CF404_REPO" --exclude='__pycache__' "$STUDY_REL/scripts" \
  || die "could not pack scripts/"
scp "${SSH_OPTS[@]}" -P "$PORT" "$TGZ" "root@$HOST:/root/cf404_r5_scripts.tgz" \
  >>"$LOG" 2>&1 || die "could not ship scripts/"
rm -f "$TGZ"
rsh "tar xzf /root/cf404_r5_scripts.tgz -C /root/cf" || die "could not unpack scripts/"
say "the box's arms table now holds: $(rsh "awk -F'\t' '!/^#/ && NF>=4 {printf \"%s \", \$1}' /root/cf/$STUDY_REL/scripts/arms.tsv")"

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
  } >"$CF404_RESULTS/round5_verify.txt" 2>&1
  sed 's/^/  /' "$CF404_RESULTS/round5_verify.txt" | tee -a "$LOG"

  used="$(awk -F', ' 'NR>1{gsub(/[^0-9]/,"",$2); s += $2} END{print s+0}' \
          <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
  apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
  say "GPU memory in use ${used:-0} MiB over $NCARD card(s), $apps compute app(s)," \
      "${#arm_list[@]} arm(s) wanted"
  [ "${used:-0}" -ge 500 ] || die "the card(s) hold ${used:-0} MiB — no trainer is on them"
  [ "$apps" -ge "${#arm_list[@]}" ] || die "$apps compute app(s) for ${#arm_list[@]} arm(s)"
  for arm in "${arm_list[@]}"; do
    cols="$(grep -A2 -- "--- $arm seed" "$CF404_RESULTS/round5_verify.txt" \
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

# ---- 8: the proof that every artefact READS, and then the teardown ----------
#
# THE ONE PLACE ROUND 5 DIFFERS FROM ROUND 4. The 97-config GIFT-Eval runs on
# elisa CPUs. The box does no work during it and bills about $1.07 for the 2.5
# hours it takes, which does not fit under MAX_SPEND. So the box goes NOW.
#
# A box that is gone cannot be pulled from again, so nothing may be wrong with
# what landed. A size floor does not prove that: a half-written checkpoint is
# large. `torch.load` does. Each backbone and each head is opened here, and the
# box lives until every one of them opens.
say "checking that every pulled artefact reads, before the box goes"
readable=1
for arm in "${arm_list[@]}"; do
  NAME="$(cf404_run_name "$arm")"
  TAG="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
  BB="$MAIN_ROOT/$arm/$CF404_CELL/leg_${KK}k/${NAME}_${KK}k.pth"
  HD="$MAIN_ROOT/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED}_final.pth"
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
  printf '%s\n' "$out" | grep -qc '^OK ' >/dev/null || readable=0
done

if [ "$readable" -eq 1 ]; then
  say "every artefact reads — tearing the box down before the evals"
  teardown
  stop_watchdog
  say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"
else
  say "an artefact did NOT read — the box STAYS UP so it can be pulled again"
  say "  (the watchdog still holds \$$MAX_SPEND and ${DEADLINE_HOURS} h)"
fi

# ---- 9: the two 97-config GIFT-Evals, on elisa ------------------------------
#
# Detached. Both evals run at the same time: #393's counting semaphore caps
# elisa's cores.
say "starting the 97-config GIFT-Evals for '$ARMS' on elisa CPUs, detached"
ARMS="$ARMS" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  nohup setsid bash "$HERE/evals_elisa.sh" \
    >"$CF404_RESULTS/evals_round5.out" 2>&1 < /dev/null &
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

# ---- 10: the box, if stage 8 left it up -------------------------------------
say "$scored of ${#arm_list[@]} arm(s) scored"
teardown
stop_watchdog
say "ROUND 5 DONE — $scored of ${#arm_list[@]} arm(s) scored"
