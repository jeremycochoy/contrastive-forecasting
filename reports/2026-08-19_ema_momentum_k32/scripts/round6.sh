#!/bin/bash
# #404 round 6 — the L_align WEIGHT, as a third lane on the round 4 box.
#
# WHY THIS ROUND EXISTS. For this loss shape the rollout depth touches the
# align term alone. A depth copy of `cosine_similarity_batch_rep_only` "has
# nothing to substitute and adds exactly zero", because that shape is
# h-anchored, and the align add-on "IS duplicated" (src/loss.py). The
# reduction is a mean, so 33 copies of L_align average back to about one
# copy's magnitude. The loss then holds ONE copy of the h-anchored repel term
# against the MEAN of 33 copies of the f-anchored pull term.
#
# `--align-loss-weight` is the only flag that sets that balance, and no arm of
# rounds 1 to 5 has moved it. This round moves it, on the best-scoring arm of
# the card and on nothing else:
#
#   w3_s08   s08 (1.1782) with --align-loss-weight 3.0 instead of 1.0
#
# Everything else is round 1's: k = 32, the mean reduction, the align target
# teacher, --ema-tau 0.8 --ema-tau-end 1.0 --ema-tau-ramp-steps 200000, 40,000
# backbone steps, 30,000 head steps, head seed 20260722, backbone seed
# 20260520, the 97-config GIFT-Eval.
#
# ---- WHAT THIS ROUND CARRIES FORWARD -----------------------------------------
#
# `r100_09` and `r100_08` started at 15:24 under round 5 and still train. This
# round adopts them: it drives all THREE arms to a score, so one driver owns
# the box and one teardown ends it. Round 5's driver and its finisher were
# stopped by pid before this one started, in that order, because the finisher
# destroys the box the moment the driver leaves the process table.
#
# ---- WHAT THIS ROUND DOES NOT RUN --------------------------------------------
#
# `s08c` and `s08d` reached 40,000 backbone steps and their checkpoints are on
# disk. Their heads and their evals stay DROPPED: they measure the collapse
# rate, and this round spends the card on the score.
#
# ---- THE MACHINE -------------------------------------------------------------
#
# ONE box, and the SAME box: instance 48192413, label `cf404-box-r4`, one RTX
# 5090 in `Default` compute mode. THREE lanes share that one card. This script
# NEVER provisions: when the box does not answer it stops, because a second
# box is not allowed.
#
# ---- TWO THINGS ROUND 5 DID PER ROUND, AND THIS ONE DOES PER ARM -------------
#
# 1. THE LAUNCH. Round 5 asked "does a trainer run on the box?" and skipped
#    the launch when one did. Three lanes make that answer useless: two
#    trainers already run, so the third would never start. This round asks the
#    question of EACH ARM, off that arm's own run name, and starts only the
#    arms that neither finished nor run.
#
# 2. THE HEADS. Round 5 waited for every backbone and then started every head.
#    A head reports 0 % GPU utilization on this card — it is data-bound, not
#    compute-bound — so a head costs the trainers beside it almost nothing.
#    This round starts each arm's head THE MOMENT that arm's backbone lands,
#    so the two 100,000-step ramps are scored while `w3_s08` still trains. It
#    takes about an hour of box time off the tail.
#
# ---- THE BUDGET --------------------------------------------------------------
#
# The limit is $6 of TOTAL box spend, which is what `vastrun-status` reports.
# The box had spent $2.61 when this round started, at $0.4278/h. So MAX_SPEND
# is $5.60 and the rest is margin for the teardown itself.
#
# THE TEARDOWN COMES BEFORE THE EVALS. The 97-config GIFT-Eval runs on elisa
# CPUs, so the box does no work for the hours it takes. Stage 8 pulls every
# artefact, LOADS each one with `torch.load` to prove it reads, and only then
# destroys the box. A size floor does not prove a checkpoint reads: a
# half-written file is large. When a file does not load, the box STAYS UP.
#
# THE WATCHDOG PULLS BEFORE IT DESTROYS. Round 5's watchdog destroyed the box
# on the spend cap and left whatever was on it there. This one pulls every
# artefact first, so a cap that fires mid-round still leaves a backbone elisa
# can train a head on.
#
# Usage:
#   nohup setsid bash scripts/round6.sh > results/round6.out 2>&1 &
#   CF404_DRY_RUN=1 bash scripts/round6.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r4}"
. "$HERE/study.sh"

ARMS="${ARMS:-r100_09 r100_08 w3_s08}"
NEW_ARMS="${NEW_ARMS:-w3_s08}"          # the arms this round itself launches
LABEL="$CF404_BOX_LABEL"
VAST_LABEL="cf404-${LABEL//_/-}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
PR="${PR:-405}"
AGENT="${AGENT:-ExperimentRunner claude-opus-5}"
HEAD_SEED="${HEAD_SEED:-20260722}"

# The canonical tree. Every arm of this card is here, and the eval, the figures
# and `collect.sh` all read this one root.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

POLL="${POLL:-300}"
BB_TIMEOUT="${BB_TIMEOUT:-32400}"      # 9 h on three lanes, against 4 h for one
HEAD_TIMEOUT="${HEAD_TIMEOUT:-14400}"  # 4 h on three heads, against 1.1 h for one
EVAL_TIMEOUT="${EVAL_TIMEOUT:-28800}"  # 8 h on three 97-config evals
DEADLINE_HOURS="${DEADLINE_HOURS:-8}"
MAX_SPEND="${MAX_SPEND:-5.60}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round6.log"
ENVF="${ENVF:-$CF404_RESULTS/round6.env}"
# The box is round 5's. Its address is copied, not re-found, so the teardown
# names one instance id and this round can never destroy another session's box.
[ -s "$ENVF" ] || cp "$CF404_RESULTS/round5.env" "$ENVF"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round6] $*" | tee -a "$LOG"; }
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

# Does a trainer for THIS arm run on the box? The run name is on the trainer's
# own command line, through --run-name and --save-dir, and it carries the arm.
# A box-wide question would answer "yes" for the two lanes already up and the
# third lane would never start.
box_arm_running(){  # <arm>
  rsh "pgrep -f '$(cf404_pgrep_pattern "$(cf404_run_name "$1")")' >/dev/null && echo yes"
}
box_head_running(){  # <arm>
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "pgrep -f '$(cf404_pgrep_pattern "qhead_${tag}_s${HEAD_SEED}")' >/dev/null && echo yes"
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "round6 arms='$ARMS' new='$NEW_ARMS' box=$VAST_LABEL stop=$STOP head=$CF404_HEAD_STEPS"
  for arm in "${arm_list[@]}"; do
    echo "  $arm alpha=$(cf404_alpha "$arm") $(cf404_schedule "$arm")" \
         "ramp=$(cf404_ramp "$arm") seed=$(cf404_seed "$arm")" \
         "align_w=$(cf404_align_weight "$arm")" \
         "holds $(cf404_momentum_at "$arm" "$STOP") at $STOP"
    echo "    score=$(cf404_score_file "$arm" "$STOP")"
  done
  echo "  head seed=$HEAD_SEED canonical root=$MAIN_ROOT"
  echo "  box: REUSED from $ENVF, never rented — $(sed -n 's/^INSTANCE=/instance /p' "$ENVF" 2>/dev/null)"
  echo "  deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND pr=#$PR"
  exit 0
fi

echo "$$" >"$CF404_RESULTS/round6.pid"
say "START arms='$ARMS' box=$VAST_LABEL deadline=${DEADLINE_HOURS}h max_spend=\$$MAX_SPEND"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- the teardown, which every exit path runs -------------------------------
#
# Only the instance THIS round holds is destroyed, and only by the id its own
# `.env` file records. `vastrun-destroy` takes the id and the label together as
# a confirmation token. The vast.ai account is shared with other sessions.
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

pull_all(){
  local arm
  for arm in "${arm_list[@]}"; do
    pull_arm "$arm" || say "$arm: WARNING — an artefact did not land"
  done
}

# The watchdog holds no other state, so it survives every failure of the stages
# below. A stage that hangs on a dead box would otherwise bill until a person
# looks.
#
# IT PULLS FIRST. The cap is a budget event, not a data event: whatever the box
# holds at that moment is still worth a head on elisa. The watchdog is forked
# before stage 1, so it carries no HOST and no PORT — it reads them back out of
# the same `.env` file the teardown names the instance from.
watchdog(){
  local secs waited=0 spent
  secs="$(awk -v h="$DEADLINE_HOURS" 'BEGIN{printf "%d", h*3600}')"
  while [ "$waited" -lt "$secs" ]; do
    sleep 600; waited=$(( waited + 600 ))
    spent="$(box_spent)"
    [ -n "$spent" ] || continue
    if awk -v s="$spent" -v m="$MAX_SPEND" 'BEGIN{exit !(s+0 >= m+0)}'; then
      say "WATCHDOG: the box has spent \$$spent of \$$MAX_SPEND — pulling, then tearing it down"
      # shellcheck disable=SC1090
      . "$ENVF"
      pull_all
      teardown
      return 0
    fi
  done
  say "WATCHDOG: ${DEADLINE_HOURS} h reached — pulling, then tearing the box down"
  # shellcheck disable=SC1090
  . "$ENVF"
  pull_all
  teardown
}
watchdog & WATCHDOG=$!
stop_watchdog(){ kill -TERM "$WATCHDOG" 2>/dev/null; }
die(){ say "ABORT: $*"; teardown; stop_watchdog; exit 1; }

# ---- 1: the box ------------------------------------------------------------
INSTANCE=""; HOST=""; PORT=""
# shellcheck disable=SC1090
. "$ENVF"
if [ -n "${HOST:-}" ] && rsh true; then
  say "reusing instance $INSTANCE at $HOST:$PORT"
else
  say "ABORT: the box in $ENVF does not answer, and this round may not rent another"
  stop_watchdog
  exit 2
fi

# ---- 2: the card the box actually carries -----------------------------------
CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
NCARD="$(printf '%s\n' "$CARD" | grep -c .)"
say "card(s): $(printf '%s' "$CARD" | tr '\n' '|')  count=$NCARD"
[ "${NCARD:-0}" -ge 1 ] || die "no card on the box"

if [ "$NCARD" -ge "${#arm_list[@]}" ]; then
  GPUS_BB="$(seq -s' ' 0 $(( ${#arm_list[@]} - 1 )))"
  say "one arm per card, lanes '$GPUS_BB'"
else
  case "$CARD" in
    *Default*) ;;
    *) die "the box carries $NCARD card(s) for ${#arm_list[@]} arm(s) and the card is not in Default compute mode — the lanes cannot share it" ;;
  esac
  GPUS_BB="$(printf '0 %.0s' "${arm_list[@]}")"; GPUS_BB="${GPUS_BB% }"
  say "${#arm_list[@]} arms on card 0, lanes '$GPUS_BB' (Default compute mode)"
fi
say "card before the new lane: $(rsh "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader")"

# ---- 3: the payload ---------------------------------------------------------
#
# The box carries round 5's scripts, and round 5's arms table has no row for
# `w3_s08` and no align-weight column at all. So SHIP the scripts directory
# again before the gate below reads it. The tar holds no results and no plots,
# and it overwrites in place, so the two lanes already up keep their
# checkpoints and their command lines.
rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh" \
  || die "the box does not carry the study — this round does not bootstrap"
say "shipping scripts/ to the box (its arms table predates $NEW_ARMS)"
TGZ="/tmp/cf404_r6_scripts.$$.tgz"
tar czf "$TGZ" -C "$CF404_REPO" --exclude='__pycache__' "$STUDY_REL/scripts" \
  || die "could not pack scripts/"
scp "${SSH_OPTS[@]}" -P "$PORT" "$TGZ" "root@$HOST:/root/cf404_r6_scripts.tgz" \
  >>"$LOG" 2>&1 || die "could not ship scripts/"
rm -f "$TGZ"
rsh "tar xzf /root/cf404_r6_scripts.tgz -C /root/cf" || die "could not unpack scripts/"
say "the box's arms table now holds: $(rsh "awk -F'\t' '!/^#/ && NF>=4 {printf \"%s \", \$1}' /root/cf/$STUDY_REL/scripts/arms.tsv")"

# The box has to hold THIS round's arms table, or it would refuse the new arm.
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  >>"$LOG" 2>&1 || die "the box refuses one of '$ARMS' — its arms table is stale"
# And it has to build the new arm's command line with the WEIGHT on it. This
# reads the box's own copy, so a table that shipped without column 6 stops the
# round here instead of training a duplicate of `s08` for six hours.
for arm in $NEW_ARMS; do
  got="$(rsh "cd /root/cf/$STUDY_REL && CF404_DRY_RUN=1 bash scripts/run_arm.sh $arm $STOP" \
         | awk '/^  align_w=/{sub(/^align_w=/, "", $1); print $1; exit}')"
  want="$(cf404_align_weight "$arm")"
  cf404_num_eq "$got" "$want" \
    || die "the box builds align_w='$got' for $arm, this table says '$want'"
  say "the box builds $arm at align_w=$got, k=$CF404_K, reduce=$CF404_REDUCE"
done

# ---- 4: the sync loop -------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop: $(cf404_sync_loops "$CF404_SYNC_DIR") for $CF404_SYNC_DIR," \
    "$(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here"

# ---- 5: the third lane, and the proof that all three run --------------------
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
    rsh "cd /root/cf/$STUDY_REL && ARMS='$arm' GPUS='0' \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}_${arm}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
    started="$started $arm"
  fi
done
say "lanes started this round:${started:- none}"

# The guard line of EVERY arm, off that arm's own trainer command line. It
# reads alpha, the reduction, the seed AND the align weight, because `w3_s08`
# differs from `s08` in the weight alone.
if [ -n "$started" ]; then
  say "waiting for the guard line of each new arm (it reads alpha, the seed"
  say "  AND the align weight, back off the trainer's own command line)"
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
  [ "$rows_ok" -ge "${#arm_list[@]}" ] && break
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
    echo "cmdline align_w $(rsh "grep -h '^Command line:' $(box_log "$arm") 2>/dev/null | tail -1" | cf404_align_of_cmdline)"
  done
} >"$CF404_RESULTS/round6_verify.txt" 2>&1
sed 's/^/  /' "$CF404_RESULTS/round6_verify.txt" | tee -a "$LOG"

used="$(awk -F', ' 'NR>1{gsub(/[^0-9]/,"",$2); s += $2} END{print s+0}' \
        <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
say "GPU memory in use ${used:-0} MiB over $NCARD card(s), $apps compute app(s)," \
    "${#arm_list[@]} arm(s) wanted"
[ "${used:-0}" -ge 500 ] || die "the card(s) hold ${used:-0} MiB — no trainer is on them"
[ "$apps" -ge "${#arm_list[@]}" ] || die "$apps compute app(s) for ${#arm_list[@]} arm(s)"
for arm in "${arm_list[@]}"; do
  cols="$(grep -A2 -- "--- $arm seed" "$CF404_RESULTS/round6_verify.txt" \
          | awk '/^depth_cols /{print $2; exit}')"
  [ "${cols:-0}" -eq $(( CF404_K + 1 )) ] \
    || die "arm $arm writes ${cols:-0} depth columns, k=$CF404_K wants $(( CF404_K + 1 ))"
  # The weight the TRAINER runs, not the weight the table wants. The two are
  # compared here for every lane, including the two this round adopted.
  got="$(grep -A9 -- "--- $arm seed" "$CF404_RESULTS/round6_verify.txt" \
         | awk '/^cmdline align_w /{print $3; exit}')"
  cf404_num_eq "$got" "$(cf404_align_weight "$arm")" \
    || die "arm $arm trains align_w '$got', the table says '$(cf404_align_weight "$arm")'"
done
say "VERIFIED — ${#arm_list[@]} trainer(s) up, each at $(( CF404_K + 1 )) depth columns,"
say "  each at the align weight its row names"
for arm in "${arm_list[@]}"; do
  say "  $arm STEP RATE $(rsh "grep -hoE '[0-9.]+ sps  ETA [0-9.]+h' $(box_log "$arm") 2>/dev/null | tail -1")"
done

# ---- 6: the heads, each one started as its own backbone lands ---------------
#
# Not "every backbone, then every head". A head reports 0 % GPU utilization on
# this card, so it costs the trainers beside it almost nothing, and the two
# ramps are scored while the third lane still trains.
head_left(){
  local arm n=0
  for arm in "${arm_list[@]}"; do [ "$(box_head "$arm")" -gt 200000 ] || n=$(( n + 1 )); done
  echo "$n"
}

say "waiting for ${STOP} steps and a head on ${#arm_list[@]} arm(s)"
waited=0
while [ "$(bb_left)" -gt 0 ] || [ "$(head_left)" -gt 0 ]; do
  [ "$waited" -ge "$BB_TIMEOUT" ] && { say "TIMEOUT after ${waited}s — $(bb_left) backbone(s) and $(head_left) head(s) missing"; break; }
  for arm in "${arm_list[@]}"; do
    [ -n "$(box_bb "$arm")" ] || continue
    [ "$(box_head "$arm")" -gt 200000 ] && continue
    [ -n "$(box_head_running "$arm")" ] && continue
    say "$arm: the backbone landed — starting its head, seed $HEAD_SEED"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$arm' GPUS='0' \
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
say "heads: $(for a in "${arm_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"

# ---- 7: the artefacts, into the canonical tree ------------------------------
pull_all

# ---- 8: the proof that every artefact READS, and then the teardown ----------
#
# The 97-config GIFT-Eval runs on elisa CPUs. The box does no work during it,
# so the box goes NOW. A box that is gone cannot be pulled from again, so
# nothing may be wrong with what landed. A size floor does not prove that: a
# half-written checkpoint is large. `torch.load` does.
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

# ---- 9: the three 97-config GIFT-Evals, on elisa ----------------------------
#
# Detached. `head_eval.sh` trains a head that is not on disk and then evals, so
# an arm whose head the box did not finish is covered here too.
say "starting the 97-config GIFT-Evals for '$ARMS' on elisa CPUs, detached"
ARMS="$ARMS" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  nohup setsid bash "$HERE/evals_elisa.sh" \
    >"$CF404_RESULTS/evals_round6.out" 2>&1 < /dev/null &
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
say "ROUND 6 DONE — $scored of ${#arm_list[@]} arm(s) scored"
