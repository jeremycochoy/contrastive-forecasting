#!/bin/bash
# #404 round 3 — the three added arms on ONE datacenter box.
#
# SUPERSEDED. This driver provisioned the box and was stopped before it started
# an arm. `scripts/round3b.sh` runs on that same box, with the two arms the
# user then asked for, one after the other. `a085` is dropped, so this script's
# own arm list no longer passes `cf404_require_arm`. It stays for the record of
# what round 3 planned.
#
# THE THREE ARMS. The review of PR #405 asks for them:
#
#   a085  alpha 0.85 fixed, the value the card names itself
#   a095  alpha 0.95 fixed, above 0.9, where the backbone metrics point
#   s08b  s08 again at backbone seed 20260521, which measures the repeat
#         spread of THIS cell
#
# Every other flag is round 1's: 40,000 backbone steps, 30,000 head steps,
# head seed 20260722, the 97-config GIFT-Eval.
#
# WHY ONE BOX. Round 2 rented three boxes, one per arm, and all three sat at
# 0% GPU because the driver's "a trainer already runs" check matched its own
# SSH shell (fixed in `study.sh`, proved by `scripts/test_trainer_check.sh`).
# Round 3 takes round 1's shape instead: one box, `launch_box.sh`, the path
# that trained four arms with no incident.
#
# THE OFFER GATE. The box must be a datacenter host at reliability 0.99 or
# better. `vastrun-search` refuses a non-datacenter host unless it is given
# `--prosumer`, which this script never passes, and `--min-reliability 0.99`
# carries the rest. The CPU is pinned as well: #373 measured this cell at 5.6
# to 6.7 steps/s on a Zen 4 desktop part against 1.1 steps/s on an EPYC 7452,
# so the CPU and not the card sets the step rate.
#
# ONE CARD, THREE ARMS. The datacenter pool held no 3-card or 4-card offer
# under $20/h when this round started, and the one 2-card offer carries a
# server CPU at 6 times the price. So the three arms share one RTX 5090.
# `gpu_gate` returns at once on a `Default`-mode card, so the three legs do
# not serialise, and stage 2 refuses a card in `Exclusive_Process` mode, where
# they would.
#
# THE VERIFICATION. Stage 5 proves the trainers RUN before the box is left
# alone: GPU memory in use, one compute app per arm, the guard line that reads
# the momentum and the seed back off each trainer's command line, and the
# first rows of each losses CSV with its 33 depth columns. A box at 0% GPU
# with no run directory is a failed launch, not a slow start.
#
# THE BUDGET. vast.ai holds $18.02 and this round may spend $12. A watchdog
# tears the box down at DEADLINE_HOURS whatever stage is running.
#
# THE TEARDOWN COMES LAST. The box lives until every score exists.
#
# Usage:
#   nohup setsid bash scripts/round3.sh > results/round3.out 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
export CF404_BOX_LABEL="${LABEL:-box_r3}"
. "$HERE/study.sh"

ARMS="${ARMS:-a085 a095 s08b}"
LABEL="$CF404_BOX_LABEL"
VAST_LABEL="cf404-${LABEL//_/-}"
STOP="${STOP:-$CF404_STOPS}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"

# The canonical tree. Round 1's four arms are here, and the eval, the figures
# and `collect.sh` all read this one root.
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
SAFE_PULL="$CF404_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

POLL="${POLL:-300}"
BB_TIMEOUT="${BB_TIMEOUT:-64800}"      # 18 h ceiling on the three backbones
HEAD_TIMEOUT="${HEAD_TIMEOUT:-21600}"  # 6 h ceiling on the three heads
# 24 h of one box at $0.3356/h is $8.05, under the $12 this round may spend.
DEADLINE_HOURS="${DEADLINE_HOURS:-24}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"
LOG="$CF404_RESULTS/round3.log"
ENVF="$CF404_RESULTS/round3.env"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round3] $*" | tee -a "$LOG"; }
rsh(){ timeout 180 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

read -r -a arm_list <<<"$ARMS"
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done
cf404_require_stop "$STOP" || exit $?

# One lane per arm, all on card 0. `launch_box.sh` deals the arms round-robin
# over this list, so three entries give three lanes and every lane names the
# only card the box has.
GPUS_BB="$(printf '0 %.0s' "${arm_list[@]}")"; GPUS_BB="${GPUS_BB% }"

say "START arms='$ARMS' box=$VAST_LABEL deadline=${DEADLINE_HOURS}h"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- the teardown, which every exit path runs -------------------------------
#
# Only the instance THIS round provisioned is destroyed, and only by the id its
# own `.env` file records. `vastrun-destroy` takes the id and the label together
# as a confirmation token. The vast.ai account is shared with other sessions.
teardown(){
  local inst
  [ -s "$ENVF" ] || { say "teardown: no address on file"; return 0; }
  inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$ENVF")"
  [ -n "$inst" ] || { say "teardown: no instance id in $ENVF"; return 0; }
  say "teardown: destroying $inst ($VAST_LABEL)"
  timeout 300 vastrun-destroy "$inst" "$VAST_LABEL" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  say "teardown: stopping the sync loop ($(cf404_stop_sync_loop "$CF404_SYNC_DIR") loop(s))"
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}

# ---- the watchdog -----------------------------------------------------------
#
# It holds no other state, so it survives every failure of the stages below. A
# stage that hangs on a dead box would otherwise bill until a person looks.
watchdog(){
  local secs
  secs="$(awk -v h="$DEADLINE_HOURS" 'BEGIN{printf "%d", h*3600}')"
  sleep "$secs"
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

if [ -z "${HOST:-}" ]; then
  say "searching — datacenter only, reliability 0.99 or better, Zen 4 desktop CPU"
  timeout 200 vastrun-search --num-gpus 1 --min-reliability 0.99 \
    --max-bid "${MAX_BID:-0.45}" --hardware --limit 20 \
    >"$CF404_RESULTS/round3_offers.txt" 2>&1
  sed 's/^/  /' "$CF404_RESULTS/round3_offers.txt" | head -8 | tee -a "$LOG"
  say "provisioning"
  out="$(VAST_SEARCH_ARGS="--num-gpus 1 --min-reliability 0.99 --max-bid ${MAX_BID:-0.45}" \
        VAST_CPU_RE="${VAST_CPU_RE:-7800X3D|9800X3D|7950X|9950X}" \
        bash "$CF404_PARENT/scripts/provision_box.sh" "$VAST_LABEL" \
          "${PROVISION_TRIES:-8}" 2>>"$LOG")"
  read -r INSTANCE HOST PORT <<<"$(printf '%s\n' "$out" | tail -1)"
  [ -n "${PORT:-}" ] || { say "ABORT: no box"; stop_watchdog; exit 2; }
  printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\n' "$INSTANCE" "$HOST" "$PORT" >"$ENVF"
  say "instance $INSTANCE at $HOST:$PORT"
fi

# ---- 2: the card the box actually carries -----------------------------------
#
# Three legs share one card here. `gpu_gate` lets them, but only on a card in
# `Default` compute mode: an `Exclusive_Process` card takes ONE CUDA context
# and the second leg would die inside `.to(device)`.
CARD="$(rsh "nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader")"
say "card: $(printf '%s' "$CARD" | tr '\n' '|')"
case "$CARD" in
  *Default*) ;;
  *) die "the card is not in Default compute mode — three legs cannot share it" ;;
esac
[ "$(printf '%s\n' "$CARD" | grep -c .)" -ge 1 ] || die "no card on the box"

# ---- 3: the payload ---------------------------------------------------------
if rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh"; then
  say "the box already carries the study"
else
  say "bootstrap"
  WT="$CF404_REPO" bash "$HERE/bootstrap_box.sh" "$HOST" "$PORT" >>"$LOG" 2>&1 \
    || die "bootstrap failed, see $LOG"
  say "bootstrap OK"
fi
rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' CF404_DRY_RUN=1 bash scripts/launch_box.sh" \
  >>"$LOG" 2>&1 || die "the box refuses one of '$ARMS'"

# ---- 4: the sync loop -------------------------------------------------------
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR="/root/cf/$STUDY_REL" \
  bash "$CF404_STUDY/sync/launch_sync.sh" "$LABEL" >>"$LOG" 2>&1
say "sync loop -> $CF404_SYNC_DIR ($(find "$CF404_SYNC_DIR" -type f 2>/dev/null | wc -l) file(s) here)"

# ---- 5: the three backbones -------------------------------------------------
KK=$(( STOP / 1000 ))
box_leg(){ printf '%s/%s/%s/leg_%dk\n' "$CF404_BOX_RUNS" "$1" "$CF404_CELL" "$KK"; }
box_bb(){  # <arm> — the checkpoint, or nothing
  rsh "ls -1 $(box_leg "$1")/$(cf404_run_name "$1")_${KK}k.pth 2>/dev/null | head -1"
}

bb_left(){  # how many arms have no bb checkpoint yet
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
    say "starting the three backbones, lanes '$GPUS_BB'"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS='$GPUS_BB' \
         nohup setsid bash scripts/launch_box.sh \
           > results/launch_${LABEL}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi

  # ---- the verification -----------------------------------------------------
  #
  # Requirement 4 of the review: prove the trainers RUN. Four facts, all read
  # off the box: the guard line per arm, the compute apps on the card, the GPU
  # memory in use, and the first rows of each losses CSV.
  say "waiting for the guard line of each arm (it reads alpha and the seed"
  say "  back off the trainer's own command line)"
  waited=0; ok_arms=0
  while [ "$waited" -lt 2400 ]; do
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

  say "GPU and the first rows of each losses CSV"
  {
    rsh "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
    rsh "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    for arm in "${arm_list[@]}"; do
      echo "--- $arm ---"
      rsh "csv=\$(ls $(box_leg "$arm")/*_losses.csv 2>/dev/null | head -1); \
           if [ -n \"\$csv\" ]; then \
             echo \"csv \$csv\"; \
             echo \"depth_cols \$(head -1 \"\$csv\" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*\$')\"; \
             echo \"rows \$(grep -c '^' \"\$csv\")\"; head -3 \"\$csv\" | cut -c1-160; \
           else echo 'csv MISSING'; fi"
    done
  } >"$CF404_RESULTS/round3_verify.txt" 2>&1
  sed 's/^/  /' "$CF404_RESULTS/round3_verify.txt" | tee -a "$LOG"

  used="$(awk -F', ' 'NR==2{gsub(/[^0-9]/,"",$2); print $2}' \
          <(rsh "nvidia-smi --query-gpu=index,memory.used --format=csv"))"
  apps="$(rsh "nvidia-smi --query-compute-apps=pid --format=csv,noheader" | grep -c .)"
  say "GPU memory in use ${used:-0} MiB, $apps compute app(s), ${#arm_list[@]} arm(s) wanted"
  [ "${used:-0}" -ge 500 ] || die "the card holds ${used:-0} MiB — no trainer is on it"
  [ "$apps" -ge "${#arm_list[@]}" ] || die "$apps compute app(s) for ${#arm_list[@]} arm(s)"
  for arm in "${arm_list[@]}"; do
    cols="$(grep -A2 -- "--- $arm ---" "$CF404_RESULTS/round3_verify.txt" \
            | awk '/^depth_cols /{print $2; exit}')"
    [ "${cols:-0}" -eq $(( CF404_K + 1 )) ] \
      || die "arm $arm writes ${cols:-0} depth columns, k=$CF404_K wants $(( CF404_K + 1 ))"
  done
  say "VERIFIED — ${#arm_list[@]} trainer(s) on the card, each at $(( CF404_K + 1 )) depth columns"

  # ---- the climb ------------------------------------------------------------
  say "waiting for ${STOP} steps on ${#arm_list[@]} arm(s)"
  waited=0
  while [ "$(bb_left)" -gt 0 ]; do
    if [ "$waited" -ge "$BB_TIMEOUT" ]; then
      die "no backbone after ${waited}s"
    fi
    if [ $(( waited % 1800 )) -eq 0 ]; then
      for arm in "${arm_list[@]}"; do
        say "  $arm $(rsh "grep -hoE '^\[ *[0-9]+\].*sps  ETA [0-9.]+h' \
             /root/cf/$STUDY_REL/results/run_$(cf404_run_name "$arm").log 2>/dev/null | tail -1")"
      done
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "the three backbones are done"
fi

# ---- 6: the three heads -----------------------------------------------------
#
# One card, so `head_vram_gate` runs them one after the other. That is its
# purpose: it holds an exclusive lock for the whole of one head training.
box_head(){  # <arm> — the head checkpoint size, or 0
  local tag; tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  rsh "wc -c <$CF404_BOX_RUNS/$1/eval/$tag/qhead_${tag}_s${HEAD_SEED:-20260722}_final.pth 2>/dev/null" \
    | tr -d ' ' | grep -E '^[0-9]+$' || echo 0
}
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
    say "starting the three heads on card 0"
    rsh "cd /root/cf/$STUDY_REL && ARMS='$ARMS' GPUS=0 \
         nohup setsid bash scripts/heads_box.sh \
           > results/heads_${LABEL}.out 2>&1 < /dev/null & echo started" \
      >>"$LOG" 2>&1
  fi
  waited=0
  while [ "$(head_left)" -gt 0 ]; do
    if [ "$waited" -ge "$HEAD_TIMEOUT" ]; then
      say "TIMEOUT: $(head_left) head(s) missing after ${waited}s — going on"
      break
    fi
    if [ $(( waited % 1800 )) -eq 0 ]; then
      say "  heads left $(head_left), sizes: $(for a in "${arm_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "heads: $(for a in "${arm_list[@]}"; do printf '%s=%s ' "$a" "$(box_head "$a")"; done)"
fi

# ---- 7: the artefacts, into the canonical tree ------------------------------
#
# A targeted pull beside the 15-minute sync loop, not instead of it. It takes
# the files an eval blocks on, straight into the root round 1 wrote.
pull(){  # <remote> <local> <floor>
  local dst="$2"
  [ -f "$dst" ] && [ "$(wc -c <"$dst")" -ge "$3" ] && { say "  have $(basename "$dst")"; return 0; }
  mkdir -p "$(dirname "$dst")"
  SSH_USER=root bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$dst" "$3" >>"$LOG" 2>&1
  [ -f "$dst" ] || { say "  MISSING $(basename "$dst")"; return 1; }
  say "  $(basename "$dst") $(wc -c <"$dst") B"
}

say "pulling into $MAIN_ROOT"
missing=0
for arm in "${arm_list[@]}"; do
  NAME="$(cf404_run_name "$arm")"
  TAG="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
  RL="$(box_leg "$arm")"; LL="$MAIN_ROOT/$arm/$CF404_CELL/leg_${KK}k"
  say " $arm"
  pull "$RL/${NAME}_${KK}k.pth"            "$LL/${NAME}_${KK}k.pth"            3000000 || missing=1
  pull "$RL/${NAME}_${KK}k_optimizer.pth"  "$LL/${NAME}_${KK}k_optimizer.pth"  4000000 || missing=1
  pull "$RL/${NAME}_losses.csv"            "$LL/${NAME}_losses.csv"            1000000 || missing=1
  pull "$RL/${NAME}_attn_amplitude.csv"    "$LL/${NAME}_attn_amplitude.csv"    1000     || missing=1
  pull "$RL/${NAME}_latent_drift.csv"      "$LL/${NAME}_latent_drift.csv"      100      || missing=1
  pull "$CF404_BOX_RUNS/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED:-20260722}_final.pth" \
       "$MAIN_ROOT/$arm/eval/$TAG/qhead_${TAG}_s${HEAD_SEED:-20260722}_final.pth" 200000 || missing=1
  pull "/root/cf/$STUDY_REL/results/run_${NAME}.log" \
       "$CF404_RESULTS/run_${NAME}.log" 1000 || missing=1
done
[ "$missing" -eq 0 ] || say "WARNING: an artefact did not land — the evals below skip that arm"

# ---- 8: the three GIFT-Evals, on elisa --------------------------------------
say "starting the 97-config GIFT-Evals for '$ARMS'"
ARMS="$ARMS" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  bash "$HERE/evals_elisa.sh" >>"$CF404_RESULTS/evals_round3.out" 2>&1
say "evals rc=$?"
scored=0
for arm in "${arm_list[@]}"; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then say "score $arm $(tr -d ' \t\r\n' <"$f")"; scored=$(( scored + 1 ))
  else say "score $arm MISSING"; fi
done

# ---- 9: the teardown --------------------------------------------------------
#
# Every score that exists, exists now. The box outlived the scores.
say "$scored of ${#arm_list[@]} arm(s) scored — tearing the box down"
teardown
stop_watchdog
say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 10: the figures --------------------------------------------------------
say "shard check"
python3 "$HERE/check_shards.py" --root "$MAIN_ROOT" \
  --out "$CF404_RESULTS/shard_check.txt" 2>&1 | tail -20 | tee -a "$LOG"
say "report assets"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" CF404_SYNC_DIR="$MAIN_DIR" \
  bash "$HERE/report_assets.sh" >>"$CF404_RESULTS/report_assets_round3.out" 2>&1
say "plots"
CF404_ROOT="$MAIN_ROOT" bash "$HERE/make_plots.sh" \
  >>"$CF404_RESULTS/make_plots_round3.out" 2>&1
say "plots rc=$?"
say "ROUND 3 DONE"
