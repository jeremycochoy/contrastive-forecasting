#!/bin/bash
# #373 round 3 — ONE work queue over FOUR GPUs.
#
# Round 2 rented one box per cell. Every operational failure in this study
# came out of that fleet: 15 failed bootstraps on one cell, a box idle 37.6 h
# at 0%, two more idle 4 h, and a duplicated run on one card. Round 3 rents
# ONE box with two cards and pairs it with elisa's two, and this loop is the
# only thing that starts work on any of them.
#
# Usage:  bash q_run.sh            # dispatch until the queue drains
#         DRY=1 bash q_run.sh      # print the plan, start nothing
#
# Slots. Two jobs share a card: a d_model=64 backbone at batch 64 is
# launch-bound, not compute-bound, so the second process buys most of its own
# throughput, and 5.4 GB + 5.4 GB + a 7 GB head still fits 24 GB. elisa's
# cards carry other projects, so an elisa slot only takes a job when that
# card has the VRAM free at that moment. The rented card is ours alone.
#
# The 97-config GIFT-Eval takes no GPU — measured 2.86 core-hours against
# 1.66x on a 4090 — so eval jobs run on elisa's cores beside everything else
# and never hold a slot.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
mkdir -p "$STATE"

export WT="${WT:-/home/jupyter/wt-cf-373-run2}"
# One flat durable root, on elisa, for every round-3 artefact. Mirrors the
# box's /root/cf373_runs exactly, so one sync loop fills it and cell_paths.sh
# resolves a checkpoint the same way on both machines.
export CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3}"
BOX_ID="${BOX_ID:?BOX_ID must be the vast.ai instance id}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"
K="${K:-3}"
POLL="${POLL:-60}"
# Free VRAM a job needs before an elisa card takes it.
BB_VRAM="${BB_VRAM:-7000}"
HEAD_VRAM="${HEAD_VRAM:-8500}"
MAX_EVALS="${MAX_EVALS:-4}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"
HEAD_SEED="${HEAD_SEED:-20260722}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o ServerAliveInterval=30)
rsh(){ ssh "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" "$@"; }

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [q] $*" | tee -a "$RES/q_run.log"; }

# ------------------------------------------------------- detached remote start
# `ssh host "cmd &"` does NOT return: sshd holds the channel open while the
# backgrounded child lives, even with the child's three descriptors redirected.
# That blocked the first dispatcher on its first job — the job ran, the loop
# never came back to place a second one.
#
# So the job body goes over as a FILE, and the start is a separate ssh under a
# hard timeout. The body is already detached with setsid before the timeout can
# fire, so a timeout here is not a failure and is not treated as one; the
# `.started` marker the body writes is what says the job began.
rlaunch(){ # <id> <body on stdin>
  local id="$1" tmp
  tmp="$(mktemp)"
  { echo '#!/bin/bash'
    echo "echo started > /root/jobs/$id.started"
    cat
    echo "echo \$? > /root/jobs/$id.rc"; } > "$tmp"
  rsh "mkdir -p /root/jobs" >/dev/null 2>&1 || { rm -f "$tmp"; return 1; }
  scp "${SSH_OPTS[@]}" -P "$BOX_PORT" "$tmp" \
      "root@$BOX_HOST:/root/jobs/$id.sh" >/dev/null 2>&1 \
    || { rm -f "$tmp"; return 1; }
  rm -f "$tmp"
  timeout 40 ssh -n "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" \
    "setsid nohup bash /root/jobs/$id.sh > /root/jobs/$id.log 2>&1 < /dev/null & echo ok" \
    >/dev/null 2>&1
  # Confirm by the marker, never by ssh's exit status.
  local t=0
  while [ "$t" -lt 60 ]; do
    [ -n "$(rsh "cat /root/jobs/$id.started 2>/dev/null" 2>/dev/null)" ] && return 0
    sleep 5; t=$(( t + 5 ))
  done
  return 1
}

# --------------------------------------------------------------- slot table
# machine:gpu — listed in the order the dispatcher offers them. The rented
# card is offered first: it is paid for by the hour whether or not it works.
# NO quotes inside the default: `${QSLOTS:-"a b"}` is ONE word, so the array
# would hold a single element and every job would be placed on it. That put
# two backbones on card 1 and left card 0 idle on the first launch.
#
# The last two entries are HEAD-ONLY, one per rented card. The rental ends
# when the LAST job ends, and 19 heads behind 9 backbones on four slots put
# every head after every extend — six hours of head time added to the end of
# a bill that is already paid by the hour. A third process fits: 5.4 GB +
# 5.4 GB + 7 GB against 24.5 GB, and the cards read 77% and 89%, not 100%.
# A head here takes no slot from a backbone; it fills what the two backbones
# leave. `r2_head_box.sh` still gates on 8.5 GB free before it allocates, so
# a card that is genuinely full makes the head wait rather than OOM.
#
# Backbones must NOT use them: three backbones on one card would slow the
# two that are on the critical path to buy a third that is not.
SLOTS=( ${QSLOTS:-rem:0 rem:0 rem:1 rem:1 loc:0 loc:1 rem:0 rem:1} )
HEAD_ONLY=" ${HEAD_ONLY_SLOTS:-6 7} "
head_only(){ case "$HEAD_ONLY" in *" $1 "*) return 0;; *) return 1;; esac; }
declare -A slot_job=()
for i in "${!SLOTS[@]}"; do slot_job[$i]=""; done
[ "${#SLOTS[@]}" -ge 2 ] || { echo "ABORT: slot table has ${#SLOTS[@]} entry" >&2; exit 2; }
echo "slots ${#SLOTS[@]}: ${SLOTS[*]} (head-only:$HEAD_ONLY)"

st(){ cat "$STATE/$1.state" 2>/dev/null || echo queued; }
setst(){ echo "$2" > "$STATE/$1.state"; }

# --------------------------------------------------------------- job fields
jf(){ awk -F'\t' -v i="$1" -v c="$2" '$1==i {print $c; exit}' "$Q"; }

# The launcher and its arm argument, from the card's own cell table.
cell_launcher(){ awk -F'\t' -v c="$1" '$1==c {print $3; exit}' "$HERE/cells.tsv"; }
cell_arg(){      awk -F'\t' -v c="$1" '$1==c {print $4; exit}' "$HERE/cells.tsv"; }

# ------------------------------------------------------------ VRAM on elisa
free_mib(){ nvidia-smi --id="$1" --query-gpu=memory.free \
              --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

# -------------------------------------------------------------- run one job
# Every job ends by writing its return code beside its log, on the machine
# that ran it. The dispatcher polls for that file and nothing else, so an
# ssh drop mid-job costs a tick, not the job.
launch_bb(){ # <id> <cell> <stop> <machine> <gpu>
  local id="$1" cell="$2" stop="$3" mach="$4" gpu="$5"
  local L A args env bb rf=""
  L="$(cell_launcher "$cell")"; A="$(cell_arg "$cell")"
  args="$A"; case "$L" in run_leg_k.sh) args="$A $stop" ;; esac
  env="K=$K TARGET_STEPS=$stop FINAL_STEPS=200000 SAVE_EVERY=20000"
  env="$env EXTRA_SAVES=$stop HEAD_SEED=$HEAD_SEED"
  # The group B launchers pick their resume with `ls -t`, so mtime decides,
  # and staging a checkpoint onto the box gives it the newest mtime there.
  # Name the checkpoint instead. run_leg_k.sh already chooses by step.
  if [ "$stop" -gt 100000 ] && [ "$L" != run_leg_k.sh ]; then
    bb="$(resume_ckpt "$cell" "$stop")"
    [ -n "$bb" ] || { log "START $cell: no checkpoint below $stop"; return 1; }
    if [ "$mach" = rem ]; then rf="/root/cf373_runs/${bb#$CF373_R3/}"; else rf="$bb"; fi
    env="$env RESUME_FROM=$rf"
    log "RESUME $cell from $(basename "$bb")"
  fi
  if [ "$mach" = rem ]; then
    stage_bb_remote "$cell" "$stop" || return 1
    rlaunch "$id" <<EOF || return 1
cd /root/cf || exit 90
export PYTHONPATH=/root/cf HF_HUB_DOWNLOAD_TIMEOUT=120
$env WT=/root/cf BB_GPU=$gpu RUNS=/root/cf373_runs \
CF373_RUNS=/root/cf373_runs CF373_ROOT=/root/cf373_runs \
  bash /root/cf/reports/2026-08-08_rollout_depth/scripts/$L $args
EOF
  else
    ( export PYTHONPATH="$WT" HF_HUB_DOWNLOAD_TIMEOUT=120
      export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
      export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
      eval "$env" WT="$WT" BB_GPU="$gpu" RUNS="$CF373_R3" \
        CF373_RUNS="$CF373_R3" CF373_ROOT="$CF373_R3" \
        bash "$HERE/$L" $args >"$RES/q_$id.log" 2>&1
      echo $? > "$STATE/$id.rc" ) &
  fi
  return 0
}

launch_head(){ # <id> <cell> <stop> <enc> <machine> <gpu>
  local id="$1" cell="$2" stop="$3" enc="$4" mach="$5" gpu="$6"
  if [ "$mach" = rem ]; then
    stage_head_remote "$cell" "$stop" || return 1
    rlaunch "$id" <<EOF || return 1
export PYTHONPATH=/root/cf CF373_ROOT=/root/cf373_runs WT=/root/cf
HEAD_GPU=$gpu HEAD_SEED=$HEAD_SEED HEAD_VRAM_MIB=$HEAD_VRAM \
  bash /root/cf/reports/2026-08-08_rollout_depth/scripts/r2_head_box.sh \
    $cell $K $stop $enc
EOF
  else
    ( export PYTHONPATH="$WT" CF373_ROOT="$CF373_R3" WT="$WT"
      HEAD_GPU="$gpu" HEAD_SEED="$HEAD_SEED" HEAD_VRAM_MIB="$HEAD_VRAM" \
        bash "$HERE/r2_head_box.sh" "$cell" "$K" "$stop" "$enc" \
        >"$RES/q_$id.log" 2>&1
      echo $? > "$STATE/$id.rc" ) &
  fi
  return 0
}

# The eval needs the backbone and the head in elisa's own tree. The sync loop
# brings both back from the box; a job that ran here wrote them directly.
launch_eval(){ # <id> <cell> <stop> <enc>
  local id="$1" cell="$2" stop="$3" enc="$4"
  local stopk=$(( stop / 1000 )) tag out bb head score
  tag="${cell}_k${K}_bb${stopk}k_${enc}"
  out="$CF373_R3/eval/$tag"
  head="$out/qhead_${tag}_s${HEAD_SEED}_final.pth"
  bb="$(CF373_ROOT="$CF373_R3" bash -c \
        ". '$HERE/cell_paths.sh'; cf373_bb_ckpt '$cell' '$K' '$stop'")"
  [ -f "$head" ] || { echo "no head yet"; return 2; }
  [ -n "$bb" ] && [ -f "$bb" ] || { echo "no backbone yet"; return 2; }
  # The head records the checkpoint it read. A mismatch means the pair is not
  # this cell's, and scoring it would attribute another run's number here.
  if [ -f "$out/backbone.txt" ]; then
    [ "$(basename "$bb")" = "$(cat "$out/backbone.txt")" ] || {
      echo "pair mismatch"; return 3; }
  fi
  score="$RES/score_${tag}.txt"
  ( EVAL_SHARDS="$EVAL_SHARDS" WT="$WT" \
      bash "$HERE/eval_local.sh" "$tag" "$stopk" "$enc" "$bb" "$head" \
        "$out" "$score" >"$RES/q_$id.log" 2>&1
    echo $? > "$STATE/$id.rc" ) &
  return 0
}

# ------------------------------------------------- stage a resume on the box
# An extend resumes the furthest checkpoint the fleet already produced below
# its stop. It has to be on the box, under the path the launcher resumes
# from, with its optimizer beside it: without the optimizer a resume loses
# the step counter, the RNG state and AdamW's moments.
#
# `stop - 100000` would name the 100k for every cell, and B1 and B2 hold a
# 140k. Resuming their 100k retrains 40k steps each that are already on disk.
resume_ckpt(){ # <cell> <stop> -> local path, or nothing
  CF373_ROOT="$CF373_R3" bash -c \
    ". '$HERE/cell_paths.sh'; cf373_bb_below '$1' '$K' '$2'"
}

# ------------------------------------- stage a head's backbone on the box
# A head reads its backbone from the tree on the machine that runs it. Four
# of the nine backbones this round train on elisa, and elisa's two cards
# carry other projects, so a head for one of them can land on the rented box
# with nothing to read: `r2_head_box.sh` exits 3 and the queue marks the job
# failed for good.
#
# So ask the box first — a backbone it trained is already there, and asking
# costs one ssh — and mirror from elisa only when it is not. No optimizer:
# a head reads weights, it does not resume.
#
# A missing local copy returns non-zero, which leaves the job queued rather
# than failed. The sync loop brings it in and the next tick places it.
stage_head_remote(){ # <cell> <stop>
  local cell="$1" stop="$2" bb rdir rname a b dst
  rdir="$(CF373_ROOT=/root/cf373_runs bash -c \
          ". '$HERE/cell_paths.sh'; cf373_runs_dir '$cell' '$K' '$stop'")"
  rname="$(bash -c ". '$HERE/cell_paths.sh'; cf373_run_name '$cell' '$K'")"
  a="$(rsh "ls $rdir/${rname}*_$(( stop / 1000 ))k.pth 2>/dev/null \
            | grep -v optimizer | head -1" 2>/dev/null | tr -d ' \r')"
  [ -n "$a" ] && return 0                       # the box trained it
  bb="$(CF373_ROOT="$CF373_R3" bash -c \
        ". '$HERE/cell_paths.sh'; cf373_bb_ckpt '$cell' '$K' '$stop'")"
  [ -n "$bb" ] && [ -f "$bb" ] || {
    log "STAGE-HEAD $cell bb$(( stop / 1000 ))k: not on the box, not on elisa yet"
    return 1; }
  dst="/root/cf373_runs/${bb#$CF373_R3/}"; dst="$(dirname "$dst")"
  log "STAGE-HEAD $cell -> $dst/$(basename "$bb")"
  rsh "mkdir -p '$dst'" >/dev/null 2>&1 || return 1
  scp "${SSH_OPTS[@]}" -P "$BOX_PORT" "$bb" "root@$BOX_HOST:$dst/" \
    >/dev/null 2>&1 || { log "STAGE-HEAD $cell: scp failed"; return 1; }
  a="$(rsh "stat -c%s '$dst/$(basename "$bb")'" 2>/dev/null | tr -d ' ')"
  b="$(stat -c%s "$bb")"
  [ "$a" = "$b" ] || { log "STAGE-HEAD $cell: size $a != $b"; return 1; }
  return 0
}

stage_bb_remote(){ # <cell> <stop>
  local cell="$1" stop="$2" bb opt dst
  [ "$stop" -le 100000 ] && return 0          # B8 starts at step 0
  bb="$(resume_ckpt "$cell" "$stop")"
  [ -n "$bb" ] && [ -f "$bb" ] || { log "STAGE $cell: no local checkpoint below $stop"; return 1; }
  opt="${bb%.pth}_optimizer.pth"
  [ -f "$opt" ] || { log "STAGE $cell: no optimizer beside $(basename "$bb")"; return 1; }
  dst="/root/cf373_runs/${bb#$CF373_R3/}"; dst="$(dirname "$dst")"
  if [ "$(rsh "stat -c%s '$dst/$(basename "$bb")' 2>/dev/null" 2>/dev/null)" = "$(stat -c%s "$bb")" ]; then
    return 0                                   # already staged
  fi
  log "STAGE $cell -> $dst"
  rsh "mkdir -p '$dst'" >/dev/null 2>&1 || return 1
  scp "${SSH_OPTS[@]}" -P "$BOX_PORT" "$bb" "$opt" "root@$BOX_HOST:$dst/" \
    >/dev/null 2>&1 || { log "STAGE $cell: scp failed"; return 1; }
  # Verify by size on the box, not by scp's exit status.
  local a b
  a="$(rsh "stat -c%s '$dst/$(basename "$bb")'" 2>/dev/null | tr -d ' ')"
  b="$(stat -c%s "$bb")"
  [ "$a" = "$b" ] || { log "STAGE $cell: size $a != $b"; return 1; }
  return 0
}

# ------------------------------------------------------------------ readiness
dep_ok(){ local d="$1"; [ "$d" = "-" ] && return 0; [ "$(st "$d")" = done ]; }

# ---------------------------------------------------------------- the loop
# Say which process is the dispatcher, in writing. Every local job runs in a
# subshell that carries this same argv, and a dispatcher restart orphans
# those subshells onto init, so neither the argv nor `ppid == 1` tells the
# supervisor whether a dispatcher is alive.
echo $$ > "$STATE/dispatcher.pid"
log "start box=$BOX_ID $BOX_HOST:$BOX_PORT slots='${SLOTS[*]}' root=$CF373_R3 pid=$$"

# Adopt what is already running. A dispatcher restart must not place a second
# copy of a job onto a second card: round 2 did exactly that once, and two
# processes trained the same run name over each other. A job whose state says
# `running` goes back onto the slot its own marker names.
for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
  [ "$(st "$id")" = running ] || continue
  want="$(cat "$STATE/$id.machine" 2>/dev/null)"
  [ -n "$want" ] || continue
  case "$want" in elisa-cpu) log "ADOPT $id (elisa cores)"; continue;; esac
  for i in "${!SLOTS[@]}"; do
    [ "${SLOTS[$i]}" = "$want" ] && [ -z "${slot_job[$i]}" ] || continue
    slot_job[$i]="$id"; log "ADOPT $id on ${SLOTS[$i]}"; break
  done
done
if [ "${DRY:-0}" = 1 ]; then
  while IFS=$'\t' read -r id type cell stop enc dep; do
    case "$id" in ''|\#*) continue;; esac
    printf '%-22s %-5s %-4s %-7s %-8s dep=%-22s %s\n' \
      "$id" "$type" "$cell" "$stop" "$enc" "$dep" "$(st "$id")"
  done < "$Q"
  exit 0
fi

evals_running(){ local n=0 id
  for id in $(awk -F'\t' '$2=="eval"{print $1}' "$Q"); do
    [ "$(st "$id")" = running ] && n=$(( n + 1 ))
  done; echo "$n"; }

while :; do
  # ---- reap: remote rc files in one ssh, local ones off disk
  remote_rc="$(rsh 'cd /root/jobs 2>/dev/null && grep -H . *.rc 2>/dev/null' 2>/dev/null || true)"
  for i in "${!SLOTS[@]}"; do
    id="${slot_job[$i]}"; [ -n "$id" ] || continue
    mach="${SLOTS[$i]%%:*}"; rc=""
    if [ "$mach" = rem ]; then
      rc="$(sed -n "s/^$id\.rc:\(.*\)$/\1/p" <<<"$remote_rc" | tr -d ' \r')"
    else
      rc="$(cat "$STATE/$id.rc" 2>/dev/null | tr -d ' ')"
    fi
    [ -n "$rc" ] || continue
    if [ "$rc" = 0 ]; then setst "$id" done; log "DONE $id (${SLOTS[$i]})"
    else                   setst "$id" failed; log "FAIL $id rc=$rc (${SLOTS[$i]})"; fi
    slot_job[$i]=""
  done
  # ---- reap evals (they hold no slot)
  for id in $(awk -F'\t' '$2=="eval"{print $1}' "$Q"); do
    [ "$(st "$id")" = running ] || continue
    rc="$(cat "$STATE/$id.rc" 2>/dev/null | tr -d ' ')"
    [ -n "$rc" ] || continue
    if [ "$rc" = 0 ]; then setst "$id" done; log "DONE $id (elisa cpu)"
    else setst "$id" failed; log "FAIL $id rc=$rc (elisa cpu)"; fi
  done

  # ---- dispatch
  while IFS=$'\t' read -r id type cell stop enc dep; do
    case "$id" in ''|\#*) continue;; esac
    [ "$(st "$id")" = queued ] || continue
    dep_ok "$dep" || continue

    if [ "$type" = eval ]; then
      [ "$(evals_running)" -lt "$MAX_EVALS" ] || continue
      why="$(launch_eval "$id" "$cell" "$stop" "$enc")" && {
        setst "$id" running; echo elisa-cpu > "$STATE/$id.machine"
        log "START $id eval on elisa cores"; }
      continue
    fi

    # A GPU job: first free slot that will take it, in the order that costs
    # least. The box is paid by the hour, so what costs money is how long the
    # rental lasts, and the rental lasts as long as the backbones do. A head
    # that takes a rented card holds it away from a backbone and pushes the
    # end of the rental out by its own runtime. So a head asks elisa's free
    # cards first and takes a rented one only when elisa has no room — never
    # leaving a card idle, and never paying for a job elisa could have run.
    #
    # Three tiers for a head, in this order: elisa's cards, which cost
    # nothing; the rented cards' head-only slots, which cost nothing extra
    # because the card is already rented and already holds its two
    # backbones; and last a rented slot a backbone could have used.
    order=("${!SLOTS[@]}")
    if [ "$type" = head ]; then
      order=(); for i in "${!SLOTS[@]}"; do
        [ "${SLOTS[$i]%%:*}" = loc ] && order+=("$i"); done
      for i in "${!SLOTS[@]}"; do
        [ "${SLOTS[$i]%%:*}" = loc ] || ! head_only "$i" || order+=("$i"); done
      for i in "${!SLOTS[@]}"; do
        [ "${SLOTS[$i]%%:*}" = loc ] || head_only "$i" || order+=("$i"); done
    fi
    for i in "${order[@]}"; do
      [ -z "${slot_job[$i]}" ] || continue
      [ "$type" = head ] || ! head_only "$i" || continue
      mach="${SLOTS[$i]%%:*}"; gpu="${SLOTS[$i]##*:}"
      if [ "$mach" = loc ]; then
        need="$BB_VRAM"; [ "$type" = head ] && need="$HEAD_VRAM"
        f="$(free_mib "$gpu")"; [ -n "$f" ] || continue
        [ "$f" -ge "$need" ] || continue
        # A head must read its backbone from the tree on the machine that
        # runs it. elisa's copy arrives by sync, so check before placing.
        if [ "$type" = head ]; then
          bb="$(CF373_ROOT="$CF373_R3" bash -c \
                ". '$HERE/cell_paths.sh'; cf373_bb_ckpt '$cell' '$K' '$stop'")"
          [ -n "$bb" ] && [ -f "$bb" ] || continue
        fi
      fi
      ok=1
      case "$type" in
        bb)   launch_bb   "$id" "$cell" "$stop" "$mach" "$gpu" || ok=0 ;;
        head) launch_head "$id" "$cell" "$stop" "$enc" "$mach" "$gpu" || ok=0 ;;
      esac
      [ "$ok" = 1 ] || { log "START $id failed on ${SLOTS[$i]}"; break; }
      slot_job[$i]="$id"; setst "$id" running
      echo "${SLOTS[$i]}" > "$STATE/$id.machine"
      date -u '+%Y-%m-%dT%H:%M:%SZ' > "$STATE/$id.started"
      log "START $id $type $cell stop=$stop ${enc/-/} on ${SLOTS[$i]}"
      break
    done
  done < "$Q"

  # ---- done?
  left=0
  while IFS=$'\t' read -r id _; do
    case "$id" in ''|\#*) continue;; esac
    case "$(st "$id")" in queued|running) left=$(( left + 1 )) ;; esac
  done < "$Q"
  [ "$left" -eq 0 ] && { log "queue drained"; break; }
  sleep "$POLL"
done
