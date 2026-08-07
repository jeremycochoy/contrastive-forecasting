#!/bin/bash
# #393 — put one rented box to work on the replicate head seeds.
#
# Usage:  WT=<checkout> bash scripts/deploy_seed_box.sh \
#           <label> <ssh_host> <ssh_port> <cell>[,<cell>...]
#
# Five steps, in this order, because each depends on the one before:
#
#   1. bootstrap_remote.sh — torch, the repo, GIFT-Eval and its data. Skipped
#      if the box already reports BOOTSTRAP_OK.
#   2. push the bb100k backbone of every cell the box was given. The
#      backbones are NOT retrained here; the replicate varies the head seed
#      and nothing else, so the box needs exactly the checkpoint elisa
#      already holds.
#   3. EVAL_PLACE=local_cpu. The box evaluates its own heads on its own
#      cores while its GPU trains the next one. The alternative, `broker`,
#      sends every eval to elisa, and elisa's 32 cores are what the whole
#      fleet would then queue behind — the eval, not the head, is the
#      binding resource for this run.
#   4. sync_loop.sh on elisa, pulling from the box, per REMOTE_LAUNCH_
#      CHECKLIST.md. Its first tick is verified by `ls` before the work
#      starts, not by reading its log.
#   5. seed_replicates.sh, restricted to the box's cells.
#
# A box is given cells whose seed-20260722 head ran on the same GPU model
# it has. seed_replicates.sh carries that map and refuses a cell outside
# the six; the pairing itself is the caller's, and results/seed_boxes.txt
# records it.
set -uo pipefail

LABEL="${1:?usage: deploy_seed_box.sh <label> <host> <port> <cells>}"
HOST="${2:?ssh host}"
PORT="${3:?ssh port}"
CELLS="${4:?comma-separated cells}"

WT="${WT:?WT must be the absolute path of the local checkout}"
EXP="$WT/experiments/2026-08-04_ema_sched_ladder"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
REMOTE_EXP=/root/cf/experiments/2026-08-04_ema_sched_ladder
REMOTE_RUNS=/root/cf393_runs
# The sync target is outside /tmp and outside the checkout — sync_loop.sh
# refuses /tmp, and `git worktree remove --force` deletes untracked files
# under a checkout (CLAUDE.md checkpoint safety rule 4).
SYNC_ROOT="${SYNC_ROOT:-/home/jupyter/cf393_seedsync_$LABEL}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [deploy $LABEL] $*"; }
rsh(){ ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

# --- 1. bootstrap --------------------------------------------------------
if rsh 'grep -q BOOTSTRAP_OK /root/bootstrap.log' 2>/dev/null; then
  say "already bootstrapped"
else
  say "bootstrapping (torch + repo + GIFT-Eval, ~8 min)"
  WT="$WT" bash "$EXP/scripts/bootstrap_remote.sh" "$HOST" "$PORT" \
    || { say "ABORT: bootstrap failed"; exit 3; }
fi

# --- 2. the backbones ----------------------------------------------------
for cell in ${CELLS//,/ }; do
  bb="$RUNS/$cell/leg_100k/cf393_${cell}_100k.pth"
  [ -f "$bb" ] || { say "ABORT: no bb100k checkpoint at $bb"; exit 4; }
  rsh "mkdir -p $REMOTE_RUNS/$cell/leg_100k" || exit 4
  # Size-checked after the fact rather than trusted: a truncated backbone
  # loads and trains, and produces a number.
  want=$(wc -c <"$bb")
  scp "${SSH_OPTS[@]}" -P "$PORT" "$bb" \
      "root@$HOST:$REMOTE_RUNS/$cell/leg_100k/" >/dev/null || exit 4
  got=$(rsh "wc -c < $REMOTE_RUNS/$cell/leg_100k/$(basename "$bb")" 2>/dev/null)
  [ "$got" = "$want" ] || { say "ABORT: $cell backbone landed $got B, want $want"; exit 4; }
  say "backbone $cell: $want B verified on the box"
done

# --- 3. placement and identity ------------------------------------------
# Cores decide how many evals the box runs at once. Four shards each, and
# two cores left for the head training's dataloader and the OS.
CORES=$(rsh nproc 2>/dev/null || echo 8)
SLOTS=$(( (CORES - 2) / 4 )); [ "$SLOTS" -lt 1 ] && SLOTS=1
[ "$SLOTS" -gt 3 ] && SLOTS=3
say "$CORES cores -> $SLOTS concurrent eval(s) x 4 shards"

rsh "mkdir -p $REMOTE_EXP/results" || exit 5
rsh "printf 'local_cpu' > $REMOTE_EXP/results/EVAL_PLACE
     printf '%s' '$LABEL' > $REMOTE_EXP/results/MACHINE" || exit 5

# Ship the scripts again over the bootstrap tarball's copy: HEAD_SEED and
# the exported WT are newer than any box image.
scp "${SSH_OPTS[@]}" -P "$PORT" \
    "$EXP/scripts/eval_stop.sh" "$EXP/scripts/eval_local.sh" \
    "$EXP/scripts/eval_slot.sh" "$EXP/scripts/seed_replicates.sh" \
    "$EXP/scripts/leg_paths.sh" "$EXP/scripts/gpu_gate.sh" \
    "$EXP/scripts/shard_configs.py" \
    "root@$HOST:$REMOTE_EXP/scripts/" >/dev/null || exit 5
scp "${SSH_OPTS[@]}" -P "$PORT" "$EXP/results/config_costs.csv" \
    "root@$HOST:$REMOTE_EXP/results/" >/dev/null || exit 5

# --- 4. sync loop, and its first tick verified by ls ----------------------
mkdir -p "$SYNC_ROOT/2026-04-27_periodic-synth-mix/scripts" \
         "$SYNC_ROOT/2026-08-04_ema_sched_ladder" || exit 6
cp -f "$WT/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh" \
      "$SYNC_ROOT/2026-04-27_periodic-synth-mix/scripts/" || exit 6

if pgrep -f "cf393_seedsync_$LABEL" >/dev/null; then
  say "sync loop already running"
else
  REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
  REMOTE_DIR="$REMOTE_EXP" REMOTE_RUNS="$REMOTE_RUNS" \
  LOCAL_DIR="$SYNC_ROOT/2026-08-04_ema_sched_ladder" INTERVAL=900 \
    nohup setsid bash "$EXP/sync/sync_loop.sh" \
      > "$SYNC_ROOT/sync_loop.log" 2>&1 < /dev/null &
  say "sync loop started -> $SYNC_ROOT"
fi

say "waiting for the first sync tick to land the backbone it just pushed"
first_cell="${CELLS%%,*}"
want_file="$SYNC_ROOT/2026-08-04_ema_sched_ladder/sync/$first_cell/leg_100k/cf393_${first_cell}_100k.pth"
for _ in $(seq 1 40); do
  [ -s "$want_file" ] && break
  sleep 15
done
if [ -s "$want_file" ]; then
  say "sync verified by ls: $(ls -l "$want_file" | awk '{print $5, $9}')"
else
  say "ABORT: no synced backbone at $want_file after 10 min — not launching"
  exit 6
fi

# --- 5. the work ---------------------------------------------------------
say "launching seed_replicates for $CELLS"
rsh "cd $REMOTE_EXP && chmod +x scripts/*.sh && \
     WT=/root/cf RUNS=$REMOTE_RUNS GIFT_EVAL=/root/workspaces/gift-eval-data \
     CF393_SEED_CELLS=$CELLS CF393_SEED_SEEDS='${CF393_SEED_SEEDS:-}' \
     CF393_SEED_JOBS=${CF393_SEED_JOBS:-2} CF393_SEED_GPUS=1 \
     CF393_EVAL_SLOTS=$SLOTS EVAL_SHARDS=4 \
     setsid nohup bash scripts/seed_replicates.sh \
       > results/seed_driver.log 2>&1 < /dev/null & echo launched" || exit 7
sleep 20
rsh "tail -5 $REMOTE_EXP/results/seed_replicates.log" 2>/dev/null
say "done"
