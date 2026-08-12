#!/bin/bash
# #373 round 2 — bring up one box, give it one cell, and start watching it.
#
# Usage: bash r2_launch_cell.sh <cell> <stop> [stop...]
#   e.g. bash r2_launch_cell.sh B5 40000 100000
#
# Provision -> bootstrap -> stage the cell's resume checkpoint if it has one
# -> start the worker detached -> start this box's sync loop -> verify the
# first tick by `ls`.
#
# Four cells resume rather than start over: A3, B1, B5 and B9 already hold a
# k = 3 checkpoint at 40k from round 1, with its optimizer companion. They
# are uploaded into the path the launcher resumes from, so the run continues
# with the step counter, the RNG state and AdamW's moments intact.
set -uo pipefail

CELL="${1:?usage: r2_launch_cell.sh <cell> <stop> [stop...]}"
shift
STOPS="$*"
[ -n "$STOPS" ] || { echo "ABORT: no stops" >&2; exit 2; }

K="${K:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
export WT="${WT:-/home/jupyter/wt-cf-373-run2}"
SYNC_BASE="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"
# vastrun-kit reads .vastrun.toml from the CWD, and every kit call here has
# to be made from the checkout root that holds it.
VASTRUN_DIR="${VASTRUN_DIR:-$(cd "$HERE" && git rev-parse --show-toplevel)}"
LABEL="cf373r2-$(tr 'A-Z' 'a-z' <<<"$CELL")"
BOXES="$RES/r2_boxes.tsv"
mkdir -p "$RES" "$SYNC_BASE/$CELL"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [$CELL] $*" | tee -a "$RES/r2_boxes.log"; }
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

case "$SYNC_BASE" in /tmp|/tmp/*) say "ABORT: sync base under /tmp"; exit 2;; esac

# ---------------------------------------------------------------- provision
# ADOPT="<id> <host> <port>" reuses a box that is already up: a launcher that
# died after provisioning, or an instance whose sshd opened after the kit
# gave up on it. Provisioning a second box for a cell that already has one
# is the expensive mistake here, not skipping the search.
if [ -n "${ADOPT:-}" ]; then
  read -r ID HOST PORT <<<"$ADOPT"
  say "adopting instance $ID at $HOST:$PORT"
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" true 2>/dev/null || {
    say "ABORT: adopted box does not answer ssh"; exit 3; }
else
  say "provisioning"
  out="$(VAST_SEARCH_ARGS="${VAST_SEARCH_ARGS:---gpu-model RTX_5090,RTX_4090 --min-vram 24 --max-bid 0.70}" \
         VAST_SEARCH_LIMIT="${VAST_SEARCH_LIMIT:-40}" \
         bash "$HERE/provision_box.sh" "$LABEL" "${PROV_TRIES:-10}")" || {
    say "ABORT: provision failed"; exit 3; }
  read -r ID HOST PORT <<<"$out"
  say "instance $ID at $HOST:$PORT"
fi

# ------------------------------------------------------------------ bootstrap
if ! WT="$WT" bash "$HERE/bootstrap_remote.sh" "$HOST" "$PORT" >>"$RES/r2_bootstrap_$CELL.log" 2>&1; then
  say "ABORT: bootstrap failed — destroying $ID (see r2_bootstrap_$CELL.log)"
  (cd "$VASTRUN_DIR" && vastrun-destroy "$ID" --force) >/dev/null 2>&1
  exit 4
fi
say "bootstrapped"

# ------------------------------------------------------- stage resume state
# The cells that carry a round-1 k = 3 checkpoint at 40k, and where that
# checkpoint is on elisa. Nothing is guessed: each path is checked before
# it is uploaded, and a cell that is not in this list starts at step 0.
R1_A="/home/jupyter/cf373_sync"
declare -A RESUME_SRC=(
  [A3]="$R1_A/b/sync/arm6_v2_combab_alignT/leg_40k"
  [B5]="$R1_A/a/sync/bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
  [B9]="$R1_A/c/sync/bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
  [B1]="/home/jupyter/checkpoints_backup/cf-373/bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
)
declare -A RESUME_DST=(
  [A3]="/root/cf373_runs/arm6_v2_combab_alignT/leg_40k"
  [B5]="/root/cf373_runs/bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
  [B9]="/root/cf373_runs/bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
  [B1]="/root/cf373_runs/bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"
)
# A cell that already ran on a box this round has its own, further,
# checkpoint in the local sync tree. Prefer it: a re-launch after a box dies
# must not throw away the steps the sync loop saved. The path inside the
# tree mirrors the box's, so the destination is the same relative path.
src=""; dst=""
newest="$(find "$SYNC_BASE/$CELL/sync" -name "*_[0-9]*k.pth" ! -name "*optimizer*" 2>/dev/null \
          | sed -E 's|.*_([0-9]+)k\.pth$|\1 &|' | sort -k1,1n | tail -1 | cut -d' ' -f2-)"
if [ -n "$newest" ]; then
  src="$(dirname "$newest")"
  dst="/root/cf373_runs/${src#$SYNC_BASE/$CELL/sync/}"
  step_k="$(sed -E 's|.*_([0-9]+)k\.pth$|\1|' <<<"$newest")"
  say "resuming from this round's own sync tree at ${step_k}k"
elif [ -n "${RESUME_SRC[$CELL]:-}" ]; then
  src="${RESUME_SRC[$CELL]}"; dst="${RESUME_DST[$CELL]}"; step_k=40
fi

if [ -n "$src" ]; then
  bb="$(ls "$src"/*_${step_k}k.pth 2>/dev/null | grep -v optimizer | head -1)"
  opt="$(ls "$src"/*_${step_k}k_optimizer.pth 2>/dev/null | head -1)"
  [ -f "$bb" ] && [ -f "$opt" ] || { say "ABORT: no ${step_k}k checkpoint+optimizer pair under $src"; (cd "$VASTRUN_DIR" && vastrun-destroy "$ID" --force) >/dev/null 2>&1; exit 5; }
  say "staging resume: $(basename "$bb") + optimizer -> $dst"
  rsh "mkdir -p '$dst'" || { say "ABORT: mkdir on box"; exit 5; }
  scp "${SSH_OPTS[@]}" -P "$PORT" "$bb" "$opt" "root@$HOST:$dst/" >/dev/null || {
    say "ABORT: staging scp failed"; (cd "$VASTRUN_DIR" && vastrun-destroy "$ID" --force) >/dev/null 2>&1; exit 5; }
  # Verify by size on the box, not by scp's exit status.
  want=$(( $(wc -c <"$bb") + $(wc -c <"$opt") ))
  got=$(rsh "cat '$dst/$(basename "$bb")' '$dst/$(basename "$opt")' 2>/dev/null | wc -c")
  [ "$got" = "$want" ] || { say "ABORT: staged $got B, want $want B"; (cd "$VASTRUN_DIR" && vastrun-destroy "$ID" --force) >/dev/null 2>&1; exit 5; }
  say "resume state verified on the box ($want B)"
fi

# --------------------------------------------------------------- start work
# The subshell, the three redirections and the bare `exit 0` are load
# bearing: ssh holds the session open until every descriptor closes, so a
# backgrounded remote job that inherits one hangs the caller forever.
rsh "(setsid env K=$K SKIP_HEAD_STOPS='${SKIP_HEAD_STOPS:-}' \
      HEAD_ENCS='${HEAD_ENCS:-}' \
      bash /root/cf/reports/2026-08-08_rollout_depth/scripts/r2_cell_worker.sh \
      $CELL $STOPS </dev/null >/root/worker.log 2>&1 &) ; exit 0" >/dev/null 2>&1 || {
  say "ABORT: could not start the worker"; exit 6; }

tmp="$BOXES.tmp.$$"
{ [ -f "$BOXES" ] && grep -v -P "^$CELL\t" "$BOXES"; \
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$CELL" "$ID" "$HOST" "$PORT" "$LABEL" "$STOPS"; } \
  | sort > "$tmp" && mv -f "$tmp" "$BOXES"

# ---------------------------------------------------------------- sync loop
LOCAL="$SYNC_BASE/$CELL"
mkdir -p "$LOCAL/sync" "$LOCAL/results"
SAFE_PULL="$WT/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
REMOTE_HOST="$HOST" REMOTE_PORT="$PORT" SSH_USER=root \
REMOTE_DIR=/root/cf/reports/2026-08-08_rollout_depth \
REMOTE_RUNS=/root/cf373_runs \
LOCAL_DIR="$LOCAL" SAFE_PULL="$SAFE_PULL" INTERVAL="${SYNC_INTERVAL:-900}" \
  nohup setsid bash "$STUDY/sync/sync_loop.sh" > "$LOCAL/sync_loop.log" 2>&1 &
# The pid, so a relaunch can stop THIS cell's loop and not every cell's.
# Its command line is `bash sync_loop.sh` on all fourteen — the host it
# polls is in the environment, not in the arguments — so a pattern kill
# would take the whole fleet's syncing with it.
echo $! > "$RES/r2_syncpid_$CELL"
say "sync loop started -> $LOCAL (pid $!)"
say "LAUNCHED instance=$ID host=$HOST port=$PORT stops=$STOPS"
