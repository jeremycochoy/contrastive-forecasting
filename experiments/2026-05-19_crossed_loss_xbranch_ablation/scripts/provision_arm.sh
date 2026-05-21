#!/bin/bash
# Provision a vast.ai box, push code, launch box_run.sh <phase> for one
# arm. Run ON elisa from the worktree root (needs .vastrun.toml + SSH key).
#   provision_arm.sh <offer_id> <arm> [phase=backbone]
# Writes state/<arm>.env  (INST HOST PORT OFFER ARM PHASE TS).
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
EXP="$WT/experiments/2026-05-19_crossed_loss_xbranch_ablation"
cd "$WT"
source "$EXP/scripts/_ssh.sh"
OFFER="${1:?offer_id}"; ARM="${2:?arm}"; PHASE="${3:-backbone}"
LABEL="xbranch-${ARM}"
ST="$EXP/scripts/state"; mkdir -p "$ST"
log(){ echo "[$(date '+%H:%M:%S')] [$ARM] $*"; }

log "provisioning offer $OFFER (label $LABEL)"
POUT="$(vastrun-provision "$OFFER" --label "$LABEL" 2>&1)"; echo "$POUT"
INST="$(echo "$POUT" | grep -oE 'Instance [0-9]+ ready' | grep -oE '[0-9]+' | head -1)"
HP="$(echo "$POUT"  | grep -oE 'ssh -p [0-9]+ root@[^ ]+' | head -1)"
PORT="$(echo "$HP" | grep -oE -- '-p [0-9]+' | grep -oE '[0-9]+')"
HOST="$(echo "$HP" | grep -oE 'root@[^ ]+' | sed 's/root@//')"
[ -n "$INST" ] || { log "FAILED to parse instance id"; exit 1; }
[ -n "$HOST" ] && [ -n "$PORT" ] || { read -r HOST PORT < <(ssh_coords "$INST"); }
[ -n "$HOST" ] && [ -n "$PORT" ] || { log "FAILED to resolve ssh for $INST"; exit 1; }
log "instance $INST @ $HOST:$PORT"
cat > "$ST/$ARM.env" <<EOF
INST=$INST
HOST=$HOST
PORT=$PORT
OFFER=$OFFER
ARM=$ARM
PHASE=$PHASE
TS=$(date -u +%FT%TZ)
EOF

# --- push code tarball (the exact worktree incl. #307 loss.py) ---
TAR=/tmp/xbranch_code.tgz
[ -f "$TAR" ] || tar czf "$TAR" -C "$WT" \
  --exclude=.git --exclude=runs --exclude=results --exclude='sync_*' \
  --exclude=.pytest_cache --exclude='*.pth' --exclude='__pycache__' \
  --exclude='experiments/*/plots' src experiments scripts README.md
log "pushing code ($(du -h "$TAR"|cut -f1))"
for try in 1 2 3 4 5; do
  ssh $SSHO -p "$PORT" "root@$HOST" 'mkdir -p /workspace/app' 2>/dev/null && \
  scp $SSHO -P "$PORT" "$TAR" "root@$HOST:/workspace/app/code.tgz" 2>/dev/null && break
  log "push retry $try (ssh not ready yet)"; sleep 20
done
ssh $SSHO -p "$PORT" "root@$HOST" \
  'cd /workspace/app && tar xzf code.tgz && ls src/loss.py experiments/2026-05-19_crossed_loss_xbranch_ablation/scripts/box_run.sh' \
  || { log "FAILED to extract code on box"; exit 1; }

# --- launch box_run under setsid (survives ssh disconnect) ---
RUNLOG="/workspace/app/box_${ARM}_${PHASE}.log"
ssh $SSHO -p "$PORT" "root@$HOST" \
  "cd /workspace/app && setsid bash -c 'bash experiments/2026-05-19_crossed_loss_xbranch_ablation/scripts/box_run.sh $PHASE $ARM > $RUNLOG 2>&1' < /dev/null & echo launched pid \$!"
sleep 4
ssh $SSHO -p "$PORT" "root@$HOST" "tail -5 $RUNLOG 2>/dev/null; echo '---'; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
log "launched $PHASE on $INST ($HOST:$PORT). state -> $ST/$ARM.env"
