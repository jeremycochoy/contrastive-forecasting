#!/bin/bash
# Variance provisioner — call from elisa worktree root. Provisions a vast
# box, pushes the worktree code, launches box_variance_run.sh for one
# (arm, seed). Writes state/variance_<arm>_<seed>.env.
#   provision_variance.sh <offer_id> <arm> <seed>
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
EXP="$WT/experiments/2026-05-19_crossed_loss_xbranch_ablation"
cd "$WT"
source "$EXP/scripts/_ssh.sh"
OFFER="${1:?offer_id}"; ARM="${2:?arm}"; SEED="${3:?seed}"
LABEL="xbranch-var-${ARM}-s${SEED:(-2)}"
ST="$EXP/scripts/state"; mkdir -p "$ST"
log(){ echo "[$(date '+%H:%M:%S')] [var $ARM s${SEED:(-2)}] $*"; }

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
cat > "$ST/variance_${ARM}_s${SEED:(-2)}.env" <<EOF
INST=$INST
HOST=$HOST
PORT=$PORT
OFFER=$OFFER
ARM=$ARM
SEED=$SEED
TS=$(date -u +%FT%TZ)
EOF

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
  'cd /workspace/app && tar xzf code.tgz && ls src/loss.py experiments/2026-05-19_crossed_loss_xbranch_ablation/scripts/box_variance_run.sh' \
  || { log "FAILED to extract code on box"; exit 1; }

RUNLOG="/workspace/app/box_variance_${ARM}_s${SEED:(-2)}.log"
ssh $SSHO -p "$PORT" "root@$HOST" \
  "cd /workspace/app && setsid bash -c 'bash experiments/2026-05-19_crossed_loss_xbranch_ablation/scripts/box_variance_run.sh $ARM $SEED > $RUNLOG 2>&1' < /dev/null & echo launched pid \$!"
sleep 4
ssh $SSHO -p "$PORT" "root@$HOST" "tail -5 $RUNLOG 2>/dev/null; echo '---'; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
log "launched variance backbone $ARM s${SEED:(-2)} on $INST ($HOST:$PORT). state -> $ST/variance_${ARM}_s${SEED:(-2)}.env"
