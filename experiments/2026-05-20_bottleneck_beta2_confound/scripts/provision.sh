#!/bin/bash
# #309 provisioner — call from worktree root. Provisions a vast box,
# pushes worktree code, launches box_run_serial.sh (α→β→γ serial).
# Writes state to scripts/state/box.env so sync/destroy helpers can find it.
#
# Usage: provision.sh <offer_id>
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
EXP="$WT/experiments/2026-05-20_bottleneck_beta2_confound"
cd "$WT"
source "$EXP/scripts/_ssh.sh"
OFFER="${1:?offer_id}"
LABEL="bbeta2-conf-309"
ST="$EXP/scripts/state"; mkdir -p "$ST"
log(){ echo "[$(date '+%H:%M:%S')] [#309 provision] $*"; }

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
cat > "$ST/box.env" <<EOF
INST=$INST
HOST=$HOST
PORT=$PORT
OFFER=$OFFER
LABEL=$LABEL
TS=$(date -u +%FT%TZ)
EOF

TAR=/tmp/bbeta2_code.tgz
rm -f "$TAR"
tar czf "$TAR" -C "$WT" \
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
  "cd /workspace/app && tar xzf code.tgz && ls src/loss.py experiments/2026-05-20_bottleneck_beta2_confound/scripts/box_run.sh" \
  || { log "FAILED to extract code on box"; exit 1; }

RUNLOG="/workspace/app/box_serial.log"
ssh $SSHO -p "$PORT" "root@$HOST" \
  "cd /workspace/app && setsid bash -c 'bash experiments/2026-05-20_bottleneck_beta2_confound/scripts/box_run_serial.sh > $RUNLOG 2>&1' < /dev/null & echo launched pid \$!"
sleep 5
ssh $SSHO -p "$PORT" "root@$HOST" "tail -10 $RUNLOG 2>/dev/null; echo '---'; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
log "launched #309 serial α→β→γ on $INST ($HOST:$PORT). state -> $ST/box.env"
