#!/bin/bash
# #404 — put this study on a bare vast.ai box.
#
# The box trains the four backbones and nothing else (`launch_box.sh`), so it
# needs #373's payload plus this study's own directory. #373's
# `bootstrap_remote.sh` already installs the wheel set, disables the CUDA
# forward-compat layer and gates the trainer, and its tarball carries the
# runner, the trainer, `src`, the shared scripts and the HF token. It does NOT
# carry `reports/2026-08-19_ema_momentum_k32`, because that directory did not
# exist when it was written.
#
# So this runs that bootstrap, then ships this study's directory beside the
# parent's. Nothing else differs.
#
# Usage: WT=<local checkout> bash scripts/bootstrap_box.sh <ssh_host> <ssh_port>
set -uo pipefail

HOST="${1:?usage: bootstrap_box.sh <ssh_host> <ssh_port>}"
PORT="${2:?usage: bootstrap_box.sh <ssh_host> <ssh_port>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
WT="${WT:-$CF404_WT}"
STUDY_REL="reports/$(basename "$CF404_STUDY")"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 bootstrap $HOST:$PORT] $*"; }
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

say "parent bootstrap (#373)"
WT="$WT" bash "$CF404_PARENT/scripts/bootstrap_remote.sh" "$HOST" "$PORT" || exit $?

say "shipping $STUDY_REL"
TGZ="/tmp/cf404_study.$$.tgz"
tar czf "$TGZ" -C "$WT" --exclude='__pycache__' --exclude='results/*' \
  --exclude='plots/*' "$STUDY_REL" || exit 3
scp "${SSH_OPTS[@]}" -P "$PORT" "$TGZ" "root@$HOST:/root/cf404_study.tgz" || exit 4
rm -f "$TGZ"
rsh 'tar xzf /root/cf404_study.tgz -C /root/cf' || exit 5

# The box has to pass the SAME checkout gate the launcher runs, before it is
# rented for hours rather than after. It also has to be able to source
# study.sh at all: an arms table that did not land is a study with no arms.
say "checking the box's checkout"
rsh "cd /root/cf/$STUDY_REL && CF404_DRY_RUN=1 bash scripts/launch_box.sh" || exit 6
say "OK"
