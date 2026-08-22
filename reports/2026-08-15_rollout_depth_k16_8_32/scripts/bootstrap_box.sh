#!/bin/bash
# #401 — put a bare vast.ai box in a state where this study's backbones run.
#
# The bootstrap is #373's `bootstrap_remote.sh`, unchanged apart from one
# additive hook: `EXTRA_PACK` names extra paths for its tarball, and it is
# empty by default, so every #373 command line still packs what it packed.
# This wrapper sets it to this study's scripts directory. A second bootstrap
# would be a second wheel set and a second GPU gate to keep correct.
#
# #373's bootstrap already does the parts that cost money to get wrong: the
# pinned wheel set (torch 2.8.0 / cu128, elisa's), pip retries over a link
# that times out, the CUDA forward-compat workaround that turns a GeForce
# card's every CUDA call into error 804, and a `--help` gate on both trainers
# so an ImportError shows up now and not at step 0 of a rented run.
#
# This wrapper adds TWO gates of its own, after that one passes.
#
#   `--train-rollout-reduce` on the box's train.py. That flag is this study's
#   whole protocol. A box bootstrapped from a checkout without it would train
#   the SUMMED objective, log nothing unusual, and produce 30 hours of the arm
#   this card already ran.
#
#   The trainer writes its own command line. `run_arm_k.sh` reads the
#   reduction off that line after a leg starts, and it is the only run-time
#   proof that the flag arrived. A trainer that writes no command line leaves
#   the check with nothing to read, on the machine where 33 hours are rented.
#
# Usage:
#   WT=/tmp/contrastive-forecasting-401 bash scripts/bootstrap_box.sh <host> <port>
set -uo pipefail

HOST="${1:?usage: bootstrap_box.sh <ssh_host> <ssh_port>}"
PORT="${2:?usage: bootstrap_box.sh <ssh_host> <ssh_port>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

WT="${WT:-$CF401_REPO}"
BOOTSTRAP="$CF401_PARENT/scripts/bootstrap_remote.sh"
[ -f "$BOOTSTRAP" ] || { echo "ABORT: no bootstrap at $BOOTSTRAP" >&2; exit 2; }

STUDY_REL="reports/$(basename "$CF401_STUDY")"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "bootstrap host=$HOST port=$PORT WT=$WT"
  echo "  bootstrap=$BOOTSTRAP"
  echo "  EXTRA_PACK=$STUDY_REL/scripts"
  echo "  gate: train.py --help names --train-rollout-reduce"
  echo "  gate: train.py writes 'Command line:'"
  exit 0
fi

echo "[#401] bootstrapping $HOST:$PORT from $WT"
WT="$WT" EXTRA_PACK="$STUDY_REL/scripts" \
  bash "$BOOTSTRAP" "$HOST" "$PORT"
rc=$?
[ $rc -eq 0 ] || { echo "ABORT: #373 bootstrap rc=$rc" >&2; exit $rc; }

# The protocol gate. `--help` already ran inside the bootstrap, so this costs
# one more ssh and it is the only check that separates this card's objective
# from the one it replaces.
echo "[#401] gate: --train-rollout-reduce on the box"
rsh "cd /root/cf && PYTHONPATH=/root/cf python3 \
  experiments/2026-04-27_freq-embedding/scripts/train.py --help \
  | grep -q -- --train-rollout-reduce" || {
  echo "ABORT: the box's train.py has no --train-rollout-reduce." >&2
  echo "  It would train the SUMMED objective. Push the branch that" >&2
  echo "  carries #401 and bootstrap again." >&2
  exit 6; }

# The run-time proof. `run_arm_k.sh` reads the reduction out of this line.
echo "[#401] gate: the box's train.py writes its own command line"
rsh "grep -q 'Command line: ' \
  /root/cf/experiments/2026-04-27_freq-embedding/scripts/train.py" || {
  echo "ABORT: the box's train.py writes no command line, so no check" >&2
  echo "  there can read which reduction a leg trains. Push the branch" >&2
  echo "  that carries #401 and bootstrap again." >&2
  exit 6; }

# The study's own scripts, and #373's runner they call.
rsh "test -f /root/cf/$STUDY_REL/scripts/study.sh" || {
  echo "ABORT: $STUDY_REL/scripts did not land on the box." >&2; exit 7; }

echo "[#401] OK — $HOST:$PORT carries the trainer, the flag and the scripts"
echo "  next: ssh -p $PORT root@$HOST"
echo "        cd /root/cf/$STUDY_REL"
echo "        nohup setsid bash scripts/launch_box.sh &   # root: $CF401_BOX_RUNS"
