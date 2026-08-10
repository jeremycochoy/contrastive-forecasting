#!/bin/bash
# #373 — put a bare vast.ai box in a state where one backbone run works.
#
# Usage: WT=<local checkout> bash bootstrap_remote.sh <ssh_host> <ssh_port>
#
# Lighter than #393's bootstrap. The box trains the backbone AND its heads;
# the GIFT-Eval stays on elisa. So no gift-eval package, no gift-eval data,
# no seasonal-naive denominator on the box — nothing on a rented card
# produces a GM-Relative MASE, so nothing there can put one on a different
# scale. Round 1 kept the heads on elisa too, and round 2 moved them here
# because other sessions hold both of elisa's cards.
#
# The repo is private, so the code is pushed rather than cloned. Versions
# are pinned to elisa's: a run's numbers must not depend on which machine
# produced them.
set -uo pipefail

HOST="${1:?usage: bootstrap_remote.sh <ssh_host> <ssh_port>}"
PORT="${2:?usage: bootstrap_remote.sh <ssh_host> <ssh_port>}"
WT="${WT:?WT must be the absolute path of the local checkout}"
STAGE="${STAGE:-/tmp/cf373_stage}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [bootstrap $HOST:$PORT] $*"; }
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

[ -f "$WT/experiments/hf_token.txt" ] || { say "ABORT: no HF token"; exit 2; }

mkdir -p "$STAGE"
say "packing code"
# Boxes are bootstrapped concurrently and share $STAGE. One writer at a
# time, and the published name only ever changes by an atomic rename, so a
# concurrent reader sees a whole tarball or the previous whole one.
(
  flock 6
  tar czf "$STAGE/cf373_code.tgz.$$" -C "$WT" --exclude='__pycache__' \
    src \
    experiments/hf_token.txt \
    experiments/2026-04-27_freq-embedding/scripts/train.py \
    experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    reports/2026-08-08_rollout_depth/scripts \
    && mv -f "$STAGE/cf373_code.tgz.$$" "$STAGE/cf373_code.tgz"
) 6>"$STAGE/.pack.lock" || exit 3

say "uploading $(du -h "$STAGE/cf373_code.tgz" | cut -f1)"
scp "${SSH_OPTS[@]}" -P "$PORT" "$STAGE/cf373_code.tgz" "root@$HOST:/root/" || exit 4

say "installing"
rsh 'cat > /root/cf373_bootstrap.sh' <<'BOOT'
#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv git curl || exit 9
PIP="python3 -m pip install --break-system-packages"
# cu128 / torch 2.8.0 is elisa's version.
$PIP -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || exit 10
# The trainer's dependency set. statsmodels is NOT eval-only: the
# `--synth-kind forked-arma` mixer every cell runs imports
# statsmodels.tsa.arima_process.ArmaProcess. Dropping it got as far as the
# `train.py --help` gate below, which is why that gate is here. gluonts and
# the gift_eval package are the eval's, and the eval runs on elisa.
$PIP -q "numpy==1.26.4" "pandas==2.2.2" "datasets==2.17.1" "scipy==1.11.4" \
        "statsmodels==0.14.2" pyarrow einops orjson toolz huggingface_hub || exit 11
mkdir -p /root/cf /root/cf373_runs
tar xzf /root/cf373_code.tgz -C /root/cf || exit 12
python3 - <<'PY' || exit 13
import torch
print("torch", torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0))
PY
# The real gate: the trainer imports and parses, on this box, with this
# wheel set. An ImportError found at step 0 of a rented run costs the run.
cd /root/cf && PYTHONPATH=/root/cf python3 \
  experiments/2026-04-27_freq-embedding/scripts/train.py --help >/dev/null || exit 14
# The head trainer is on the box in round 2, so it gets the same gate.
cd /root/cf && PYTHONPATH=/root/cf python3 \
  experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py --help >/dev/null || exit 15
nvidia-smi --query-gpu=compute_mode,name --format=csv,noheader
echo BOOTSTRAP_OK
BOOT
rsh 'chmod +x /root/cf373_bootstrap.sh && setsid nohup bash /root/cf373_bootstrap.sh > /root/bootstrap.log 2>&1 < /dev/null & echo started'

say "waiting for BOOTSTRAP_OK (install is ~4 min)"
for _ in $(seq 1 80); do
  if rsh 'grep -q BOOTSTRAP_OK /root/bootstrap.log' 2>/dev/null; then
    say "OK"; rsh 'tail -n 3 /root/bootstrap.log'; exit 0
  fi
  if rsh 'grep -qE "^\+ exit (9|1[0-5])$" /root/bootstrap.log' 2>/dev/null; then
    say "FAILED"; rsh 'tail -n 25 /root/bootstrap.log'; exit 5
  fi
  sleep 15
done
say "TIMEOUT"; rsh 'tail -n 25 /root/bootstrap.log'; exit 5
