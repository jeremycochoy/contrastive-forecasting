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
# Paths to add to the tarball, relative to $WT, space separated. Empty by
# default, so every #373 command line packs exactly what it always packed.
# #401 reuses this bootstrap and adds its own scripts directory, rather than
# writing a second remote pipeline.
read -r -a EXTRA_PACK_ARR <<<"${EXTRA_PACK:-}"
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
    scripts \
    "${EXTRA_PACK_ARR[@]}" \
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
# A rented box reaches PyPI over a link this study has watched time out six
# times in one fleet launch: every one of those boxes died on
# `ReadTimeoutError` from files.pythonhosted.org, not on a bad wheel set.
# pip's own retry is the fix, and a wheel that already landed is cached, so
# a retry resumes rather than restarts.
PIP="python3 -m pip install --break-system-packages --retries 10 --timeout 120"
retry_pip(){
  for t in 1 2 3; do
    $PIP "$@" && return 0
    echo "pip attempt $t failed; retrying in 30s" >&2; sleep 30
  done
  return 1
}
# cu128 / torch 2.8.0 is elisa's version.
retry_pip -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || exit 10
# The trainer's dependency set. statsmodels is NOT eval-only: the
# `--synth-kind forked-arma` mixer every cell runs imports
# statsmodels.tsa.arima_process.ArmaProcess. Dropping it got as far as the
# `train.py --help` gate below, which is why that gate is here. gluonts and
# the gift_eval package are the eval's, and the eval runs on elisa.
retry_pip -q "numpy==1.26.4" "pandas==2.2.2" "datasets==2.17.1" "scipy==1.11.4" \
        "statsmodels==0.14.2" pyarrow einops orjson toolz huggingface_hub || exit 11
mkdir -p /root/cf /root/cf373_runs
tar xzf /root/cf373_code.tgz -C /root/cf || exit 12
# The GPU gate. It runs before any training and it is hard: a box that
# cannot count a device is destroyed, not trained on.
#
# B8 died on this fifteen times with
#   Error 804: forward compatibility was attempted on non supported HW
# The host driver was never the problem. Every offer taken carried driver
# 580 or 595, which is CUDA 13. The image ships /usr/local/cuda/compat and
# puts it first on LD_LIBRARY_PATH. That directory holds a forward-compat
# libcuda, and forward compat runs on data-center cards only. On a GeForce
# card the loader takes the compat libcuda and every CUDA call returns 804,
# while the card and its driver are fine.
#
# So take the compat directory off the path and out of the way. A path
# entry that does not exist is skipped, so moving the directory fixes every
# later process on the box, not only this shell. The trainer and the head
# trainer start over a separate ssh and would not inherit an export.
gpu_count(){ python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null | tail -1; }
if [ "$(gpu_count)" = "0" ] || [ -z "$(gpu_count)" ]; then
  echo "GPU-GATE: device_count 0 — disabling the CUDA forward-compat layer"
  for d in /usr/local/cuda/compat /usr/local/cuda-*/compat /usr/local/nvidia/compat; do
    [ -d "$d" ] && mv "$d" "$d.cf373-disabled" && echo "GPU-GATE: moved $d aside"
  done
  ldconfig 2>/dev/null || true
fi
n=$(gpu_count)
echo "GPU-GATE: device_count $n"
[ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null || exit 17
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
# `run_arm_lalign_k.sh` — the launcher four teacher-align cells run — sources
# the repo's shared checkpoint resolver and aborts without it. B2 and B4 got
# as far as `ABORT: no checkpoint resolver` before the tarball carried it.
test -f /root/cf/scripts/resolve_eval_checkpoint.sh || exit 16
# The driver and the card of every box the study keeps, in its own log.
nvidia-smi --query-gpu=driver_version,name,compute_mode --format=csv,noheader \
  | sed 's/^/BOX-ENV driver=/'
echo BOOTSTRAP_OK
BOOT
rsh 'chmod +x /root/cf373_bootstrap.sh && setsid nohup bash /root/cf373_bootstrap.sh > /root/bootstrap.log 2>&1 < /dev/null & echo started'

say "waiting for BOOTSTRAP_OK (install is ~4 min)"
for _ in $(seq 1 80); do
  if rsh 'grep -q BOOTSTRAP_OK /root/bootstrap.log' 2>/dev/null; then
    say "OK"; rsh 'tail -n 3 /root/bootstrap.log'; exit 0
  fi
  if rsh 'grep -qE "^\+ exit (9|1[0-9])$" /root/bootstrap.log' 2>/dev/null; then
    say "FAILED"; rsh 'tail -n 25 /root/bootstrap.log'; exit 5
  fi
  sleep 15
done
say "TIMEOUT"; rsh 'tail -n 25 /root/bootstrap.log'; exit 5
