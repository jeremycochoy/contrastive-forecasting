#!/bin/bash
# #393 — put a bare vast.ai box in a state where `ladder.py` runs.
#
# Usage (from the checkout, on the machine that owns it):
#   bash scripts/bootstrap_remote.sh <ssh_host> <ssh_port> [cell,cell,...]
#
# The vast.ai images the kit provisions are bare CUDA runtimes: no torch, no
# repo, no GIFT-Eval. Everything the ladder needs is pushed from here rather
# than pulled from the network, for two reasons. The repo is private, so a
# clone would need a credential on the box; and the GIFT-Eval data and the
# seasonal-naive denominator have to be BYTE-IDENTICAL to elisa's or the
# GM-Relative MASE values from the two machines are not on one scale.
#
# Versions are pinned to elisa's, so a run's numbers do not depend on which
# machine produced them. torch 2.8.0+cu128 is also the floor for Blackwell
# (RTX 5090, sm_120): an older wheel has no kernels for it and fails at the
# first `.to(device)`.
#
# One thing this cannot fix, and the caller has to plan around: vast.ai
# boxes come up with the GPU in `Exclusive_Process` compute mode and the
# container is not privileged enough to change it (`nvidia-smi -c 0` →
# "Insufficient Permissions"). Only ONE CUDA context can exist at a time, so
# cells run one after another there — see queue_remote.sh — where on elisa
# two share a GPU.
set -uo pipefail

HOST="${1:?usage: bootstrap_remote.sh <ssh_host> <ssh_port> [cells]}"
PORT="${2:?usage: bootstrap_remote.sh <ssh_host> <ssh_port> [cells]}"
CELLS="${3:-}"

WT="${WT:?WT must be the absolute path of the local checkout}"
GIFT_SRC="${GIFT_SRC:-$HOME/workspaces/gift-eval}"
GIFT_DATA="${GIFT_DATA:-$HOME/workspaces/gift-eval-data}"
STAGE="${STAGE:-/tmp/cf393_stage}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [bootstrap $HOST:$PORT] $*"; }
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

for p in "$GIFT_SRC/results/seasonal_naive/all_results.csv" "$GIFT_DATA" \
         "$WT/experiments/hf_token.txt"; do
  [ -e "$p" ] || { say "ABORT: missing $p"; exit 2; }
done

mkdir -p "$STAGE"
say "packing code"
tar czf "$STAGE/cf393_code.tgz" -C "$WT" --exclude='__pycache__' \
  src \
  experiments/hf_token.txt \
  experiments/2026-04-27_freq-embedding/scripts/train.py \
  experiments/2026-04-13_gift-eval/scripts \
  experiments/2026-08-04_ema_sched_ladder/scripts \
  experiments/2026-08-04_ema_sched_ladder/sync \
  experiments/2026-08-04_ema_sched_ladder/README.md || exit 3
# The GIFT-Eval package carries the seasonal-naive denominator the whole
# study is normalised by; it ships with the code, not with the data.
[ -f "$STAGE/gift_eval_pkg.tgz" ] || \
  tar czf "$STAGE/gift_eval_pkg.tgz" -C "$(dirname "$GIFT_SRC")" \
      --exclude='__pycache__' --exclude='.git' "$(basename "$GIFT_SRC")" || exit 3
[ -f "$STAGE/gift_eval_data.tgz" ] || \
  tar czf "$STAGE/gift_eval_data.tgz" -C "$(dirname "$GIFT_DATA")" \
      "$(basename "$GIFT_DATA")" || exit 3

say "uploading $(du -ch "$STAGE"/*.tgz | tail -1 | cut -f1)"
scp "${SSH_OPTS[@]}" -P "$PORT" "$STAGE"/cf393_code.tgz \
    "$STAGE"/gift_eval_pkg.tgz "$STAGE"/gift_eval_data.tgz \
    "root@$HOST:/root/" || exit 4

say "installing"
rsh 'cat > /root/cf393_bootstrap.sh' <<'BOOT'
#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv git curl || exit 9
PIP="python3 -m pip install --break-system-packages"
# cu128 / torch 2.8.0: elisa's version, and the Blackwell (sm_120) floor.
$PIP -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || exit 10
$PIP -q "numpy==1.26.4" "pandas==2.2.2" "gluonts==0.15.1" "datasets==2.17.1" \
        "scipy==1.11.4" "statsmodels==0.14.2" pyarrow einops orjson toolz \
        matplotlib huggingface_hub || exit 11
mkdir -p /root/cf /root/workspaces /root/cf393_runs
tar xzf /root/cf393_code.tgz      -C /root/cf         || exit 12
tar xzf /root/gift_eval_pkg.tgz   -C /root/workspaces || exit 13
tar xzf /root/gift_eval_data.tgz  -C /root/workspaces || exit 14
$PIP -q -e /root/workspaces/gift-eval || exit 15
rm -f /root/gift_eval_data.tgz
python3 - <<'PY' || exit 16
import torch, gluonts, gift_eval, pandas
print("torch", torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0))
print("gift_eval", gift_eval.__file__)
PY
nvidia-smi --query-gpu=compute_mode --format=csv,noheader
echo BOOTSTRAP_OK
BOOT
rsh 'chmod +x /root/cf393_bootstrap.sh && setsid nohup bash /root/cf393_bootstrap.sh > /root/bootstrap.log 2>&1 < /dev/null & echo started'

say "waiting for BOOTSTRAP_OK (install is ~5 min)"
for _ in $(seq 1 120); do
  if rsh 'grep -q BOOTSTRAP_OK /root/bootstrap.log' 2>/dev/null; then
    say "OK"; rsh 'tail -n 4 /root/bootstrap.log'; break
  fi
  if rsh 'grep -qE "^\+ exit (9|1[0-6])$" /root/bootstrap.log' 2>/dev/null; then
    say "FAILED"; rsh 'tail -n 20 /root/bootstrap.log'; exit 5
  fi
  sleep 15
done
rsh 'grep -q BOOTSTRAP_OK /root/bootstrap.log' || { say "TIMEOUT"; exit 5; }

# The code tarball carries scripts/ and sync/, not results/, so the session
# ceiling has to be pushed separately. Without it a fresh box climbs past
# bb100k while another cell has not reached bb40k.
HOLD="$WT/experiments/2026-08-04_ema_sched_ladder/results/HOLD_ABOVE"
if [ -f "$HOLD" ]; then
  say "pushing HOLD_ABOVE ($(tr -d '[:space:]' <"$HOLD"))"
  rsh 'mkdir -p /root/cf/experiments/2026-08-04_ema_sched_ladder/results'
  scp "${SSH_OPTS[@]}" -P "$PORT" "$HOLD" \
      "root@$HOST:/root/cf/experiments/2026-08-04_ema_sched_ladder/results/" || exit 6
fi

if [ -n "$CELLS" ]; then
  say "queueing cells: $CELLS"
  scp "${SSH_OPTS[@]}" -P "$PORT" \
      "$WT/experiments/2026-08-04_ema_sched_ladder/scripts/queue_remote.sh" \
      "root@$HOST:/root/" || exit 6
  rsh "chmod +x /root/queue_remote.sh && setsid nohup bash /root/queue_remote.sh \
       $(echo "$CELLS" | tr ',' ' ') > /root/queue.log 2>&1 < /dev/null & echo queued"
fi
say "done"
