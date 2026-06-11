#!/bin/bash
# Crash-resilient sync loop for the #339 stop-grad run on vast.ai. Adapted from
# experiments/2026-04-27_periodic-synth-mix/scripts/sync_loop.sh (atomic .tmp +
# per-class size floor + one-deep rotation, scp over ssh, fixed 15-min cadence).
#
# Usage: sync_loop_sgpos.sh <ssh_host> <ssh_port> <local_dir> [remote_dir]
set -u

SSH_HOST=${1:?ssh_host}
SSH_PORT=${2:?ssh_port}
LOCAL_DIR=${3:?local_dir}
REMOTE_DIR=${4:-/workspace/out}

mkdir -p "$LOCAL_DIR/runs" "$LOCAL_DIR/results"

# (glob, min_bytes) pairs — min_bytes ≈ 50% of expected size so a truncated
# transfer is rejected without rejecting legitimately small files.
#   backbone model ~67 MB, backbone optimizer ~132 MB,
#   q-head 2L ~14.4 MB (6L ~3x), q-head optimizer 2L ~28.9 MB,
#   CSVs / summaries are KB-class.
WANTED_PATTERNS=(
    "runs/bb_*sgpos*_optimizer.pth      90000000"
    "runs/bb_*sgpos*.pth                35000000"
    "runs/bb_*sgpos*_losses.csv         32"
    "runs/bb_*sgpos*_attn_amplitude.csv 32"
    "runs/qhead_*_optimizer.pth         14000000"
    "runs/qhead_*.pth                   7000000"
    "runs/qhead_*_losses.csv            32"
    "results/gift_eval_full_*/all_results.csv 32"
    "results/gift_eval_full_*/summary.txt     32"
    "results/run_*.log                  1"
)
LOG_PATTERNS=(
    "run_all.log"
)

_fetch_atomic() {
    local remote_path=$1 local_path=$2 min_bytes=$3
    scp -q -o StrictHostKeyChecking=no -P "$SSH_PORT" \
        "root@$SSH_HOST:$remote_path" "${local_path}.tmp" 2>/dev/null
    if [[ ! -s "${local_path}.tmp" ]]; then
        rm -f "${local_path}.tmp"; return 1
    fi
    local sz; sz=$(wc -c < "${local_path}.tmp")
    if (( sz < min_bytes )); then
        rm -f "${local_path}.tmp"; return 1
    fi
    if [[ -s "$local_path" ]]; then
        mv -f "$local_path" "${local_path}.prev"
    fi
    mv "${local_path}.tmp" "$local_path"
    return 0
}

_list_remote() {
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" \
        "cd $REMOTE_DIR && ls -1 $1 2>/dev/null" 2>/dev/null
}

_sync_once() {
    echo "[$(date -u +%H:%M:%S)] sync tick"
    for pat in "${LOG_PATTERNS[@]}"; do
        for remote in $(_list_remote "$pat"); do
            _fetch_atomic "$REMOTE_DIR/$remote" "$LOCAL_DIR/$(basename "$remote")" 1 \
                && echo "    ✓ $remote" || echo "    ✗ $remote"
        done
    done
    for entry in "${WANTED_PATTERNS[@]}"; do
        local pat min
        pat=$(echo "$entry" | awk '{print $1}')
        min=$(echo "$entry" | awk '{print $2}')
        for remote in $(_list_remote "$pat"); do
            # A *.pth glob also matches *_optimizer.pth — skip those here so
            # each file is fetched once, under its own size class.
            [[ "$pat" != *optimizer* && "$remote" == *_optimizer.pth ]] && continue
            local_path="$LOCAL_DIR/$remote"
            mkdir -p "$(dirname "$local_path")"
            _fetch_atomic "$REMOTE_DIR/$remote" "$local_path" "$min" \
                && echo "    ✓ $remote ($(wc -c < "$local_path") bytes)" \
                || echo "    ✗ $remote (skip / failed, min=${min}B)"
        done
    done
    local log="$LOCAL_DIR/run_all.log"
    if [[ -s "$log" ]]; then
        if tail -200 "$log" | grep -qE "NaN/Inf DETECTED|Traceback|FAILED|ALL STAGES COMPLETE"; then
            echo "    ⚠️  Signal in run_all.log tail:"
            tail -200 "$log" | grep -E "NaN/Inf|Traceback|FAILED|ALL STAGES COMPLETE" | head -5 | sed 's/^/      /'
        fi
    fi
}

while true; do
    _sync_once
    sleep 900
done
