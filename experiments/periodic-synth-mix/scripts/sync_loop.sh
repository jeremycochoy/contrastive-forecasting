#!/bin/bash
# Crash-resilient sync loop for periodic-synth-mix runs on vast.ai.
# Mirrors the v3b sync strategy: atomic .tmp + size-check + mv, scp over
# ssh, first hour @ 5min cadence then 15min.
#
# Usage:
#   SSH_HOST=ssh5.vast.ai SSH_PORT=11930 \
#   LOCAL_DIR=sync_periodic_synth/ \
#   REMOTE_DIR=/workspace/app \
#   ./experiments/periodic-synth-mix/scripts/sync_loop.sh
#
# Or pass directly:
#   ./sync_loop.sh <ssh_host> <ssh_port> <local_dir> <remote_dir>

set -u

SSH_HOST=${1:-${SSH_HOST:-}}
SSH_PORT=${2:-${SSH_PORT:-}}
LOCAL_DIR=${3:-${LOCAL_DIR:-sync_periodic_synth}}
REMOTE_DIR=${4:-${REMOTE_DIR:-/workspace/app}}

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo "usage: sync_loop.sh <ssh_host> <ssh_port> [local_dir] [remote_dir]"
    exit 1
fi

mkdir -p "$LOCAL_DIR/checkpoints"

# Unique files we want to pull. We glob each on the remote side.
WANTED_PATTERNS=(
    "checkpoints/tiny_v3c_*.pth"
    "checkpoints/tiny_v3c_*_losses.csv"
    "checkpoints/R1v3c_*.pth"
    "checkpoints/R1v3c_*_losses.csv"
)
LOG_PATTERNS=(
    "run_all.log"
)

# Atomic download: scp to .tmp, verify non-empty, atomic mv.
_fetch_atomic() {
    local remote_path=$1
    local local_path=$2
    local min_bytes=$3   # Below this size we assume partial/failed transfer.

    scp -q -o StrictHostKeyChecking=no -P "$SSH_PORT" \
        "root@$SSH_HOST:$remote_path" "${local_path}.tmp" 2>/dev/null
    if [[ ! -s "${local_path}.tmp" ]]; then
        rm -f "${local_path}.tmp"
        return 1
    fi
    local sz
    sz=$(wc -c < "${local_path}.tmp")
    if (( sz < min_bytes )); then
        rm -f "${local_path}.tmp"
        return 1
    fi
    mv "${local_path}.tmp" "$local_path"
    return 0
}

_list_remote() {
    local pattern=$1
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" \
        "cd $REMOTE_DIR && ls -1 $pattern 2>/dev/null" 2>/dev/null
}

_sync_once() {
    local ts
    ts=$(date -u +%H:%M:%S)
    echo "[$ts] sync tick"

    # Logs — small, always fetch fresh
    for pat in "${LOG_PATTERNS[@]}"; do
        for remote in $(_list_remote "$pat"); do
            local_path="$LOCAL_DIR/$(basename "$remote")"
            _fetch_atomic "$REMOTE_DIR/$remote" "$local_path" 1 \
                && echo "    ✓ $remote" \
                || echo "    ✗ $remote"
        done
    done

    # Checkpoints + loss CSVs
    for pat in "${WANTED_PATTERNS[@]}"; do
        for remote in $(_list_remote "$pat"); do
            local_path="$LOCAL_DIR/checkpoints/$(basename "$remote")"
            # Heuristic: .pth must be >70 MB (20M param backbone, fp32)
            #            .csv needs at least header + one row
            local min=1024
            if [[ "$remote" == *.pth ]]; then
                # optimizer files can be smaller (~80MB) and model files ~80MB+
                if [[ "$remote" == *_optimizer.pth ]]; then
                    min=70000000
                else
                    min=70000000
                fi
            elif [[ "$remote" == *.csv ]]; then
                min=32
            fi
            _fetch_atomic "$REMOTE_DIR/$remote" "$local_path" "$min" \
                && echo "    ✓ $remote ($(wc -c < "$local_path") bytes)" \
                || echo "    ✗ $remote (skip / failed)"
        done
    done

    # NaN / failure sniffer on the log
    local log="$LOCAL_DIR/run_all.log"
    if [[ -s "$log" ]]; then
        if tail -200 "$log" | grep -qE "NaN/Inf DETECTED|Traceback|FAILED|ALL STAGES COMPLETE"; then
            echo "    ⚠️  Signal detected in run_all.log tail:"
            tail -200 "$log" | grep -E "NaN/Inf|Traceback|FAILED|ALL STAGES COMPLETE" | head -5 | sed 's/^/      /'
        fi
    fi
}

start_s=$(date +%s)
tick=0
while true; do
    _sync_once
    now=$(date +%s)
    elapsed=$(( now - start_s ))
    if (( elapsed < 3600 )); then
        interval=300   # 5 min for first hour
    else
        interval=900   # 15 min thereafter
    fi
    tick=$((tick + 1))
    sleep "$interval"
done
