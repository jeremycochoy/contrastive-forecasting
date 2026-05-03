#!/bin/bash
# Crash-resilient sync loop for 2026-04-27_periodic-synth-mix runs on vast.ai.
# Mirrors the v3b sync strategy: atomic .tmp + size-check + mv, scp over
# ssh, first hour @ 5min cadence then 15min.
#
# Usage:
#   SSH_HOST=ssh5.vast.ai SSH_PORT=11930 \
#   LOCAL_DIR=sync_periodic_synth/ \
#   REMOTE_DIR=/workspace/app \
#   ./experiments/2026-04-27_periodic-synth-mix/scripts/sync_loop.sh
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

# (glob, min_bytes) pairs. min_bytes is chosen so a partial/aborted transfer
# is rejected, without also rejecting legitimately small files. Head
# checkpoints are ~2.4 MB, so they MUST NOT share the backbone's 70 MB floor
# — that was a bug in the first version of this script, which silently
# dropped every R1*_head file. See issue / PR for fix.
#
# Rule of thumb: min_bytes ≈ 50% of the expected final size, floored at 1 KB.
WANTED_PATTERNS=(
    # pattern                                     min_bytes
    # Backbone optimisers are consistently ~155 MB. Threshold at 130 MB
    # (~84%) so a partial transfer at 78 MB or 154 MB is rejected. The
    # earlier 70 MB threshold let through truncated files that then got
    # promoted over the previous good copy (rotation saved us, but we'd
    # rather not depend on .prev).
    "checkpoints/tiny_*_optimizer.pth             130000000"  # backbone optimiser ~155 MB
    "checkpoints/tiny_*.pth                       70000000"   # backbone model ~80 MB
    "checkpoints/tiny_*_losses.csv                32"
    "checkpoints/R1_*_optimizer.pth               1000000"    # head optimiser ~5 MB
    "checkpoints/R1_*.pth                         1000000"    # head model ~2.4 MB
    "checkpoints/R1_*_losses.csv                  32"
    # Multi-experiment naming variants from run_all_experiments.sh:
    # R1q_* (quantile heads), R1r_* (reconstruction heads). Same size
    # class as R1_*: ~2.5 MB head, ~5 MB optimizer.
    "checkpoints/R1q_*_optimizer.pth              1000000"
    "checkpoints/R1q_*.pth                        1000000"
    "checkpoints/R1q_*_losses.csv                 32"
    "checkpoints/R1r_*_optimizer.pth              1000000"
    "checkpoints/R1r_*.pth                        1000000"
    "checkpoints/R1r_*_losses.csv                 32"
    "checkpoints/R1v3c_*_optimizer.pth            1000000"    # legacy v3c head optimiser
    "checkpoints/R1v3c_*.pth                      1000000"    # legacy v3c head
    "checkpoints/R1v3c_*_losses.csv               32"
    # GIFT-Eval result CSVs land in results/<run_name>/ on the remote.
    # Tiny files (KBs); floor 32 B.
    "results/*/all_results.csv                    32"
    "results/*/summary.txt                        32"
)
LOG_PATTERNS=(
    "run_all.log"
)

# Atomic download with one-deep rotation.
#   1. scp remote → ${local_path}.tmp
#   2. verify non-empty AND >= min_bytes
#   3. if a current ${local_path} exists, rotate it to ${local_path}.prev
#      (so we never overwrite the only good copy with a fresh transfer)
#   4. atomic mv ${local_path}.tmp → ${local_path}
# A failed transfer (truncated SCP / SSH drop / size below floor) leaves
# the previous ${local_path} untouched and removes the broken .tmp.
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
    # One-deep rotation: keep the previous good copy as a safety net.
    if [[ -s "$local_path" ]]; then
        mv -f "$local_path" "${local_path}.prev"
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

    # Checkpoints, loss CSVs, GIFT-Eval results — each pattern carries
    # its own min_bytes. Local destination preserves the remote *directory*
    # of each match (e.g. results/<run>/all_results.csv lands in
    # LOCAL_DIR/results/<run>/all_results.csv) so that nested run-specific
    # outputs don't collide on the basename.
    for entry in "${WANTED_PATTERNS[@]}"; do
        # Split "glob  min_bytes" on whitespace.
        local pat min
        pat=$(echo "$entry" | awk '{print $1}')
        min=$(echo "$entry" | awk '{print $2}')
        for remote in $(_list_remote "$pat"); do
            # Preserve the remote directory under LOCAL_DIR so nested
            # outputs (results/<run>/file) don't collide on basename.
            local_path="$LOCAL_DIR/$remote"
            mkdir -p "$(dirname "$local_path")"
            _fetch_atomic "$REMOTE_DIR/$remote" "$local_path" "$min" \
                && echo "    ✓ $remote ($(wc -c < "$local_path") bytes)" \
                || echo "    ✗ $remote (skip / failed, min=${min}B)"
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

tick=0
while true; do
    _sync_once
    tick=$((tick + 1))
    # Fixed 15-min cadence — simpler than ramped schedule, and a full tick
    # (many files over the vast.ai scp proxy) can itself take ~2-5 min.
    sleep 900
done
