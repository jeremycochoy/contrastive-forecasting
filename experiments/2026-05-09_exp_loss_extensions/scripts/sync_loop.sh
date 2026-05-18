#!/usr/bin/env bash
# Sync loop: pulls training artifacts from elisa every 15 min.
# Run on the laptop, keep alive for the full training duration.
# Usage: bash sync_loop.sh
set -euo pipefail

REMOTE=jupyter@elisa
REMOTE_DIR=workspaces/contrastive-forecasting/experiments/2026-05-09_exp_loss_extensions/results/loss_ext_square_tau_0_10
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/loss_ext_square_tau_0_10"
RUN_NAME=loss_ext_square_tau_0_10
INTERVAL=900  # 15 min

mkdir -p "$LOCAL_DIR"

pull_file() {
    local remote_path="$1"
    local local_path="$2"
    local min_bytes="${3:-0}"
    local tmp="${local_path}.tmp"

    if ssh "$REMOTE" "test -f ${remote_path}"; then
        scp "${REMOTE}:${remote_path}" "$tmp" 2>/dev/null || { echo "⚠️  scp failed: $remote_path"; return 1; }
        local sz
        sz=$(stat -f%z "$tmp" 2>/dev/null || stat -c%s "$tmp")
        if [[ "$sz" -ge "$min_bytes" ]]; then
            [[ -f "$local_path" ]] && mv "$local_path" "${local_path}.prev"
            mv "$tmp" "$local_path"
            echo "✓ $(basename "$local_path") (${sz} B)"
        else
            echo "⚠️  $(basename "$local_path") too small (${sz} < ${min_bytes}), keeping prior"
            rm -f "$tmp"
        fi
    else
        echo "  (not yet) $remote_path"
    fi
}

tick() {
    echo "=== sync tick $(date '+%H:%M:%S') ==="

    # Losses CSV (a few KB)
    pull_file "${REMOTE_DIR}/${RUN_NAME}_losses.csv" "${LOCAL_DIR}/${RUN_NAME}_losses.csv" 10

    # Training log
    pull_file "${REMOTE_DIR}/run_${RUN_NAME}.log" "${LOCAL_DIR}/run_${RUN_NAME}.log" 10

    # Periodic checkpoints and best checkpoints (backbone ~80MB, optimizer ~150MB)
    for suffix in _best_gap _best_loss; do
        pull_file "${REMOTE_DIR}/${RUN_NAME}${suffix}.pth"           "${LOCAL_DIR}/${RUN_NAME}${suffix}.pth"           81920000
        pull_file "${REMOTE_DIR}/${RUN_NAME}${suffix}_optimizer.pth" "${LOCAL_DIR}/${RUN_NAME}${suffix}_optimizer.pth" 150000000
    done

    # Scan for periodic Nk checkpoints
    for f in $(ssh "$REMOTE" "ls ${REMOTE_DIR}/${RUN_NAME}_*k.pth 2>/dev/null" || true); do
        base=$(basename "$f")
        pull_file "${REMOTE_DIR}/${base}" "${LOCAL_DIR}/${base}" 81920000
        opt="${f%%.pth}_optimizer.pth"
        pull_file "${REMOTE_DIR}/$(basename "$opt")" "${LOCAL_DIR}/$(basename "$opt")" 150000000
    done

    # Check for NaN or completion
    if [[ -f "${LOCAL_DIR}/run_${RUN_NAME}.log" ]]; then
        if grep -q "NaN\|nan loss\|diverge" "${LOCAL_DIR}/run_${RUN_NAME}.log" 2>/dev/null; then
            echo "⚠️  NaN or divergence detected in log!"
        fi
        if grep -q "Training complete\|Finished training" "${LOCAL_DIR}/run_${RUN_NAME}.log" 2>/dev/null; then
            echo "✓ Training complete — final sync done."
        fi
    fi
}

echo "Sync loop started. REMOTE=${REMOTE}:${REMOTE_DIR}"
echo "LOCAL=${LOCAL_DIR}"
echo "Interval: ${INTERVAL}s. Ctrl-C to stop."

while true; do
    tick
    sleep "$INTERVAL"
done
