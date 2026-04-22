#!/bin/bash
# Atomic sync loop for v3b-final training on Vast.ai.
# Usage: ./sync_loop.sh <SSH_HOST> <SSH_PORT> <INSTANCE_ID>
# Syncs checkpoints (model+optimizer), CSV, log, and results at 5-min cadence
# for first hour, then 15-min.

set -u

SSH_HOST="${1:?missing SSH_HOST}"
SSH_PORT="${2:?missing SSH_PORT}"
INSTANCE_ID="${3:?missing INSTANCE_ID}"

REMOTE="root@${SSH_HOST}"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_RESULTS_DIR="/workspace/app/results"
REMOTE_LOG="/workspace/app/run_all.log"
LOCAL_BASE="/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_v3b_final"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOCAL_RESULTS="${LOCAL_BASE}/results"
LOG_FILE="${LOCAL_BASE}/sync.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

mkdir -p "${LOCAL_CKPT}" "${LOCAL_RESULTS}"

echo "[$(date)] Sync loop start (instance ${INSTANCE_ID} @ ${SSH_HOST}:${SSH_PORT})" | tee -a "${LOG_FILE}"

# atomic_scp <remote_path> <local_dest> <min_size_bytes>
# scp to .tmp, verify non-empty (>min_size), then mv over dest.
atomic_scp() {
    local remote_path="$1"
    local local_dest="$2"
    local min_size="${3:-0}"
    local tmp="${local_dest}.tmp"

    if scp ${SSH_OPTS} -P "${SSH_PORT}" -q "${REMOTE}:${remote_path}" "${tmp}" 2>/dev/null; then
        if [ -s "${tmp}" ]; then
            local sz=$(stat -f%z "${tmp}" 2>/dev/null || stat -c%s "${tmp}" 2>/dev/null)
            if [ "${sz}" -ge "${min_size}" ]; then
                mv "${tmp}" "${local_dest}"
                echo "  synced $(basename "${remote_path}") (${sz} bytes)" | tee -a "${LOG_FILE}"
                return 0
            else
                echo "  TOO SMALL $(basename "${remote_path}") (${sz} < ${min_size})" | tee -a "${LOG_FILE}"
                rm -f "${tmp}"
                return 1
            fi
        else
            rm -f "${tmp}"
            return 1
        fi
    fi
    rm -f "${tmp}"
    return 1
}

# list_remote <glob>
list_remote() {
    ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" "ls -1 $1 2>/dev/null" 2>/dev/null
}

cycle_count=0
while true; do
    cycle_count=$((cycle_count + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date)] === Sync cycle #${cycle_count} ===" | tee -a "${LOG_FILE}"

    # 1. Best model + optimizer pairs (size > 70MB for model, > 140MB for optimizer)
    for kind in best_gap best_loss; do
        for prefix in tiny_v3b tiny_v3b_r2 tiny_v3b_r3 tiny_v3b_r4 R1v3b; do
            atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_${kind}.pth" "${LOCAL_CKPT}/${prefix}_${kind}.pth" 70000000 || true
            atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_${kind}_optimizer.pth" "${LOCAL_CKPT}/${prefix}_${kind}_optimizer.pth" 140000000 || true
        done
    done
    # Also R1v3b head's default "best" suffix
    atomic_scp "${REMOTE_CKPT_DIR}/R1v3b_best.pth" "${LOCAL_CKPT}/R1v3b_best.pth" 1000000 || true

    # 2. Periodic saves (10k, 20k, ..., 500k) discovered dynamically
    PERIODIC=$(list_remote "${REMOTE_CKPT_DIR}/tiny_v3b*k.pth")
    for rp in ${PERIODIC}; do
        fname=$(basename "${rp}")
        if [ ! -f "${LOCAL_CKPT}/${fname}" ] || [ "${rp}" = "${REMOTE_CKPT_DIR}/tiny_v3b_r3_40k.pth" ]; then
            # Sync if missing locally. Skip the uploaded resume checkpoint.
            if [ "${fname}" != "tiny_v3b_r3_40k.pth" ]; then
                atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" 70000000 || true
                opt_remote="${rp%.pth}_optimizer.pth"
                opt_fname="${fname%.pth}_optimizer.pth"
                atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" 140000000 || true
            fi
        fi
    done

    # 3. Loss CSVs and head CSV
    for csv in tiny_v3b_losses.csv tiny_v3b_r2_losses.csv tiny_v3b_r3_losses.csv tiny_v3b_r4_losses.csv R1v3b_losses.csv; do
        atomic_scp "${REMOTE_CKPT_DIR}/${csv}" "${LOCAL_CKPT}/${csv}" 1 || true
    done

    # 4. Training log (append-only on remote; mirror latest)
    atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run_all.log" 1 || true

    # 5. Final results (only exist after Stage 3)
    atomic_scp "${REMOTE_RESULTS_DIR}/R1v3b/all_results.csv" "${LOCAL_RESULTS}/R1v3b_all_results.csv" 1 || true
    atomic_scp "${REMOTE_RESULTS_DIR}/R1v3b/summary.txt" "${LOCAL_RESULTS}/R1v3b_summary.txt" 1 || true

    # 6. Health checks
    if grep -qi "nan detected\|Traceback\|Error:" "${LOCAL_BASE}/run_all.log" 2>/dev/null; then
        echo "  !!! NaN/Error/Traceback detected in run_all.log !!!" | tee -a "${LOG_FILE}"
    fi

    ALIVE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" 'pgrep -af "python3.*train.py\|train_forecasting_head.py\|eval_gift_eval" >/dev/null && echo ALIVE || echo DEAD' 2>/dev/null | tail -1)
    echo "  remote training process: ${ALIVE:-UNREACHABLE}" | tee -a "${LOG_FILE}"

    LATEST=$(tail -5 "${LOCAL_BASE}/run_all.log" 2>/dev/null | grep -E '^\[' | tail -1)
    echo "  latest: ${LATEST}" | tee -a "${LOG_FILE}"

    # Completion check
    if grep -q "ALL STAGES COMPLETE" "${LOCAL_BASE}/run_all.log" 2>/dev/null; then
        echo "  === ALL STAGES COMPLETE detected; final sync ===" | tee -a "${LOG_FILE}"
        # One last pass
        atomic_scp "${REMOTE_RESULTS_DIR}/R1v3b/all_results.csv" "${LOCAL_RESULTS}/R1v3b_all_results.csv" 1 || true
        atomic_scp "${REMOTE_RESULTS_DIR}/R1v3b/summary.txt" "${LOCAL_RESULTS}/R1v3b_summary.txt" 1 || true
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run_all.log" 1 || true
        echo "[$(date)] Sync loop finished (completion)" | tee -a "${LOG_FILE}"
        exit 0
    fi

    # Cadence: 5 min for first 12 cycles (≈1h), then 15 min
    if [ "${cycle_count}" -lt 12 ]; then
        SLEEP=300
    else
        SLEEP=900
    fi
    echo "  next sync in ${SLEEP}s" | tee -a "${LOG_FILE}"
    sleep "${SLEEP}"
done
