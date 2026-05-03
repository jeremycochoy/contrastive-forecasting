#!/bin/bash
# Atomic sync loop for exp_compositesynth_2arm (Vast.ai).
# Usage: ./sync_loop.sh <SSH_HOST> <SSH_PORT> <INSTANCE_ID>
# Cadence: 15-minute fixed (per CLAUDE.md).

set -u

SSH_HOST="${1:?missing SSH_HOST}"
SSH_PORT="${2:?missing SSH_PORT}"
INSTANCE_ID="${3:?missing INSTANCE_ID}"

REMOTE="root@${SSH_HOST}"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_RESULTS_DIR="/workspace/app/experiments/exp_compositesynth_2arm/results"
REMOTE_LOG="/workspace/app/run_all.log"
LOCAL_BASE="/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_compositesynth"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOCAL_RESULTS="${LOCAL_BASE}/results"
LOG_FILE="${LOCAL_BASE}/sync.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

mkdir -p "${LOCAL_CKPT}" "${LOCAL_RESULTS}"

echo "[$(date)] Sync loop start (instance ${INSTANCE_ID} @ ${SSH_HOST}:${SSH_PORT})" | tee -a "${LOG_FILE}"

# atomic_scp <remote_path> <local_dest> <min_size_bytes>
# scp to .tmp, verify size >= min_size, rotate existing → .prev, atomic mv.
atomic_scp() {
    local remote_path="$1"
    local local_dest="$2"
    local min_size="${3:-0}"
    local tmp="${local_dest}.tmp"

    if scp ${SSH_OPTS} -P "${SSH_PORT}" -q "${REMOTE}:${remote_path}" "${tmp}" 2>/dev/null; then
        if [ -s "${tmp}" ]; then
            local sz
            sz=$(stat -f%z "${tmp}" 2>/dev/null || stat -c%s "${tmp}" 2>/dev/null)
            if [ "${sz}" -ge "${min_size}" ]; then
                # Append-regression guard for *.csv / *.log destinations.
                # See docs/SYNC_PROTOCOL_REVIEW.md §2.1 / §3.2 — a fresh-instance
                # remote can replace a long historical CSV/log with a short one;
                # rotating the long local good copy to .prev and then overwriting
                # .prev on the next cycle destroys it. Refuse a shrinking pull and
                # divert it to <LOCAL_BASE>/archive/ for forensics.
                local _bn
                _bn="$(basename "${local_dest}")"
                case "${_bn}" in
                    *.csv|*.log)
                        if [ -s "${local_dest}" ]; then
                            local _new_lines _old_lines
                            _new_lines=$(wc -l < "${tmp}" 2>/dev/null | tr -d ' ')
                            _old_lines=$(wc -l < "${local_dest}" 2>/dev/null | tr -d ' ')
                            : "${_new_lines:=0}"
                            : "${_old_lines:=0}"
                            if [ "${_new_lines}" -lt "${_old_lines}" ]; then
                                local _archive_dir _stamp _archived
                                _archive_dir="${LOCAL_BASE}/archive"
                                mkdir -p "${_archive_dir}"
                                _stamp=$(date +%Y%m%dT%H%M%S)
                                _archived="${_archive_dir}/${_stamp}_${_bn}.regression"
                                mv "${tmp}" "${_archived}"
                                echo "  ⚠️ APPEND REGRESSION on ${_bn}: remote=${_new_lines} lines, local=${_old_lines} lines — archived to ${_archived}, NOT rotating." | tee -a "${LOG_FILE}"
                                return 0
                            fi
                        fi
                        ;;
                esac
                if [ -s "${local_dest}" ]; then
                    mv -f "${local_dest}" "${local_dest}.prev"
                fi
                mv "${tmp}" "${local_dest}"
                echo "  ✓ $(basename "${remote_path}") (${sz} bytes)" | tee -a "${LOG_FILE}"
                return 0
            else
                echo "  ✗ TOO SMALL $(basename "${remote_path}") (${sz} < ${min_size})" | tee -a "${LOG_FILE}"
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

list_remote() {
    ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" "ls -1 $1 2>/dev/null" 2>/dev/null
}

# Per-file-class thresholds (CLAUDE.md):
BB_MIN=70000000          # backbone ~80MB
BB_OPT_MIN=120000000     # bb optimizer ~130MB observed; threshold 95% × 130MB
HEAD_MIN=1000000         # head ~2.4MB; loose floor
HEAD_OPT_MIN=1000000     # head optimizer; loose floor
CSV_MIN=1                # losses CSV: any non-empty
LOG_MIN=1                # log: any non-empty

BB_NAMES=(tiny_compsyn_revin tiny_compsyn_ewma128)
HEAD_NAMES=(R1q_compsyn_revin R1q_compsyn_ewma128)

cycle_count=0
while true; do
    cycle_count=$((cycle_count + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date)] === Sync cycle #${cycle_count} ===" | tee -a "${LOG_FILE}"

    # 1. Best backbone + optimizer pairs (best_loss, best_gap, FINAL)
    for kind in best_loss best_gap FINAL; do
        for prefix in "${BB_NAMES[@]}"; do
            atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_${kind}.pth" \
                       "${LOCAL_CKPT}/${prefix}_${kind}.pth" "${BB_MIN}" || true
            atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_${kind}_optimizer.pth" \
                       "${LOCAL_CKPT}/${prefix}_${kind}_optimizer.pth" "${BB_OPT_MIN}" || true
        done
    done

    # 2. Best head + optimizer (best, FINAL)
    for kind in best FINAL; do
        for prefix in "${HEAD_NAMES[@]}"; do
            atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_${kind}.pth" \
                       "${LOCAL_CKPT}/${prefix}_${kind}.pth" "${HEAD_MIN}" || true
            atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_${kind}_optimizer.pth" \
                       "${LOCAL_CKPT}/${prefix}_${kind}_optimizer.pth" "${HEAD_OPT_MIN}" || true
        done
    done

    # 3. Periodic backbone saves (Nk.pth) discovered dynamically
    for prefix in "${BB_NAMES[@]}"; do
        for rp in $(list_remote "${REMOTE_CKPT_DIR}/${prefix}_*k.pth"); do
            fname=$(basename "${rp}")
            if [ ! -f "${LOCAL_CKPT}/${fname}" ]; then
                atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" "${BB_MIN}" || true
                opt_remote="${rp%.pth}_optimizer.pth"
                opt_fname="${fname%.pth}_optimizer.pth"
                atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" "${BB_OPT_MIN}" || true
            fi
        done
    done

    # 4. Loss CSVs (always re-pull; they grow)
    for prefix in "${BB_NAMES[@]}" "${HEAD_NAMES[@]}"; do
        atomic_scp "${REMOTE_CKPT_DIR}/${prefix}_losses.csv" \
                   "${LOCAL_CKPT}/${prefix}_losses.csv" "${CSV_MIN}" || true
    done

    # 5. Training log (append-only on remote; mirror latest)
    atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run_all.log" "${LOG_MIN}" || true

    # 6. Eval results (per-arm directories)
    for arm in revin ewma128; do
        atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${arm}/all_results.csv" \
                   "${LOCAL_RESULTS}/gift_eval_${arm}_all_results.csv" "${CSV_MIN}" || true
        atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${arm}/summary.txt" \
                   "${LOCAL_RESULTS}/gift_eval_${arm}_summary.txt" "${CSV_MIN}" || true
    done

    # 7. Health checks
    if grep -qiE "nan detected|Traceback|Error:" "${LOCAL_BASE}/run_all.log" 2>/dev/null; then
        echo "  !!! NaN/Error/Traceback detected in run_all.log !!!" | tee -a "${LOG_FILE}"
    fi

    ALIVE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" \
        'pgrep -af "python3.*train.py|train_forecasting_head.py|eval_gift_eval" >/dev/null && echo ALIVE || echo DEAD' \
        2>/dev/null | tail -1)
    echo "  remote training process: ${ALIVE:-UNREACHABLE}" | tee -a "${LOG_FILE}"

    LATEST=$(tail -5 "${LOCAL_BASE}/run_all.log" 2>/dev/null | grep -E "^\[|^===" | tail -1)
    echo "  latest: ${LATEST}" | tee -a "${LOG_FILE}"

    # Completion: arm B's eval is the last stage
    if grep -q "run_compositesynth_2arm: ALL DONE" "${LOCAL_BASE}/run_all.log" 2>/dev/null; then
        echo "  === ALL DONE detected; final sync ===" | tee -a "${LOG_FILE}"
        # Final pass for results
        for arm in revin ewma128; do
            atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${arm}/all_results.csv" \
                       "${LOCAL_RESULTS}/gift_eval_${arm}_all_results.csv" "${CSV_MIN}" || true
            atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${arm}/summary.txt" \
                       "${LOCAL_RESULTS}/gift_eval_${arm}_summary.txt" "${CSV_MIN}" || true
        done
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run_all.log" "${LOG_MIN}" || true
        echo "[$(date)] Sync loop finished (completion)" | tee -a "${LOG_FILE}"
        exit 0
    fi

    # CLAUDE.md mandates 15-minute fixed sync cadence.
    SLEEP=900
    echo "  next sync in ${SLEEP}s" | tee -a "${LOG_FILE}"
    sleep "${SLEEP}"
done
