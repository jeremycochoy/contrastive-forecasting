#!/bin/bash
# Atomic sync loop for one arm of exp_compositesynth_v4combined_2arm.
# Mirrors sync_compositesynth_v2pulse/sync_loop.sh but with v2b run-name
# pattern (tiny_compsyn_v4_<arm>) and v2b experiment dir.
# Usage: ./sync_loop.sh <SSH_HOST> <SSH_PORT> <INSTANCE_ID> <ARM>

set -u

SSH_HOST="${1:?missing SSH_HOST}"
SSH_PORT="${2:?missing SSH_PORT}"
INSTANCE_ID="${3:?missing INSTANCE_ID}"
ARM="${4:?missing ARM}"

REMOTE="root@${SSH_HOST}"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_RESULTS_DIR="/workspace/app/experiments/exp_compositesynth_v4combined_2arm/results"
REMOTE_LOG="/workspace/app/run_${ARM}.log"
LOCAL_BASE="/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_compositesynth_v4combined/${ARM}"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOCAL_RESULTS="${LOCAL_BASE}/results"
LOG_FILE="${LOCAL_BASE}/sync.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

mkdir -p "${LOCAL_CKPT}" "${LOCAL_RESULTS}"

echo "[$(date)] Sync loop start arm=${ARM} v2b instance=${INSTANCE_ID} @ ${SSH_HOST}:${SSH_PORT}" | tee -a "${LOG_FILE}"

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

BB_MIN=70000000
BB_OPT_MIN=120000000
HEAD_MIN=1000000
HEAD_OPT_MIN=1000000
CSV_MIN=1
LOG_MIN=1

BB="tiny_compsyn_v4_${ARM}"
HEAD="R1q_compsyn_v4_${ARM}"

cycle_count=0
while true; do
    cycle_count=$((cycle_count + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date)] === arm=${ARM} v2b cycle #${cycle_count} ===" | tee -a "${LOG_FILE}"

    for kind in best_loss best_gap FINAL; do
        atomic_scp "${REMOTE_CKPT_DIR}/${BB}_${kind}.pth" \
                   "${LOCAL_CKPT}/${BB}_${kind}.pth" "${BB_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${BB}_${kind}_optimizer.pth" \
                   "${LOCAL_CKPT}/${BB}_${kind}_optimizer.pth" "${BB_OPT_MIN}" || true
    done

    for kind in best FINAL; do
        atomic_scp "${REMOTE_CKPT_DIR}/${HEAD}_${kind}.pth" \
                   "${LOCAL_CKPT}/${HEAD}_${kind}.pth" "${HEAD_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${HEAD}_${kind}_optimizer.pth" \
                   "${LOCAL_CKPT}/${HEAD}_${kind}_optimizer.pth" "${HEAD_OPT_MIN}" || true
    done

    for rp in $(list_remote "${REMOTE_CKPT_DIR}/${BB}_*k.pth"); do
        fname=$(basename "${rp}")
        if [ ! -f "${LOCAL_CKPT}/${fname}" ]; then
            atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" "${BB_MIN}" || true
            opt_remote="${rp%.pth}_optimizer.pth"
            opt_fname="${fname%.pth}_optimizer.pth"
            atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" "${BB_OPT_MIN}" || true
        fi
    done

    atomic_scp "${REMOTE_CKPT_DIR}/${BB}_losses.csv" \
               "${LOCAL_CKPT}/${BB}_losses.csv" "${CSV_MIN}" || true
    atomic_scp "${REMOTE_CKPT_DIR}/${HEAD}_losses.csv" \
               "${LOCAL_CKPT}/${HEAD}_losses.csv" "${CSV_MIN}" || true

    atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true

    atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${ARM}/all_results.csv" \
               "${LOCAL_RESULTS}/all_results.csv" "${CSV_MIN}" || true
    atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${ARM}/summary.txt" \
               "${LOCAL_RESULTS}/summary.txt" "${CSV_MIN}" || true

    if grep -qiE "nan detected|Traceback|Error:" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  !!! NaN/Error/Traceback in run.log !!!" | tee -a "${LOG_FILE}"
    fi

    ALIVE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" \
        'pgrep -af "python3.*train.py|train_forecasting_head|eval_gift_eval" >/dev/null && echo ALIVE || echo DEAD' \
        2>/dev/null | tail -1)
    echo "  remote training process: ${ALIVE:-UNREACHABLE}" | tee -a "${LOG_FILE}"

    if grep -q "ARM ${ARM} ALL DONE" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  === ARM ${ARM} v2b ALL DONE detected; final sync ===" | tee -a "${LOG_FILE}"
        atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${ARM}/all_results.csv" \
                   "${LOCAL_RESULTS}/all_results.csv" "${CSV_MIN}" || true
        atomic_scp "${REMOTE_RESULTS_DIR}/gift_eval_${ARM}/summary.txt" \
                   "${LOCAL_RESULTS}/summary.txt" "${CSV_MIN}" || true
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true
        echo "[$(date)] Sync loop arm=${ARM} v2b finished" | tee -a "${LOG_FILE}"
        exit 0
    fi

    SLEEP=900
    echo "  next sync in ${SLEEP}s" | tee -a "${LOG_FILE}"
    sleep "${SLEEP}"
done
