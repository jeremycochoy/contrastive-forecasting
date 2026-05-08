#!/bin/bash
# Atomic sync loop for the τ-sweep (5 arms × 50k steps, single nohup process).
# The launcher run_tau_sweep.sh trains all 5 arms sequentially with run names
# tau_sweep_0_03 ... tau_sweep_0_20 in /workspace/app/checkpoints. Output is
# captured to /workspace/app/run_tau_sweep.log. This loop pulls every 15 min
# to a local mirror under the main contrastive-forecasting checkout on elisa.
#
# Usage: ./sync_loop.sh <SSH_HOST> <SSH_PORT> <INSTANCE_ID>
#
# Notes
# - Local base lives in the MAIN checkout (CLAUDE.md rule 4 — never under a
#   worktree, since `git worktree remove --force` would destroy untracked .pth
#   bundles).
# - One shared remote log file for all 5 arms (sequential launcher).
# - Per-class size thresholds (CLAUDE.md). C=1, d_model=384 backbone is ~46 MB
#   (.pth) and ~90 MB (_optimizer.pth) at this arch.
# - Append-regression guard for CSV/log: a fresh-instance state can replace
#   the long historical file with a short one; refuse and divert.
# - Completion marker: launcher prints "=== τ sweep ALL DONE ===" on success.

set -u

SSH_HOST="${1:?missing SSH_HOST}"
SSH_PORT="${2:?missing SSH_PORT}"
INSTANCE_ID="${3:?missing INSTANCE_ID}"

REMOTE="root@${SSH_HOST}"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_LOG="/workspace/app/run_tau_sweep.log"
LOCAL_BASE="/home/jupyter/contrastive-forecasting/sync_tau_sweep"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOG_FILE="${LOCAL_BASE}/sync.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

mkdir -p "${LOCAL_CKPT}"

echo "[$(date)] Sync loop start tau_sweep instance=${INSTANCE_ID} @ ${SSH_HOST}:${SSH_PORT}" | tee -a "${LOG_FILE}"

atomic_scp() {
    local remote_path="$1"
    local local_dest="$2"
    local min_size="${3:-0}"
    local tmp="${local_dest}.tmp"

    if scp ${SSH_OPTS} -P "${SSH_PORT}" -q "${REMOTE}:${remote_path}" "${tmp}" 2>/dev/null; then
        if [ -s "${tmp}" ]; then
            local sz
            sz=$(stat -c%s "${tmp}" 2>/dev/null || stat -f%z "${tmp}" 2>/dev/null)
            if [ "${sz}" -ge "${min_size}" ]; then
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

# Per-class size thresholds. C=1, d_model=384, 6L backbone is ~46 MB; AdamW
# optimizer ~90 MB; CSV / log are KB-scale (must just be non-empty).
BB_MIN=40000000
BB_OPT_MIN=70000000
CSV_MIN=1
LOG_MIN=1

ARMS=(0_03 0_05 0_07 0_10 0_20)

cycle_count=0
while true; do
    cycle_count=$((cycle_count + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date)] === tau_sweep cycle #${cycle_count} ===" | tee -a "${LOG_FILE}"

    for ARM in "${ARMS[@]}"; do
        BB="tau_sweep_${ARM}"
        for kind in best_loss best_gap FINAL; do
            atomic_scp "${REMOTE_CKPT_DIR}/${BB}_${kind}.pth" \
                       "${LOCAL_CKPT}/${BB}_${kind}.pth" "${BB_MIN}" || true
            atomic_scp "${REMOTE_CKPT_DIR}/${BB}_${kind}_optimizer.pth" \
                       "${LOCAL_CKPT}/${BB}_${kind}_optimizer.pth" "${BB_OPT_MIN}" || true
        done
        # Periodic safety checkpoints (every 5k steps).
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
    done

    atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true

    if grep -qiE "nan detected|Traceback|Error:" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  !!! NaN/Error/Traceback in run.log !!!" | tee -a "${LOG_FILE}"
    fi

    ALIVE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" \
        'pgrep -af "python3.*train.py|run_tau_sweep" >/dev/null && echo ALIVE || echo DEAD' \
        2>/dev/null | tail -1)
    echo "  remote training process: ${ALIVE:-UNREACHABLE}" | tee -a "${LOG_FILE}"

    if grep -q "τ sweep ALL DONE" "${LOCAL_BASE}/run.log" 2>/dev/null \
       || grep -q "tau sweep ALL DONE" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  === τ sweep ALL DONE detected; final sync ===" | tee -a "${LOG_FILE}"
        for ARM in "${ARMS[@]}"; do
            BB="tau_sweep_${ARM}"
            atomic_scp "${REMOTE_CKPT_DIR}/${BB}_FINAL.pth" \
                       "${LOCAL_CKPT}/${BB}_FINAL.pth" "${BB_MIN}" || true
            atomic_scp "${REMOTE_CKPT_DIR}/${BB}_FINAL_optimizer.pth" \
                       "${LOCAL_CKPT}/${BB}_FINAL_optimizer.pth" "${BB_OPT_MIN}" || true
            atomic_scp "${REMOTE_CKPT_DIR}/${BB}_losses.csv" \
                       "${LOCAL_CKPT}/${BB}_losses.csv" "${CSV_MIN}" || true
        done
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true
        echo "[$(date)] Sync loop tau_sweep finished" | tee -a "${LOG_FILE}"
        exit 0
    fi

    SLEEP=900
    echo "  next sync in ${SLEEP}s" | tee -a "${LOG_FILE}"
    sleep "${SLEEP}"
done
