#!/bin/bash
# Atomic sync loop for the qhead-improvements Round 1 (E1 + E2) on backbone
# beta. Heads only — backbone is already local and never trained here.
# Usage: ./sync_loop.sh <SSH_HOST> <SSH_PORT> <INSTANCE_ID>

set -u

SSH_HOST="${1:?missing SSH_HOST}"
SSH_PORT="${2:?missing SSH_PORT}"
INSTANCE_ID="${3:?missing INSTANCE_ID}"

REMOTE="root@${SSH_HOST}"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_LOG="/workspace/app/run_qhead_beta_rd1.log"
LOCAL_BASE="/home/jupyter/contrastive-forecasting/sync_qhead_beta_rd1"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOCAL_LOG_DIR="${LOCAL_BASE}/logs"
LOG_FILE="${LOCAL_BASE}/sync.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

mkdir -p "${LOCAL_CKPT}" "${LOCAL_LOG_DIR}"

echo "[$(date)] Sync loop start qhead_beta_rd1 instance=${INSTANCE_ID} @ ${SSH_HOST}:${SSH_PORT}" | tee -a "${LOG_FILE}"

atomic_scp() {
    local remote_path="$1"
    local local_dest="$2"
    local min_size="${3:-0}"
    local tmp="${local_dest}.tmp"

    if scp ${SSH_OPTS} -P "${SSH_PORT}" -q "${REMOTE}:${remote_path}" "${tmp}" 2>/dev/null; then
        if [ -s "${tmp}" ]; then
            local sz
            sz=$(stat -c%s "${tmp}" 2>/dev/null)
            if [ "${sz}" -ge "${min_size}" ]; then
                local _bn
                _bn="$(basename "${local_dest}")"
                # CSV/log append-regression guard.
                case "${_bn}" in
                    *.csv|*.log)
                        if [ -s "${local_dest}" ]; then
                            local _new _old
                            _new=$(wc -l < "${tmp}" 2>/dev/null | tr -d ' ')
                            _old=$(wc -l < "${local_dest}" 2>/dev/null | tr -d ' ')
                            : "${_new:=0}"; : "${_old:=0}"
                            if [ "${_new}" -lt "${_old}" ]; then
                                local _stamp _archived _adir
                                _adir="${LOCAL_BASE}/archive"
                                mkdir -p "${_adir}"
                                _stamp=$(date +%Y%m%dT%H%M%S)
                                _archived="${_adir}/${_stamp}_${_bn}.regression"
                                mv "${tmp}" "${_archived}"
                                echo "  ⚠️ APPEND REGRESSION on ${_bn}: remote=${_new} local=${_old} — archived to ${_archived}, NOT rotating." | tee -a "${LOG_FILE}"
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

# Per-class size floors. Linear head ~220KB, GRU head ~2.5MB; optimizer ~2x.
HEAD_MIN_LIN=100000
HEAD_OPT_MIN_LIN=300000
HEAD_MIN_GRU=1500000
HEAD_OPT_MIN_GRU=3000000
CSV_MIN=1
LOG_MIN=1

E1="R1_E1_linear_quant_lr3e4"
E2="R1_E2_gru_quant_moirai_wsd"

cycle_count=0
while true; do
    cycle_count=$((cycle_count + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date)] === qhead_beta_rd1 cycle #${cycle_count} ===" | tee -a "${LOG_FILE}"

    # E1 (linear) — best, FINAL
    for kind in best FINAL; do
        atomic_scp "${REMOTE_CKPT_DIR}/${E1}_${kind}.pth" \
                   "${LOCAL_CKPT}/${E1}_${kind}.pth" "${HEAD_MIN_LIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${E1}_${kind}_optimizer.pth" \
                   "${LOCAL_CKPT}/${E1}_${kind}_optimizer.pth" "${HEAD_OPT_MIN_LIN}" || true
    done

    # E2 (gru) — best, FINAL, STABLE (24k branchable)
    for kind in best FINAL STABLE; do
        atomic_scp "${REMOTE_CKPT_DIR}/${E2}_${kind}.pth" \
                   "${LOCAL_CKPT}/${E2}_${kind}.pth" "${HEAD_MIN_GRU}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${E2}_${kind}_optimizer.pth" \
                   "${LOCAL_CKPT}/${E2}_${kind}_optimizer.pth" "${HEAD_OPT_MIN_GRU}" || true
    done

    # Periodic snapshots (e.g. *_24k.pth) — pick up anything new.
    for prefix in "${E1}" "${E2}"; do
        case "${prefix}" in
            *linear*) MIN=${HEAD_MIN_LIN}; OPT=${HEAD_OPT_MIN_LIN} ;;
            *)        MIN=${HEAD_MIN_GRU}; OPT=${HEAD_OPT_MIN_GRU} ;;
        esac
        for rp in $(list_remote "${REMOTE_CKPT_DIR}/${prefix}_*k.pth"); do
            fname=$(basename "${rp}")
            if [ ! -f "${LOCAL_CKPT}/${fname}" ]; then
                atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" "${MIN}" || true
                opt_remote="${rp%.pth}_optimizer.pth"
                opt_fname="${fname%.pth}_optimizer.pth"
                atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" "${OPT}" || true
            fi
        done
    done

    # Loss CSVs (always pull, append-only — guarded)
    atomic_scp "${REMOTE_CKPT_DIR}/${E1}_losses.csv" \
               "${LOCAL_CKPT}/${E1}_losses.csv" "${CSV_MIN}" || true
    atomic_scp "${REMOTE_CKPT_DIR}/${E2}_losses.csv" \
               "${LOCAL_CKPT}/${E2}_losses.csv" "${CSV_MIN}" || true

    # Run log
    atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true

    # Surface any error / NaN signal in the log
    if grep -qiE "nan detected|Traceback|Error:" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  !!! NaN/Error/Traceback in run.log !!!" | tee -a "${LOG_FILE}"
    fi

    ALIVE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" \
        'pgrep -af "train_forecasting_head" >/dev/null && echo ALIVE || echo DEAD' \
        2>/dev/null | tail -1)
    echo "  remote training process: ${ALIVE:-UNREACHABLE}" | tee -a "${LOG_FILE}"

    if grep -q "R1 ALL DONE" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  === R1 ALL DONE detected; final sync ===" | tee -a "${LOG_FILE}"
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true
        echo "[$(date)] Sync loop qhead_beta_rd1 finished" | tee -a "${LOG_FILE}"
        exit 0
    fi

    SLEEP=900
    echo "  next sync in ${SLEEP}s" | tee -a "${LOG_FILE}"
    sleep "${SLEEP}"
done
