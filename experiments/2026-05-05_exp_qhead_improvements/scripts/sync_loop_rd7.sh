#!/bin/bash
# Atomic sync loop for R3_E4 (transformer head on backbone-beta).
# Single experiment — heads are ~10.7M params (~43MB).
# Usage: ./sync_loop.sh <SSH_HOST> <SSH_PORT> <INSTANCE_ID>

set -u

SSH_HOST="${1:?missing SSH_HOST}"
SSH_PORT="${2:?missing SSH_PORT}"
INSTANCE_ID="${3:?missing INSTANCE_ID}"

REMOTE="root@${SSH_HOST}"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_LOG="/workspace/app/run_qhead_beta_rd7.log"
LOCAL_BASE="/home/jupyter/contrastive-forecasting/sync_qhead_beta_rd7"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOG_FILE="${LOCAL_BASE}/sync.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

mkdir -p "${LOCAL_CKPT}"
echo "[$(date)] Sync loop start qhead_beta_rd7 instance=${INSTANCE_ID} @ ${SSH_HOST}:${SSH_PORT}" | tee -a "${LOG_FILE}"

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
                case "${_bn}" in
                    *.csv|*.log)
                        if [ -s "${local_dest}" ]; then
                            local _new _old
                            _new=$(wc -l < "${tmp}" 2>/dev/null | tr -d ' ')
                            _old=$(wc -l < "${local_dest}" 2>/dev/null | tr -d ' ')
                            : "${_new:=0}"; : "${_old:=0}"
                            if [ "${_new}" -lt "${_old}" ]; then
                                mv "${tmp}" "${LOCAL_BASE}/$(date +%Y%m%dT%H%M%S)_${_bn}.regression"
                                echo "  ⚠️ APPEND REGRESSION on ${_bn}: remote=${_new} local=${_old}" | tee -a "${LOG_FILE}"
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

# Transformer head ~43 MB; optimizer ~86 MB.
HEAD_MIN=40000000
HEAD_OPT_MIN=80000000
CSV_MIN=1
LOG_MIN=1

E4="R7_E9_xfmr12L_quant_moirai_cosine_100k"

cycle_count=0
while true; do
    cycle_count=$((cycle_count + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date)] === qhead_beta_rd7 cycle #${cycle_count} ===" | tee -a "${LOG_FILE}"

    for kind in best FINAL; do
        atomic_scp "${REMOTE_CKPT_DIR}/${E4}_${kind}.pth" \
                   "${LOCAL_CKPT}/${E4}_${kind}.pth" "${HEAD_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${E4}_${kind}_optimizer.pth" \
                   "${LOCAL_CKPT}/${E4}_${kind}_optimizer.pth" "${HEAD_OPT_MIN}" || true
    done

    for rp in $(list_remote "${REMOTE_CKPT_DIR}/${E4}_*k.pth"); do
        fname=$(basename "${rp}")
        if [ ! -f "${LOCAL_CKPT}/${fname}" ]; then
            atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" "${HEAD_MIN}" || true
            opt_remote="${rp%.pth}_optimizer.pth"
            opt_fname="${fname%.pth}_optimizer.pth"
            atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" "${HEAD_OPT_MIN}" || true
        fi
    done

    atomic_scp "${REMOTE_CKPT_DIR}/${E4}_losses.csv" \
               "${LOCAL_CKPT}/${E4}_losses.csv" "${CSV_MIN}" || true
    atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true

    if grep -qiE "nan detected|Traceback|Error:" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  !!! NaN/Error/Traceback in run.log !!!" | tee -a "${LOG_FILE}"
    fi

    ALIVE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" \
        'pgrep -af "train_forecasting_head" >/dev/null && echo ALIVE || echo DEAD' \
        2>/dev/null | tail -1)
    echo "  remote training process: ${ALIVE:-UNREACHABLE}" | tee -a "${LOG_FILE}"

    if grep -q "R7 ALL DONE" "${LOCAL_BASE}/run.log" 2>/dev/null; then
        echo "  === R7 ALL DONE detected; final sync ===" | tee -a "${LOG_FILE}"
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run.log" "${LOG_MIN}" || true
        echo "[$(date)] Sync loop qhead_beta_rd7 finished" | tee -a "${LOG_FILE}"
        exit 0
    fi

    SLEEP=900
    echo "  next sync in ${SLEEP}s" | tee -a "${LOG_FILE}"
    sleep "${SLEEP}"
done
