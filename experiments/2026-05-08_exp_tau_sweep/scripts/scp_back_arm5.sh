#!/bin/bash
# scp-back watcher for τ-sweep arm5 (τ=0.20) running on a vast.ai instance.
#
# Polls the vast-side launcher log for the "ARM τ=0.20 DONE" marker every
# 60 s. Once seen, atomically scp's the FINAL.pth (+ losses.csv + optimizer
# + final log) back to elisa's sync_tau_sweep/checkpoints/ so the elisa
# launcher's idempotency check skips τ=0.20 when its sequential loop reaches
# that arm.
#
# Usage (on elisa):
#   nohup bash experiments/2026-05-08_exp_tau_sweep/scripts/scp_back_arm5.sh \
#       <SSH_HOST> <SSH_PORT> > sync_tau_sweep/scp_back_arm5.log 2>&1 &
#   disown
#
# Atomic write: scp to .tmp, size-check, rotate existing file to .prev, mv.
# Exits 0 on success, non-zero on permanent failure (e.g. > 6 h with no
# DONE marker — vast probably crashed; alert the user).

set -u

SSH_HOST="${1:?usage: $0 <SSH_HOST> <SSH_PORT>}"
SSH_PORT="${2:?usage: $0 <SSH_HOST> <SSH_PORT>}"

REMOTE="root@${SSH_HOST}"
REMOTE_LOG="/workspace/app/run_tau_0_20.log"
REMOTE_CKPT_DIR="/workspace/app/checkpoints"
NAME="tau_sweep_0_20"

LOCAL_BASE="/home/jupyter/contrastive-forecasting/sync_tau_sweep"
LOCAL_CKPT="${LOCAL_BASE}/checkpoints"
LOG_FILE="${LOCAL_BASE}/scp_back_arm5.log"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o ConnectTimeout=30 -o ServerAliveInterval=30"

POLL_INTERVAL=60
MAX_WAIT_SEC=$((6 * 3600))   # 6 h hard cap; τ=0.20 takes ~1 h on a 5090
BB_MIN=40000000              # backbone .pth ~46 MB at d_model=384, C=1
BB_OPT_MIN=70000000          # AdamW optimizer ~90 MB

mkdir -p "${LOCAL_CKPT}"
echo "[$(date)] scp_back_arm5 start; remote=${SSH_HOST}:${SSH_PORT}" | tee -a "${LOG_FILE}"

atomic_scp() {
    local remote_path="$1"
    local local_dest="$2"
    local min_size="${3:-0}"
    local tmp="${local_dest}.tmp"

    if scp ${SSH_OPTS} -P "${SSH_PORT}" -q "${REMOTE}:${remote_path}" "${tmp}" 2>/dev/null; then
        if [ -s "${tmp}" ]; then
            local sz
            sz=$(stat -c%s "${tmp}" 2>/dev/null || echo 0)
            if [ "${sz}" -ge "${min_size}" ]; then
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
        fi
    fi
    rm -f "${tmp}"
    return 1
}

elapsed=0
while [ "${elapsed}" -lt "${MAX_WAIT_SEC}" ]; do
    DONE=$(ssh ${SSH_OPTS} -p "${SSH_PORT}" "${REMOTE}" \
        "grep -qE '=== ARM .{0,5}=0\\.20 DONE ===' ${REMOTE_LOG} && echo DONE || echo PENDING" \
        2>/dev/null | tail -1)

    if [ "${DONE}" = "DONE" ]; then
        echo "[$(date)] DONE marker seen on remote log; pulling files." | tee -a "${LOG_FILE}"

        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_FINAL.pth" \
                   "${LOCAL_CKPT}/${NAME}_FINAL.pth" "${BB_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_FINAL_optimizer.pth" \
                   "${LOCAL_CKPT}/${NAME}_FINAL_optimizer.pth" "${BB_OPT_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_best_loss.pth" \
                   "${LOCAL_CKPT}/${NAME}_best_loss.pth" "${BB_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_best_loss_optimizer.pth" \
                   "${LOCAL_CKPT}/${NAME}_best_loss_optimizer.pth" "${BB_OPT_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_best_gap.pth" \
                   "${LOCAL_CKPT}/${NAME}_best_gap.pth" "${BB_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_best_gap_optimizer.pth" \
                   "${LOCAL_CKPT}/${NAME}_best_gap_optimizer.pth" "${BB_OPT_MIN}" || true
        atomic_scp "${REMOTE_CKPT_DIR}/${NAME}_losses.csv" \
                   "${LOCAL_CKPT}/${NAME}_losses.csv" 1 || true
        atomic_scp "${REMOTE_LOG}" "${LOCAL_BASE}/run_tau_0_20_vast.log" 1 || true

        if [ -s "${LOCAL_CKPT}/${NAME}_FINAL.pth" ]; then
            echo "[$(date)] scp_back_arm5 SUCCESS — FINAL.pth present locally" | tee -a "${LOG_FILE}"
            exit 0
        else
            echo "[$(date)] scp_back_arm5 PARTIAL — FINAL.pth missing locally; will retry next cycle" | tee -a "${LOG_FILE}"
            # don't exit; loop again — maybe scp transiently failed.
        fi
    fi

    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo "[$(date)] scp_back_arm5 TIMEOUT after ${MAX_WAIT_SEC}s with no DONE marker" | tee -a "${LOG_FILE}"
exit 2
