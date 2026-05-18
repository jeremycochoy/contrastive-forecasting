#!/usr/bin/env bash
# Sync loop for loss_ext_square experiment on elisa.
# Pulls all square arms (15k initial + 50k + 100k extensions) every 15 min.
# CSVs are snapshotted with step-count suffix so no data is ever overwritten.
REMOTE=jupyter@elisa
REMOTE_DIR=/home/jupyter/contrastive-forecasting/sync_loss_ext_square/checkpoints
LOCAL_DIR=/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_loss_ext_square/checkpoints
INTERVAL=900

mkdir -p "$LOCAL_DIR"

pull_pth() {
    local rp="$1" lp="$2" min="${3:-40000000}" tmp="${lp}.tmp"
    if ssh "$REMOTE" "test -f ${rp}" 2>/dev/null; then
        scp "${REMOTE}:${rp}" "$tmp" 2>/dev/null || { echo "⚠️  scp failed: $rp"; return 1; }
        local sz; sz=$(stat -f%z "$tmp" 2>/dev/null || stat -c%s "$tmp")
        if [[ "$sz" -ge "$min" ]]; then
            [[ -f "$lp" ]] && mv "$lp" "${lp}.prev"
            mv "$tmp" "$lp"
            echo "✓ $(basename "$lp") (${sz}B)"
        else
            echo "⚠️  $(basename "$lp") too small (${sz} < ${min}), keeping prior"
            rm -f "$tmp"
        fi
    fi
}

pull_csv() {
    local rp="$1" lp="$2" tmp="${lp}.tmp"
    if ssh "$REMOTE" "test -f ${rp}" 2>/dev/null; then
        scp "${REMOTE}:${rp}" "$tmp" 2>/dev/null || { echo "⚠️  scp failed: $rp"; return 1; }
        local sz; sz=$(stat -f%z "$tmp" 2>/dev/null || stat -c%s "$tmp")
        if [[ "$sz" -ge 10 ]]; then
            local last_step
            last_step=$(tail -1 "$tmp" | cut -d',' -f1)
            mv "$tmp" "$lp"
            local snap="${lp%.csv}_step${last_step}.csv"
            if [[ ! -f "$snap" ]]; then
                cp "$lp" "$snap"
                echo "✓ $(basename "$lp") (${sz}B, step=${last_step}) → snapshot"
            else
                echo "✓ $(basename "$lp") (${sz}B, step=${last_step})"
            fi
        else
            echo "⚠️  $(basename "$lp") too small, skipping"
            rm -f "$tmp"
        fi
    fi
}

tick() {
    echo "=== sync $(date '+%Y-%m-%d %H:%M:%S') ==="
    for NAME in \
        loss_ext_square_tau_0_10 \
        loss_ext_square_tau_0_20 \
        loss_ext_square_tau_0_10_50k \
        loss_ext_square_tau_0_20_50k \
        loss_ext_square_tau_0_10_100k \
        loss_ext_square_tau_0_20_100k; do
        pull_csv "${REMOTE_DIR}/${NAME}_losses.csv" "${LOCAL_DIR}/${NAME}_losses.csv"
        for suffix in _best_gap _best_loss; do
            pull_pth "${REMOTE_DIR}/${NAME}${suffix}.pth"           "${LOCAL_DIR}/${NAME}${suffix}.pth"
            pull_pth "${REMOTE_DIR}/${NAME}${suffix}_optimizer.pth" "${LOCAL_DIR}/${NAME}${suffix}_optimizer.pth"
        done
        for f in $(ssh "$REMOTE" "ls ${REMOTE_DIR}/${NAME}_*k.pth 2>/dev/null" 2>/dev/null || true); do
            base=$(basename "$f")
            opt="$(basename "${f%%.pth}")_optimizer.pth"
            pull_pth "${REMOTE_DIR}/${base}" "${LOCAL_DIR}/${base}"
            pull_pth "${REMOTE_DIR}/${opt}"  "${LOCAL_DIR}/${opt}"
        done
    done
}

echo "Sync loop started — interval ${INTERVAL}s"
echo "Remote: ${REMOTE}:${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"

while true; do tick; sleep "$INTERVAL"; done
