#!/bin/bash
# Watchdog + periodic plotter + git-pusher for the enc_fcst_dk09_hs_b512_50k run.
#
# Emits one stdout line per significant event (each becomes a Monitor
# notification). Selective: silent during normal training; speaks up on
# every 5k-step boundary (after replotting + pushing to GitHub) or on
# errors/completion.
#
# Lines emitted:
#   "[plot+push @ NNNNN] loss=… auc=… top1=… egc=…"  every 5k step boundary
#   "[ERROR] <line>"                                  error / NaN / OOM
#   "[DEATH] training process gone (last step NNNNN)"
#   "[DONE] FINAL.pth written"
#
set -uo pipefail

REPO=/home/jupyter/cf-encoder-forecaster-v2
LOG=$REPO/experiments/2026-05-11_exp_encoder_forecaster/results/run_enc_fcst_dk09_hs_b512_50k.log
CSV=/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dk09_hs_b512_50k_losses.csv
PLOT_DIR=$REPO/experiments/2026-05-11_exp_encoder_forecaster
PLOT_SCRIPT=$PLOT_DIR/scripts/plot_progress.py
COMMIT_PATHS=(
    experiments/2026-05-11_exp_encoder_forecaster/plots/
    experiments/2026-05-11_exp_encoder_forecaster/results/
)

last_marker=0           # last 5k step boundary we plotted
last_err_pos=0          # bytes already scanned in the log
heartbeat_step=0        # for liveness check

step_5k=5000

while true; do
    sleep 60

    # ---------- step-based plot trigger ----------
    if [ -f "$CSV" ]; then
        cur_step=$(tail -1 "$CSV" 2>/dev/null | cut -d, -f1)
        # guard non-numeric (header)
        case "$cur_step" in
            ''|*[!0-9]*) cur_step=0 ;;
        esac
        next_marker=$(( (cur_step / step_5k) * step_5k ))
        if [ "$next_marker" -gt "$last_marker" ] && [ "$next_marker" -ge "$step_5k" ]; then
            cd "$REPO" && PYTHONPATH=. python3 "$PLOT_SCRIPT" \
                >/tmp/plot_$$.out 2>&1 || true
            summary=$(grep "encoder+forecaster v2" /tmp/plot_$$.out | head -1)
            rm -f /tmp/plot_$$.out

            # commit + push: plots/ + results/ (log file).
            cd "$REPO" && git add "${COMMIT_PATHS[@]}" >/dev/null 2>&1 || true
            if ! git -C "$REPO" diff --cached --quiet; then
                # Use whatever branch the worktree is currently on. After
                # the v2 PR was merged we moved to v4-fresh, hardcoding
                # was wrong.
                cur_branch=$(git -C "$REPO" branch --show-current)
                git -C "$REPO" commit -m "exp(enc-fcst-v4): progress @ step $next_marker" \
                    >/dev/null 2>&1 || true
                git -C "$REPO" push origin "$cur_branch" \
                    >/tmp/push_$$.out 2>&1
                push_status=$?
                rm -f /tmp/push_$$.out
                push_tag="pushed"
                [ "$push_status" -ne 0 ] && push_tag="push-FAILED"
            else
                push_tag="no-change"
            fi
            echo "[plot+$push_tag @ $next_marker] $summary"
            last_marker=$next_marker
        fi
        heartbeat_step=$cur_step
    fi

    # ---------- error / completion scanning ----------
    if [ -f "$LOG" ]; then
        cur_size=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
        if [ "$cur_size" -gt "$last_err_pos" ]; then
            new=$(tail -c +$((last_err_pos + 1)) "$LOG" 2>/dev/null)
            last_err_pos=$cur_size
            # Errors / fatal signals
            echo "$new" | grep -E '(Traceback|Error|FAILED|OOM|Killed|RuntimeError|CUDA)' \
                | grep -vE '(Warning|warning|UserWarning)' \
                | head -3 \
                | while read -r line; do echo "[ERROR] $line"; done
            # NaN
            echo "$new" | grep -iE 'nan' | head -2 \
                | while read -r line; do echo "[NaN?] $line"; done
            # Completion
            echo "$new" | grep -E '=== DONE' \
                | while read -r line; do echo "[DONE] $line"; done
        fi
    fi

    # ---------- liveness ----------
    if ! pgrep -f 'experiments/2026-04-27_freq-embedding/scripts/train.py.*enc_fcst_dk09_hs_b512_50k' >/dev/null 2>&1; then
        # process is gone — emit DEATH once and exit
        echo "[DEATH] training process gone (last step $heartbeat_step)"
        exit 0
    fi
done
