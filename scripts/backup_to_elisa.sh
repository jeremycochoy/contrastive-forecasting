#!/usr/bin/env bash
# Mirror gitignored working data (sync targets, hf_token, etc.) from this
# host to jupyter@elisa:~/contrastive-forecasting_backup/.
#
# Skips itself when running on elisa (so post-migration runs of this
# script are no-ops). Best-effort: SSH/rsync failures log a ⚠️ line and
# exit 0 so callers using this in EXIT traps don't get spurious errors.
#
# Usage:
#   bash scripts/backup_to_elisa.sh                # mirror all (full set)
#   bash scripts/backup_to_elisa.sh --priority     # only the live + recent full4096 work
#
# The --priority mode is what you want when the laptop is about to be
# turned off and you need the live training's data preserved on elisa
# RIGHT NOW. Full-set runs get the rest (older experiments).

set -u

if [[ "$(hostname -s 2>/dev/null)" == "elisa" ]]; then
    echo "⚠️  running on elisa; backup_to_elisa is a no-op here" >&2
    exit 0
fi

REMOTE="jupyter@elisa"
REMOTE_BASE="/home/jupyter/contrastive-forecasting_backup"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
# macOS rsync is v2.6.9 — does not support --info=progress2. Use plain
# verbose (-v) which works on both ancient macOS rsync and modern Linux.
RSYNC_OPTS=(-azv --partial -e "ssh ${SSH_OPTS[*]}")

PRIORITY_PATHS=(
    "experiments/hf_token.txt"
    "sync_realonly_full4096_moirai_hp_FRESH"
    "sync_realonly_full4096_moirai_hp_FINAL_run1"
    "sync_realonly_full4096_moirai_hp_FINAL_run2"
    "sync_realonly_full4096_moirai_hp"
    "sync_realonly_full4096_learnable_tau"
)

# Everything else worth preserving but not blocking the migration cutover.
ALL_EXTRA_PATHS=(
    "sync_compositesynth_v3primitives"
    "sync_compositesynth_v4combined"
    "sync_compositesynth_v5envboost"
    "sync_csb_pair_ewma"
    "sync_csb_pair_revin"
    "sync_csb_synth_legacy"
    "sync_dualemb_3arm"
    "sync_freqemb"
    "sync_multiexp"
    "sync_periodic_synth"
    "sync_realonly_4096"
    "sync_realonly_4096_smaller"
    "sync_realonly_4096_smaller_learnable_tau"
    "sync_realonly_4096_smaller_tau_sweep"
    "sync_v3b"
    "sync_v3b_final"
)

mode="${1:-all}"
case "$mode" in
    --priority|priority) PATHS=("${PRIORITY_PATHS[@]}") ;;
    --all|all|"")        PATHS=("${PRIORITY_PATHS[@]}" "${ALL_EXTRA_PATHS[@]}") ;;
    *) echo "usage: $0 [--priority|--all]" >&2; exit 0 ;;
esac

ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_BASE'" 2>/dev/null \
    || { echo "⚠️  ssh to $REMOTE failed; aborting (best-effort, exit 0)" >&2; exit 0; }

ok=0; fail=0
for p in "${PATHS[@]}"; do
    src="$LOCAL_BASE/$p"
    if [[ ! -e "$src" ]]; then
        echo "⚠️  skip (missing locally): $p"
        continue
    fi
    # macOS rsync 2.6.9 doesn't auto-create parent dirs on remote, so
    # mkdir explicitly first.
    parent="$REMOTE_BASE/$(dirname "$p")"
    ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$parent'" 2>/dev/null || true
    # rsync with trailing slash on dir source = copy contents into dst path.
    if [[ -d "$src" ]]; then
        # shellcheck disable=SC2029
        if rsync "${RSYNC_OPTS[@]}" "$src/" "$REMOTE:$REMOTE_BASE/$p/"; then
            echo "✓ $p"; ok=$((ok+1))
        else
            echo "⚠️  rsync failed: $p"; fail=$((fail+1))
        fi
    else
        if rsync "${RSYNC_OPTS[@]}" "$src" "$REMOTE:$REMOTE_BASE/$p"; then
            echo "✓ $p (file)"; ok=$((ok+1))
        else
            echo "⚠️  rsync failed: $p"; fail=$((fail+1))
        fi
    fi
done

echo ""
echo "[backup_to_elisa] mode=$mode  ok=$ok  fail=$fail"
exit 0
