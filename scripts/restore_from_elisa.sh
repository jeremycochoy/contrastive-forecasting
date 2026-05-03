#!/usr/bin/env bash
# Inverse of backup_to_elisa.sh. Pulls the gitignored working data from
# the elisa backup into the current checkout's repo root.
#
# Auto-detect:
#   - If running on elisa AND $HOME/contrastive-forecasting_backup/ exists,
#     use it as a local rsync source (fast, no SSH).
#   - Otherwise rsync from jupyter@elisa over SSH.
#
# Usage:
#   bash scripts/restore_from_elisa.sh             # restore all
#   bash scripts/restore_from_elisa.sh --priority  # only live + recent full4096
#
# Idempotent: re-runs are delta-only.

set -u

LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="jupyter@elisa"
REMOTE_BASE="/home/jupyter/contrastive-forecasting_backup"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
RSYNC_OPTS=(-azv --partial)

# Detect local-vs-remote mode.
mode_label="ssh"
src_prefix="$REMOTE:$REMOTE_BASE"
if [[ "$(hostname -s 2>/dev/null)" == "elisa" ]]; then
    if [[ -d "$HOME/contrastive-forecasting_backup" ]]; then
        mode_label="local"
        src_prefix="$HOME/contrastive-forecasting_backup"
    fi
fi
[[ "$mode_label" == "ssh" ]] && RSYNC_OPTS+=(-e "ssh ${SSH_OPTS[*]}")

PRIORITY_PATHS=(
    "experiments/hf_token.txt"
    "sync_realonly_full4096_moirai_hp_FRESH"
    "sync_realonly_full4096_moirai_hp_FINAL_run1"
    "sync_realonly_full4096_moirai_hp_FINAL_run2"
    "sync_realonly_full4096_moirai_hp"
    "sync_realonly_full4096_learnable_tau"
)

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

ok=0; fail=0; missing=0
for p in "${PATHS[@]}"; do
    src="$src_prefix/$p"
    dst="$LOCAL_BASE/$p"
    # For local mode, check existence directly. For ssh, attempt rsync and
    # let it report missing.
    if [[ "$mode_label" == "local" ]] && [[ ! -e "$src" ]]; then
        echo "⚠️  skip (missing on backup): $p"
        missing=$((missing+1))
        continue
    fi
    if [[ "$mode_label" == "local" ]] && [[ -d "$src" ]]; then
        mkdir -p "$dst"
        if rsync "${RSYNC_OPTS[@]}" "$src/" "$dst/"; then
            echo "✓ $p"; ok=$((ok+1))
        else
            echo "⚠️  rsync failed: $p"; fail=$((fail+1))
        fi
    elif [[ "$mode_label" == "local" ]]; then
        mkdir -p "$(dirname "$dst")"
        if rsync "${RSYNC_OPTS[@]}" "$src" "$dst"; then
            echo "✓ $p (file)"; ok=$((ok+1))
        else
            echo "⚠️  rsync failed: $p"; fail=$((fail+1))
        fi
    else
        # SSH mode: try as directory first (trailing slash, with --rsync-path
        # to test existence cheaply). We just attempt rsync; missing entries
        # produce an rsync error which we count as fail.
        mkdir -p "$dst" 2>/dev/null
        if rsync "${RSYNC_OPTS[@]}" "$src/" "$dst/" 2>/dev/null; then
            echo "✓ $p (dir)"; ok=$((ok+1))
        elif rsync "${RSYNC_OPTS[@]}" "$src" "$dst" 2>/dev/null; then
            echo "✓ $p (file)"; ok=$((ok+1))
        else
            echo "⚠️  rsync failed: $p"; fail=$((fail+1))
        fi
    fi
done

echo ""
echo "[restore_from_elisa] mode=$mode_label  ok=$ok  fail=$fail  missing=$missing"
exit 0
