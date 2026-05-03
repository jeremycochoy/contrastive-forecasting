#!/usr/bin/env bash
# Bidirectional convenience wrapper around backup_to_elisa.sh /
# restore_from_elisa.sh.
#
# Usage:
#   bash scripts/mirror_with_elisa.sh --push       # this host → elisa
#   bash scripts/mirror_with_elisa.sh --pull       # elisa → this host
#   bash scripts/mirror_with_elisa.sh --push --priority
#   bash scripts/mirror_with_elisa.sh --pull --all
#
# Default direction depends on host:
#   - laptop / non-elisa → --push (safer; treat the gitignored data as
#     authoritative on the active workstation, mirror to elisa as backup)
#   - elisa              → --pull (this is the post-migration operating mode)

set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
direction=""
mode="all"

for arg in "$@"; do
    case "$arg" in
        --push)     direction="push" ;;
        --pull)     direction="pull" ;;
        --priority) mode="priority" ;;
        --all)      mode="all" ;;
        *) echo "usage: $0 [--push|--pull] [--priority|--all]" >&2; exit 0 ;;
    esac
done

if [[ -z "$direction" ]]; then
    if [[ "$(hostname -s 2>/dev/null)" == "elisa" ]]; then
        direction="pull"
    else
        direction="push"
    fi
fi

case "$direction" in
    push) exec bash "$DIR/backup_to_elisa.sh"   "--$mode" ;;
    pull) exec bash "$DIR/restore_from_elisa.sh" "--$mode" ;;
esac
