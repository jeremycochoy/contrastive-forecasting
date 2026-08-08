#!/bin/bash
# #373 — get one usable vast.ai box, or give up cleanly.
#
# Usage: bash provision_box.sh <label> [max attempts]
#
# Two failures are routine on vast.ai and both waste money if unhandled:
#
#   "Offer N is no longer available"  — the listing churns in seconds. Cost
#       nothing; search again and take the next one.
#   "Instance N created but SSH unreachable" — the instance EXISTS and is
#       BILLING. It has to be destroyed, or it bills for nothing until
#       somebody notices. The study's whole GPU budget is $7.31.
#
# Prints `<instance id> <ssh host> <ssh port>` on success.
#
# Only instances this script created are ever destroyed, and only by the id
# the failed provision printed: the vast.ai account is shared with other
# agent sessions (CLAUDE.md), and a broad cleanup would take their work.
set -uo pipefail

LABEL="${1:?usage: provision_box.sh <label> [max attempts]}"
TRIES="${2:-8}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [provision $LABEL] $*" >&2; }

try_offer(){ # <offer id> -> prints "id host port" on success
  local off="$1" out id hostport
  out=$(timeout 900 vastrun-provision "$off" --label "$LABEL" 2>&1)

  if grep -qi "no longer available" <<<"$out"; then
    say "  offer $off went away"; return 1
  fi

  # Both of these leave an instance that EXISTS and BILLS. Destroy it by the
  # id in the message before trying another offer.
  #   SSH unreachable      — booted, no route in
  #   did not reach 'running' (reason: doa) — the host never brought it up
  if grep -qiE "SSH unreachable|did not reach" <<<"$out"; then
    id=$(grep -oE "Instance [0-9]+" <<<"$out" | head -1 | awk '{print $2}')
    say "  offer $off: instance ${id:-?} came up dead — destroying it"
    [ -n "$id" ] && timeout 300 vastrun-destroy "$id" --force >/dev/null 2>&1
    return 1
  fi

  id=$(grep -oE "Instance [0-9]+" <<<"$out" | head -1 | awk '{print $2}')
  hostport=$(grep -oE "ssh[0-9]*\.vast\.ai:[0-9]+" <<<"$out" | head -1)
  if [ -z "$id" ] || [ -z "$hostport" ]; then
    say "  offer $off: unreadable provision output:"
    printf '%s\n' "$out" >&2
    return 1
  fi
  say "READY instance $id at $hostport"
  printf '%s %s %s\n' "$id" "${hostport%%:*}" "${hostport##*:}"
  return 0
}

for (( a = 1; a <= TRIES; a++ )); do
  # Every offer of one search, not just the first. The listing churns in
  # seconds, so re-searching per offer spends the whole budget of attempts
  # racing the same disappearing row.
  mapfile -t offers < <(timeout 200 vastrun-search --max-bid 0.60 2>/dev/null \
        | awk 'NR>1 && $1 ~ /^[0-9]+$/ {print $1}')
  [ "${#offers[@]}" -gt 0 ] || { say "attempt $a: no offer matched"; sleep 20; continue; }
  say "attempt $a: ${#offers[@]} offer(s)"
  for off in "${offers[@]}"; do
    try_offer "$off" && exit 0
  done
  sleep 10
done

say "gave up after $TRIES attempts"
exit 1
