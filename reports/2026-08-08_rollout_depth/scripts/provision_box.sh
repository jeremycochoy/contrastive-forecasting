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

# vastrun-kit reads `.vastrun.toml` from the CWD and exits if it is not
# there, and its error goes to stdout, where it reads as "unreadable
# provision output" rather than as the one-line fix it is. Run from the
# checkout root, which holds the file.
VASTRUN_DIR="${VASTRUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$VASTRUN_DIR" ] && [ -f "$VASTRUN_DIR/.vastrun.toml" ] || {
  say "ABORT: no .vastrun.toml under '${VASTRUN_DIR:-?}' — vastrun-kit needs one in the CWD"
  exit 2; }
cd "$VASTRUN_DIR" || exit 2

# The endpoint out of a kit message, as "host port", or nothing. The kit
# prints TWO forms and this script has to read both: the success banner says
#   SSH: ssh -p 13680 root@ssh8.vast.ai
# and the failure messages say
#   ... SSH unreachable at ssh8.vast.ai:13680.
# Round 2 matched only the second and read every SUCCESS as unreadable
# output, so the retry loop left three billing instances behind before the
# log was read. Never parse one form of a tool's output when it prints two.
endpoint_of(){ # <text> -> "host port"
  local t="$1" hp
  hp=$(grep -oE "ssh[0-9]*\.vast\.ai:[0-9]+" <<<"$t" | head -1)
  [ -n "$hp" ] && { printf '%s %s\n' "${hp%%:*}" "${hp##*:}"; return 0; }
  awk 'match($0, /ssh +-p +[0-9]+ +root@ssh[0-9]*\.vast\.ai/) {
         s = substr($0, RSTART, RLENGTH)
         split(s, f, /[ \t]+/)
         for (i = 1; i <= length(f); i++) {
           if (f[i] == "-p") port = f[i+1]
           if (f[i] ~ /^root@/) { host = f[i]; sub(/^root@/, "", host) }
         }
         if (host != "" && port != "") { print host, port; exit }
       }' <<<"$t"
}

try_offer(){ # <offer id> -> prints "id host port" on success
  local off="$1" out id ep
  out=$(timeout 900 vastrun-provision "$off" --label "$LABEL" 2>&1)

  if grep -qi "no longer available" <<<"$out"; then
    say "  offer $off went away"; return 1
  fi

  # `SSH unreachable` is usually not a dead box. vastrun-kit gives sshd
  # 15 x 2 s = 30 s after boot, and a vast.ai container routinely takes
  # longer than that to open the port. Round 2 threw away two RTX 5090s in
  # ten seconds on this message before the probe below existed.
  #
  # So probe it. The instance exists and is billing either way; the only
  # question is whether it is usable. An instance adopted here carries no
  # vastrun ownership marker, because the kit writes the marker after this
  # check — destroying it later needs --force, and `reap` already handles
  # that case by name.
  if grep -qi "SSH unreachable" <<<"$out"; then
    id=$(grep -oE "Instance [0-9]+" <<<"$out" | head -1 | awk '{print $2}')
    ep=$(endpoint_of "$out")
    if [ -n "$id" ] && [ -n "$ep" ]; then
      local h p w=0; read -r h p <<<"$ep"
      say "  offer $off: instance $id has no ssh yet — probing $h:$p for ${SSH_PROBE_WAIT:-300}s"
      while [ "$w" -lt "${SSH_PROBE_WAIT:-300}" ]; do
        if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=10 -o BatchMode=yes -p "$p" "root@$h" true 2>/dev/null; then
          say "ADOPTED instance $id at $h:$p after ${w}s (no vastrun marker)"
          printf '%s %s %s\n' "$id" "$h" "$p"
          return 0
        fi
        sleep 15; w=$(( w + 15 ))
      done
      say "  offer $off: instance $id still unreachable after ${w}s — destroying it"
    else
      say "  offer $off: instance ${id:-?} unreachable and unparsable — destroying it"
    fi
    [ -n "$id" ] && timeout 300 vastrun-destroy "$id" --force >/dev/null 2>&1
    return 1
  fi

  # The host never brought it up. It exists and it bills, so it goes.
  if grep -qi "did not reach" <<<"$out"; then
    id=$(grep -oE "Instance [0-9]+" <<<"$out" | head -1 | awk '{print $2}')
    say "  offer $off: instance ${id:-?} never reached running — destroying it"
    [ -n "$id" ] && timeout 300 vastrun-destroy "$id" --force >/dev/null 2>&1
    return 1
  fi

  id=$(grep -oE "Instance [0-9]+" <<<"$out" | head -1 | awk '{print $2}')
  ep=$(endpoint_of "$out")
  if [ -z "$id" ] || [ -z "$ep" ]; then
    # An instance may exist behind an output this script cannot read, and it
    # bills whether or not the caller ever learns its address. Destroy it if
    # the id is legible, and say so loudly if it is not.
    say "  offer $off: unreadable provision output:"
    printf '%s\n' "$out" >&2
    if [ -n "$id" ]; then
      say "  offer $off: destroying instance $id, which the output above created"
      timeout 300 vastrun-destroy "$id" --force >/dev/null 2>&1
    else
      say "  offer $off: NO INSTANCE ID IN THAT OUTPUT — check vastrun-status by hand"
    fi
    return 1
  fi
  say "READY instance $id at ${ep/ /:}"
  printf '%s %s\n' "$id" "$ep"
  return 0
}

for (( a = 1; a <= TRIES; a++ )); do
  # Every offer of one search, not just the first. The listing churns in
  # seconds, so re-searching per offer spends the whole budget of attempts
  # racing the same disappearing row.
  # Round 2 pins the GPU class. The cells run 60k to 100k steps each and the
  # 5090 measured 136 ms/step against the 4090's 443 ms on this study's own
  # runs, so a class filter is worth more than the price spread.
  read -r -a search_args <<<"${VAST_SEARCH_ARGS:---max-bid 0.60}"
  mapfile -t offers < <(timeout 200 vastrun-search "${search_args[@]}" --limit "${VAST_SEARCH_LIMIT:-20}" 2>/dev/null \
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
