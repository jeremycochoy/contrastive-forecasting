#!/bin/bash
# The Hub-outage gate. Sourced, never run.
#
# Training data streams from the Hugging Face Hub, so a lane's leg dies in
# about 3 seconds when the box loses DNS. On 2026-08-23 at 18:48 elisa lost
# DNS for some minutes. A lane read those 3-second deaths as failed arms,
# spent its whole retry ladder in two minutes, declared the arm dead and moved
# to the next one. Three arms went that way in seven minutes, and the card
# then sat idle for 27 hours.
#
# A network failure is not a failed arm. This library gives a lane the two
# things it needs to tell them apart and to ride an outage out.
#
#   hub_outage_in_text / hub_outage_in_log   read a dead leg's tail
#   hub_is_up / hub_wait_up                  read the Hub itself
#
# Usage:
#   . "$(dirname "${BASH_SOURCE[0]}")/../scripts/hub_gate.sh"
#   hub_wait_up "$HUB_GATE_DEADLINE" || exit 1      # before an arm starts
#   hub_outage_in_log "$tlog" && ...                # after a leg dies

# The exit code a leg gives when the HUB, not the leg, is what failed. It
# lives here so the leg that raises it and the lane that reads it cannot
# drift apart. It is none of the codes a leg already uses (0, 1, 2, 4, 9, 10).
HUB_GATE_RC=20

# The host a probe reads, and how long one probe may take.
HUB_GATE_HOST="${HUB_GATE_HOST:-huggingface.co}"
HUB_GATE_PROBE_TIMEOUT="${HUB_GATE_PROBE_TIMEOUT:-10}"
# The delay before the second probe, and the ceiling each doubling stops at.
HUB_GATE_BASE_WAIT="${HUB_GATE_BASE_WAIT:-60}"
HUB_GATE_MAX_WAIT="${HUB_GATE_MAX_WAIT:-900}"
# How long a caller waits in total, by default. Hours, not minutes: a DNS
# outage of 30 minutes must not end a study, and a lane that gives up hands
# the card back to nobody.
HUB_GATE_DEADLINE="${HUB_GATE_DEADLINE:-21600}"
# How many lines of a trainer log hold the failure that ended it.
HUB_GATE_TAIL_LINES="${HUB_GATE_TAIL_LINES:-40}"

hub_gate_log(){ echo "[$(date '+%m-%d %H:%M:%S')] [hub-gate] $*"; }

# Every string that says "the Hub was unreachable", one per line.
#
# The first four come from the 2026-08-23 outage. The rest are the other ways
# a stream drops: a timeout, a reset, and the Hub's own 5xx. Each one is a
# fault that goes away by itself, which is what makes a re-fire worth a card.
#
# Nothing here matches a CUDA fault, a NaN or a missing file. A lane must
# still count those against its ladder, because a re-fire trains the same
# fault.
hub_outage_patterns(){
  cat <<'PATTERNS'
Failed to resolve
Temporary failure in name resolution
NameResolutionError
Couldn't reach
Max retries exceeded with url
MaxRetryError
requests.exceptions.ConnectionError
ConnectTimeoutError
ReadTimeoutError
Connection reset by peer
Connection aborted
Name or service not known
Network is unreachable
LocalEntryNotFoundError
502 Server Error
503 Server Error
504 Server Error
PATTERNS
}

# Does this text hold a Hub connection error? Reads the argument, or stdin
# when there is none.
hub_outage_in_text(){  # [text]
  local text
  if [ "$#" -gt 0 ]; then text="$1"; else text="$(cat)"; fi
  printf '%s\n' "$text" | grep -qF -f <(hub_outage_patterns)
}

# The same question over the END of a log. A trainer log grows across
# re-fires, so only the last lines belong to the leg that just died.
hub_outage_in_log(){  # <log> [lines]
  [ -f "${1:?log}" ] || return 1
  tail -n "${2:-$HUB_GATE_TAIL_LINES}" "$1" | hub_outage_in_text
}

# Does the Hub answer right now? `HUB_GATE_PROBE` replaces the read, which is
# how a test runs this without a network.
hub_is_up(){
  if [ -n "${HUB_GATE_PROBE:-}" ]; then
    eval "$HUB_GATE_PROBE" >/dev/null 2>&1
    return $?
  fi
  curl -fsS -I --max-time "$HUB_GATE_PROBE_TIMEOUT" -o /dev/null \
    "https://$HUB_GATE_HOST" 2>/dev/null
}

# The delay before probe number <try>. It doubles and then holds at the cap,
# so a short outage costs one minute and a long one costs one probe every
# quarter hour.
hub_backoff_delay(){  # <try> [base] [cap]
  local try="${1:?try}" base="${2:-$HUB_GATE_BASE_WAIT}"
  local cap="${3:-$HUB_GATE_MAX_WAIT}" d="$base" i=1
  while [ "$i" -lt "$try" ]; do
    d=$(( d * 2 ))
    [ "$d" -ge "$cap" ] && { d="$cap"; break; }
    i=$(( i + 1 ))
  done
  printf '%d\n' "$d"
}

# Block until the Hub answers. Returns 0 when it does, 1 at the deadline.
hub_wait_up(){  # [deadline seconds]
  local deadline="${1:-$HUB_GATE_DEADLINE}" waited=0 try=1 delay
  hub_is_up && return 0
  hub_gate_log "$HUB_GATE_HOST is down — waiting up to ${deadline}s for it"
  while [ "$waited" -lt "$deadline" ]; do
    delay="$(hub_backoff_delay "$try")"
    [ $(( waited + delay )) -gt "$deadline" ] && delay=$(( deadline - waited ))
    sleep "$delay"
    waited=$(( waited + delay ))
    try=$(( try + 1 ))
    if hub_is_up; then
      hub_gate_log "$HUB_GATE_HOST is up again after ${waited}s"
      return 0
    fi
    hub_gate_log "$HUB_GATE_HOST still down after ${waited}s (probe $try)"
  done
  hub_gate_log "$HUB_GATE_HOST is still down after ${waited}s — giving up"
  return 1
}
