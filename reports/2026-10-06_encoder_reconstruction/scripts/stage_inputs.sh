#!/bin/bash
# #425 — copy each input checkpoint of jobs.tsv that the box lacks from
# elisa's mirror of the box, ~/checkpoints_backup/cf-412/vast_lr100x.
#
# The box deleted only the checkpoints that elisa holds, so the mirror has
# each input. A copy lands as `<name>.tmp`, and it takes its name only when
# its size is the size in jobs.tsv. A cut transfer leaves no file under the
# real name. Run it on elisa, before queue.sh, and again after a failure:
# it copies only what is missing.
#
# Usage, on elisa:  bash stage_inputs.sh            # CF425_DRY_RUN=1 lists only
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${CF425_JOBS:-$HERE/jobs.tsv}"
MIRROR="${CF425_MIRROR:-$HOME/checkpoints_backup/cf-412/vast_lr100x}"
HOST="${CF425_HOST:-root@ssh5.vast.ai}"
PORT="${CF425_PORT:-31200}"
CK="${CF425_CK:-/workspace/ckpt}"
SSH=(ssh -p "$PORT" -o ConnectTimeout=20 -o ServerAliveInterval=30 "$HOST")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 stage] $*"; }

# "<bytes> <ckpt>" of each input, from the table.
inputs(){ awk -F'\t' '!/^#/ && NF >= 6 { print $6, $5 }' "$JOBS"; }

# The box size of each input: "<ckpt> <bytes or missing>".
box_sizes(){
  inputs | "${SSH[@]}" "cd '$CK' && while read -r want ckpt; do
      echo \"\$ckpt \$(stat -c %s \"\$ckpt\" 2>/dev/null || echo missing)\"; done" \
    2>/dev/null
}

copy_one(){  # <ckpt> <bytes>
  local ckpt="$1" want="$2" have
  have=$(stat -c %s "$MIRROR/$ckpt" 2>/dev/null || echo missing)
  [ "$have" = "$want" ] || { log "ABORT: elisa holds $have bytes of $ckpt, want $want"; return 1; }
  # The loop of the caller reads stdin, so no command here may read it.
  "${SSH[@]}" "mkdir -p '$CK/$(dirname "$ckpt")'" </dev/null || return 1
  scp -q -P "$PORT" -o ConnectTimeout=20 "$MIRROR/$ckpt" "$HOST:$CK/$ckpt.tmp" \
    </dev/null || return 1
  "${SSH[@]}" "s=\$(stat -c %s '$CK/$ckpt.tmp'); [ \"\$s\" = '$want' ] && mv '$CK/$ckpt.tmp' '$CK/$ckpt'" \
    </dev/null || { log "size check failed for $ckpt"; return 1; }
  log "copied $ckpt ($want bytes)"
}

listing="$(box_sizes)"
[ -n "$listing" ] || { log "ABORT: no listing from $HOST"; exit 2; }
missing=0; failed=0
while read -r want ckpt; do
  have=$(awk -v c="$ckpt" '$1 == c { print $2 }' <<<"$listing")
  [ "$have" = "$want" ] && continue
  missing=$(( missing + 1 ))
  log "needs $ckpt (box: ${have:-?}, want $want)"
  [ -n "${CF425_DRY_RUN:-}" ] && continue
  copy_one "$ckpt" "$want" || failed=$(( failed + 1 ))
done < <(inputs)
log "done: $missing input(s) to copy, $failed failed"
[ "$failed" -eq 0 ]
