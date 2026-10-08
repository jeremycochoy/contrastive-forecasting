#!/bin/bash
# #425 — compare each box file of the card with its copy on elisa, by byte
# size. Run it on elisa when the queue of the box has ended and a tick of
# sync_box.sh has passed.
#
# It prints each box file that elisa lacks or holds at another size, the
# count of equal files, and the count of the `.pth` files that stay on the
# box. The prune deletes the box copy of a `.pth` only when elisa holds it
# at the same byte size. So "0 not equal" and "0 `.pth` on the box" mean
# that elisa holds each head and each score, and the box has no more work.
#
# Usage, on elisa:  bash verify_mirror.sh | tee ../results/mirror_check.txt
set -uo pipefail

HOST="${CF425_HOST:-root@ssh5.vast.ai}"
PORT="${CF425_PORT:-31200}"
CK_ROOT="${CF425_BOX_ROOT:-/workspace/ckpt/cf-425}"
BOX_RES="${CF425_RES:-/workspace/results/cf-425}"
MIRROR="${CF425_MIRROR:-$HOME/checkpoints_backup/cf-412/vast_lr100x}/cf-425"
RES_MIRROR="${CF425_RESULTS_MIRROR:-$HOME/checkpoints_backup/cf-425/box_results}"
read -r -a SSH <<<"${CF425_SSH:-ssh -p $PORT -o ConnectTimeout=20 $HOST}"

compare(){  # <box dir> <elisa dir>
  local equal=0 other=0 bytes rel got
  while IFS=$'\t' read -r bytes rel; do
    [ -n "$rel" ] || continue
    got=$(stat -c %s "$2/$rel" 2>/dev/null || echo missing)
    if [ "$got" = "$bytes" ]; then
      equal=$(( equal + 1 ))
    else
      other=$(( other + 1 )); echo "NOT EQUAL $1/$rel: box $bytes, elisa $got"
    fi
  done < <("${SSH[@]}" "cd '$1' && find . -type f ! -path './locks/*' \
    ! -name '*.tmp' ! -name '*.lock' ! -name 'elisa_manifest.tsv' \
    -printf '%s\t%P\n'" </dev/null)
  echo "$1: $equal equal, $other not equal"
}

echo "[$(date -u '+%m-%d %H:%M:%S') UTC] box files against elisa"
compare "$BOX_RES" "$RES_MIRROR"
compare "$CK_ROOT" "$MIRROR"
echo "$CK_ROOT: $("${SSH[@]}" "find '$CK_ROOT' -name '*.pth' | wc -l" </dev/null) .pth on the box"
