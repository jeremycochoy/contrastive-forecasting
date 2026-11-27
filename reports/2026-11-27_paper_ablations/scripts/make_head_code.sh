#!/bin/bash
# Make the score code of the rows 6 and 8 of #482 on elisa: the score code of
# #425 (hp64/ff/code, the head code that scored the reference) with the patch
# of the commit HEAD_SHA (branch run/cf-482-heads). With the patch, the head
# trainer and the GIFT-Eval script read the patch encoder and the kernel of
# the convolution from the checkpoint. check_head_code.py shows that the
# patched code builds the same backbone for the reference checkpoint. The
# folder gets the read token of the card (cf-482/secrets), not the shared one.
#   HEAD_SHA=<sha> [TOKEN=<token file>] bash make_head_code.sh
set -euo pipefail
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
REPO="${REPO:-/home/jupyter/contrastive-forecasting}"
HC=/home/jupyter/cf_runs/hp64/ff/code
SHA="$(git -C "$REPO" rev-parse "${HEAD_SHA:?HEAD_SHA: the patch commit}")"
TOKEN="${TOKEN:-$CF/secrets/hf_token_agent.txt}"
[ -s "$TOKEN" ] || { echo "no token in $TOKEN"; exit 2; }
DIR="$CF/ff_code_${SHA:0:8}"
[ ! -e "$DIR" ] || { echo "$DIR exists"; exit 2; }
rsync -a --exclude '__pycache__' "$HC/" "$DIR/"
git -C "$REPO" diff "$SHA~1" "$SHA" -- src experiments | (cd "$DIR" && patch -p1 --forward --no-backup-if-mismatch --quiet)
for f in $(git -C "$REPO" diff --name-only "$SHA~1" "$SHA" -- src experiments); do
  cmp -s <(git -C "$REPO" show "$SHA:$f") "$DIR/$f" || { echo "ABORT: $f differs from $SHA"; exit 2; }
done
echo "$(cat "$HC/DEPLOYED_COMMIT") + #482 patch $SHA" > "$DIR/DEPLOYED_COMMIT"
install -m 600 "$TOKEN" "$DIR/experiments/hf_token.txt"
sum=$(cd "$DIR" && find . -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum | cut -c1-32)
echo "$(date -u +%FT%TZ) $DIR: $HC with the patch $SHA, md5 of the tree $sum"
