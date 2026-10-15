#!/bin/bash
# #425 — put the code of the last commit in the code folder of the linear
# queue of elisa, ~/checkpoints_backup/cf-425-lin/code.
#
# queue_elisa.sh reads its code there, not in a checkout: a checkout in /tmp
# goes away at a restart of elisa, and a checkout can change while a queue
# runs. The folder holds the scripts of the queue, `src`, the name of the
# commit (DEPLOYED_COMMIT) and the Hugging Face token of the data streams.
#
# bash reads a script while it runs it. So the deploy refuses when a queue
# holds the queue lock of the results folder, or when a process runs a file
# of the code folder.
#
# Usage, on elisa, in a checkout of the branch:  bash deploy_elisa.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE="${CF425_ELISA_BASE:-$HOME/checkpoints_backup/cf-425-lin}"
CODE="$BASE/code"
TOKEN="${CF425_HF_TOKEN_FILE:-$REPO/experiments/hf_token.txt}"
# What a queue reads: the head trainer and the eval, the B4 runner with its
# config costs, the scripts of this report, and the library.
PATHS=(experiments/2026-04-13_gift-eval/scripts
       reports/2026-08-08_rollout_depth/scripts
       reports/2026-08-08_rollout_depth/results/config_costs.csv
       reports/2026-10-06_encoder_reconstruction/scripts
       scripts src)

[ -s "$TOKEN" ] || { echo "ABORT: no Hugging Face token at $TOKEN" >&2; exit 2; }
if [ -e "$BASE/results/queue.lock" ] && ! flock -n "$BASE/results/queue.lock" true; then
  echo "ABORT: a queue runs with the code of $CODE (it holds results/queue.lock). Stop it first." >&2
  exit 3
fi
if pgrep -f "$CODE/" >/dev/null; then
  echo "ABORT: a process of a queue runs a file of $CODE. Wait for its end." >&2
  exit 3
fi
[ -z "$(git -C "$REPO" status --porcelain -- "${PATHS[@]}")" ] \
  || echo "NOTE: the checkout has changes with no commit. The deploy takes the last commit only."

new="$CODE.new.$$"
rm -rf "$new" "$CODE.old"
mkdir -p "$new"
git -C "$REPO" archive HEAD "${PATHS[@]}" | tar -xf - -C "$new"
git -C "$REPO" rev-parse --short=8 HEAD >"$new/DEPLOYED_COMMIT"
cp "$TOKEN" "$new/experiments/hf_token.txt"
[ ! -e "$CODE" ] || mv "$CODE" "$CODE.old"
mv "$new" "$CODE"
rm -rf "$CODE.old"
echo "deployed $(cat "$CODE/DEPLOYED_COMMIT") to $CODE"
