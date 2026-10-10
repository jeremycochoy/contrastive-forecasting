#!/bin/bash
# freq_family — put the code of the last commit in the code folder of the
# waves, /home/jupyter/cf_runs/freq_family/code.
#
# run_wave.sh and follow_abc_gift.sh read their code there, not in a
# checkout: a checkout in /tmp goes away at a restart of elisa, and a
# checkout can change while a wave runs. The folder holds the head trainer
# and the eval, the B4 runner with its config costs, the scripts of this
# report, the library, the name of the commit (DEPLOYED_COMMIT) and the
# Hugging Face token of the data stream.
#
# bash reads a script while it runs it. So the deploy refuses when a process
# runs a file of the code folder: a wave, a score or the follower.
#
# Usage, on elisa, in a checkout of the branch:  bash deploy.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
CODE="$BASE/code"
TOKEN="${FF_HF_TOKEN_FILE:-$REPO/experiments/hf_token.txt}"
PATHS=(experiments/2026-04-13_gift-eval/scripts
       reports/2026-08-08_rollout_depth/scripts
       reports/2026-08-08_rollout_depth/results/config_costs.csv
       reports/2026-10-10_frequency_family_head/scripts
       scripts src)

[ -s "$TOKEN" ] || { echo "ABORT: no Hugging Face token at $TOKEN" >&2; exit 2; }
if pgrep -f "$CODE/" >/dev/null; then
  echo "ABORT: a process runs a file of $CODE (a wave, a score or the follower). Wait for its end." >&2
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
