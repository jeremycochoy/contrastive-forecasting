#!/bin/bash
# Make the code folder of the ablation runs (#482) on elisa: the src/ folder and
# the trainer folder of the run commit (git archive), and the Hugging Face token
# file. The layout is the layout of hp64/code_038ea26c, so the run scripts and
# setup_box.sh use it the same way. The script prints no token.
# The token is the read token of the card (cf-482/secrets), not the shared one.
#   SHA=<run commit> [TOKEN=<token file>] bash make_code_folder.sh
set -euo pipefail
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
REPO="${REPO:-/home/jupyter/contrastive-forecasting}"
SHA="${SHA:?SHA: the run commit (branch run/cf-482-ablations)}"
TOKEN="${TOKEN:-$CF/secrets/hf_token_agent.txt}"
[ -s "$TOKEN" ] || { echo "no token in $TOKEN"; exit 2; }
DIR="$CF/code_$(git -C "$REPO" rev-parse --short=8 "$SHA")"
[ ! -e "$DIR" ] || { echo "$DIR exists"; exit 2; }
mkdir -p "$DIR"
git -C "$REPO" archive "$SHA" src experiments/2026-04-27_freq-embedding/scripts | tar -x -C "$DIR"
install -m 600 "$TOKEN" "$DIR/experiments/hf_token.txt"
sum=$(cd "$DIR" && find . -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum | cut -c1-32)
echo "$(date -u +%FT%TZ) $DIR: commit $(git -C "$REPO" rev-parse "$SHA"), md5 of the tree $sum"
