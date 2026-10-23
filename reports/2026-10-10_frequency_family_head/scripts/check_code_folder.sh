#!/bin/bash
# freq_family — check a deployed code folder before the waves read it.
#
# The script starts run_wave.sh, follow_abc_gift.sh and collect_scores.py of
# the code folder on a scratch base folder. It shows that the code reads the
# files of the code of ad1984a6, which wrote no protocol file:
#   A. The test wave of that code (<base>/smoke): 6 configs on the CPU.
#   B. A score as its BLK wave leaves it: 97 configs on the GPU. The eval
#      table is the table of b4_gpu_check.sh, and the log line is the line
#      that head_eval_bb.sh writes at an eval start.
#   C. The follower on the checkpoint of `abc_gift` at 40k.
#
# A stand-in trainer and a stand-in runner replace the head trainer and the
# eval. So the script uses no GPU and no data stream, and starts no wave. It
# reads the base folder of the waves and the folder of `abc_gift`, and
# writes in the scratch folder only.
#
# Usage, on elisa:  bash check_code_folder.sh <code folder> [scratch folder]
set -uo pipefail

C="$(realpath -ms "${1:?usage: check_code_folder.sh <code folder> [scratch folder]}")"
X="${2:-$(mktemp -d /tmp/ff_code_check.XXXXXX)}"
S="$C/reports/2026-10-10_frequency_family_head/scripts"
LIVE=/home/jupyter/cf_runs/freq_family
BB="$HOME/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k/leg_665k/cf419_cos200k_r2_200k.pth"
FILTER='^(bizitobs_service/short|ett1/15T/short|ett1/H/short|us_births/D/short|m4_weekly/short|m4_yearly/short)$'
[ -f "$C/DEPLOYED_COMMIT" ] || { echo "ABORT: $C is not a deployed code folder" >&2; exit 2; }
mkdir -p "$X" || exit 2
export CUDA_VISIBLE_DEVICES=
unset FF_CODE FF_BASE FF_EVAL_DEVICE FF_EVAL_FILTER FF_EVAL_EXPECT FF_RESCORE

show(){ cut -f1-7 "$1" | column -t -s "$(printf '\t')"; }
short(){ sed "s|$X|<scratch>|g" | cut -c1-210; }

# The stand-in runner: the flags of a head, or a score with the eval table
# and the eval log of one config.
cat >"$X/runner.sh" <<'EOF'
#!/bin/bash
tag="$1"; out="$CF373_ROOT/eval/$tag"; mkdir -p "$out"
if [ -n "${CF_HEAD_ARGV_TO:-}" ]; then
  printf '["--save-dir", "%s", "--run-name", "qhead_%s"]\n' "$out" "$tag" >>"$CF_HEAD_ARGV_TO"
  exit 0
fi
echo "[$(date '+%m-%d %H:%M:%S')] [$tag] eval start (97 configs, B4, forecast-len 16, ${EVAL_DEVICE:-cpu})" >>"$out/stop.log"
mkdir -p "$out/gift/shard_0"
printf 'dataset,model,eval_metrics/MASE[0.5]\nett1/H/short,m,1.5\n' >"$out/gift/shard_0/all_results.csv"
cp "$out/gift/shard_0/all_results.csv" "$out/gift/all_results.csv"
echo "  [eval] ett1/H/short: frequency H, family member 32" >"$out/gift/shard_0/shard.log"
echo 1.1000 >"$CF_RESULTS/score_$tag.txt"
EOF
# The stand-in trainer: a stand-in final head for each job.
cat >"$X/trainer.py" <<'EOF'
import json
import os
import sys

jobs = [json.loads(line) for line in open(sys.argv[sys.argv.index("--jobs") + 1])]
for argv in jobs:
    out, name = argv[argv.index("--save-dir") + 1], argv[argv.index("--run-name") + 1]
    open(os.path.join(out, f"{name}_final.pth"), "w").write("stand-in head")
token = "set" if os.environ.get("HF_TOKEN") else "missing"
print(f"[shared] {len(jobs)} stand-in heads, token {token}")
EOF

echo "code folder: $C ($(cat "$C/DEPLOYED_COMMIT")). The waves that run read $(cat "$LIVE/code/DEPLOYED_COMMIT")."
echo
echo "== A. The test wave of the old code (6 configs on the CPU, no protocol file)"
mkdir -p "$X/a/results"
cp "$LIVE/smoke/results/arms.tsv" "$LIVE"/smoke/results/score_*.txt "$X/a/results/"
for dir in "$LIVE"/smoke/heads/eval/*/; do
  tag="$(basename "$dir")"
  mkdir -p "$X/a/heads/eval/$tag"
  cp "$dir/stop.log" "$X/a/heads/eval/$tag/"
  [ ! -d "$dir/gift" ] || cp -r "$dir/gift" "$X/a/heads/eval/$tag/"
  for head in "$dir"/*_final.pth; do
    echo "stand-in head" >"$X/a/heads/eval/$tag/$(basename "$head")"
  done
done
echo "-- A1. collect_scores.py of the code folder"
PYTHONPATH="$C" python3 "$S/collect_scores.py" --base "$X/a" | short
show "$X/a/results/scores.tsv"
echo "-- the table of the old code"
cut -f1-6 "$LIVE/smoke/results/scores.tsv" | column -t -s "$(printf '\t')"
echo "-- A2. run_wave.sh with the filter of the test wave: an old test score names no filter"
FF_BASE="$X/a" FF_HEAD_STEPS=500 FF_TRAIN=0 FF_EVAL_FILTER="$FILTER" FF_EVAL_EXPECT=6 \
  bash "$S/run_wave.sh" blk 200 "$BB" control shared_draw 2>&1 | short
echo "rc=${PIPESTATUS[0]}"

echo
echo "== B. A score as the BLK wave of the old code leaves it (97 configs on the GPU, no protocol file)"
tag=blk_bb200k_h30k_control
mkdir -p "$X/b/results" "$X/b/heads/eval/$tag"
cp -r "$LIVE/timing/blk200_b4_gpu/gift" "$X/b/heads/eval/$tag/gift"
cp "$LIVE/timing/score_blk200_b4_gpu.txt" "$X/b/results/score_$tag.txt"
echo "stand-in head" >"$X/b/heads/eval/$tag/qhead_${tag}_s20260722_final.pth"
echo "[10-10 14:10:00] [$tag] eval start (97 configs, B4, forecast-len 16, cuda)" \
  >"$X/b/heads/eval/$tag/stop.log"
echo "-- B1. FF_EVAL_DEVICE=cuda: the score is the work that is done"
FF_BASE="$X/b" FF_TRAIN=0 FF_EVAL_DEVICE=cuda bash "$S/run_wave.sh" blk 200 "$BB" control 2>&1 | short
echo "rc=${PIPESTATUS[0]}"
show "$X/b/results/scores.tsv"
echo "-- B2. the CPU default: the start refuses the GPU score"
FF_BASE="$X/b" FF_TRAIN=0 bash "$S/run_wave.sh" blk 200 "$BB" control 2>&1 | short
echo "rc=${PIPESTATUS[0]}"

echo
echo "== C. follow_abc_gift.sh on the checkpoint of abc_gift at 40k"
follow(){
  FF_BASE="$X/c" FF_STOPS=40 FF_WAIT_MAX=120 FF_POLL=5 FF_WAVE_VRAM_MIB=0 \
    FF_RUNNER="$X/runner.sh" FF_TRAINER="$X/trainer.py" \
    FF_EVAL_DEVICE=cuda FF_EVAL_FILTER='^ett1/H/short$' FF_EVAL_EXPECT=1 "$@" \
    bash "$S/follow_abc_gift.sh" control shared_strict 2>&1 | short
  echo "rc=${PIPESTATUS[0]}"
}
echo "-- C1. with FF_CODE=$C"
follow env FF_CODE="$C"
show "$X/c/results/scores.tsv"
echo "-- the trainer log of the wave"
cat "$X"/c/results/waves/*/train.log | short
echo "-- C2. a second start, with no FF_CODE: the script reads its own code folder"
follow env
cut -f1-3,5 "$X/c/results/checkpoints.tsv" | short
echo "-- C3. a start on another device"
follow env FF_EVAL_DEVICE=cpu
