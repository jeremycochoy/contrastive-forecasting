#!/bin/bash
# #393 — the bb100k heads again, at two more seeds, for the six cells whose
# branch turned on a sub-noise difference.
#
# Usage:  WT=<checkout> [RUNS=<durable root>] [CF393_SEED_JOBS=4] \
#           bash scripts/seed_replicates.sh [--list] [--status]
#         bash scripts/seed_replicates.sh --one <cell> <enc> <seed> <gpu>
#
# WHY. The extend rule reads one number per (cell, stop, head) and compares
# bb40k against bb100k. The 2026-08-04 study measured head-seed ranges up to
# 0.0908 on this protocol. Six of this card's ten cells moved less than that
# on BOTH heads, `arm6_v2_combab_alignT` by 0.0026 on the student, so with
# one seed there is nothing in the run that separates a move from noise.
# Three seeds per cell per head give an in-study spread, which is what the
# report plots as error bars and what decides which branches survive.
#
# WHAT IS HELD FIXED. Everything except the head seed: the same bb100k
# backbone checkpoint (backbones are NOT retrained), the same encoder
# pairing rule, 30,000 head steps, 97 GIFT-Eval configs, B4, forecast
# horizon 16, --grad-clip 1.0, and the same seasonal-naive denominator.
# eval_stop.sh takes HEAD_SEED and files a non-default seed under
# `bb100k_<enc>_s<seed>`, so seed 20260722's artefacts are untouched.
#
# THE SEEDS. 20260723 and 20260724 — the protocol seed 20260722 plus one and
# plus two. They are written here, in the job list, and they appear in every
# path and every row they produce, so a rerun reproduces them exactly.
#
# ORDER. `arm6_v2_combab_alignS` and `alignT` first, both seeds, both heads:
# they are the study's two best cells and the card names them first. Then
# arm5, then arm6_v2_nse.
#
# CONCURRENCY. Each job is a GPU phase (the head) then a CPU phase (the
# eval), so N jobs destagger themselves rather than all wanting the same
# resource. CF393_SEED_JOBS caps how many run here. Launching this script a
# second time while the first is still running is safe and is how the pool
# is widened: a job is claimed by `mkdir`, which is atomic, so no two
# workers take the same one.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="${WT:-$(dirname "$(dirname "$EXP")")}"
export WT
RES="$EXP/results"
LOG="$RES/seed_replicates.log"

# The two replicate seeds and the protocol seed they are compared against.
PROTOCOL_SEED=20260722
SEEDS=(20260723 20260724)

# The six cells, in the order the card asks for. Only bb100k: that is the
# stop the extend rule fires at, and the bb40k side of every delta is the
# same number for all three seeds' comparison.
CELLS=(arm6_v2_combab_alignS arm6_v2_combab_alignT
       arm5_combab_alignS arm5_combab_alignT
       arm6_v2_nse_alignT arm6_v2_nse_alignS)
STOP=100000
HEAD_STEPS=30000

CLAIMS="${CF393_SEED_CLAIMS:-/tmp/cf393_seed_claims}"
JOBS="${CF393_SEED_JOBS:-4}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [seeds] $*" | tee -a "$LOG"; }

# The job list, in priority order. One line per (cell, encoder, seed, gpu).
# Both seeds of a cell sit next to each other so a cell completes rather
# than half-completing across the whole grid.
job_list(){
  local i=0 cell enc seed
  for cell in "${CELLS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for enc in student teacher; do
        echo "$cell $enc $seed $(( i % 2 ))"
        i=$(( i + 1 ))
      done
    done
  done
}

# Where the score lands. Matches scores_from_evals.py's `score_<stopdir>.txt`
# convention, so the file names stay readable next to the seed-20260722 ones.
score_path(){  # <cell> <enc> <seed>
  local root="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
  local sfx=""; [ "$3" != "$PROTOCOL_SEED" ] && sfx="_s$3"
  printf '%s/%s/eval/score_bb%dk_%s%s.txt\n' "$root" "$1" "$(( STOP / 1000 ))" "$2" "$sfx"
}

run_one(){  # <cell> <enc> <seed> <gpu>
  local cell="$1" enc="$2" seed="$3" gpu="$4"
  local out; out="$(score_path "$cell" "$enc" "$seed")"

  if [ -s "$out" ]; then
    say "SKIP $cell $enc s$seed — score $(cat "$out")"
    return 0
  fi
  # Atomic claim. mkdir fails if the directory exists, so exactly one worker
  # wins a job even when two drivers race on the same line.
  mkdir -p "$CLAIMS" 2>/dev/null
  if ! mkdir "$CLAIMS/${cell}_${enc}_s${seed}" 2>/dev/null; then
    say "claimed already: $cell $enc s$seed"
    return 0
  fi

  say "START $cell $enc s$seed gpu=$gpu"
  local t0=$SECONDS
  HEAD_SEED="$seed" BB_GPU="$gpu" \
    bash "$HERE/eval_stop.sh" "$cell" "$STOP" "$enc" "$HEAD_STEPS" "$out" \
    >>"$RES/seed_${cell}_${enc}_s${seed}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    say "DONE  $cell $enc s$seed in $(( (SECONDS - t0) / 60 ))m — $(cat "$out")"
  else
    # Release the claim so a later pass retries. A head that trained but
    # whose eval died is picked up where it stopped: eval_stop.sh skips a
    # head whose _final.pth exists and eval_local.sh resumes its shards.
    rmdir "$CLAIMS/${cell}_${enc}_s${seed}" 2>/dev/null
    say "FAIL  $cell $enc s$seed rc=$rc — claim released, see seed_${cell}_${enc}_s${seed}.log"
  fi
  return $rc
}

status(){
  local cell enc seed out n=0 done_=0
  printf '%-24s %-8s %-9s %s\n' cell head seed score
  for cell in "${CELLS[@]}"; do
    for seed in "$PROTOCOL_SEED" "${SEEDS[@]}"; do
      for enc in student teacher; do
        out="$(score_path "$cell" "$enc" "$seed")"
        n=$(( n + 1 ))
        if [ -s "$out" ]; then
          done_=$(( done_ + 1 ))
          printf '%-24s %-8s %-9s %s\n' "$cell" "$enc" "$seed" "$(cat "$out")"
        else
          printf '%-24s %-8s %-9s %s\n' "$cell" "$enc" "$seed" \
            "$([ -d "$CLAIMS/${cell}_${enc}_s${seed}" ] && echo running || echo -)"
        fi
      done
    done
  done
  echo
  echo "$done_/$n scored (36 = six cells x two heads x three seeds)"
}

case "${1:-}" in
  --list)   job_list; exit 0 ;;
  --status) status; exit 0 ;;
  --one)    shift; run_one "$@"; exit $? ;;
esac

[ -d "${RUNS:-/home/jupyter/checkpoints_backup/cf-393}" ] || {
  echo "ABORT: RUNS is not a directory" >&2; exit 2; }
say "pool of $JOBS over $(job_list | wc -l) job(s); claims in $CLAIMS"
job_list | xargs -P "$JOBS" -L 1 bash "$HERE/seed_replicates.sh" --one
say "pool drained"
status | tee -a "$LOG"
