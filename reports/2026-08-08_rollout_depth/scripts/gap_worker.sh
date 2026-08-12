#!/bin/bash
# #373 — run one row of gap_jobs.tsv end to end: backbone, then both heads
# and both GIFT-Evals.
#
# The study's first pass rented GPUs for the backbones and kept elisa for
# the heads and the evals. This pass has no credit, so everything runs here.
# One 4090 carries two backbones at a time — the models are d_model=64 at
# batch 64 and a single one leaves the card about half idle — and the evals
# stay on the cores, where they cost no VRAM.
#
# The heads do not wait for the next backbone: as soon as a row's checkpoint
# lands, its two heads are handed to a background subshell and the worker
# takes the next row. Head training is ~35 GPU-minutes and the eval is ~1 CPU
# hour, so serialising them behind the queue would idle the cores for hours.
#
# Usage: bash gap_worker.sh <worker index> <total workers>
#        BB_GPU=0 bash gap_worker.sh 0 2
#
# Both arguments are labels for the log. The workers share one claim queue
# and take rows top-down, so the reviewer's order is what runs first and a
# cheap row cannot leave a worker idle behind an expensive one.
set -uo pipefail

WORKER="${1:?usage: gap_worker.sh <worker index> <total workers>}"
NWORKERS="${2:?total workers}"
: "$NWORKERS"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$WT/reports/2026-08-08_rollout_depth/results"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
JOBS="${JOBS:-$HERE/gap_jobs.tsv}"
BB_GPU="${BB_GPU:-0}"
mkdir -p "$RES"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [w$WORKER] $*" \
  | tee -a "$RES/gaps_driver.log"; }

# The run name the launcher will build. Group A's is a template; group B's
# lives in the launcher's own `case` block, so it is read out of the
# launcher rather than copied into a second table that can drift.
run_name(){ # <launcher> <arg> <k> <suffix>
  local launcher="$1" arg="$2" k="$3" suf="$4" name
  case "$launcher" in
    run_leg_k.sh) printf 'cf393_%s_cf373k%s%s\n' "$arg" "$k" "$suf" ;;
    *)
      name="$(awk -v a="  $arg)" '
        $0 == a { grab = 1; next }
        grab && /^ *NAME=/ { sub(/^ *NAME="/, ""); sub(/"$/, ""); print; exit }
      ' "$HERE/$launcher")"
      [ -n "$name" ] || return 2
      printf '%s_cf373k%s%s\n' "$name" "$k" "$suf" ;;
  esac
}

runs_dir(){ # <launcher> <arg> <k> <suffix> <steps>
  case "$1" in
    run_leg_k.sh) printf '%s/%s/leg_%dk\n' "$CF373_ROOT" "$2" "$(( $5 / 1000 ))" ;;
    *)            printf '%s/%s\n' "$CF373_ROOT" "$(run_name "$1" "$2" "$3" "$4")" ;;
  esac
}

# The card with the most free memory right now. Two backbones hold ~11 GB of
# GPU 0 between them and a head needs 5.3 GB, so a head pinned to the
# training card would wait behind the queue it is supposed to run beside.
# head_eval_bb.sh's own VRAM gate then polls whichever card this returns.
pick_gpu(){
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '{ if ($2 > best) { best = $2; g = $1 } } END { print (g == "" ? 0 : g) }'
}

# Both heads and both evals of one finished backbone. Runs detached from the
# queue, so the next backbone starts immediately.
heads_for(){ # <id> <ckpt>
  local id="$1" bb="$2" enc rc hgpu
  for enc in student teacher; do
    for attempt in 1 2; do
      hgpu="$(pick_gpu)"
      log "HEAD START $id $enc (attempt $attempt, gpu $hgpu)"
      BB_GPU="$hgpu" HEAD_VRAM_MIB="${HEAD_VRAM_MIB:-6000}" \
      EVAL_SHARDS="${EVAL_SHARDS:-6}" \
      CF393_EVAL_SLOTS="${CF393_EVAL_SLOTS:-3}" \
        bash "$HERE/head_eval_bb.sh" "${id}_bb40k_${enc}" "$bb" "$enc" 15000
      rc=$?
      log "HEAD END   $id $enc rc=$rc"
      [ "$rc" -eq 0 ] && break
      # head_eval_bb.sh is idempotent in both halves — it skips a head whose
      # final checkpoint exists and the eval resumes per shard — so a retry
      # costs only what did not finish, and not retrying throws away 15,000
      # GPU steps for a transient.
      [ "$attempt" -eq 2 ] && log "GIVING UP on $id $enc"
    done
  done
}

# A shared claim queue rather than a round-robin split of the table. The
# rows differ by 3x in cost — B5 at k = 3 is the study's most expensive
# backbone and the two `L_align`-only rows its cheapest — so a fixed split
# leaves one worker idle for hours while the other still has the queue.
# `mkdir` is the atomic claim: it succeeds for exactly one caller.
CLAIMS="${CLAIMS:-$CF373_ROOT/gap_claims}"
mkdir -p "$CLAIMS"
claim(){ mkdir "$CLAIMS/$1" 2>/dev/null; }

while :; do
  taken=""
  while IFS=$'\t' read -r id gap launcher arg k suf seed gargs steps; do
    case "$id" in ''|\#*) continue;; esac
    if claim "$id"; then taken="$id"; break; fi
  done < "$JOBS"
  [ -n "$taken" ] || break

  # Re-read the claimed row: the loop above ended on it, but `read` inside a
  # pipeline leaves the fields set only by luck of the break point.
  IFS=$'\t' read -r id gap launcher arg k suf seed gargs steps < <(
    awk -F'\t' -v t="$taken" '$1 == t { print; exit }' "$JOBS")

  [ "$suf"   = "-" ] && suf=""
  [ "$gargs" = "-" ] && gargs=""
  name="$(run_name "$launcher" "$arg" "$k" "$suf")" || { log "ABORT: no name for $id"; continue; }
  dir="$(runs_dir "$launcher" "$arg" "$k" "$suf" "$steps")"
  ck="$dir/${name}_$(( steps / 1000 ))k.pth"

  if [ -f "$ck" ]; then
    log "BB SKIP $id — $(basename "$ck") already on disk"
  else
    log "BB START $id (gap $gap) launcher=$launcher arg=$arg k=$k seed=$seed suffix='${suf:-none}' gap_args='${gargs:-none}' gpu=$BB_GPU"
    args=("$arg")
    case "$launcher" in run_leg_k.sh) args+=("$steps") ;; esac
    K="$k" SEED="$seed" RUN_SUFFIX="$suf" GAP_ARGS="$gargs" \
    TARGET_STEPS="$steps" FINAL_STEPS=200000 \
    WT="$WT" BB_GPU="$BB_GPU" \
      bash "$HERE/$launcher" "${args[@]}"
    rc=$?
    log "BB END   $id rc=$rc"
    if [ ! -f "$ck" ]; then
      log "FAILED $id — no checkpoint at $ck; skipping its heads"
      continue
    fi
  fi

  ( heads_for "$id" "$ck" >/dev/null 2>&1 & )
  log "heads for $id handed off"
done < "$JOBS"

log "worker $WORKER drained"
