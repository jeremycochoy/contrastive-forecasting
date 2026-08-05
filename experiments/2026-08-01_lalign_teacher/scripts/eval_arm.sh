#!/bin/bash
# #390 — 2L quantile head + GIFT-Eval B4 on one arm's backbone snapshot.
#
# The measurement half of a wave. #379's `eval_2L_gm_mase.sh` verbatim except
# for three things it hardcodes and #390 cannot share: the checkout, the
# experiment dir, and the arm → run-name resolution (that one now comes from
# `arm_names.sh` instead of an awk over run_arm.sh's case block).
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train ARM=arm5 BB_GPU=0 \
#     BB_STEP_K=40 HEAD_STEPS=15000 [BB_CHECKPOINT=<path>] bash eval_arm.sh
#
# BB_CHECKPOINT names the backbone file directly. Needed only when a run was
# resumed and both `<name>_<K>k.pth` and `<name>_r<N>_<K>k.pth` exist: the
# resolver refuses to guess between them. It must be one of those two names —
# a file from another step or another run is refused rather than published
# under this cell's name.
#
# The cell carries what decides its number — the backbone replicate and the
# head seed — so two measurements never share a directory, a head or an
# aggregate:
#
#   arm5_bb40k_hd15000s            the base run's 40k backbone, wave seed
#   arm5_bb40k_r3_hd15000s         the same arm's third resume, same step
#   arm5_s20260723_bb40k_hd15000s  the same backbone, another head seed
#
# The base run's token and the wave seed's token are both empty, so every
# cell name the report cites is unchanged. A cell already measured from one
# replicate, or under one seed, is never reused for another.
#
# The wave decides (BB_STEP_K, HEAD_STEPS):
#   wave 1 |  40k backbone, 15 000-step head
#   wave 2 | 100k backbone, 30 000-step head  (fresh head, new output dir)
#   wave 3 | 200k backbone, 30 000-step head  (fresh head, new output dir)
#
# A head is "fresh" because the output dir is keyed on the cell name, which
# carries both numbers — nothing from an earlier wave is on that path to
# resume from.
#
# GIFT-Eval must cover all 97 B4 configs. `--resume` fills in from the
# partial all_results.csv, so a killed eval continues rather than restarting;
# this script retries until the row count is 97 and exits non-zero if it
# never gets there. It never reports a partial.
#
# Exit codes. 2-6 are `resolve_eval_checkpoint.sh`'s, propagated verbatim
# (5 = ambiguous checkpoint, 6 = the override names another run or step —
# both mean "name the replicate you mean"). This script's own start at 20, so
# no number stands for two different operator actions:
#
#   20 the environment is wrong (checkout, trainer, token, resolver, library)
#   21 the head trainer exited 0 and wrote no checkpoint
#   22 GIFT-Eval never reached 97 configs — re-run it
#   23 97 rows on disk and no Aggregate line in gift/summary.txt
#   24 the aggregate is not over 97 configs
#   25 the resolver returned a path this run cannot name a cell from
#      (E_BAD_TAG, from the cell-identity library — #379's eval uses the
#      same number for the same condition)
#
# Anything else is the head trainer's own status, and the `head-train rc=`
# line right above it in the log says so.
set -uo pipefail

E_SETUP=20; E_NO_HEAD=21; E_PARTIAL=22; E_NO_AGGREGATE=23
E_AGG_SCOPE=24

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit $E_SETUP ;;
esac

# `cd -P`, because the orchestrators and the documented usage reach this file
# through a `scripts/` symlink inside $WT: the logical path would walk back up
# to $WT and load that checkout's resolver, which can sit on any commit. A
# missing one aborts loudly; an older one answers, silently.
HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -P "$HERE/../../.." && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

# The cell-identity library, loaded before the knobs below because it holds
# the head-seed default they take: the same value decides whether the cell
# name carries a `_s<seed>` token, so a second copy of it here would rename
# every cell the moment the two drifted. It also carries E_BAD_TAG.
CELL_IDENTITY="$ROOT/scripts/eval_cell_identity.sh"
[ -f "$CELL_IDENTITY" ] || { echo "ABORT: no cell-identity library at $CELL_IDENTITY" >&2; exit $E_SETUP; }
# shellcheck source=/dev/null
source "$CELL_IDENTITY"

ARM="${ARM:?set ARM=<slug>}"
BB_GPU="${BB_GPU:?set BB_GPU=0 or 1}"
BB_STEP_K="${BB_STEP_K:-40}"
HEAD_STEPS="${HEAD_STEPS:-15000}"
N_CONFIGS_EXPECTED=97
MAX_EVAL_TRIES="${MAX_EVAL_TRIES:-3}"
# Two knobs the review added, both defaulted to what every wave already ran:
#   HEAD_SEED — the head's init/data seed. Review item 4 measures the spread
#               over seeds; the waves all used the library's default,
#               #379's value. It is part of the cell name, so a seed cannot
#               land on another seed's directory, head or aggregate.
#   CELL_TAG  — appended to the arm slug, for a variation the seed does not
#               name: the student control's own backbone (`_alignstudent`).
HEAD_SEED="${HEAD_SEED:-$EVAL_DEFAULT_HEAD_SEED}"
CELL_TAG="${CELL_TAG:-}"
# The replicate to evaluate, when a resumed run left more than one backbone at
# this step. Empty means "resolve it", which only succeeds when there is one.
BB_CHECKPOINT="${BB_CHECKPOINT:-}"

EXP="$WT/experiments/2026-08-01_lalign_teacher"
RUNS="$EXP/runs"
OUT_ROOT="$EXP/eval_gm_mase"
HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
GEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"

export PYTHONPATH="$WT"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"

[ -f "$HEAD_TRAIN" ] || { echo "ABORT: head trainer not at $HEAD_TRAIN" >&2; exit $E_SETUP; }
[ -f "$GEVAL" ]      || { echo "ABORT: gift-eval not at $GEVAL" >&2; exit $E_SETUP; }
[ -n "$HF_TOKEN" ]   || { echo "ABORT: empty HF_TOKEN" >&2; exit $E_SETUP; }

NAME="$(bb_name "$ARM")" || exit $E_SETUP

# The snapshot for this wave. A resumed run writes a fresh `_r<N>` safe run
# name, so one (name, step) pair can leave several backbones on disk; the
# resolver aborts rather than pick between them, and prints what it chose.
# It comes from this script's own checkout ($ROOT), never from $WT — see the
# `cd -P` note above.
CKPT_RESOLVER="$ROOT/scripts/resolve_eval_checkpoint.sh"
[ -f "$CKPT_RESOLVER" ] || { echo "ABORT: no checkpoint resolver at $CKPT_RESOLVER" >&2; exit $E_SETUP; }
BB=$(bash "$CKPT_RESOLVER" "$RUNS" "$NAME" "$BB_STEP_K" "$BB_CHECKPOINT") || exit $?

# The cell is named after everything that decides its number: the backbone it
# cites, replicate included, and the head seed. Without a token a second
# replicate or a second seed lands in the first one's directory, where
# head-train SKIPs on the first one's head and the 97-row check lifts its
# aggregate — the run then reports a number it never measured and exits 0.
# Both tokens are empty at the wave's own values, so the committed cell names
# do not move.
REPL_TAG="$(replicate_tag "$NAME" "$BB_STEP_K" "$BB")" || {
  echo "ABORT: resolved checkpoint is not '$NAME' at ${BB_STEP_K}k: $BB" >&2
  exit $E_BAD_TAG; }

CELL="$(eval_cell_name "${ARM}${CELL_TAG}" "$BB_STEP_K" "$REPL_TAG" "$HEAD_STEPS" "$HEAD_SEED")"
OUT="$OUT_ROOT/$CELL"; mkdir -p "$OUT"
HEAD_NAME="qhead_2L_${NAME}_bb${BB_STEP_K}k${REPL_TAG}"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/eval.log"
CSV="$OUT/gift/all_results.csv"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [$CELL] $*" | tee -a "$LOG"; }

# Data rows in all_results.csv (header excluded); 0 when the file is absent.
n_rows(){ [ -f "$CSV" ] && awk 'END{print (NR>0 ? NR-1 : 0)}' "$CSV" || echo 0; }

say "start on GPU $BB_GPU (backbone=$BB)"

# Backbone arch args accepted by train_forecasting_head.py (it reads the
# remaining backbone flags out of the checkpoint config).
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)
# GIFT-Eval reconstructs freq/seasonality embedding dims from the checkpoint
# and rejects the flags; everything else it takes.
ARCH_EVAL=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

# --- 1) fresh 2L transformer quantile head ---------------------------------
if [ ! -f "$HEAD_CKPT" ]; then
  say "head-train ${HEAD_STEPS} steps"
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$HEAD_TRAIN" \
    --backbone-path "$BB" \
    --device cuda \
    --quantile-head --grad-clip 1.0 \
    --forecast-len 16 --batch-size 256 --lr 1e-3 \
    --total-steps "$HEAD_STEPS" --save-every 5000 --log-every 500 \
    --save-dir "$OUT" --run-name "$HEAD_NAME" --seed "$HEAD_SEED" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --head-arch transformer --head-num-layers 2 --head-nhead 8 \
    --head-ffn-mult 4.0 --head-causal true --head-train-input e_then_f \
    --head-dropout 0.1 \
    "${ARCH_HEAD[@]}" >>"$LOG" 2>&1
  rc=$?
  say "head-train rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  [ -f "$HEAD_CKPT" ] || { say "ABORT: head trainer exited 0 but wrote no $HEAD_CKPT"; exit $E_NO_HEAD; }
else
  say "head-train SKIP (final head already on disk)"
fi

# --- 2) GIFT-Eval B4, all 97 configs ---------------------------------------
# `--resume` reads the partial CSV, so a retry costs only the missing configs.
try=1
while [ "$try" -le "$MAX_EVAL_TRIES" ]; do
  have=$(n_rows)
  if [ "$have" -ge "$N_CONFIGS_EXPECTED" ]; then break; fi
  say "gift-eval try $try/$MAX_EVAL_TRIES (have $have/$N_CONFIGS_EXPECTED configs)"
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$GEVAL" \
    --backbone-path "$BB" \
    --head-path "$HEAD_CKPT" \
    --output-dir "$OUT/gift" \
    --strategy B4 --forecast-len 16 --resume \
    "${ARCH_EVAL[@]}" \
    --head-nhead 8 --head-causal true \
    >>"$LOG" 2>&1
  say "gift-eval try $try rc=$? rows=$(n_rows)"
  try=$((try + 1))
done

have=$(n_rows)
if [ "$have" -ne "$N_CONFIGS_EXPECTED" ]; then
  say "FAILED — $have/$N_CONFIGS_EXPECTED configs after $MAX_EVAL_TRIES tries. No partial reported."
  exit $E_PARTIAL
fi

# Aggregate: the eval script tees one `Aggregate GM-Relative MASE (N configs)`
# line into its own summary.txt. Lift it to the cell root and one level up, so
# a wave's numbers are one `cat` away.
AGG=$(grep -h "Aggregate GM-Relative MASE" "$OUT/gift/summary.txt" 2>/dev/null | tail -1)
if [ -z "$AGG" ]; then
  say "FAILED — 97 rows on disk but no Aggregate line in gift/summary.txt"
  exit $E_NO_AGGREGATE
fi
case "$AGG" in
  *"($N_CONFIGS_EXPECTED configs)"*) : ;;
  *) say "FAILED — aggregate is not over $N_CONFIGS_EXPECTED configs: $AGG"; exit $E_AGG_SCOPE ;;
esac
# The number and the file it came from, together. `eval.log` is appended to
# across retries and is not what the analysis reads; these two are, and
# `collect_artefacts.sh` copies them into the report directory. The aggregate
# stays the first line, so every reader keeps working.
{ echo "$AGG"; echo "backbone: $BB"; } > "$OUT/summary.txt"
cp -f "$OUT/summary.txt" "$OUT_ROOT/${CELL}_summary.txt"
say "DONE — $AGG (backbone=$BB)"
