#!/bin/bash
# #393 — run one GIFT-Eval on elisa's CPUs, sharded, and write its score.
#
# Usage: eval_local.sh <cell> <stop_k> <student|teacher> \
#                      <backbone> <head_ckpt> <out_dir> <score_out>
#
# This is the CPU half of the split decided on PR #394: head training
# stays on a rented GPU, GIFT-Eval moves here. Two measurements put it
# here.
#
#   The eval does not need a GPU. Six configs on elisa: 58.7 s on a 4090,
#   97.3 s on one CPU core. 1.66x for a job that then takes no VRAM, no SM
#   time, and leaves both cards to the cells training on them. Per core,
#   32 cores beat two contended GPUs by a wide margin.
#
#   The eval is not 5.2 h. That figure came from extrapolating the first
#   four configs, which are among the slowest of the 97. A completed
#   97-config eval of this protocol totals 3.70 h, and on this study's
#   d_model=64 backbone it is 2.86 core-hours here. Config cost spans
#   0.4 s to 1537 s, so the shard split is by measured cost — see
#   scripts/shard_configs.py.
#
# Each shard writes its own all_results.csv. They are concatenated and the
# aggregate is produced by re-running the official script with --resume
# over the full 97: it finds every config already done, computes nothing,
# and writes summary.txt with the same code that would have written it
# unsharded. Re-deriving GM-Relative MASE here instead would be a second
# implementation of the study's one metric.
set -uo pipefail

CELL="${1:?usage: eval_local.sh <cell> <stop_k> <enc> <bb> <head> <out> <score>}"
STOP_K="${2:?stop in thousands}"
ENC="${3:?student|teacher}"
BB="${4:?backbone checkpoint}"
HEAD_CKPT="${5:?head checkpoint}"
OUT="${6:?output dir}"
SCORE_OUT="${7:?score output path}"

case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac
[ -f "$BB" ] || { echo "ABORT: no backbone at $BB" >&2; exit 3; }
[ -f "$HEAD_CKPT" ] || { echo "ABORT: no head at $HEAD_CKPT" >&2; exit 3; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/leg_paths.sh"
. "$HERE/eval_slot.sh"

WT="${WT:-/tmp/contrastive-forecasting-393}"
GEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
[ -f "$GEVAL" ] || { echo "ABORT: no eval script at $GEVAL" >&2; exit 2; }

EVAL_SHARDS="${EVAL_SHARDS:-4}"
# The full protocol is 97 configs, and the merge count below is what proves
# the split lost none. A caller that runs a SUBSET says so through these two,
# together: the regex it wants, and how many configs that regex covers.
#
# #401 uses them to run one config, so the head half of its pipeline can be
# exercised end to end in minutes rather than in three core-hours. Unset,
# this is the 97-config protocol, unchanged, and every published number of
# both studies comes from that path.
EVAL_CONFIG_FILTER="${EVAL_CONFIG_FILTER:-}"
EVAL_EXPECT_CONFIGS="${EVAL_EXPECT_CONFIGS:-97}"
# One filter cannot be split across shards: each shard would run the same
# regex and the merge would count every config as many times as there are
# shards.
[ -n "$EVAL_CONFIG_FILTER" ] && EVAL_SHARDS=1
GIFT="$OUT/gift"
LOG="$OUT/eval_local.log"
mkdir -p "$GIFT" "$(dirname "$SCORE_OUT")" || exit 2

export PYTHONPATH="$WT"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
# The eval reads GIFT_EVAL from local disk, so it needs no HF token and no
# network. That is also why sharding does not multiply into a rate limit.
[ -d "$GIFT_EVAL" ] || { echo "ABORT: no GIFT_EVAL data at $GIFT_EVAL" >&2; exit 2; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Matches eval_stop.sh. GIFT-Eval rebuilds the freq / seasonality embedding
# dims from the checkpoint, so it takes the shorter list.
ARCH=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
      --num-layers 3 --encoder-type gru
      --rev-norm-kind ewma --rev-norm-span 128
      --head-nhead 8 --head-causal true)

if [ -s "$SCORE_OUT" ]; then
  log "SKIP $CELL bb${STOP_K}k $ENC — score already at $SCORE_OUT"
  exit 0
fi

eval_slot || { log "ABORT: no eval slot for $CELL bb${STOP_K}k $ENC"; exit 1; }
log "START $CELL bb${STOP_K}k enc=$ENC shards=$EVAL_SHARDS slot=${CF393_EVAL_SLOT_HELD:-?}"

# Repair a shard CSV that an interrupted run left unreadable.
#
# eval_gift_eval_official.py reads all_results.csv for --resume and then
# REOPENS it "w", so a process killed in that window leaves a zero-byte
# file. The next --resume calls next(reader) on it and dies with
# StopIteration — and so does every retry after that, forever, on a head
# that already cost 15,000 GPU steps. Three shards were in exactly this
# state after a broker was killed mid-eval on 2026-08-05.
#
# A kill can also truncate the row being written, so short rows go too.
# Dropping a good row costs one config's recompute; keeping a malformed
# one puts a wrong number in the aggregate.
sanitise_shard_csv() {  # <all_results.csv>
  local f="$1" ncol
  [ -e "$f" ] || return 0
  if [ ! -s "$f" ] || [ -z "$(head -1 "$f")" ]; then
    log "  shard CSV $f is empty — removing so --resume starts clean"
    rm -f "$f"
    return 0
  fi
  ncol=$(head -1 "$f" | awk -F, '{print NF}')
  awk -F, -v n="$ncol" 'NR==1 || NF==n' "$f" > "$f.clean" || return 0
  if ! cmp -s "$f" "$f.clean"; then
    log "  shard CSV $f had $(( $(wc -l <"$f") - $(wc -l <"$f.clean") )) malformed row(s) — dropped"
    mv -f "$f.clean" "$f"
  else
    rm -f "$f.clean"
  fi
}

pids=(); tags=()
for (( s = 0; s < EVAL_SHARDS; s++ )); do
  if [ -n "$EVAL_CONFIG_FILTER" ]; then
    filt="$EVAL_CONFIG_FILTER"
  else
    filt="$(python3 "$HERE/shard_configs.py" --shards "$EVAL_SHARDS" --shard "$s")" \
      || { log "ABORT: shard $s plan failed"; exit 4; }
  fi
  sdir="$GIFT/shard_$s"
  mkdir -p "$sdir"
  sanitise_shard_csv "$sdir/all_results.csv"
  # One core each: torch defaults to a thread per core, and 4 shards x 32
  # threads on a shared box is how you take a machine down. Measured: 1
  # thread 97.3 s, 4 threads 81.4 s for the same six configs — four times
  # the cores for 16% less wall clock, so per-core throughput says one.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES="" \
  python3 -u "$GEVAL" \
    --backbone-path "$BB" --head-path "$HEAD_CKPT" \
    --encoder-source "$ENC" --output-dir "$sdir" \
    --strategy B4 --forecast-len 16 --resume --device cpu \
    --config-filter "$filt" "${ARCH[@]}" >>"$sdir/shard.log" 2>&1 &
  pids+=($!); tags+=("$s")
done

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { log "shard ${tags[$i]} rc=$? — see $GIFT/shard_${tags[$i]}/shard.log"; fail=1; }
done
[ "$fail" -eq 0 ] || { log "ABORT: a shard failed"; exit 5; }

# Merge. The header is taken from the first shard; every shard writes the
# same CSV_HEADER, so a differing one means the shards did not run the
# same code and the merge must not paper over it.
MERGED="$GIFT/all_results.csv"
hdr=""
: >"$MERGED.tmp"
for (( s = 0; s < EVAL_SHARDS; s++ )); do
  f="$GIFT/shard_$s/all_results.csv"
  [ -s "$f" ] || { log "ABORT: shard $s wrote no all_results.csv"; exit 5; }
  h="$(head -1 "$f")"
  if [ -z "$hdr" ]; then hdr="$h"; printf '%s\n' "$hdr" >>"$MERGED.tmp"
  elif [ "$h" != "$hdr" ]; then log "ABORT: shard $s header differs"; exit 5; fi
  tail -n +2 "$f" >>"$MERGED.tmp"
done

# Exactly EVAL_EXPECT_CONFIGS distinct configs — 97 for the protocol — or
# the split lost or doubled one. The shard regexes are anchored and generated
# from one partition, so this should never fire; it is here because a merged
# CSV that is quietly short still produces a plausible-looking aggregate over
# fewer configs.
n_rows=$(( $(wc -l <"$MERGED.tmp") - 1 ))
n_uniq=$(tail -n +2 "$MERGED.tmp" | cut -d, -f1 | sort -u | wc -l)
if [ "$n_rows" -ne "$EVAL_EXPECT_CONFIGS" ] || [ "$n_uniq" -ne "$EVAL_EXPECT_CONFIGS" ]; then
  log "ABORT: merged CSV has $n_rows rows / $n_uniq configs, want $EVAL_EXPECT_CONFIGS/$EVAL_EXPECT_CONFIGS"
  exit 6
fi
mv "$MERGED.tmp" "$MERGED"
log "merged $n_rows configs -> $MERGED"

# All the configs are on disk, so this pass evaluates nothing. It exists to
# write summary.txt through the official aggregation, against the same
# seasonal-naive denominator every other stop in this study used.
#
# The filter goes to this pass too. Without it a subset run would find its
# own configs done and then evaluate the other 96 here, which is the whole
# cost the subset exists to avoid.
AGG_FILTER=()
[ -n "$EVAL_CONFIG_FILTER" ] && AGG_FILTER=(--config-filter "$EVAL_CONFIG_FILTER")
python3 -u "$GEVAL" \
  --backbone-path "$BB" --head-path "$HEAD_CKPT" \
  --encoder-source "$ENC" --output-dir "$GIFT" \
  --strategy B4 --forecast-len 16 --resume --device cpu \
  "${AGG_FILTER[@]}" "${ARCH[@]}" >>"$LOG" 2>&1
rc=$?
[ $rc -eq 0 ] || { log "ABORT: aggregate pass rc=$rc"; exit 7; }

score_from_summary "$GIFT/summary.txt" > "$SCORE_OUT.tmp" \
  || { rm -f "$SCORE_OUT.tmp"; log "ABORT: no score in $GIFT/summary.txt"; exit 4; }
mv "$SCORE_OUT.tmp" "$SCORE_OUT"
log "DONE $CELL bb${STOP_K}k $ENC — $(grep -h 'Aggregate GM-Relative MASE' \
  "$GIFT/summary.txt") -> $(cat "$SCORE_OUT")"
