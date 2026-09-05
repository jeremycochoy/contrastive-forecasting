#!/bin/bash
# #407 review gap 1 — the head-seed band at one stop.
#
# The card compares six numbers. Each one is one head, drawn with one head
# seed, and evaluated once. Nothing in that plan measures the spread of a
# score, so a 0.005 move between two stops reads as a result when it can be
# a head draw.
#
# This script draws the SAME head again on the SAME backbone, and changes
# only `--seed`. The backbone costs nothing: it is already on disk. One
# draw costs 30,000 head steps (about 35 min on a 4090) and one 97-config
# GIFT-Eval (about 72 min on 4 of elisa's cores).
#
# The seeds are 20260723 and 20260724, the two replicate seeds
# `experiments/2026-08-04_ema_sched_ladder/scripts/seed_replicates.sh` drew
# against the protocol seed 20260722. A seed from that set keeps this band
# inside the family the project's published +-0.0384 band was measured over.
#
# Everything else is `head_eval_bb.sh`'s, unchanged: same trainer, same
# 30,000 steps, same --grad-clip 1.0, same ARCH lists, same 97 configs
# under the official B4 strategy, same forecast horizon 16.
#
# Layout. One chain per seed: student, then teacher. The chains run at the
# same time. `head_vram_gate` holds a per-card flock, so only one head
# trains on the GPU at once and the other chain sits in its GIFT-Eval on
# the CPU. Two chains cap this study at 2 concurrent evals, which is 8 of
# elisa's 32 cores.
#
# Usage: replicate_heads.sh <stop steps> [seed ...]
#
#   BB_GPU=1 WT=<checkout> CF373_ROOT=<durable root> \
#     nohup setsid bash replicate_heads.sh 200000 > rep_200k.out 2>&1 &
#
# Exit codes: 1 a draw failed, 2 bad input, 3 no backbone at that stop.
set -uo pipefail

STOP="${1:?usage: replicate_heads.sh <stop steps> [seed ...]}"
case "$STOP" in ''|*[!0-9]*) echo "ABORT: bad stop '$STOP'" >&2; exit 2;; esac
shift
SEEDS=("$@")
[ "${#SEEDS[@]}" -gt 0 ] || SEEDS=(20260723 20260724)
STOP_K=$(( STOP / 1000 ))

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
PARENT="$WT/reports/2026-08-08_rollout_depth/scripts"
export CF373_ROOT="${CF373_ROOT:-/home/jupyter/cf373_r3/sync}"
export BB_GPU="${BB_GPU:-1}"
# elisa's card 1 carries another session's job. The default gate of 7,000
# MiB does not fit beside it, and the head's own peak is under this.
export HEAD_VRAM_MIB="${HEAD_VRAM_MIB:-6400}"
export HEAD_STEPS_N="${HEAD_STEPS_N:-30000}"

CELL_ID=A4
K=3
LOG="$RES/replicate_${STOP_K}k.log"
log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [rep${STOP_K}k] $*" | tee -a "$LOG"; }

. "$PARENT/cell_paths.sh"
BB="$(cf373_bb_ckpt "$CELL_ID" "$K" "$STOP")"
[ -n "$BB" ] && [ -f "$BB" ] || {
  log "ABORT: no bb${STOP_K}k backbone for $CELL_ID k=$K under $CF373_ROOT"; exit 3; }

# Pin the copy every draw read. The protocol-seed draw at 200k ran on a
# rented box and read that box's copy, which was released before it could
# be checksummed. So this records THIS band's input, and the report says
# which draws share a machine.
md5sum "$BB" | tee -a "$RES/replicate_${STOP_K}k_backbone_md5.txt"
log "backbone $BB"
log "seeds ${SEEDS[*]}  heads student teacher  ${HEAD_STEPS_N} steps  gpu $BB_GPU"

# One seed's two draws, in series.
chain(){ # <seed>
  local seed="$1" enc tag rc
  for enc in student teacher; do
    tag="${CELL_ID}_k${K}_bb${STOP_K}k_${enc}_s${seed}"
    log "DRAW $tag start"
    HEAD_SEED="$seed" bash "$PARENT/head_eval_bb.sh" \
      "$tag" "$BB" "$enc" "$HEAD_STEPS_N"
    rc=$?
    if [ $rc -ne 0 ]; then
      log "DRAW $tag rc=$rc — the band loses this draw"
      return 1
    fi
    log "DRAW $tag DONE $(cat "$WT/reports/2026-08-08_rollout_depth/results/score_${tag}.txt" 2>/dev/null)"
  done
  return 0
}

pids=(); names=()
for seed in "${SEEDS[@]}"; do
  chain "$seed" & pids+=($!); names+=("$seed")
done

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { log "chain seed ${names[$i]} failed"; fail=1; }
done

# The read-back, the moment this band drains. `read_back.sh` brings the
# draws into the checkout, refreshes the band, the teacher track and the
# figure, and ends with the mirror. The watchdog runs the same script every
# hour, so a band that dies before this line still reads back inside an hour.
bash "$HERE/read_back.sh" "$STOP_K" >>"$LOG" 2>&1 || log "WARN: read_back rc=$?"
python3 "$HERE/head_band.py" --stop "$STOP" --results "$RES" \
  --parent "$WT/reports/2026-08-08_rollout_depth/results" | tee -a "$LOG"

[ "$fail" -eq 0 ] || exit 1
log "band drained at ${STOP_K}k"
