#!/bin/bash
# #425 — the parity test of the shared trainer on the box, before the queue.
#
# OMB 25k (patch sizes 8 to 128, mean/std) trains 1,000 steps in two ways:
#   solo  alone, with the round-1 code (5b15fe22 in /workspace/cf-425-r1),
#   wave  in queue.sh with the new code, beside BLK 200k, BMS 40k and OAF 40k
#         on GiftEvalPretrain, while the old-data lane trains LOW 665k and
#         TWN 100k.
# Then R scores the two OMB heads on the 4 configs of box_smoke.sh.
# parity_compare.py compares the loss CSVs: OMB 25k against its solo run,
# and BLK 200k, BMS 40k, OAF 40k and LOW 665k against the 500-step solo
# smoke heads of round 1.
#
# Every file goes to test folders that the queue never reads.
#
# Usage, on the box:  bash parity.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW="${CF425_CODE:-/workspace/cf-425}"
OLD="${CF425_OLD_CODE:-/workspace/cf-425-r1}"
CK=/workspace/ckpt
OUT="$CK/cf-425/parity"
RES=/workspace/results/cf-425/parity
SMOKE="$CK/cf-425/smoke/eval"
B4=reports/2026-08-08_rollout_depth/scripts
FILTER='^(m4_hourly/short|electricity/H/short|ett1/15T/long|us_births/D/short)$'
OMB=cf-412om/leg_25k/cf412om_k3_25k.pth
SHAPE="--d-model 384 --n-heads 8 --num-layers 3"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 parity] $*"; }

mkdir -p "$OUT" "$RES"
awk -F'\t' -v want=" OMB:25 BLK:200 BMS:40 OAF:40 LOW:665 TWN:100 " \
  '/^#code/ || index(want, " " $1 ":" $3 " ")' "$HERE/jobs.tsv" >"$RES/jobs.tsv"

SOLO="head_eval_bb.sh omb25_solo_recon"
if [ -f "$OUT/solo/eval/omb25_solo_recon/qhead_omb25_solo_recon_s20260722_final.pth" ] \
    || pgrep -f "$SOLO" >/dev/null; then
  log "solo: it ended or it runs, so it does not start again"
else
  log "solo: OMB 25k, 1,000 steps, code $(cat "$OLD/DEPLOYED_COMMIT")"
  WT="$OLD" CF373_ROOT="$OUT/solo" CF_RESULTS="$RES" CF_STOP_K=25 \
    CF_RECONSTRUCTION=encoder HEAD_SAVE_EVERY=1000000 CF_SKIP_EVAL=1 \
    CF_BB_SHAPE="$SHAPE" BB_GPU=0 HEAD_VRAM_MIB=4000 \
    GPU_GATE_LOCKDIR=/tmp/cf425_parity_solo GIFT_EVAL=/workspace/gift-eval-data \
    bash "$OLD/$B4/head_eval_bb.sh" omb25_solo_recon "$CK/$OMB" student 1000 \
    >"$RES/solo.out" 2>&1 </dev/null &
fi

log "waves: $(grep -vc '^#' "$RES/jobs.tsv") jobs, 1,000 steps, code $(cat "$NEW/DEPLOYED_COMMIT")"
CF425_CODE="$NEW" CF425_JOBS="$RES/jobs.tsv" CF425_HEAD_STEPS=1000 \
  CF425_SCORE=0 CF425_ROOT="$OUT/queue" CF425_RES="$RES/queue" \
  CF425_LANE_STAGGER=30 CF425_GPU_LOCK=/tmp/cf425_parity_gpu.lock \
  bash "$NEW/reports/2026-10-06_encoder_reconstruction/scripts/queue.sh" \
  >"$RES/queue.out" 2>&1
log "waves rc=$?"
while pgrep -f "$SOLO" >/dev/null; do sleep 20; done
log "solo: $(grep -h 'head-train rc' "$RES/stops.log" | tail -1)"

for name in omb25_solo_recon cf412om_bb25k_h30k_recon; do
  dir=solo; [ "$name" = omb25_solo_recon ] || dir=queue
  head="$OUT/$dir/eval/$name/qhead_${name}_s20260722_final.pth"
  WT="$NEW" EVAL_STRATEGY=R EVAL_DEVICE=cuda BB_GPU=0 EVAL_SHARDS=1 \
    EVAL_CONFIG_FILTER="$FILTER" EVAL_EXPECT_CONFIGS=4 CF_BB_SHAPE="$SHAPE" \
    CF393_EVAL_SLOTDIR=/tmp/cf425_parity_slots GIFT_EVAL=/workspace/gift-eval-data \
    bash "$NEW/$B4/eval_local.sh" "$name" 25 student "$CK/$OMB" "$head" \
    "$OUT/r4/$name" "$RES/score_r4_$name.txt" >>"$RES/r4.out" 2>&1 </dev/null
  log "R, 4 configs, $name: $(cat "$RES/score_r4_$name.txt" 2>/dev/null || echo FAILED)"
done

csv(){ echo "$1/qhead_$2_s20260722_losses.csv"; }
python3 "$HERE/parity_compare.py" \
  "OMB 25k" "$(csv "$OUT/queue/eval/cf412om_bb25k_h30k_recon" cf412om_bb25k_h30k_recon)" \
    "$(csv "$OUT/solo/eval/omb25_solo_recon" omb25_solo_recon)" \
  "BLK 200k" "$(csv "$OUT/queue/eval/cf419_cos200k_bb200k_h30k_recon" cf419_cos200k_bb200k_h30k_recon)" \
    "$(csv "$SMOKE/smoke_blk_s500_recon" smoke_blk_s500_recon)" \
  "BMS 40k" "$(csv "$OUT/queue/eval/cf419ms_bb40k_h30k_recon" cf419ms_bb40k_h30k_recon)" \
    "$(csv "$SMOKE/smoke_bms_s500_recon" smoke_bms_s500_recon)" \
  "OAF 40k" "$(csv "$OUT/queue/eval/cf412oa2_bb40k_h30k_recon" cf412oa2_bb40k_h30k_recon)" \
    "$(csv "$SMOKE/smoke_bank_s500_recon" smoke_bank_s500_recon)" \
  "LOW 665k" "$(csv "$OUT/queue/eval/k3_r100_09_lr56_fix09_dec10k_lr30x_bb665k_h30k_recon" k3_r100_09_lr56_fix09_dec10k_lr30x_bb665k_h30k_recon)" \
    "$(csv "$SMOKE/smoke_old_s500_recon" smoke_old_s500_recon)" \
  | tee "$RES/compare.txt"
grep -h '^\[shared\]' "$RES"/queue/waves/*/train.log | tail -4
