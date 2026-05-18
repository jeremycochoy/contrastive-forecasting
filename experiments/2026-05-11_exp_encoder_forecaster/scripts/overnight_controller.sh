#!/bin/bash
# Overnight 150k controller — executes OVERNIGHT_PLAN_2026-05-15.md as an
# idempotent state machine. Never resets an optimizer (from-scratch =
# continuous optimizer; v27c = warm-resume from its intact companion).
# Re-entrant: every step skips if its FINAL/summary already exists.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
CK="$MAIN/checkpoints"
EXP="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"
RES="$EXP/results"
LOG="$RES/overnight_controller.log"
SEED=20260516
mkdir -p "$RES"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
dgate(){ local f; f=$(df -BG / | awk 'NR==2{gsub("G","",$4);print $4}'); if [ "$f" -lt 40 ]; then log "DISK LOW (${f}G) — pausing 600s for cron cleanup"; sleep 600; f=$(df -BG / | awk 'NR==2{gsub("G","",$4);print $4}'); if [ "$f" -lt 40 ]; then log "DISK STILL LOW (${f}G) — ABORT step"; return 1; fi; fi; return 0; }

COMMON=(--device cuda --batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --save-every 5000 --save-dir "$CK" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1"
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 --num-encoder-layers 6
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers --depthwise-conv 3 --deprecated-depthwise-conv 0
  --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3
  --rev-norm-kind ewma --rev-norm-span 128 --tau 0.10 --loss-shape cosine_similarity_batch --encoder-type gru)

# ---- backbone builders (each leaves <name>_FINAL.pth = best_loss) ----
backbone_done(){ [ -f "$CK/$1_FINAL.pth" ]; }

train_v20_150k(){ # dk0.9 from scratch, two-phase, new seed; GPU $1
  local gpu="$1" A="enc_fcst_v20_phaseA_fp32warmup_5k_v150" B="enc_fcst_v20_freshwarmup_fp16_150k"
  backbone_done "$B" && { log "v20-150k FINAL exists — skip"; return 0; }
  dgate || return 1
  if [ ! -f "$CK/${A}_5k.pth" ]; then
    log "v20-150k Phase A (fp32 0->5k, seed $SEED) GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
      --seed $SEED --encoder-dropkey 0.9 --total-steps 5000 --run-name "$A" \
      --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
      >>"$RES/run_${B}.log" 2>&1 || { log "v20-150k Phase A FAILED"; return 1; }
  fi
  [ -f "$CK/${A}_5k.pth" ] || { log "v20-150k Phase A _5k missing"; return 1; }
  dgate || return 1
  log "v20-150k Phase B (warm-resume ${A}_5k, fp16 body 5k->150k) GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey 0.9 --total-steps 150000 --run-name "$B" --resume "$CK/${A}_5k.pth" \
    --patch-emb-dtype fp32 --residual-dtype fp16 --attn-dtype fp16 --ffn-dtype fp16 \
    >>"$RES/run_${B}.log" 2>&1 || { log "v20-150k Phase B FAILED"; return 1; }
  cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth" && log "v20-150k DONE"
}

train_v27c_150k(){ # warm-resume v27c 50k(+opt) -> 150k; GPU $1
  local gpu="$1" B="enc_fcst_v27c_dk08_ffnfp16_150k"
  backbone_done "$B" && { log "v27c-150k FINAL exists — skip"; return 0; }
  dgate || return 1
  log "v27c-150k warm-resume 50k(+optimizer) -> 150k GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --encoder-dropkey 0.8 --total-steps 150000 --run-name "$B" \
    --resume "$CK/enc_fcst_v27c_dk08_ffnfp16_resume25k_50k_50k.pth" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp16 \
    --log-attn-amplitude --log-attn-amplitude-every 1000 \
    >>"$RES/run_${B}.log" 2>&1 || { log "v27c-150k FAILED"; return 1; }
  cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth" && log "v27c-150k DONE"
}

train_fp32_from0_150k(){ # v16/v17 from scratch new seed pure-fp32; $1=dk $2=name $3=gpu
  local dk="$1" B="$2" gpu="$3"
  backbone_done "$B" && { log "$B FINAL exists — skip"; return 0; }
  dgate || return 1
  log "$B from-scratch (seed $SEED, dk$dk, fp32) 0->150k GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py "${COMMON[@]}" \
    --seed $SEED --encoder-dropkey "$dk" --total-steps 150000 --run-name "$B" \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    >>"$RES/run_${B}.log" 2>&1 || { log "$B FAILED"; return 1; }
  cp -f "$CK/${B}_best_loss.pth" "$CK/${B}_FINAL.pth" && log "$B DONE"
}

# ---- q-head + triage + full-eval at a given backbone checkpoint ----
qhead_eval(){ # $1=backbone_path $2=tag $3=gpu
  local bb="$1" tag="$2" gpu="$3"
  local qn="${tag}_qhead_xfmr2L_quant_30k" qf="$CK/${qn}_FINAL.pth"
  local fout="$RES/gift_eval_full_${tag}"
  [ -f "$fout/summary.txt" ] && { log "$tag full-eval exists — skip (GM=$(gm "$fout/summary.txt"))"; return 0; }
  [ -f "$bb" ] || { log "$tag backbone missing ($bb) — skip"; return 1; }
  dgate || return 1
  if [ ! -f "$qf" ]; then
    log "$tag q-head train (30k) GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
      --backbone-path "$bb" --forecast-len 16 --quantile-head --head-arch transformer --head-causal true \
      --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f \
      --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
      --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 --save-every 5000 --log-every 200 \
      --save-dir "$CK" --run-name "$qn" --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
      --device cuda --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
      --encoder-type gru --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
      --reconstruction forecaster --amp-dtype bf16 >>"$RES/run_${qn}.log" 2>&1 || { log "$tag qhead FAILED"; return 1; }
    if [ -f "$CK/${qn}_best.pth" ]; then cp -f "$CK/${qn}_best.pth" "$qf"
    elif [ -f "$CK/${qn}_final.pth" ]; then cp -f "$CK/${qn}_final.pth" "$qf"
    else cp -f "$(ls -t "$CK/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  fi
  [ -f "$qf" ] || { log "$tag qhead FINAL missing"; return 1; }
  dgate || return 1
  log "$tag full-eval (97 cfg) GPU$gpu"
  mkdir -p "$fout"
  CUDA_VISIBLE_DEVICES=$gpu HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
    python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$bb" --head-path "$qf" --output-dir "$fout" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_full_eval_${tag}.log" 2>&1
  [ -f "$fout/summary.txt" ] && log "$tag full-eval DONE GM=$(gm "$fout/summary.txt")" || { log "$tag full-eval NO SUMMARY"; return 1; }
}

# per-arm pipeline: backbone -> evals at 50k,150k,100k (user order). $1=arm
arm_pipeline(){
  local arm="$1" gpu="$2" bbname
  case "$arm" in
    v20)  bbname="enc_fcst_v20_freshwarmup_fp16_150k"; train_v20_150k "$gpu" || return 1 ;;
    v27c) bbname="enc_fcst_v27c_dk08_ffnfp16_150k";   train_v27c_150k "$gpu" || return 1 ;;
    v16)  bbname="enc_fcst_v16_dk07_150k";  train_fp32_from0_150k 0.7  "$bbname" "$gpu" || return 1 ;;
    v17)  bbname="enc_fcst_v17_dk095_150k"; train_fp32_from0_150k 0.95 "$bbname" "$gpu" || return 1 ;;
  esac
  # 50k: v27c reuse existing gift_eval_full_v27c; others use own _50k periodic
  if [ "$arm" = v27c ]; then log "v27c 50k = existing gift_eval_full_v27c (GM=$(gm "$RES/gift_eval_full_v27c/summary.txt"))"
  else qhead_eval "$CK/${bbname}_50k.pth"  "${arm}x50k"  "$gpu"; fi
  qhead_eval "$CK/${bbname}_FINAL.pth" "${arm}x150k" "$gpu"
  qhead_eval "$CK/${bbname}_100k.pth"  "${arm}x100k" "$gpu"
  log "arm $arm pipeline COMPLETE"
}

# ================= ORCHESTRATION =================
# Fastest valid 150k = v27c warm-resume (optimizer intact, only +100k steps):
# run it NOW on the free GPU0. GPU1 is busy producing the v27c-50k full GM
# (missing #2); when that frees GPU1, run v20 dk0.9 from-scratch (the plan),
# then v17 from-scratch (spare, to adjudicate the user's bet).
log "=== overnight controller START ==="

# --- GPU0 track: v27c 150k warm-continue NOW (+ its 100k/150k evals) ---
( arm_pipeline v27c 0 ) >>"$LOG" 2>&1 &
P_GPU0=$!
log "GPU0: v27c 150k warm-resume launched NOW (pid $P_GPU0) — fastest valid 150k"

# --- GPU1 track: wait for v27c-50k chain to free GPU1, then v20 then v17 ---
(
  while :; do
    [ -f "$RES/gift_eval_full_v27c/summary.txt" ] \
      && ! pgrep -f 'post_qhead_chain_v27c|run_qhead_v27c' >/dev/null 2>&1 && break
    sleep 300
  done
  log "GPU1 free: v27c-50k chain done (v27c_full=$(gm "$RES/gift_eval_full_v27c/summary.txt")); v20R_full=$(gm "$RES/gift_eval_full_v20R/summary.txt")"
  log "GPU1: v20 dk0.9 from-scratch 150k (plan)"
  arm_pipeline v20 1
  log "GPU1: v17 dk0.95 from-scratch 150k (spare — adjudicates the v17&v27c<v11c bet)"
  arm_pipeline v17 1
) >>"$LOG" 2>&1 &
P_GPU1=$!

wait "$P_GPU0"; log "GPU0 track (v27c 150k) finished rc=$?"
wait "$P_GPU1"; log "GPU1 track (v20 + v17) finished rc=$?"

# ================= PHASE 4: scorecard =================
g(){ gm "$RES/gift_eval_full_$1/summary.txt"; }
{
  echo "=== 150k ordering scorecard ($(date)) ==="
  echo "dk0.9 (v20 recipe, fp32-warmup->fp16, ~3% penalty vs pure-fp32 v11c):"
  echo "  50k=$(g v20x50k)  100k=$(g v20x100k)  150k=$(g v20x150k)   [v11c pure-fp32 50k ref=1.292]"
  echo "v27c dk0.8:  50k=$(g v27c)  100k=$(g v27cx100k)  150k=$(g v27cx150k)"
  echo "v17  dk0.95: 50k=1.409  100k=$(g v17x100k)  150k=$(g v17x150k)"
  echo "v16  dk0.7:  50k=1.335  100k=$(g v16x100k)  150k=$(g v16x150k)"
  echo "--- BET (user): v17 & v27c < v11c(dk0.9) @150k ---"
  echo "  dk0.9@150k ref = $(g v20x150k) (note: v20 recipe carries ~3% fp16 penalty)"
  echo "  v17@150k = $(g v17x150k) ; v27c@150k = $(g v27cx150k)"
} | tee -a "$RES/overnight_scorecard.txt" | tee -a "$LOG"
log "=== overnight controller DONE ==="
