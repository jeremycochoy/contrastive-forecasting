#!/bin/bash
# #390 arm launcher — the 10 #379 cells that carry an L_align term, rerun
# with L_align pointed at the EMA teacher (`--align-target teacher`).
#
# Every arm below is its #379 counterpart's command line verbatim plus that
# one flag. Same backbone (d_model=64, n_heads=8, num_encoder_layers=3,
# num_layers=3), same batch size 64, same seed, same dataset. #379's other
# 20 cells contain no L_align term and are copied, not rerun.
#
# Backbone-only: trains the backbone, saves checkpoints, records training
# dynamics (loss / ff / u_batchtime / u_batchtime_e). Head training and
# GIFT-Eval are separate stages.
#
# Usage:
#   WT=/absolute/path/to/checkout BB_GPU=1 bash run_arm.sh <arm>
#
# where <arm> ∈ {arm5, arm5_tr1, arm5_nse, arm5_ncpc, arm5_combab,
#                arm6_v2, arm6_v2_tr1, arm6_v2_nse, arm6_v2_ncpc,
#                arm6_v2_combab}.
#
# Checkpoints land in this experiment's own runs/ dir AND carry the
# `_alignteacher` name suffix, so no #379 artefact can be overwritten.
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm5|arm5_tr1|arm5_nse|arm5_ncpc|arm5_combab|arm6_v2|arm6_v2_tr1|arm6_v2_nse|arm6_v2_ncpc|arm6_v2_combab>}"

# WT MUST be an absolute path under a persistent checkout — never /tmp.
# /tmp gets wiped by reboots and by `git worktree remove --force`
# (CLAUDE.md § Checkpoint Safety Rule 4, Apr-2026 incident). Fall back
# to the documented elisa layout when unset.
WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
case "$WT" in
  /tmp/*|/tmp)
    echo "ABORT: WT=$WT is under /tmp. Set WT to an absolute path under a" >&2
    echo "  persistent checkout (e.g. \$HOME/workspaces/contrastive-forecasting)." >&2
    exit 2
    ;;
esac

OUT="$WT/reports/2026-08-08_rollout_depth"
RES="$OUT/results"; mkdir -p "$RES"

# ---- Per-arm dispatch --------------------------------------------------------
# NAME:       base run name (all backbone artefacts prefix on it).
# ARM_DESC:   human-readable one-line description used in the BB START log line.
# LOSS_ARGS:  extra CLI flags added AFTER --loss-shape (loss-shape itself included).
# EXTRA_ARGS: flags appended at the END of the trainer invocation, so a
#             repeated flag (e.g. `--sigreg-embedding-weight 0.0` or
#             `--cpc-infonce-weight 0.0`) overrides the earlier default —
#             Python argparse keeps the LAST value on repeat.
#
# `--align-target teacher` sits in LOSS_ARGS of every arm: it is the one
# difference from #379 and must never be arm-specific.
EXTRA_ARGS=()
case "$ARM" in
  arm5)
    NAME="bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 5: rep_only + align-loss-weight 1.0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --align-target teacher)
    ;;
  arm6_v2)
    NAME="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 6 v2: rep_only + align-loss-weight 1.0 + moco-rep-keys + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --align-target teacher)
    ;;
  # ---- tau_rep=1.0 setting -------------------------------------------------
  arm5_tr1)
    NAME="bb_small_arm5_tr1_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 5_tr1: rep_only + align-loss-weight 1.0 + tau_rep=1.0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --tau-rep 1.0 --align-target teacher)
    ;;
  arm6_v2_tr1)
    NAME="bb_small_arm6_v2_tr1_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 6 v2_tr1: rep_only + align-loss-weight 1.0 + moco-rep-keys + tau_rep=1.0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --tau-rep 1.0 \
               --align-target teacher)
    ;;
  # ---- no-sigreg-embedding (nse) setting -----------------------------------
  arm5_nse)
    NAME="bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 5_nse: rep_only + align-loss-weight 1.0 + sigreg_embedding=0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --align-target teacher)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  arm6_v2_nse)
    NAME="bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 6 v2_nse: rep_only + align-loss-weight 1.0 + moco-rep-keys + sigreg_embedding=0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --align-target teacher)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  # ---- no-CPC (ncpc) setting -----------------------------------------------
  arm5_ncpc)
    NAME="bb_small_arm5_ncpc_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 5_ncpc: rep_only + align-loss-weight 1.0 + cpc_infonce_weight=0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --align-target teacher)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm6_v2_ncpc)
    NAME="bb_small_arm6_v2_ncpc_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 6 v2_ncpc: rep_only + align-loss-weight 1.0 + moco-rep-keys + cpc_infonce_weight=0 + align-target teacher"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --align-target teacher)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  # ---- combined-ablation (combab) setting ----------------------------------
  # combab = τ_rep raised to 1.0 + CPC-InfoNCE off. #379 skipped nse on
  # these two arms (nse hurt them there); that choice is copied verbatim.
  arm5_combab)
    NAME="bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 5_combab: rep_only + align + τ_rep=1.0 + cpc=0 + align-target teacher (nse SKIPPED: nse hurt arm5)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --tau-rep 1.0 --align-target teacher)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm6_v2_combab)
    NAME="bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
    ARM_DESC="arm 6 v2_combab: rep_only + align + moco-rep-keys + τ_rep=1.0 + cpc=0 + align-target teacher (nse SKIPPED: nse hurt arm6_v2)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --tau-rep 1.0 \
               --align-target teacher)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  *)
    echo "ABORT: unknown arm '$ARM'" >&2
    echo "  valid: arm5 arm5_tr1 arm5_nse arm5_ncpc arm5_combab" >&2
    echo "         arm6_v2 arm6_v2_tr1 arm6_v2_nse arm6_v2_ncpc arm6_v2_combab" >&2
    exit 2
    ;;
esac

SEED=20260520
# #373: rollout depth. Every cell of this study takes it from the SHARED
# flag block below, never from EXTRA_ARGS.
K="${K:-3}"
# Artefacts of this study never share a name with a published k = 0 one.
NAME="${NAME}_cf373k${K}"
# Checkpoints live on the durable root, never inside the checkout.
RUNS="${CF373_RUNS:-/home/jupyter/checkpoints_backup/cf-373}/$NAME"
mkdir -p "$RUNS"
LOG_EVERY="${LOG_EVERY:-200}"
# Staged-wave support, copied from #379's launcher.
#   TARGET_STEPS: per-launch training target (total_steps for this run).
#   FINAL_STEPS:  the arm's true final step. `_FINAL.pth` is only written
#                 when TARGET_STEPS ≥ FINAL_STEPS. During intermediate
#                 waves (40k / 100k), leave FINAL_STEPS=200000 so we
#                 finish this wave, stop cleanly, and let the next
#                 orchestrator invocation resume from the saved
#                 `_<N>k.pth` intermediate.
# STEPS is kept as an alias so callers using the older env var still work.
TARGET_STEPS="${TARGET_STEPS:-${STEPS:-200000}}"
STEPS="$TARGET_STEPS"
FINAL_STEPS="${FINAL_STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-25000}"
EXTRA_SAVES="${EXTRA_SAVES:-2500}"

# Wave-endpoint auto-backfill: on an intermediate wave, the next wave's
# launcher resumes from the newest `_<N>k.pth`, so we must have a snapshot
# at TARGET_STEPS. `--save-every` covers the endpoint when TARGET_STEPS is
# a multiple, but a caller passing an off-cadence target (or forgetting
# the extras field) would otherwise stop 1 save-every short. Idempotent —
# same-value entries are dedup'd by parse_extra_save_steps.
if [ "$TARGET_STEPS" -lt "$FINAL_STEPS" ]; then
  case ",$EXTRA_SAVES," in
    *",$TARGET_STEPS,"*) : ;;
    *) EXTRA_SAVES="${EXTRA_SAVES:+$EXTRA_SAVES,}$TARGET_STEPS" ;;
  esac
fi
NENC=3; NLAY=3
BB_GPU="${BB_GPU:-0}"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
DL_LOG="$RES/dl_${ARM}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [elisa-$ARM] $*" | tee -a "$DL_LOG"; }
[ -f "$TRAIN" ]  || { log "ABORT: TRAIN not at $TRAIN"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token missing at $HF_TOKEN_PATH"; exit 2; }
# The `_FINAL.pth` fallback at the end of this script picks a snapshot at a
# named step, and a resumed run can leave two of those — so it goes through
# the same resolver as the eval sites (#390). Taken from this script's own
# checkout, never from $WT, which can sit on a commit that predates it.
# Checked here, before the run: finding it missing after a 200k-step training
# run costs the whole run. `cd -P`, because the orchestrators reach this file
# through a `scripts/` symlink inside $WT — the logical path would walk back
# up to $WT and land on exactly the stale-checkout resolver this avoids.
CKPT_RESOLVER="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/resolve_eval_checkpoint.sh"
[ -f "$CKPT_RESOLVER" ] || { log "ABORT: no checkpoint resolver at $CKPT_RESOLVER"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

BB="$RUNS/${NAME}_FINAL.pth"
tlog="$RES/run_${NAME}.log"

if [ -f "$BB" ] && [ "${FORCE:-0}" != "1" ]; then
  log "BB SKIP ($NAME FINAL exists — set FORCE=1 to override)"
  exit 0
fi

# The highest step this run has a periodic snapshot for, over the base name
# and every `_r<N>` resume; -1 when there is none. Optimizer sidecars and
# non-numeric matches are not snapshots.
latest_step_k(){
  local f k best=-1
  for f in "$RUNS/${NAME}"_*k.pth; do
    [ -e "$f" ] || continue
    case "$f" in *_optimizer.pth) continue;; esac
    k=$(basename "$f" | sed -E 's/.*_([0-9]+)k\.pth$/\1/')
    case "$k" in ''|*[!0-9]*) continue;; esac
    (( k > best )) && best=$k
  done
  echo "$best"
}

# Wave idempotency: if an existing periodic checkpoint has already reached
# (or exceeded) TARGET_STEPS and this is an intermediate wave
# (TARGET_STEPS < FINAL_STEPS), skip re-running the trainer.
target_k=$(( TARGET_STEPS / 1000 ))
best_ck_k=$(latest_step_k)
if [ "$TARGET_STEPS" -lt "$FINAL_STEPS" ] && [ "$best_ck_k" -ge "$target_k" ] && [ "${FORCE:-0}" != "1" ]; then
  log "WAVE SKIP: existing _${best_ck_k}k.pth ≥ target ${target_k}k (final=$FINAL_STEPS not reached — set FORCE=1 to override)"
  exit 0
fi

RESUME=""; latest="${RESUME_FROM:-$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)}"
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START ($ARM_DESC) gpu=$BB_GPU target=$TARGET_STEPS final=$FINAL_STEPS bs=64 ${RESUME}"
CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 64 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --extra-save-steps "$EXTRA_SAVES" \
  --save-dir "$RUNS" --run-name "$NAME" --log-every "$LOG_EVERY" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8 \
  --num-encoder-layers "$NENC" --num-layers "$NLAY" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  "${LOSS_ARGS[@]}" \
  --ema-embedding --ema-encoder --ema-tau 0.9 --cpc-infonce-weight 1.0 \
  --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
  --sigreg-embedding-weight 1.0 --sigreg-encoding-weight 1.0 \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  --train-rollout-depth "$K" \
  "${EXTRA_ARGS[@]}" \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL. tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi

# Intermediate wave — do not write _FINAL.pth (the run_arm.sh skip
# sentinel). The next wave's launcher resumes from the latest
# `_<N>k.pth` on disk.
if [ "$TARGET_STEPS" -lt "$FINAL_STEPS" ]; then
  latest_ck=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
  log "$ARM wave complete: target=$TARGET_STEPS (< final $FINAL_STEPS) — latest ck $(basename "${latest_ck:-<none>}")"
  exit 0
fi

if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else
  # Neither named checkpoint exists — fall back to the last periodic
  # snapshot. A resumed run can leave two of those at that step; they are
  # different models and mtime does not say which one `_FINAL.pth` should
  # carry, so resolve instead of picking (#390). Recomputed rather than
  # reusing the preflight scan: the training run just wrote snapshots that
  # were not on disk when it ran.
  last_k=$(latest_step_k)
  [ "$last_k" -ge 0 ] || { log "BB FAILED no checkpoint"; exit 1; }
  errf="$(mktemp)"
  snap=$(bash "$CKPT_RESOLVER" "$RUNS" "$NAME" "$last_k" 2>"$errf"); rc=$?
  if [ $rc -ne 0 ]; then
    log "BB FAILED: no single ${last_k}k snapshot to name _FINAL.pth from (rc=$rc):"
    while IFS= read -r line; do log "  $line"; done <"$errf"
    log "  Copy the replicate you mean to ${NAME}_FINAL.pth."
    rm -f "$errf"; exit $rc
  fi
  rm -f "$errf"
  log "FINAL from $snap"
  cp -f "$snap" "$BB"
fi
[ -f "$BB" ] || { log "BB FAILED no checkpoint"; exit 1; }
log "$ARM complete: BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"
