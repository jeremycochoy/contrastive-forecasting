#!/bin/bash
# #379 arm launcher — small backbone × 200k steps on elisa, one script per arm.
# Backbone-only: trains the backbone, saves checkpoints, records training
# dynamics (loss / ff / u_batchtime / u_batchtime_e). No q-head training,
# no GIFT-Eval — the deliverable is the training-dynamics trajectories.
#
# Usage:
#   WT=/absolute/path/to/checkout BB_GPU=1 bash run_arm.sh <arm>
#
# where <arm> ∈ {arm1, arm3, arm4, arm5, arm6_v2, bimoco,
#                arm1_tr1, arm3_tr1, arm5_tr1, arm6_v2_tr1, bimoco_tr1}.
# The _tr1 arms are the #379 tau_rep=1.0 reruns of the L_rep-bearing
# arms (arm 1/3/5/6_v2 + bimoco). Each arm shares the same backbone
# architecture and training schedule (d_model=64, n_heads=8,
# num_encoder_layers=3, num_layers=3, bs=64, 200k steps,
# save-every=25000, extra snapshot at 2500); only the arm-specific bits
# (run name, human label, extra CLI flags after --loss-shape) come from
# the per-arm case block below. See `README.md` for the parent-experiment
# mapping to #374.
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm1|arm3|arm4|arm5|arm6_v2|bimoco|arm1_tr1|arm3_tr1|arm5_tr1|arm6_v2_tr1|bimoco_tr1|arm1_nse|arm3_nse|arm4_nse|arm5_nse|arm6_v2_nse|bimoco_nse|arm1_ncpc|arm3_ncpc|arm4_ncpc|arm5_ncpc|arm6_v2_ncpc|bimoco_ncpc>}"

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
#             Python argparse keeps the LAST value on repeat. Empty for
#             base + tau_rep arms.
#
# When adding a 7th arm, add one case here and one entry in the launcher
# shape test (tests/test_small_long_launcher_shape.py). Nothing else in
# this script needs to change.
EXTRA_ARGS=()
case "$ARM" in
  arm1)
    NAME="bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1: split_pred_rep"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep)
    ;;
  arm3)
    NAME="bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3: split_pred_rep + moco-negatives"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep --moco-negatives)
    ;;
  arm4)
    NAME="bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4: xshh_allt + moco-negatives"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives)
    ;;
  arm5)
    NAME="bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5: rep_only + align-loss-weight 1.0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only --align-loss-weight 1.0)
    ;;
  arm6_v2)
    NAME="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2: rep_only + align-loss-weight 1.0 + moco-rep-keys"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    ;;
  bimoco)
    NAME="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="bimoco: split_pred_rep + moco-negatives + moco-rep-keys"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --moco-rep-keys)
    ;;
  # ---- #379 tau_rep=1.0 reruns ---------------------------------------------
  # Every arm with an L_rep term (i.e. all but arm 4), rerun with the
  # temperature of L_rep raised to 1.0 while L_pred stays at 0.10.
  # `--tau-rep 1.0` is the only per-arm change; loss flags mirror the base
  # arm 1:1. Name suffix `_tr1` is threaded through the checkpoint filename
  # so the base and rerun artefacts never collide on disk.
  arm1_tr1)
    NAME="bb_small_arm1_tr1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1_tr1: split_pred_rep + tau=1.0 + tau_rep=1.0 (all tau at 1.0)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --tau 1.0 --tau-rep 1.0)
    ;;
  arm3_tr1)
    NAME="bb_small_arm3_tr1_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3_tr1: split_pred_rep + moco-negatives + tau=1.0 + tau_rep=1.0 (all tau at 1.0)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --tau 1.0 --tau-rep 1.0)
    ;;
  arm4_tr1)
    NAME="bb_small_arm4_tr1_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4_tr1: xshh_allt + moco-negatives + tau=1.0 (single pooled tau)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives \
               --tau 1.0)
    ;;
  arm5_tr1)
    NAME="bb_small_arm5_tr1_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5_tr1: rep_only + align-loss-weight 1.0 + tau_rep=1.0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --tau-rep 1.0)
    ;;
  arm6_v2_tr1)
    NAME="bb_small_arm6_v2_tr1_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2_tr1: rep_only + align-loss-weight 1.0 + moco-rep-keys + tau_rep=1.0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --tau-rep 1.0)
    ;;
  bimoco_tr1)
    NAME="bb_small_bimoco_tr1_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="bimoco_tr1: split_pred_rep + moco-negatives + moco-rep-keys + tau=1.0 + tau_rep=1.0 (all tau at 1.0)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --moco-rep-keys --tau 1.0 --tau-rep 1.0)
    ;;
  # ---- #379 no-sigreg-embedding (nse) reruns -------------------------------
  # Each `_nse` arm mirrors its base arm 1:1 and appends
  # `--sigreg-embedding-weight 0.0` via EXTRA_ARGS (placed AFTER the shared
  # `--sigreg-embedding-weight 1.0` in the trainer call so argparse's
  # last-wins rule zeroes the e_t regulariser). The h_t regulariser
  # (`--sigreg-encoding-weight 1.0`) is kept as in the base.
  arm1_nse)
    NAME="bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1_nse: split_pred_rep + sigreg_embedding=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  arm3_nse)
    NAME="bb_small_arm3_nse_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3_nse: split_pred_rep + moco-negatives + sigreg_embedding=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep --moco-negatives)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  arm4_nse)
    NAME="bb_small_arm4_nse_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4_nse: xshh_allt + moco-negatives + sigreg_embedding=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  arm5_nse)
    NAME="bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5_nse: rep_only + align-loss-weight 1.0 + sigreg_embedding=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only --align-loss-weight 1.0)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  arm6_v2_nse)
    NAME="bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2_nse: rep_only + align-loss-weight 1.0 + moco-rep-keys + sigreg_embedding=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  bimoco_nse)
    NAME="bb_small_bimoco_nse_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="bimoco_nse: split_pred_rep + moco-negatives + moco-rep-keys + sigreg_embedding=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --moco-rep-keys)
    EXTRA_ARGS=(--sigreg-embedding-weight 0.0)
    ;;
  # ---- #379 no-CPC (ncpc) reruns -------------------------------------------
  # Each `_ncpc` arm mirrors its base arm 1:1 and appends
  # `--cpc-infonce-weight 0.0` via EXTRA_ARGS (placed AFTER the shared
  # `--cpc-infonce-weight 1.0` in the trainer call so argparse's
  # last-wins rule disables the CPC auxiliary while keeping SIGReg on).
  arm1_ncpc)
    NAME="bb_small_arm1_ncpc_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1_ncpc: split_pred_rep + cpc_infonce_weight=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm3_ncpc)
    NAME="bb_small_arm3_ncpc_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3_ncpc: split_pred_rep + moco-negatives + cpc_infonce_weight=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep --moco-negatives)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm4_ncpc)
    NAME="bb_small_arm4_ncpc_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4_ncpc: xshh_allt + moco-negatives + cpc_infonce_weight=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm5_ncpc)
    NAME="bb_small_arm5_ncpc_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5_ncpc: rep_only + align-loss-weight 1.0 + cpc_infonce_weight=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only --align-loss-weight 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm6_v2_ncpc)
    NAME="bb_small_arm6_v2_ncpc_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2_ncpc: rep_only + align-loss-weight 1.0 + moco-rep-keys + cpc_infonce_weight=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  bimoco_ncpc)
    NAME="bb_small_bimoco_ncpc_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="bimoco_ncpc: split_pred_rep + moco-negatives + moco-rep-keys + cpc_infonce_weight=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --moco-rep-keys)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  # ---- #379 combined-ablation (combab) reruns ------------------------------
  # combab = all τ raised to 1.0 + CPC-InfoNCE off + (conditionally) SIGReg
  # on the embedding off. nse is added ONLY for arms where the Wave-D nse
  # stat test showed a reduction in latent movement (arm1/3/4). arm5, arm6_v2
  # and bimoco keep sigreg_e=1.0 because nse hurt them there.
  arm1_combab)
    NAME="bb_small_arm1_combab_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 1_combab: split_pred_rep + all τ=1.0 + cpc=0 + sigreg_e=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --tau 1.0 --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0 --sigreg-embedding-weight 0.0)
    ;;
  arm3_combab)
    NAME="bb_small_arm3_combab_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 3_combab: split_pred_rep + moco-negatives + all τ=1.0 + cpc=0 + sigreg_e=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --tau 1.0 --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0 --sigreg-embedding-weight 0.0)
    ;;
  arm4_combab)
    NAME="bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 4_combab: xshh_allt + moco-negatives + τ=1.0 + cpc=0 + sigreg_e=0"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
               --pos-in-denominator --subtract-contrastive-floor --moco-negatives \
               --tau 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0 --sigreg-embedding-weight 0.0)
    ;;
  arm5_combab)
    NAME="bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 5_combab: rep_only + align + τ_rep=1.0 + cpc=0 (nse SKIPPED: nse hurt arm5)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  arm6_v2_combab)
    NAME="bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="arm 6 v2_combab: rep_only + align + moco-rep-keys + τ_rep=1.0 + cpc=0 (nse SKIPPED: nse hurt arm6_v2)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_rep_only \
               --align-loss-weight 1.0 --moco-rep-keys --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  bimoco_combab)
    NAME="bb_small_bimoco_combab_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
    ARM_DESC="bimoco_combab: split_pred_rep + moco-negatives + moco-rep-keys + all τ=1.0 + cpc=0 (nse SKIPPED: nse hurt bimoco)"
    LOSS_ARGS=(--loss-shape cosine_similarity_batch_split_pred_rep \
               --moco-negatives --moco-rep-keys --tau 1.0 --tau-rep 1.0)
    EXTRA_ARGS=(--cpc-infonce-weight 0.0)
    ;;
  *)
    echo "ABORT: unknown arm '$ARM'" >&2
    echo "  valid: arm1 arm3 arm4 arm5 arm6_v2 bimoco" >&2
    echo "         arm1_tr1 arm3_tr1 arm4_tr1 arm5_tr1 arm6_v2_tr1 bimoco_tr1" >&2
    echo "         arm1_nse arm3_nse arm4_nse arm5_nse arm6_v2_nse bimoco_nse" >&2
    echo "         arm1_ncpc arm3_ncpc arm4_ncpc arm5_ncpc arm6_v2_ncpc bimoco_ncpc" >&2
    echo "         arm1_combab arm3_combab arm4_combab arm5_combab arm6_v2_combab bimoco_combab" >&2
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
# Staged-wave support (issue #379 refinement).
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
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

BB="$RUNS/${NAME}_FINAL.pth"
tlog="$RES/run_${NAME}.log"

if [ -f "$BB" ] && [ "${FORCE:-0}" != "1" ]; then
  log "BB SKIP ($NAME FINAL exists — set FORCE=1 to override)"
  exit 0
fi

# Wave idempotency (issue #379): if an existing periodic checkpoint has
# already reached (or exceeded) TARGET_STEPS and this is an intermediate
# wave (TARGET_STEPS < FINAL_STEPS), skip re-running the trainer. This
# saves the ~30 s of load / no-op that the trainer's own start_step ≥
# total_steps handling would do anyway when the orchestrator re-fires a
# completed wave (e.g. after a re-launch mid-crash).
target_k=$(( TARGET_STEPS / 1000 ))
best_ck_k=-1
for f in "$RUNS/${NAME}"_*k.pth; do
  [ -e "$f" ] || continue
  case "$f" in *_optimizer.pth) continue;; esac
  k=$(basename "$f" | sed -E 's/.*_([0-9]+)k\.pth$/\1/')
  case "$k" in ''|*[!0-9]*) continue;; esac
  (( k > best_ck_k )) && best_ck_k=$k
done
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
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] || { log "BB FAILED no checkpoint"; exit 1; }
log "$ARM complete: BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"
