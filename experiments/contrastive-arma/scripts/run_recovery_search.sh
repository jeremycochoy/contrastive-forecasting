#!/bin/bash
# =============================================================================
# Recovery Architecture Search
# =============================================================================
# Systematic search for the best parameter recovery head architecture and
# hyperparameters, tested on both V1 (SimpleModel) and V2 (ConfigurableModel)
# backbones.
#
# Phases:
#   1. Baseline sweep (7 model types, 5k epochs) on V2 backbone
#   2. Baseline sweep (7 model types, 5k epochs) on V1 backbone
#   3. Hyperparameter sweep (hidden_dim x num_gru_layers) on V2, gru & deepgru
#   4. Loss function sweep (4 losses) on V2, deepgru
#   5. (Placeholder) Full training of best configs on both backbones
#
# Usage:
#   bash run_recovery_search.sh          # run all phases
#   bash run_recovery_search.sh 3        # run only phase 3
#   bash run_recovery_search.sh 1 2      # run phases 1 and 2
# =============================================================================

set -e
cd ~/workspaces/contrastive-forecasting

# ---- GPU selection: pick the GPU with the most free memory ----
echo "Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    # Get free memory for each GPU (in MiB), pick the one with the most
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -n -r \
        | head -1 \
        | cut -d',' -f1 \
        | tr -d ' ')
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    echo "Selected GPU ${GPU_ID} (most free memory)"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader \
        | while read line; do echo "  $line"; done
else
    echo "nvidia-smi not found, defaulting to CUDA_VISIBLE_DEVICES=1"
    export CUDA_VISIBLE_DEVICES=1
fi

# ---- Backbone definitions ----
V2_MODEL="v2_2M_model_best.pth"
V2_BACKBONE_ARGS="--encoder-type gru --H 1024 --num-layers 12 --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3"

V1_MODEL="trained_simple_model_H1024.pth"
V1_BACKBONE_ARGS="--H 1024 --num-layers 12"

# ---- Common settings ----
DEVICE="cuda"
BATCH_SIZE=32
LR="1e-3"
LOG_EVERY=100
SAVE_EVERY=5000
NUM_ARMA=4
DIMENSION=4

# ---- Model types ----
ALL_MODELS="mlp gru resmlp attention grupool deepgru deepgrupool"

# ---- Output directory ----
LOGDIR="recovery_search_logs"
mkdir -p "$LOGDIR"

# ---- Helper: extract best val loss from a log file ----
extract_best_val() {
    local logfile="$1"
    if [ -f "$logfile" ]; then
        # Look for "best=X.XXXXXX@EPOCH" pattern in log output
        grep -oP 'best=\K[0-9]+\.[0-9]+' "$logfile" | tail -1
    else
        echo "N/A"
    fi
}

# ---- Helper: print a summary line after an experiment ----
print_summary() {
    local label="$1"
    local logfile="$2"
    local best_val
    best_val=$(extract_best_val "$logfile")
    echo "  >> ${label}: best_val_loss=${best_val}"
}

# ---- Phase selection ----
# If arguments are given, run only those phases; otherwise run all
if [ $# -gt 0 ]; then
    PHASES="$@"
else
    PHASES="1 2 3 4 5"
fi

should_run_phase() {
    local phase="$1"
    for p in $PHASES; do
        if [ "$p" = "$phase" ]; then
            return 0
        fi
    done
    return 1
}

echo ""
echo "============================================================"
echo " Recovery Architecture Search"
echo " Started: $(date)"
echo " Phases to run: ${PHASES}"
echo "============================================================"
echo ""

SEARCH_START=$(date +%s)

# =====================================================================
# Phase 1: Baseline sweep on V2 backbone (5k epochs)
# =====================================================================
if should_run_phase 1; then
    echo "============================================================"
    echo " Phase 1: Baseline sweep on V2 backbone"
    echo "============================================================"
    PHASE1_START=$(date +%s)
    date

    for model in $ALL_MODELS; do
        TAG="p1_v2_${model}"
        LOGFILE="${LOGDIR}/recovery_search_p1_${model}.log"
        HEADPATH="${LOGDIR}/recovery_search_p1_${model}.pth"

        echo ""
        echo "--- Phase 1: ${model} on V2 backbone ---"
        date

        python3 -u train_parameter_recovery_v2.py \
            --device ${DEVICE} \
            --model-path ${V2_MODEL} \
            ${V2_BACKBONE_ARGS} \
            --model-type ${model} \
            --hidden-dim 256 \
            --num-gru-layers 2 \
            --num-arma-params ${NUM_ARMA} \
            --dimension ${DIMENSION} \
            --epochs 5000 \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --log-every ${LOG_EVERY} \
            --save-every ${SAVE_EVERY} \
            --head-path ${HEADPATH} \
            2>&1 | tee ${LOGFILE}

        print_summary "Phase1 V2 ${model}" "${LOGFILE}"
        sleep 5
    done

    PHASE1_END=$(date +%s)
    PHASE1_ELAPSED=$(( PHASE1_END - PHASE1_START ))
    echo ""
    echo "============================================================"
    echo " Phase 1 Complete! (${PHASE1_ELAPSED}s / $(( PHASE1_ELAPSED / 60 ))min)"
    echo "============================================================"
    echo ""
    echo "=== Phase 1 Summary ==="
    for model in $ALL_MODELS; do
        LOGFILE="${LOGDIR}/recovery_search_p1_${model}.log"
        print_summary "V2 ${model}" "${LOGFILE}"
    done
    echo ""
fi

# =====================================================================
# Phase 2: Baseline sweep on V1 backbone (5k epochs)
# Uses train_parameter_recovery.py (NOT v2)
# V1 args: --model-path, --H, --num-layers, --model-type, --head-path,
#           --epochs, --batch-size, --lr, --num-arma-params, --dimension,
#           --device, --log-every, --hidden-dim, --save-every
# (No --encoder-type, --nhead, --ffn-mult, --activation, --depthwise-conv,
#  --num-gru-layers, --loss-type)
# =====================================================================
if should_run_phase 2; then
    echo "============================================================"
    echo " Phase 2: Baseline sweep on V1 backbone"
    echo "============================================================"
    PHASE2_START=$(date +%s)
    date

    for model in $ALL_MODELS; do
        TAG="p2_v1_${model}"
        LOGFILE="${LOGDIR}/recovery_search_p2_${model}.log"
        HEADPATH="${LOGDIR}/recovery_search_p2_${model}.pth"

        echo ""
        echo "--- Phase 2: ${model} on V1 backbone ---"
        date

        python3 -u train_parameter_recovery.py \
            --device ${DEVICE} \
            --model-path ${V1_MODEL} \
            ${V1_BACKBONE_ARGS} \
            --model-type ${model} \
            --hidden-dim 256 \
            --num-arma-params ${NUM_ARMA} \
            --dimension ${DIMENSION} \
            --epochs 5000 \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --log-every ${LOG_EVERY} \
            --save-every ${SAVE_EVERY} \
            --head-path ${HEADPATH} \
            2>&1 | tee ${LOGFILE}

        print_summary "Phase2 V1 ${model}" "${LOGFILE}"
        sleep 5
    done

    PHASE2_END=$(date +%s)
    PHASE2_ELAPSED=$(( PHASE2_END - PHASE2_START ))
    echo ""
    echo "============================================================"
    echo " Phase 2 Complete! (${PHASE2_ELAPSED}s / $(( PHASE2_ELAPSED / 60 ))min)"
    echo "============================================================"
    echo ""
    echo "=== Phase 2 Summary ==="
    for model in $ALL_MODELS; do
        LOGFILE="${LOGDIR}/recovery_search_p2_${model}.log"
        print_summary "V1 ${model}" "${LOGFILE}"
    done
    echo ""
fi

# =====================================================================
# Phase 3: Hyperparameter sweep on GRU-based heads (V2 backbone, 5k epochs)
# Model types: gru, deepgru
# hidden_dim: 128, 256, 512
# num_gru_layers: 1, 2, 3, 4
# Total: 2 x 3 x 4 = 24 experiments
# =====================================================================
if should_run_phase 3; then
    echo "============================================================"
    echo " Phase 3: Hyperparameter sweep (gru/deepgru, V2 backbone)"
    echo " 2 models x 3 hidden_dims x 4 num_gru_layers = 24 experiments"
    echo "============================================================"
    PHASE3_START=$(date +%s)
    date

    P3_COUNT=0
    P3_TOTAL=24

    for model in gru deepgru; do
        for hdim in 128 256 512; do
            for nlayers in 1 2 3 4; do
                P3_COUNT=$(( P3_COUNT + 1 ))
                TAG="p3_${model}_h${hdim}_l${nlayers}"
                LOGFILE="${LOGDIR}/recovery_search_p3_${model}_h${hdim}_l${nlayers}.log"
                HEADPATH="${LOGDIR}/recovery_search_p3_${model}_h${hdim}_l${nlayers}.pth"

                echo ""
                echo "--- Phase 3 [${P3_COUNT}/${P3_TOTAL}]: ${model} hidden=${hdim} gru_layers=${nlayers} ---"
                date

                python3 -u train_parameter_recovery_v2.py \
                    --device ${DEVICE} \
                    --model-path ${V2_MODEL} \
                    ${V2_BACKBONE_ARGS} \
                    --model-type ${model} \
                    --hidden-dim ${hdim} \
                    --num-gru-layers ${nlayers} \
                    --num-arma-params ${NUM_ARMA} \
                    --dimension ${DIMENSION} \
                    --epochs 5000 \
                    --batch-size ${BATCH_SIZE} \
                    --lr ${LR} \
                    --log-every ${LOG_EVERY} \
                    --save-every ${SAVE_EVERY} \
                    --head-path ${HEADPATH} \
                    2>&1 | tee ${LOGFILE}

                print_summary "Phase3 ${TAG}" "${LOGFILE}"
                sleep 5
            done
        done
    done

    PHASE3_END=$(date +%s)
    PHASE3_ELAPSED=$(( PHASE3_END - PHASE3_START ))
    echo ""
    echo "============================================================"
    echo " Phase 3 Complete! (${PHASE3_ELAPSED}s / $(( PHASE3_ELAPSED / 60 ))min)"
    echo "============================================================"
    echo ""
    echo "=== Phase 3 Summary ==="
    for model in gru deepgru; do
        for hdim in 128 256 512; do
            for nlayers in 1 2 3 4; do
                TAG="p3_${model}_h${hdim}_l${nlayers}"
                LOGFILE="${LOGDIR}/recovery_search_p3_${model}_h${hdim}_l${nlayers}.log"
                print_summary "${TAG}" "${LOGFILE}"
            done
        done
    done
    echo ""
fi

# =====================================================================
# Phase 4: Loss function sweep (V2 backbone, deepgru, 5k epochs)
# Losses: mse, huber, l1, weighted_mse
# Uses default hidden_dim=256
# =====================================================================
if should_run_phase 4; then
    echo "============================================================"
    echo " Phase 4: Loss function sweep (deepgru, V2 backbone)"
    echo "============================================================"
    PHASE4_START=$(date +%s)
    date

    for loss in mse huber l1 weighted_mse; do
        TAG="p4_deepgru_${loss}"
        LOGFILE="${LOGDIR}/recovery_search_p4_deepgru_${loss}.log"
        HEADPATH="${LOGDIR}/recovery_search_p4_deepgru_${loss}.pth"

        echo ""
        echo "--- Phase 4: deepgru loss=${loss} ---"
        date

        python3 -u train_parameter_recovery_v2.py \
            --device ${DEVICE} \
            --model-path ${V2_MODEL} \
            ${V2_BACKBONE_ARGS} \
            --model-type deepgru \
            --hidden-dim 256 \
            --num-gru-layers 2 \
            --loss-type ${loss} \
            --num-arma-params ${NUM_ARMA} \
            --dimension ${DIMENSION} \
            --epochs 5000 \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --log-every ${LOG_EVERY} \
            --save-every ${SAVE_EVERY} \
            --head-path ${HEADPATH} \
            2>&1 | tee ${LOGFILE}

        print_summary "Phase4 deepgru ${loss}" "${LOGFILE}"
        sleep 5
    done

    PHASE4_END=$(date +%s)
    PHASE4_ELAPSED=$(( PHASE4_END - PHASE4_START ))
    echo ""
    echo "============================================================"
    echo " Phase 4 Complete! (${PHASE4_ELAPSED}s / $(( PHASE4_ELAPSED / 60 ))min)"
    echo "============================================================"
    echo ""
    echo "=== Phase 4 Summary ==="
    for loss in mse huber l1 weighted_mse; do
        LOGFILE="${LOGDIR}/recovery_search_p4_deepgru_${loss}.log"
        print_summary "deepgru ${loss}" "${LOGFILE}"
    done
    echo ""
fi

# =====================================================================
# Phase 5: Full training of best configurations
# =====================================================================
if should_run_phase 5; then
    echo "============================================================"
    echo " Phase 5: Full training of best configurations"
    echo "============================================================"
    date

    # Phase 5: Edit this after reviewing Phase 1-4 results
    #
    # After reviewing the summaries from Phases 1-4, select the top 5
    # configurations and train them for 20k epochs on both V1 and V2
    # backbones. Example template:
    #
    # BEST_CONFIGS=(
    #     "deepgru --hidden-dim 256 --num-gru-layers 3 --loss-type mse"
    #     "deepgru --hidden-dim 512 --num-gru-layers 2 --loss-type huber"
    #     "gru --hidden-dim 256 --num-gru-layers 2 --loss-type mse"
    #     "grupool --hidden-dim 256 --loss-type mse"
    #     "deepgrupool --hidden-dim 256 --num-gru-layers 3 --loss-type mse"
    # )
    #
    # for i in "${!BEST_CONFIGS[@]}"; do
    #     CONFIG="${BEST_CONFIGS[$i]}"
    #     MODEL_TYPE=$(echo "$CONFIG" | awk '{print $1}')
    #     EXTRA_ARGS=$(echo "$CONFIG" | cut -d' ' -f2-)
    #
    #     # --- V2 backbone ---
    #     TAG="p5_v2_${MODEL_TYPE}_cfg${i}"
    #     LOGFILE="${LOGDIR}/recovery_search_p5_v2_${MODEL_TYPE}_cfg${i}.log"
    #     HEADPATH="${LOGDIR}/recovery_search_p5_v2_${MODEL_TYPE}_cfg${i}.pth"
    #     python3 -u train_parameter_recovery_v2.py \
    #         --device ${DEVICE} --model-path ${V2_MODEL} ${V2_BACKBONE_ARGS} \
    #         --model-type ${MODEL_TYPE} ${EXTRA_ARGS} \
    #         --num-arma-params ${NUM_ARMA} --dimension ${DIMENSION} \
    #         --epochs 20000 --batch-size ${BATCH_SIZE} --lr ${LR} \
    #         --log-every ${LOG_EVERY} --save-every ${SAVE_EVERY} \
    #         --head-path ${HEADPATH} 2>&1 | tee ${LOGFILE}
    #     print_summary "Phase5 V2 ${TAG}" "${LOGFILE}"
    #     sleep 5
    #
    #     # --- V1 backbone (no v2-only args) ---
    #     TAG="p5_v1_${MODEL_TYPE}_cfg${i}"
    #     LOGFILE="${LOGDIR}/recovery_search_p5_v1_${MODEL_TYPE}_cfg${i}.log"
    #     HEADPATH="${LOGDIR}/recovery_search_p5_v1_${MODEL_TYPE}_cfg${i}.pth"
    #     python3 -u train_parameter_recovery.py \
    #         --device ${DEVICE} --model-path ${V1_MODEL} ${V1_BACKBONE_ARGS} \
    #         --model-type ${MODEL_TYPE} --hidden-dim 256 \
    #         --num-arma-params ${NUM_ARMA} --dimension ${DIMENSION} \
    #         --epochs 20000 --batch-size ${BATCH_SIZE} --lr ${LR} \
    #         --log-every ${LOG_EVERY} --save-every ${SAVE_EVERY} \
    #         --head-path ${HEADPATH} 2>&1 | tee ${LOGFILE}
    #     print_summary "Phase5 V1 ${TAG}" "${LOGFILE}"
    #     sleep 5
    # done

    echo ""
    echo "Phase 5 is a placeholder. Edit the script after reviewing Phase 1-4 results."
    echo ""
fi

# =====================================================================
# Final summary
# =====================================================================
SEARCH_END=$(date +%s)
SEARCH_ELAPSED=$(( SEARCH_END - SEARCH_START ))

echo ""
echo "============================================================"
echo " Recovery Architecture Search Complete!"
echo " Total time: ${SEARCH_ELAPSED}s / $(( SEARCH_ELAPSED / 60 ))min"
echo " Finished: $(date)"
echo "============================================================"
echo ""
echo "All logs and checkpoints are in: ${LOGDIR}/"
echo ""
echo "To review results, run:"
echo "  grep 'best_val_loss' ${LOGDIR}/recovery_search_*.log"
echo "  # or look at the _results.json files"
echo ""
