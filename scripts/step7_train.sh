#!/bin/bash
# Step 7: Train Router Models
# Usage: ./scripts/step7_train.sh <dataset> <result_model> [options]
#
# Options:
#   --label-name NAME      Label file name (default: hard_llm_correct_rule)
#   --split-name NAME      Split file name (default: split_8_1_1)
#   --aggregated-name NAME Aggregated file name (default: query_metrics_filtered)
#   --text-only            Train only text router (default)
#   --feat-only            Train only feature routers
#   --all                  Train both text and feature routers
#   --batch-size N         Override batch size
#   --epochs N             Override epochs (default: 10)
#   --seed N               Random seed (default: 42)
#   --device DEVICE        Specify device (cuda/cpu)
#   --save-prefix PREFIX   Model save name prefix (default: router)

set -e

DATASET="graphragBench_medical"
RESULT_MODEL="llama-3.1-8b-awq-int4"

# Positional args are optional:
#   1) dataset
#   2) result_model
# If omitted, defaults are used. Options can be passed directly.
if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET="$1"
    shift
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
    RESULT_MODEL="$1"
    shift
fi

# Default settings
LABEL_NAME="hard_llm_correct_rule"
SPLIT_NAME="split_08_01_1"
AGGREGATED_NAME="query_metrics_filtered"
SAVE_PREFIX="router"
TRAIN_TEXT=true
TRAIN_FEAT=true
TEXT_BATCH=128
FEAT_BATCH=128
EPOCHS=10
LR="1e-4"
SEED=42
DEVICE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --label-name) LABEL_NAME="$2"; shift 2 ;;
        --split-name) SPLIT_NAME="$2"; shift 2 ;;
        --aggregated-name) AGGREGATED_NAME="$2"; shift 2 ;;
        --save-prefix) SAVE_PREFIX="$2"; shift 2 ;;
        --text-only) TRAIN_TEXT=true; TRAIN_FEAT=false; shift ;;
        --feat-only) TRAIN_TEXT=false; TRAIN_FEAT=true; shift ;;
        --all) TRAIN_TEXT=true; TRAIN_FEAT=true; shift ;;
        --batch-size) TEXT_BATCH="$2"; FEAT_BATCH="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Step 7: Training Router Models"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Label Name: $LABEL_NAME"
echo "Split Name: $SPLIT_NAME"
echo "Aggregated Name: $AGGREGATED_NAME"
echo "Save Prefix: $SAVE_PREFIX"
echo "Train Text: $TRAIN_TEXT"
echo "Train Feature: $TRAIN_FEAT"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo ""

# Train text router
if [ "$TRAIN_TEXT" = true ]; then
    MODEL_NAME="${SAVE_PREFIX}_text"
    echo "--- Training Text Router ---"

    CMD="python Run/Router/run_train_router.py \
        --dataset $DATASET \
        --result-model $RESULT_MODEL \
        --model-type text_router \
        --split-name $SPLIT_NAME \
        --label-name $LABEL_NAME \
        --aggregated-name $AGGREGATED_NAME \
        --batch-size $TEXT_BATCH \
        --learning-rate $LR \
        --epochs $EPOCHS \
        --seed $SEED \
        --save-name $MODEL_NAME"

    if [ -n "$DEVICE" ]; then
        CMD="$CMD --device $DEVICE"
    fi

    eval $CMD
    echo ""
fi

# Train feature routers
if [ "$TRAIN_FEAT" = true ]; then
    FEATURES=("mean_hidden" "last_hidden" "last_5_mean")

    for FEATURE in "${FEATURES[@]}"; do
        # Shorten feature name for model name
        SHORT_NAME=$(echo "$FEATURE" | sed 's/_hidden//' | sed 's/_5_mean/5/')
        MODEL_NAME="${SAVE_PREFIX}_feat_${SHORT_NAME}"

        echo "--- Training Feature Router ($FEATURE) ---"

        CMD="python Run/Router/run_train_router.py \
            --dataset $DATASET \
            --result-model $RESULT_MODEL \
            --model-type feature_router \
            --feature-name $FEATURE \
            --split-name $SPLIT_NAME \
            --label-name $LABEL_NAME \
            --aggregated-name $AGGREGATED_NAME \
            --batch-size $FEAT_BATCH \
            --learning-rate $LR \
            --epochs $EPOCHS \
            --seed $SEED \
            --save-name $MODEL_NAME"

        if [ -n "$DEVICE" ]; then
            CMD="$CMD --device $DEVICE"
        fi

        eval $CMD
        echo ""
    done
fi

echo "Step 7 completed!"
echo "Models saved to: Dataset/RouterTrainingData/Models/${SAVE_PREFIX}_*/"
