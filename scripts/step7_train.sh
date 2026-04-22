#!/bin/bash
# Step 7: Train Router Models
# Usage: ./scripts/step7_train.sh <dataset> <result_model> <version> [options]
#
# Options:
#   --text-only        Train only text router (default)
#   --feat-only        Train only feature routers
#   --all              Train both text and feature routers
#   --batch-size N     Override batch size
#   --epochs N         Override epochs (default: 10)
#   --device DEVICE    Specify device (cuda/cpu)

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}
shift 3 2>/dev/null || true

# Default settings
TRAIN_TEXT=true
TRAIN_FEAT=false
TEXT_BATCH=64
FEAT_BATCH=128
EPOCHS=10
LR="1e-4"
DEVICE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --text-only) TRAIN_TEXT=true; TRAIN_FEAT=false; shift ;;
        --feat-only) TRAIN_TEXT=false; TRAIN_FEAT=true; shift ;;
        --all) TRAIN_TEXT=true; TRAIN_FEAT=true; shift ;;
        --batch-size) TEXT_BATCH="$2"; FEAT_BATCH="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LABEL_NAME="hard_llm_correct_rule_${VERSION}"
SPLIT_NAME="split_${VERSION}_8_1_1"

echo "============================================================"
echo "Step 7: Training Router Models"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Version: $VERSION"
echo "Train Text: $TRAIN_TEXT"
echo "Train Feature: $TRAIN_FEAT"
echo "Epochs: $EPOCHS"
echo ""

# Train text router
if [ "$TRAIN_TEXT" = true ]; then
    MODEL_NAME="router_${VERSION}_text"
    echo "--- Training Text Router ---"

    CMD="python Run/Router/run_train_router.py \
        --dataset $DATASET \
        --result-model $RESULT_MODEL \
        --model-type text_router \
        --split-name $SPLIT_NAME \
        --label-name $LABEL_NAME \
        --batch-size $TEXT_BATCH \
        --learning-rate $LR \
        --epochs $EPOCHS \
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
        MODEL_NAME="router_${VERSION}_feat_${SHORT_NAME}"

        echo "--- Training Feature Router ($FEATURE) ---"

        CMD="python Run/Router/run_train_router.py \
            --dataset $DATASET \
            --result-model $RESULT_MODEL \
            --model-type feature_router \
            --feature-name $FEATURE \
            --split-name $SPLIT_NAME \
            --label-name $LABEL_NAME \
            --batch-size $FEAT_BATCH \
            --learning-rate $LR \
            --epochs $EPOCHS \
            --save-name $MODEL_NAME"

        if [ -n "$DEVICE" ]; then
            CMD="$CMD --device $DEVICE"
        fi

        eval $CMD
        echo ""
    done
fi

echo "Step 7 completed!"
echo "Models saved to: Dataset/RouterTrainingData/Models/router_${VERSION}_*/"
