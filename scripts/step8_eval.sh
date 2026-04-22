#!/bin/bash
# Step 8: Comprehensive Router Evaluation
# Usage: ./scripts/step8_eval.sh <dataset> <result_model> <version> [options]
#
# Options:
#   --text-only        Evaluate only text router (default)
#   --feat-only        Evaluate only feature routers
#   --all              Evaluate both text and feature routers
#   --device DEVICE    Specify device (cuda/cpu)

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}
shift 3 2>/dev/null || true

# Default settings
EVAL_TEXT=true
EVAL_FEAT=false
DEVICE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --text-only) EVAL_TEXT=true; EVAL_FEAT=false; shift ;;
        --feat-only) EVAL_TEXT=false; EVAL_FEAT=true; shift ;;
        --all) EVAL_TEXT=true; EVAL_FEAT=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LABEL_NAME="hard_llm_correct_rule_${VERSION}"
SPLIT_NAME="split_${VERSION}_8_1_1"
AGGREGATED_NAME="query_metrics_${VERSION}_filtered"

echo "============================================================"
echo "Step 8: Comprehensive Router Evaluation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Version: $VERSION"
echo "Evaluate Text: $EVAL_TEXT"
echo "Evaluate Feature: $EVAL_FEAT"
echo ""

# Evaluate text router
if [ "$EVAL_TEXT" = true ]; then
    MODEL_NAME="router_${VERSION}_text"
    echo "--- Evaluating Text Router ---"

    CMD="python Run/Router/run_comprehensive_router_eval.py \
        --dataset $DATASET \
        --result-model $RESULT_MODEL \
        --model-name $MODEL_NAME \
        --model-type text_router \
        --split-name $SPLIT_NAME \
        --label-name $LABEL_NAME \
        --aggregated-name $AGGREGATED_NAME"

    if [ -n "$DEVICE" ]; then
        CMD="$CMD --device $DEVICE"
    fi

    eval $CMD
    echo ""
fi

# Evaluate feature routers
if [ "$EVAL_FEAT" = true ]; then
    FEATURES=("mean_hidden" "last_hidden" "last_5_mean")

    for FEATURE in "${FEATURES[@]}"; do
        SHORT_NAME=$(echo "$FEATURE" | sed 's/_hidden//' | sed 's/_5_mean/5/')
        MODEL_NAME="router_${VERSION}_feat_${SHORT_NAME}"

        echo "--- Evaluating Feature Router ($FEATURE) ---"

        CMD="python Run/Router/run_comprehensive_router_eval.py \
            --dataset $DATASET \
            --result-model $RESULT_MODEL \
            --model-name $MODEL_NAME \
            --model-type feature_router \
            --feature-name $FEATURE \
            --split-name $SPLIT_NAME \
            --label-name $LABEL_NAME \
            --aggregated-name $AGGREGATED_NAME"

        if [ -n "$DEVICE" ]; then
            CMD="$CMD --device $DEVICE"
        fi

        eval $CMD
        echo ""
    done
fi

echo "Step 8 completed!"
echo "Results saved to: Dataset/RouterTrainingData/Evaluation/router_${VERSION}_*/"
