#!/bin/bash
# Step 8: Comprehensive Router Evaluation
# Usage: ./scripts/step8_eval.sh <dataset> <result_model> [options]
#
# Options:
#   --label-name NAME      Label file name (default: hard_llm_correct_rule)
#   --split-name NAME      Split file name (default: split_8_1_1)
#   --aggregated-name NAME Aggregated file name (default: query_metrics_filtered)
#   --save-prefix PREFIX   Model save name prefix (default: router)
#   --text-only            Evaluate only text router (default)
#   --feat-only            Evaluate only feature routers
#   --all                  Evaluate both text and feature routers
#   --device DEVICE        Specify device (cuda/cpu)
#   --register-summary     Append result to summary registry JSON
#   --summary-path PATH    Custom summary registry path
#   --official             Mark registered record as official
#   --tags TAGS            Comma-separated tags for registered record
#   --experiment-id ID     Base experiment id for registry entries

set -e

DATASET="musique"
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
EVAL_TEXT=true
EVAL_FEAT=true
DEVICE=""
REGISTER_SUMMARY=false
SUMMARY_PATH=""
OFFICIAL=false
TAGS=""
EXPERIMENT_ID=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --label-name) LABEL_NAME="$2"; shift 2 ;;
        --split-name) SPLIT_NAME="$2"; shift 2 ;;
        --aggregated-name) AGGREGATED_NAME="$2"; shift 2 ;;
        --save-prefix) SAVE_PREFIX="$2"; shift 2 ;;
        --text-only) EVAL_TEXT=true; EVAL_FEAT=false; shift ;;
        --feat-only) EVAL_TEXT=false; EVAL_FEAT=true; shift ;;
        --all) EVAL_TEXT=true; EVAL_FEAT=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --register-summary) REGISTER_SUMMARY=true; shift ;;
        --summary-path) SUMMARY_PATH="$2"; shift 2 ;;
        --official) OFFICIAL=true; shift ;;
        --tags) TAGS="$2"; shift 2 ;;
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Step 8: Comprehensive Router Evaluation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Label Name: $LABEL_NAME"
echo "Split Name: $SPLIT_NAME"
echo "Aggregated Name: $AGGREGATED_NAME"
echo "Save Prefix: $SAVE_PREFIX"
echo "Evaluate Text: $EVAL_TEXT"
echo "Evaluate Feature: $EVAL_FEAT"
echo "Register Summary: $REGISTER_SUMMARY"
echo ""

# Evaluate text router
if [ "$EVAL_TEXT" = true ]; then
    MODEL_NAME="${SAVE_PREFIX}_text"
    echo "--- Evaluating Text Router ---"

    CMD="python Run/Router/run_comprehensive_router_eval.py \
        --dataset $DATASET \
        --result-model $RESULT_MODEL \
        --model-name $MODEL_NAME \
        --model-type text_router \
        --split-name $SPLIT_NAME \
        --label-name $LABEL_NAME \
        --aggregated-name $AGGREGATED_NAME"

    if [ "$REGISTER_SUMMARY" = true ]; then
        CMD="$CMD --register-summary"
    fi
    if [ -n "$SUMMARY_PATH" ]; then
        CMD="$CMD --summary-path $SUMMARY_PATH"
    fi
    if [ "$OFFICIAL" = true ]; then
        CMD="$CMD --official"
    fi
    if [ -n "$TAGS" ]; then
        CMD="$CMD --tags $TAGS"
    fi
    if [ -n "$EXPERIMENT_ID" ]; then
        CMD="$CMD --experiment-id ${EXPERIMENT_ID}_text"
    fi

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
        MODEL_NAME="${SAVE_PREFIX}_feat_${SHORT_NAME}"

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

        if [ "$REGISTER_SUMMARY" = true ]; then
            CMD="$CMD --register-summary"
        fi
        if [ -n "$SUMMARY_PATH" ]; then
            CMD="$CMD --summary-path $SUMMARY_PATH"
        fi
        if [ "$OFFICIAL" = true ]; then
            CMD="$CMD --official"
        fi
        if [ -n "$TAGS" ]; then
            CMD="$CMD --tags $TAGS"
        fi
        if [ -n "$EXPERIMENT_ID" ]; then
            CMD="$CMD --experiment-id ${EXPERIMENT_ID}_feat_${SHORT_NAME}"
        fi

        if [ -n "$DEVICE" ]; then
            CMD="$CMD --device $DEVICE"
        fi

        eval $CMD
        echo ""
    done
fi

echo "Step 8 completed!"
echo "Results saved to: Dataset/RouterTrainingData/Evaluation/${SAVE_PREFIX}_*/"
