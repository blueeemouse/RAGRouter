#!/bin/bash
# Run Full Router Training Pipeline
# Usage: ./scripts/run_all.sh <dataset> <result_model> <version> [options]
#
# Options:
#   --skip-retrieve    Skip Step 1 (retrieval)
#   --skip-eval        Skip Step 2 (evaluation)
#   --start-step N     Start from step N (default: 1)
#   --train-feat       Also train feature routers
#   --device DEVICE    Specify device

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}
shift 3 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default settings
START_STEP=1
SKIP_RETRIEVE=false
SKIP_EVAL=false
TRAIN_FEAT=false
DEVICE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-retrieve) SKIP_RETRIEVE=true; START_STEP=2; shift ;;
        --skip-eval) SKIP_EVAL=true; START_STEP=3; shift ;;
        --start-step) START_STEP="$2"; shift 2 ;;
        --train-feat) TRAIN_FEAT=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Router Training Pipeline - Full Run"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Version: $VERSION"
echo "Start Step: $START_STEP"
echo "Skip Retrieve: $SKIP_RETRIEVE"
echo "Skip Eval: $SKIP_EVAL"
echo "Train Feature: $TRAIN_FEAT"
echo "Device: ${DEVICE:-auto}"
echo ""

# Build extra args for train/eval scripts
TRAIN_ARGS=""
EVAL_ARGS=""
if [ "$TRAIN_FEAT" = true ]; then
    TRAIN_ARGS="--all"
    EVAL_ARGS="--all"
fi
if [ -n "$DEVICE" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --device $DEVICE"
    EVAL_ARGS="$EVAL_ARGS --device $DEVICE"
fi

# Step 1: Retrieval
if [ $START_STEP -le 1 ] && [ "$SKIP_RETRIEVE" = false ]; then
    "$SCRIPT_DIR/step1_retrieve.sh" "$DATASET" "$RESULT_MODEL" "$DEVICE"
fi

# Step 2: Evaluation
if [ $START_STEP -le 2 ] && [ "$SKIP_EVAL" = false ]; then
    "$SCRIPT_DIR/step2_eval.sh" "$DATASET" "$RESULT_MODEL" "$DEVICE"
fi

# Step 3: Aggregation
if [ $START_STEP -le 3 ]; then
    "$SCRIPT_DIR/step3_aggregate.sh" "$DATASET" "$RESULT_MODEL" "$VERSION"
fi

# Step 4: Filtering
if [ $START_STEP -le 4 ]; then
    "$SCRIPT_DIR/step4_filter.sh" "$DATASET" "$RESULT_MODEL" "$VERSION"
fi

# Step 5: Label Building
if [ $START_STEP -le 5 ]; then
    "$SCRIPT_DIR/step5_labels.sh" "$DATASET" "$RESULT_MODEL" "$VERSION"
fi

# Step 6: Split Building
if [ $START_STEP -le 6 ]; then
    "$SCRIPT_DIR/step6_split.sh" "$DATASET" "$RESULT_MODEL" "$VERSION"
fi

# Step 7: Training
if [ $START_STEP -le 7 ]; then
    "$SCRIPT_DIR/step7_train.sh" "$DATASET" "$RESULT_MODEL" "$VERSION" $TRAIN_ARGS
fi

# Step 8: Evaluation
if [ $START_STEP -le 8 ]; then
    "$SCRIPT_DIR/step8_eval.sh" "$DATASET" "$RESULT_MODEL" "$VERSION" $EVAL_ARGS
fi

echo ""
echo "============================================================"
echo "Pipeline Completed Successfully!"
echo "============================================================"
