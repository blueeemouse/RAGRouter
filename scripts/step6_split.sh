#!/bin/bash
# Step 6: Build Train/Val/Test Splits
# Usage: ./scripts/step6_split.sh <dataset> <result_model> <version> [train_ratio] [val_ratio] [seed]

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}
TRAIN_RATIO=${4:-"0.8"}
VAL_RATIO=${5:-"0.1"}
SEED=${6:-"42"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LABEL_NAME="hard_llm_correct_rule_${VERSION}"
SPLIT_NAME="split_${VERSION}_$(echo "$TRAIN_RATIO" | tr -d '.')_$(echo "$VAL_RATIO" | tr -d '.')_1"

echo "============================================================"
echo "Step 6: Split Building"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Label Name: $LABEL_NAME"
echo "Split Name: $SPLIT_NAME"
echo "Train Ratio: $TRAIN_RATIO"
echo "Val Ratio: $VAL_RATIO"
echo "Seed: $SEED"
echo ""

python Run/Router/run_build_router_split.py \
    --dataset "$DATASET" \
    --result-model "$RESULT_MODEL" \
    --label-name "$LABEL_NAME" \
    --split-name "$SPLIT_NAME" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --seed "$SEED"

echo ""
echo "Step 6 completed!"
echo "Output: Dataset/RouterTrainingData/Splits/$DATASET/${SPLIT_NAME}.json"
