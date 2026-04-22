#!/bin/bash
# Step 5: Build Training Labels
# Usage: ./scripts/step5_labels.sh <dataset> <result_model> <version>

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

AGGREGATED_NAME="query_metrics_${VERSION}_filtered"
LABEL_NAME="hard_llm_correct_rule_${VERSION}"

echo "============================================================"
echo "Step 5: Label Building"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Aggregated: $AGGREGATED_NAME"
echo "Label Name: $LABEL_NAME"
echo ""

python Run/Router/run_build_router_labels_with_source.py \
    --dataset "$DATASET" \
    --result-model "$RESULT_MODEL" \
    --aggregated-name "$AGGREGATED_NAME" \
    --label-name "$LABEL_NAME"

echo ""
echo "Step 5 completed!"
echo "Output: Dataset/RouterTrainingData/Labels/$DATASET/$RESULT_MODEL/${LABEL_NAME}.json"
