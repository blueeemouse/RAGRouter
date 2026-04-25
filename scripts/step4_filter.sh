#!/bin/bash
# Step 4: Filter All-Failed Samples
# Usage: ./scripts/step4_filter.sh <dataset> <result_model> <version>

set -e

DATASET=${1:-"quality"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# AGGREGATED_NAME="query_metrics_${VERSION}"
AGGREGATED_NAME="query_metrics_3class"
# SAVE_NAME="query_metrics_${VERSION}_filtered"
SAVE_NAME="query_metrics_filtered"

echo "============================================================"
echo "Step 4: Filtering All-Failed Samples"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Input: $AGGREGATED_NAME"
echo "Output: $SAVE_NAME"
echo ""

python Run/Router/run_filter_all_failed.py \
    --dataset "$DATASET" \
    --result-model "$RESULT_MODEL" \
    --aggregated-name "$AGGREGATED_NAME" \
    --save-name "$SAVE_NAME"

echo ""
echo "Step 4 completed!"
echo "Output: Dataset/RouterTrainingData/Aggregated/$DATASET/$RESULT_MODEL/${SAVE_NAME}.json"
