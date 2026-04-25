#!/bin/bash
# Step 3: Aggregate Evaluation Data with Token Counts
# Usage: ./scripts/step3_aggregate.sh <dataset> <result_model> <version>

set -e

DATASET=${1:-"quality"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
VERSION=${3:-"v5"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# SAVE_NAME="query_metrics_${VERSION}"
SAVE_NAME="query_metrics_3class"

echo "============================================================"
echo "Step 3: Aggregation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Save Name: $SAVE_NAME"
echo ""

python Run/Router/run_aggregate_router_data_with_tokens.py \
    --dataset "$DATASET" \
    --result-model "$RESULT_MODEL" \
    --strategies llm_direct naive_rag graph_rag \
    --save-name "$SAVE_NAME"

echo ""
echo "Step 3 completed!"
echo "Output: Dataset/RouterTrainingData/Aggregated/$DATASET/$RESULT_MODEL/${SAVE_NAME}.json"
