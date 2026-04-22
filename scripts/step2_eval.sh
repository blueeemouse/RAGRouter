#!/bin/bash
# Step 2: Evaluate RAG Answers
# Usage: ./scripts/step2_eval.sh <dataset> <result_model>

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Step 2: Evaluation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo ""

STRATEGIES=("llm_direct" "naive_rag" "graph_rag")

for strategy in "${STRATEGIES[@]}"; do
    echo "Evaluating $strategy..."
    python Run/Evaluation/run_result_eval.py \
        --dataset "$DATASET" \
        --method "$strategy" \
        --result-model "$RESULT_MODEL"
    echo ""
done

echo "Step 2 completed!"
