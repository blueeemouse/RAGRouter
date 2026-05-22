#!/bin/bash
# Step 2: Evaluate RAG Answers
# Usage: ./scripts/step2_eval.sh <dataset> [result_model] [device]

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
DEVICE=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Step 2: Evaluation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model: $RESULT_MODEL"
echo "Device: ${DEVICE:-auto}"
echo "Note: actual model is controlled by Config/LLMConfig.py"
echo ""

STRATEGIES=("llm_direct" "naive_rag" "graph_rag")

for strategy in "${STRATEGIES[@]}"; do
    echo "Evaluating $strategy..."
    if [ -n "$DEVICE" ]; then
        CUDA_VISIBLE_DEVICES="$DEVICE" python main.py evaluate result \
            --dataset "$DATASET" \
            --method "$strategy"
    else
        python main.py evaluate result \
            --dataset "$DATASET" \
            --method "$strategy"
    fi
    echo ""
done

echo "Step 2 completed!"
