#!/bin/bash
# Step 1: Collect raw RAG answers (RAGRouter-Bench style)
# Usage: ./scripts/step1_retrieve.sh <dataset> [result_model] [device]

set -e

DATASET=${1:-"musique"}
RESULT_MODEL=${2:-"llama-3.1-8b-awq-int4"}
DEVICE=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Step 1: Retrieval (Collect Raw Answers)"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Result Model (info): $RESULT_MODEL"
echo "Device: ${DEVICE:-auto}"
echo "Note: actual model is controlled by Config/LLMConfig.py"
echo ""

# Use RAGRouter-Bench style retrieve commands
STRATEGIES=("llm_direct" "naive" "graph")

for strategy in "${STRATEGIES[@]}"; do
    echo "Running retrieve: $strategy ..."
    if [ -n "$DEVICE" ]; then
        CUDA_VISIBLE_DEVICES="$DEVICE" python main.py retrieve "$strategy" --dataset "$DATASET"
    else
        python main.py retrieve "$strategy" --dataset "$DATASET"
    fi
    echo ""
done

echo "Step 1 completed!"
