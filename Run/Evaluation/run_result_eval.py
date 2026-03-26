"""
Run Result Evaluation

Evaluates RAG results with all metrics:
- semantic_f1: BERTScore-based semantic similarity
- coverage: How well the prediction covers ground truth
- faithfulness_hard/soft: How well the answer is grounded in retrieval
- llm_label: LLM-based correctness classification

Usage:
    python run_result_eval.py --dataset quality --method naive_rag
    python run_result_eval.py --dataset quality --method graph_rag
    python run_result_eval.py --dataset quality --method hybrid_rag
    python run_result_eval.py --dataset quality --method iterative_rag --retriever-type naive
    python run_result_eval.py --dataset quality --method llm_direct
    python run_result_eval.py --dataset quality --method naive_rag --skip-llm  # Skip LLM evaluation (faster)
    python run_result_eval.py --dataset quality --method naive_rag --no-resume  # Don't resume from existing

    # Evaluate Llama results using DeepSeek as judge:
    python run_result_eval.py --dataset quality --method naive_rag --result-model llama-3.1-8b-awq
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from BenchCore.Evaluation.ResultEvaluation.EvaluationDo import ResultEvaluator
from Config.PathConfig import PathConfig
from Config.LLMConfig import LLMConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG results with all metrics",
        epilog="Model is configured in LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., quality)")
    parser.add_argument("--method", type=str, required=True,
                        choices=["naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"],
                        help="Method to evaluate")
    parser.add_argument("--retriever-type", type=str, default="graph",
                        choices=["naive", "graph"],
                        help="Base retriever for iterative_rag (default: graph)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-based evaluation (faster)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls (default: 10)")
    parser.add_argument("--result-model", type=str, default=None,
                        help="Model name for loading results (e.g., llama-3.1-8b-awq). If not specified, uses current LLM config.")
    args = parser.parse_args()

    # Get evaluation LLM config
    model_config = LLMConfig.get_model_config()
    eval_model = model_config["model"]

    # Determine which model's results to evaluate
    # If --result-model is specified, use it; otherwise use config model_name
    if args.result_model:
        result_model = args.result_model
    else:
        result_model = model_config.get("model_name", model_config["model"])

    print("=" * 60)
    print("Result Evaluation (All Metrics)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Results from model: {result_model}")
    print(f"Evaluation LLM: {eval_model}")
    if args.method == "iterative_rag":
        print(f"Retriever Type: {args.retriever_type}")
    print(f"Resume: {not args.no_resume}")
    print(f"Skip LLM: {args.skip_llm}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print("=" * 60)

    # Run evaluation
    evaluator = ResultEvaluator()
    results = evaluator.evaluate_all_metrics(
        model_name=result_model,
        dataset_name=args.dataset,
        method=args.method,
        retriever_type=args.retriever_type,
        resume=not args.no_resume,
        skip_llm=args.skip_llm,
        max_concurrent=args.max_concurrent
    )

    # Print output path
    if results:
        retriever_type = args.retriever_type if args.method == "iterative_rag" else None
        output_path = PathConfig.get_result_eval_path(result_model, args.dataset, args.method, retriever_type)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
