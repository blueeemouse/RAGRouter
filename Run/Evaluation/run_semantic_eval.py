"""
Run Semantic Evaluation

This script loads embeddings and evaluates semantic quality.

Usage:
    python Run/Evaluation/run_semantic_eval.py
    python Run/Evaluation/run_semantic_eval.py --dataset test
    python Run/Evaluation/run_semantic_eval.py --dataset test --hubness_k 10
"""
import argparse
import sys

from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationDo import SemanticEvaluator
from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationSave import SemanticEvaluationSaver


def run_semantic_eval(dataset_name: str, hubness_k: int = 10):
    """Run semantic evaluation for a dataset

    Args:
        dataset_name: Name of the dataset (e.g., "hotpotqa", "test")
        hubness_k: K value for hubness calculation (default: 10)
    """
    try:
        # Step 1: Load embeddings
        print(f"Loading embeddings for dataset '{dataset_name}'...")
        embeddings_by_doc = EmbeddingSaver.load(dataset_name)

        # Step 2: Evaluate semantic quality
        print("Evaluating semantic quality...")
        evaluator = SemanticEvaluator(hubness_k=hubness_k)
        metrics = evaluator.evaluate(embeddings_by_doc)

        # Check if evaluation succeeded
        if metrics.get('status') != 'success':
            raise RuntimeError(f"Evaluation failed: {metrics.get('error')}")

        # Step 3: Save evaluation results
        SemanticEvaluationSaver.save(metrics, dataset_name)

        print(f"Semantic evaluation completed successfully")
        return 0
    except Exception as e:
        print(f"Semantic evaluation failed: {e}")
        return 1


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Run semantic evaluation")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    parser.add_argument("--hubness_k", type=int, default=10, help="K value for hubness calculation")
    args = parser.parse_args()

    exit_code = run_semantic_eval(args.dataset, args.hubness_k)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
