"""
Run Structure Evaluation

This script loads the graph and evaluates its structure.

Usage:
    python Run/Evaluation/run_structure_eval.py
    python Run/Evaluation/run_structure_eval.py --dataset test
"""
import argparse
import sys

from RAGCore.Graph.GraphSave import GraphSaver
from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationDo import StructureEvaluator
from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationSave import StructureEvaluationSaver


def run_structure_eval(dataset_name: str):
    """Run structure evaluation for a dataset

    Args:
        dataset_name: Name of the dataset (e.g., "hotpotqa", "test")
    """
    try:
        # Step 1: Load graph
        print(f"Loading graph for dataset '{dataset_name}'...")
        result = GraphSaver.load(dataset_name)
        graph = result['graph']

        # Step 2: Evaluate structure
        print("Evaluating graph structure...")
        evaluator = StructureEvaluator()
        metrics = evaluator.evaluate(graph)

        # Step 3: Save evaluation results
        StructureEvaluationSaver.save(metrics, dataset_name)

        print(f"Structure evaluation completed successfully")
        return 0
    except Exception as e:
        print(f"Structure evaluation failed: {e}")
        return 1


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Run structure evaluation")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    args = parser.parse_args()

    exit_code = run_structure_eval(args.dataset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
