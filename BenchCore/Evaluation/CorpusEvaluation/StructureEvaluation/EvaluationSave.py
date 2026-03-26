import os
import json
from typing import Dict, Any
from Config.PathConfig import PathConfig


class StructureEvaluationSaver:
    """Save and load structure evaluation results"""

    @staticmethod
    def save(metrics: Dict[str, Any], dataset_name: str):
        """Save structure evaluation metrics to JSON file

        Args:
            metrics: Output from StructureEvaluator.evaluate()
            dataset_name: Name of the dataset (e.g., "hotpotqa")
        """
        # Get save directory
        save_dir = PathConfig.get_corpus_eval_path(dataset_name, "StructureEvaluation")
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(save_dir, "structure_evaluation.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"Saved structure evaluation to: {metrics_path}")

        # Print file size
        file_size_kb = os.path.getsize(metrics_path) / 1024
        print(f"File size: {file_size_kb:.2f} KB")

    @staticmethod
    def load(dataset_name: str) -> Dict[str, Any]:
        """Load structure evaluation metrics from JSON file

        Args:
            dataset_name: Name of the dataset (e.g., "hotpotqa")

        Returns:
            Dict containing structure evaluation metrics
        """
        # Get load directory
        load_dir = PathConfig.get_corpus_eval_path(dataset_name, "StructureEvaluation")
        metrics_path = os.path.join(load_dir, "structure_evaluation.json")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Structure evaluation file not found: {metrics_path}")

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        print(f"Loaded structure evaluation from: {metrics_path}")
        return metrics


# Usage example
if __name__ == "__main__":
    from RAGCore.Graph.GraphSave import GraphSaver
    from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationDo import StructureEvaluator

    dataset_name = "hotpotqa"

    # Step 1: Load graph
    print("Loading graph...")
    graph = GraphSaver.load(dataset_name)

    # Step 2: Evaluate structure
    print("\nEvaluating graph structure...")
    evaluator = StructureEvaluator()
    metrics = evaluator.evaluate(graph)

    print(f"\nEvaluation complete:")
    print(f"  Nodes: {metrics['basic_stats']['num_nodes']}")
    print(f"  Edges: {metrics['basic_stats']['num_edges']}")
    print(f"  Connected: {metrics['connectivity']['is_connected']}")

    # Step 3: Save metrics
    print("\nSaving metrics...")
    StructureEvaluationSaver.save(metrics, dataset_name)

    # Step 4: Load metrics (to verify)
    print("\nLoading metrics to verify...")
    loaded_metrics = StructureEvaluationSaver.load(dataset_name)

    print(f"\nVerification:")
    print(f"  Original nodes: {metrics['basic_stats']['num_nodes']}")
    print(f"  Loaded nodes: {loaded_metrics['basic_stats']['num_nodes']}")
    print(f"  Match: {metrics == loaded_metrics}")
