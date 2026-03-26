import os
import json
from typing import Dict, Any
from Config.PathConfig import PathConfig


class SemanticEvaluationSaver:
    """Save and load semantic evaluation results"""

    @staticmethod
    def save(metrics: Dict[str, Any], dataset_name: str):
        """Save semantic evaluation metrics to JSON file

        Args:
            metrics: Output from SemanticEvaluator.evaluate()
            dataset_name: Name of the dataset (e.g., "hotpotqa")
        """
        # Get save directory
        save_dir = PathConfig.get_corpus_eval_path(dataset_name, "SemanticEvaluation")
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(save_dir, "semantic_evaluation.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"Saved semantic evaluation to: {metrics_path}")

        # Print file size
        file_size_kb = os.path.getsize(metrics_path) / 1024
        print(f"File size: {file_size_kb:.2f} KB")

    @staticmethod
    def load(dataset_name: str) -> Dict[str, Any]:
        """Load semantic evaluation metrics from JSON file

        Args:
            dataset_name: Name of the dataset (e.g., "hotpotqa")

        Returns:
            Dict containing semantic evaluation metrics
        """
        # Get load directory
        load_dir = PathConfig.get_corpus_eval_path(dataset_name, "SemanticEvaluation")
        metrics_path = os.path.join(load_dir, "semantic_evaluation.json")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Semantic evaluation file not found: {metrics_path}")

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        print(f"Loaded semantic evaluation from: {metrics_path}")
        return metrics


# Usage example
if __name__ == "__main__":
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
    from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationDo import SemanticEvaluator

    dataset_name = "hotpotqa"

    # Step 1: Load embeddings
    print("Loading embeddings...")
    embeddings_by_doc = EmbeddingSaver.load(dataset_name)

    # Step 2: Evaluate semantic quality
    print("\nEvaluating semantic quality...")
    evaluator = SemanticEvaluator(hubness_k=10)
    metrics = evaluator.evaluate(embeddings_by_doc)

    print(f"\nEvaluation complete:")
    print(f"  Status: {metrics.get('status')}")
    print(f"  Total Embeddings: {metrics.get('total_embeddings')}")
    print(f"  Embedding Dimension: {metrics.get('embedding_dimension')}")

    # Step 3: Save metrics
    print("\nSaving metrics...")
    SemanticEvaluationSaver.save(metrics, dataset_name)

    # Step 4: Load metrics (to verify)
    print("\nLoading metrics to verify...")
    loaded_metrics = SemanticEvaluationSaver.load(dataset_name)

    print(f"\nVerification:")
    print(f"  Original status: {metrics.get('status')}")
    print(f"  Loaded status: {loaded_metrics.get('status')}")
    print(f"  Original total_embeddings: {metrics.get('total_embeddings')}")
    print(f"  Loaded total_embeddings: {loaded_metrics.get('total_embeddings')}")
    print(f"  Match: {metrics == loaded_metrics}")
