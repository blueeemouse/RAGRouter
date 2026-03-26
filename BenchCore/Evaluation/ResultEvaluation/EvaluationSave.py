"""
Result Evaluation Saver

Handles saving and loading of evaluation results.

Output path: /Dataset/EvaluationData/ResultEvaluation/{model}/{dataset}/{method}.json
"""
import json
import os
from typing import Dict, List, Any

from Config.PathConfig import PathConfig


class EvaluationSaver:
    """Save and load evaluation results"""

    @staticmethod
    def save(results: List[Dict[str, Any]], model_name: str, dataset_name: str, method: str, retriever_type: str = None):
        """Save evaluation results to JSON file

        Args:
            results: List of evaluation results
                [{"id": 0, "llm_retrieval_score": 7, "llm_ans_score": 5}, ...]
            model_name: Model name (e.g., "deepseek-chat")
            dataset_name: Dataset name (e.g., "musique")
            method: "naive_rag" or "graph_rag"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")

        Output path: /Dataset/EvaluationData/ResultEvaluation/{model}/{dataset}/{method}.json
                     For iterative_rag: {method}_{retriever_type}.json
        """
        output_path = PathConfig.get_result_eval_path(model_name, dataset_name, method, retriever_type)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} evaluation results to: {output_path}")

    @staticmethod
    def load(model_name: str, dataset_name: str, method: str, retriever_type: str = None) -> List[Dict[str, Any]]:
        """Load existing evaluation results

        Args:
            model_name: Model name
            dataset_name: Dataset name
            method: "naive_rag" or "graph_rag"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")

        Returns:
            List of evaluation results, or empty list if not exists
        """
        output_path = PathConfig.get_result_eval_path(model_name, dataset_name, method, retriever_type)

        if not os.path.exists(output_path):
            return []

        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def get_evaluated_ids(model_name: str, dataset_name: str, method: str, retriever_type: str = None) -> set:
        """Get set of already evaluated question IDs

        Args:
            model_name: Model name
            dataset_name: Dataset name
            method: "naive_rag" or "graph_rag"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")

        Returns:
            Set of evaluated question IDs
        """
        results = EvaluationSaver.load(model_name, dataset_name, method, retriever_type)
        return {r['id'] for r in results}
