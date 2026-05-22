"""Aggregate query-level evaluation outputs with complete token counts.

This aggregator extends EvaluationAggregator to also read token counts from
RetrievalResultData/{Strategy}/{model}/{dataset}/answer.jsonl files.

The issue being addressed:
- EvaluationData/ResultEvaluation/{model}/{dataset}/{strategy}.json may have None for
  input_tokens/output_tokens
- RetrievalResultData has complete token counts in answer.jsonl files
- This aggregator merges both sources

Usage:
    from RouterCore.Data.EvaluationAggregatorWithTokens import EvaluationAggregatorWithTokens
    aggregator = EvaluationAggregatorWithTokens()
    aggregated = aggregator.build(dataset_name="musique", result_model="llama-3.1-8b-awq-int4")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from Config.PathConfig import PathConfig
from RouterCore.Data.DatasetSchema import (
    STRATEGY_NAMES,
    build_empty_method_metrics,
    validate_sample_id,
)
from RouterCore.RouterPathConfig import RouterPathConfig


# Mapping from strategy_name to RetrievalResultData directory
STRATEGY_TO_DIR = {
    "llm_direct": "LLMDirect",
    "naive_rag": "NaiveRAG",
    "graph_rag": "GraphRAG",
    "hybrid_rag": "HybridRAG",
}

# Iterative RAG strategies have retriever_type in path
ITERATIVE_STRATEGY_TO_DIR = {
    "iterative_rag_naive": ("IterativeRAG", "naive"),
    "iterative_rag_graph": ("IterativeRAG", "graph"),
}


class EvaluationAggregatorWithTokens:
    """Aggregate evaluation data with token counts from RetrievalResultData.

    This aggregator:
    1. Reads quality metrics from EvaluationData/ResultEvaluation/
    2. Reads token counts from RetrievalResultData/answer.jsonl
    3. Merges them, preferring token counts from RetrievalResultData when eval file has None
    """

    def __init__(
        self,
        eval_file_suffix: str = "",
        strategies: Optional[List[str]] = None,
    ):
        """Initialize aggregator.

        Args:
            eval_file_suffix: Optional suffix for evaluation files, e.g., "_full" to look for
                             "graph_rag_full.json" instead of "graph_rag.json"
            strategies: Optional list of strategy names to process. If None, uses all STRATEGY_NAMES.
                       Useful for 3-class experiments (llm_direct, naive_rag, graph_rag).
        """
        self.eval_file_suffix = eval_file_suffix
        self.strategies = strategies if strategies is not None else STRATEGY_NAMES

    def build(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Build aggregated query-level metrics with complete token counts."""
        questions = self.load_questions(dataset_name)
        method_results = self.load_all_method_results(dataset_name, result_model)
        retrieval_tokens = self.load_all_retrieval_tokens(dataset_name, result_model)

        samples = self.aggregate_samples(questions, method_results, retrieval_tokens)

        return {
            "metadata": {
                "dataset": dataset_name,
                "result_model": result_model,
                "strategies": self.strategies,
                "note": "Token counts merged from RetrievalResultData",
            },
            "samples": samples,
        }

    def load_questions(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Load question records and index them by string id."""
        question_path = Path(PathConfig.get_question_path(dataset_name))
        with question_path.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                questions = json.load(f)
            else:
                questions = [json.loads(line) for line in f if line.strip()]

        indexed: Dict[str, Dict[str, Any]] = {}
        for record in questions:
            sample_id = record.get("id")
            validate_sample_id(sample_id)
            indexed[sample_id] = record
        return indexed

    def load_all_method_results(
        self, dataset_name: str, result_model: str
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load evaluation results for all strategies."""
        loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for strategy_name in self.strategies:
            loaded[strategy_name] = self.load_method_results(dataset_name, result_model, strategy_name)
        return loaded

    def load_method_results(
        self, dataset_name: str, result_model: str, strategy_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load evaluation results for one strategy."""
        # Determine eval file path
        if strategy_name in STRATEGY_TO_DIR:
            method_name = strategy_name
            retriever_type = None
        elif strategy_name in ITERATIVE_STRATEGY_TO_DIR:
            method_name = "iterative_rag"
            retriever_type = ITERATIVE_STRATEGY_TO_DIR[strategy_name][1]
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        eval_path = Path(
            PathConfig.get_result_eval_path(result_model, dataset_name, method_name, retriever_type)
        )
        # Apply suffix if specified
        if self.eval_file_suffix:
            eval_path = eval_path.parent / (eval_path.stem + self.eval_file_suffix + ".json")

        if not eval_path.exists():
            raise FileNotFoundError(f"Missing evaluation file for strategy '{strategy_name}': {eval_path}")

        with eval_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        indexed: Dict[str, Dict[str, Any]] = {}
        for record in records:
            sample_id = record.get("id")
            validate_sample_id(sample_id)
            indexed[sample_id] = record
        return indexed

    def load_all_retrieval_tokens(
        self, dataset_name: str, result_model: str
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load token counts from RetrievalResultData for all strategies."""
        loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for strategy_name in self.strategies:
            loaded[strategy_name] = self.load_retrieval_tokens(dataset_name, result_model, strategy_name)
        return loaded

    def load_retrieval_tokens(
        self, dataset_name: str, result_model: str, strategy_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load token counts from RetrievalResultData/answer.jsonl for one strategy."""
        # Determine path to answer.jsonl
        if strategy_name in STRATEGY_TO_DIR:
            strategy_dir = STRATEGY_TO_DIR[strategy_name]
            answer_path = Path(PathConfig.RETRIEVAL_RESULT_DIR) / strategy_dir / result_model / dataset_name / "answer.jsonl"
        elif strategy_name in ITERATIVE_STRATEGY_TO_DIR:
            strategy_dir, retriever_type = ITERATIVE_STRATEGY_TO_DIR[strategy_name]
            answer_path = (
                Path(PathConfig.RETRIEVAL_RESULT_DIR)
                / strategy_dir
                / result_model
                / retriever_type
                / dataset_name
                / "answer.jsonl"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        if not answer_path.exists():
            print(f"Warning: RetrievalResultData not found for strategy '{strategy_name}': {answer_path}")
            return {}

        tokens_by_id: Dict[str, Dict[str, Any]] = {}
        with answer_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sample_id = record.get("id")
                    validate_sample_id(sample_id)

                    # Extract token_usage.total
                    token_usage = record.get("token_usage", {})
                    if isinstance(token_usage, dict):
                        total = token_usage.get("total", {})
                    else:
                        total = {}

                    tokens_by_id[sample_id] = {
                        "input_tokens": total.get("in_tokens"),
                        "output_tokens": total.get("out_tokens"),
                        "total_tokens": total.get("total_tokens"),
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse line in {answer_path}: {e}")

        return tokens_by_id

    def aggregate_samples(
        self,
        questions: Dict[str, Dict[str, Any]],
        method_results: Dict[str, Dict[str, Dict[str, Any]]],
        retrieval_tokens: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Aggregate question records with per-method metrics and token counts."""
        samples: List[Dict[str, Any]] = []
        for sample_id, question_record in questions.items():
            validate_sample_id(sample_id)
            samples.append(
                self.aggregate_one_sample(sample_id, question_record, method_results, retrieval_tokens)
            )
        return samples

    def aggregate_one_sample(
        self,
        sample_id: str,
        question_record: Dict[str, Any],
        method_results: Dict[str, Dict[str, Dict[str, Any]]],
        retrieval_tokens: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Aggregate one query's per-method metrics with token counts."""
        method_metrics = build_empty_method_metrics()

        for strategy_name in self.strategies:
            eval_record = method_results[strategy_name].get(sample_id, {})
            token_record = retrieval_tokens[strategy_name].get(sample_id, {})

            method_metrics[strategy_name] = self.extract_method_metrics(eval_record, token_record)

        return {
            "id": sample_id,
            "question": question_record.get("question", ""),
            "ground_truth": question_record.get("answer", ""),
            "method_metrics": method_metrics,
        }

    def extract_method_metrics(
        self, evaluation_record: Dict[str, Any], token_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract canonical metrics from evaluation record, filling token counts from retrieval tokens.

        Token counts are taken from RetrievalResultData (token_record) when the evaluation
        file (evaluation_record) has None for input_tokens/output_tokens.
        """
        llm_label = evaluation_record.get("llm_label")

        # Get token counts from evaluation record
        eval_input_tokens = evaluation_record.get("input_tokens")
        eval_output_tokens = evaluation_record.get("output_tokens")

        # Get token counts from retrieval result
        retrieval_input_tokens = token_record.get("input_tokens")
        retrieval_output_tokens = token_record.get("output_tokens")

        # Prefer retrieval tokens when eval has None
        input_tokens = eval_input_tokens if eval_input_tokens is not None else retrieval_input_tokens
        output_tokens = eval_output_tokens if eval_output_tokens is not None else retrieval_output_tokens

        return {
            "llm_label": llm_label,
            "llm_reason": evaluation_record.get("llm_reason"),
            "llm_judge_correct": 1 if llm_label == "correct" else 0 if llm_label is not None else None,
            "semantic_f1": evaluation_record.get("semantic_f1"),
            "token_f1": evaluation_record.get("token_f1"),
            "bleu1": evaluation_record.get("bleu1"),
            "rouge1_f": evaluation_record.get("rouge1_f"),
            "rouge2_f": evaluation_record.get("rouge2_f"),
            "rougeL_f": evaluation_record.get("rougeL_f"),
            "meteor": evaluation_record.get("meteor"),
            "coverage": evaluation_record.get("coverage"),
            "faithfulness_hard": evaluation_record.get("faithfulness_hard"),
            "faithfulness_soft": evaluation_record.get("faithfulness_soft"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def save(
        self,
        aggregated: Dict[str, Any],
        dataset_name: str,
        result_model: str,
        save_name: Optional[str] = None,
    ) -> Path:
        """Save aggregated query metrics under RouterTrainingData/Aggregated.

        Args:
            aggregated: The aggregated data dict
            dataset_name: Dataset name
            result_model: Result model name
            save_name: Optional custom filename (without .json). If None, uses default.

        Returns:
            Path to saved file
        """
        output_path = RouterPathConfig.get_aggregated_path(dataset_name, result_model)
        if save_name:
            output_path = output_path.with_name(f"{save_name}.json")

        RouterPathConfig.ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        return output_path
