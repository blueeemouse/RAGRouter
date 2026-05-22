"""Aggregate query-level evaluation outputs into a unified router training dataset.

The aggregator is responsible for reorganizing per-method query-level evaluation
results into a single query-centric view. It does not build hard/soft labels,
splits, or training inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from Config.PathConfig import PathConfig
from RouterCore.Data.DatasetSchema import (
    STRATEGY_NAMES,
    build_empty_method_metrics,
    normalize_strategy_name,
    validate_sample_id,
)
from RouterCore.RouterPathConfig import RouterPathConfig


@dataclass(frozen=True)
class AggregatorInputSpec:
    """Describe one upstream evaluation source to load."""

    strategy_name: str
    method: str
    retriever_type: Optional[str] = None


class EvaluationAggregator:
    """Aggregate per-method result evaluation files into unified query metrics."""

    DEFAULT_INPUT_SPECS: List[AggregatorInputSpec] = [
        AggregatorInputSpec(strategy_name="llm_direct", method="llm_direct"),
        AggregatorInputSpec(strategy_name="naive_rag", method="naive_rag"),
        AggregatorInputSpec(strategy_name="graph_rag", method="graph_rag"),
        AggregatorInputSpec(strategy_name="hybrid_rag", method="hybrid_rag"),
        AggregatorInputSpec(strategy_name="iterative_rag_naive", method="iterative_rag", retriever_type="naive"),
        AggregatorInputSpec(strategy_name="iterative_rag_graph", method="iterative_rag", retriever_type="graph"),
    ]

    def __init__(self, input_specs: Optional[Iterable[AggregatorInputSpec]] = None):
        self.input_specs = list(input_specs) if input_specs is not None else list(self.DEFAULT_INPUT_SPECS)

    def build(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Build aggregated query-level metrics for a dataset/model pair.

        Current expectation:
        - questions come from RawData/{dataset}/Question.json
        - per-method evaluation results come from EvaluationData/ResultEvaluation/... when available
        - all required strategies must be present before a full aggregation is considered valid
        """
        questions = self.load_questions(dataset_name)
        method_results = self.load_all_method_results(dataset_name, result_model)
        samples = self.aggregate_samples(questions, method_results)

        aggregated = {
            "metadata": {
                "dataset": dataset_name,
                "result_model": result_model,
                "strategies": STRATEGY_NAMES,
            },
            "samples": samples,
        }
        return aggregated

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

    def load_all_method_results(self, dataset_name: str, result_model: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load query-level evaluation results for all configured strategies.

        Returns:
            strategy_name -> sample_id -> evaluation_record
        """
        loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for spec in self.input_specs:
            loaded[spec.strategy_name] = self.load_method_results(dataset_name, result_model, spec)
        return loaded

    def load_method_results(
        self,
        dataset_name: str,
        result_model: str,
        spec: AggregatorInputSpec,
    ) -> Dict[str, Dict[str, Any]]:
        """Load one method's result-evaluation file and index by string id."""
        eval_path = Path(PathConfig.get_result_eval_path(result_model, dataset_name, spec.method, spec.retriever_type))
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Missing evaluation file for strategy '{spec.strategy_name}': {eval_path}"
            )

        with eval_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        indexed: Dict[str, Dict[str, Any]] = {}
        for record in records:
            sample_id = record.get("id")
            validate_sample_id(sample_id)
            indexed[sample_id] = record
        return indexed

    def aggregate_samples(
        self,
        questions: Dict[str, Dict[str, Any]],
        method_results: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Aggregate question records with per-method metrics."""
        samples: List[Dict[str, Any]] = []
        for sample_id, question_record in questions.items():
            validate_sample_id(sample_id)
            samples.append(self.aggregate_one_sample(sample_id, question_record, method_results))
        return samples

    def aggregate_one_sample(
        self,
        sample_id: str,
        question_record: Dict[str, Any],
        method_results: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Aggregate one query's per-method metrics into the canonical sample schema."""
        method_metrics = build_empty_method_metrics()

        for strategy_name in STRATEGY_NAMES:
            strategy_records = method_results.get(strategy_name, {})
            if sample_id not in strategy_records:
                raise KeyError(f"Sample '{sample_id}' missing strategy results for '{strategy_name}'")
            method_metrics[strategy_name] = self.extract_method_metrics(strategy_records[sample_id])

        return {
            "id": sample_id,
            "question": question_record.get("question", ""),
            "ground_truth": question_record.get("answer", ""),
            "method_metrics": method_metrics,
        }

    def extract_method_metrics(self, evaluation_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the canonical first-stage metrics from one evaluation record."""
        llm_label = evaluation_record.get("llm_label")
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
            "input_tokens": evaluation_record.get("input_tokens"),
            "output_tokens": evaluation_record.get("output_tokens"),
        }

    def save(self, aggregated: Dict[str, Any], dataset_name: str, result_model: str) -> Path:
        """Save aggregated query metrics under RouterTrainingData/Aggregated."""
        output_path = RouterPathConfig.get_aggregated_path(dataset_name, result_model)
        RouterPathConfig.ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        return output_path
