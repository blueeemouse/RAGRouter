"""Build hard labels from aggregated router query metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from RouterCore.Data.DatasetSchema import (
    STRATEGY_NAMES,
    get_strategy_index_from_list,
    validate_sample_id,
    validate_strategy_names,
)
from RouterCore.RouterPathConfig import RouterPathConfig


class HardLabelBuilder:
    """Construct hard-label router supervision data from aggregated metrics."""

    LABEL_NAME = "hard_llm_correct_rule_v1"
    TIE_BREAK_RULE = "priority_order_v1"
    PRIMARY_METRIC = "llm_judge_correct"
    FALLBACK_METRIC = "semantic_f1"
    ALL_FAILED_STRATEGY = "all_failed"
    ALL_FAILED_LABEL_SUFFIX = "_v2_all_failed_class"
    ALL_FAILED_RULE = {
        "llm_judge_correct": "!= 1",
        "semantic_f1": "== 0",
    }
    PRIORITY_ORDER: List[str] = [
        "llm_direct",
        "naive_rag",
        "graph_rag",
        "iterative_rag_naive",
        "iterative_rag_graph",
        "hybrid_rag",
    ]

    def __init__(self, label_name: str | None = None):
        self.label_name = label_name or self.LABEL_NAME
        self.enable_all_failed = self.label_name.endswith(self.ALL_FAILED_LABEL_SUFFIX)
        self.active_strategies = STRATEGY_NAMES.copy()
        if self.enable_all_failed:
            self.active_strategies.append(self.ALL_FAILED_STRATEGY)

    def load_aggregated(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Load aggregated query metrics from RouterTrainingData/Aggregated."""
        aggregated_path = RouterPathConfig.get_aggregated_path(dataset_name, result_model)
        if not aggregated_path.exists():
            raise FileNotFoundError(f"Missing aggregated router data file: {aggregated_path}")

        with aggregated_path.open("r", encoding="utf-8") as f:
            aggregated = json.load(f)

        metadata = aggregated.get("metadata", {})
        validate_strategy_names(metadata.get("strategies", []))
        samples = aggregated.get("samples")
        if not isinstance(samples, list):
            raise ValueError("Aggregated router data must contain a 'samples' list")
        return aggregated

    def build(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Build hard-label data for a dataset/model pair."""
        aggregated = self.load_aggregated(dataset_name, result_model)
        samples = aggregated["samples"]

        hard_samples = [self.build_one_sample(sample) for sample in samples]
        return {
            "metadata": {
                "dataset": dataset_name,
                "result_model": result_model,
                "label_name": self.label_name,
                "source_aggregated_file": RouterPathConfig.get_aggregated_path(dataset_name, result_model).name,
                "strategies": self.active_strategies,
                "primary_metric": self.PRIMARY_METRIC,
                "fallback_metric": self.FALLBACK_METRIC,
                "tie_break_rule": self.TIE_BREAK_RULE,
                "priority_order": self.PRIORITY_ORDER,
                "all_failed_rule": self.ALL_FAILED_RULE if self.enable_all_failed else None,
            },
            "samples": hard_samples,
        }

    def build_one_sample(self, aggregated_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Build one hard-label sample from one aggregated query sample."""
        sample_id = aggregated_sample.get("id")
        validate_sample_id(sample_id)

        method_metrics = aggregated_sample.get("method_metrics")
        if not isinstance(method_metrics, dict):
            raise ValueError(f"Aggregated sample '{sample_id}' must contain a method_metrics dict")

        all_failed = self.is_all_failed_sample(method_metrics)
        if self.enable_all_failed and all_failed:
            optimal_strategy = self.ALL_FAILED_STRATEGY
            candidate_best_strategies = [optimal_strategy]
            decision_source = "all_failed_gate"
        else:
            candidate_best_strategies, decision_source = self.select_candidate_strategies(sample_id, method_metrics)
            optimal_strategy = self.apply_priority_order(candidate_best_strategies)

        return {
            "id": sample_id,
            "optimal_strategy": optimal_strategy,
            "label_index": get_strategy_index_from_list(optimal_strategy, self.active_strategies),
            "candidate_best_strategies": candidate_best_strategies,
            "decision_source": decision_source,
        }

    def is_all_failed_sample(self, method_metrics: Dict[str, Dict[str, Any]]) -> bool:
        """Return True when all base strategies are judged failed by strict rule."""
        for strategy_name in STRATEGY_NAMES:
            metrics = method_metrics.get(strategy_name, {})
            if metrics.get("llm_judge_correct") == 1:
                return False
            if metrics.get("semantic_f1") != 0:
                return False
        return True

    def select_candidate_strategies(
        self,
        sample_id: str,
        method_metrics: Dict[str, Dict[str, Any]],
    ) -> tuple[List[str], str]:
        """Select pre-tie-break candidate strategies for one sample."""
        self.validate_method_metrics(sample_id, method_metrics)

        llm_correct_candidates = [
            strategy_name
            for strategy_name in STRATEGY_NAMES
            if method_metrics[strategy_name].get("llm_judge_correct") == 1
        ]
        if llm_correct_candidates:
            return llm_correct_candidates, "llm_judge_correct"

        semantic_f1_values = {
            strategy_name: method_metrics[strategy_name].get("semantic_f1")
            for strategy_name in STRATEGY_NAMES
        }
        if any(value is None for value in semantic_f1_values.values()):
            missing = [name for name, value in semantic_f1_values.items() if value is None]
            raise ValueError(
                f"Sample '{sample_id}' has no llm_judge_correct winner and missing semantic_f1 for: {missing}"
            )

        max_semantic_f1 = max(semantic_f1_values.values())
        semantic_f1_candidates = [
            strategy_name
            for strategy_name, value in semantic_f1_values.items()
            if value == max_semantic_f1
        ]
        return semantic_f1_candidates, "semantic_f1"

    def apply_priority_order(self, candidate_best_strategies: List[str]) -> str:
        """Collapse candidate best strategies into one hard label using the agreed order."""
        candidate_set = set(candidate_best_strategies)
        for strategy_name in self.PRIORITY_ORDER:
            if strategy_name in candidate_set:
                return strategy_name
        raise ValueError(
            "Could not resolve optimal strategy from candidate_best_strategies: "
            f"{candidate_best_strategies}"
        )

    def validate_method_metrics(self, sample_id: str, method_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Validate the minimal schema required for hard-label construction."""
        missing_strategies = [strategy_name for strategy_name in STRATEGY_NAMES if strategy_name not in method_metrics]
        if missing_strategies:
            raise ValueError(f"Sample '{sample_id}' missing method_metrics for strategies: {missing_strategies}")

        for strategy_name in STRATEGY_NAMES:
            metrics = method_metrics[strategy_name]
            if not isinstance(metrics, dict):
                raise ValueError(f"Sample '{sample_id}' strategy '{strategy_name}' metrics must be a dict")
            if "llm_judge_correct" not in metrics:
                raise ValueError(
                    f"Sample '{sample_id}' strategy '{strategy_name}' missing llm_judge_correct"
                )
            if "semantic_f1" not in metrics:
                raise ValueError(f"Sample '{sample_id}' strategy '{strategy_name}' missing semantic_f1")

    def save(self, hard_labels: Dict[str, Any], dataset_name: str, result_model: str) -> Path:
        """Save hard labels under RouterTrainingData/Labels."""
        output_path = RouterPathConfig.get_hard_label_path(
            dataset_name=dataset_name,
            result_model=result_model,
            label_name=self.label_name,
        )
        RouterPathConfig.ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(hard_labels, f, ensure_ascii=False, indent=2)
        return output_path
