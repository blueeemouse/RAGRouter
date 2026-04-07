"""Collect baseline and router offline performance on test split.

Usage examples:

1. 只计算 baseline + oracle：
   python Run/Router/run_collect_rag_baseline.py --dataset musique --result-model llama-3.1-8b-awq-int4

2. 基于已保存 prediction 计算 router routed performance：
   python Run/Router/run_collect_rag_baseline.py \
       --dataset musique \
       --result-model llama-3.1-8b-awq-int4 \
       --prediction-file Dataset/RouterTrainingData/Evaluation/text_router_baseline_v1/musique/musique_test_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RouterCore.RouterPathConfig import RouterPathConfig


METRIC_KEYS = [
    "llm_judge_correct",
    "semantic_f1",
    "coverage",
    "faithfulness_hard",
    "faithfulness_soft",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect RAG baseline or router offline performance on test split")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument("--result-model", type=str, required=True, help="Result model name, e.g. llama-3.1-8b-awq-int4")
    parser.add_argument("--split-name", type=str, default="split_v1", help="Split name (default: split_v1)")
    parser.add_argument(
        "--prediction-file",
        type=str,
        default=None,
        help="Optional saved test prediction file. If omitted, only baseline + oracle are collected.",
    )
    return parser.parse_args()


def load_aggregated_and_split(dataset: str, result_model: str, split_name: str):
    aggregated_path = RouterPathConfig.get_aggregated_path(dataset, result_model)
    split_path = RouterPathConfig.get_split_path(dataset, split_name)

    with aggregated_path.open("r", encoding="utf-8") as f:
        aggregated = json.load(f)
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    strategies = aggregated["metadata"]["strategies"]
    sample_lookup = {sample["id"]: sample for sample in aggregated["samples"]}
    test_ids = split["splits"]["test"]
    test_samples = [sample_lookup[qid] for qid in test_ids if qid in sample_lookup]
    return aggregated, split, strategies, sample_lookup, test_samples


def compute_average_metrics_for_strategies(samples: List[Dict[str, Any]], strategies: List[str]) -> Dict[str, Any]:
    """Compute average metrics for each fixed strategy across the provided samples."""
    strategy_metrics = defaultdict(lambda: {key: [] for key in METRIC_KEYS})

    for sample in samples:
        for strategy in strategies:
            metrics = sample.get("method_metrics", {}).get(strategy, {})
            for key in METRIC_KEYS:
                value = metrics.get(key)
                if value is not None:
                    strategy_metrics[strategy][key].append(value)

    avg_metrics: Dict[str, Any] = {}
    for strategy, metrics in strategy_metrics.items():
        avg_metrics[strategy] = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[strategy][key] = {
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }
    return avg_metrics


def compute_oracle_metrics(samples: List[Dict[str, Any]], strategies: List[str]) -> Dict[str, Any]:
    """Compute a semantic_f1-based oracle upper bound on the provided samples."""
    best_per_query = []
    strategy_distribution = defaultdict(int)

    # Metrics to compute for oracle
    oracle_metric_keys = ["llm_judge_correct", "semantic_f1", "coverage", "faithfulness_hard", "faithfulness_soft"]
    metric_accumulators = {key: [] for key in oracle_metric_keys}

    for sample in samples:
        best_strategy = None
        best_semantic_f1 = float("-inf")
        best_metrics = None
        for strategy in strategies:
            metrics = sample.get("method_metrics", {}).get(strategy, {})
            semantic_f1 = metrics.get("semantic_f1")
            if semantic_f1 is None:
                continue
            if semantic_f1 > best_semantic_f1:
                best_semantic_f1 = semantic_f1
                best_strategy = strategy
                best_metrics = metrics

        if best_strategy is None:
            continue

        strategy_distribution[best_strategy] += 1
        record = {
            "id": sample["id"],
            "best_strategy": best_strategy,
            "best_semantic_f1": best_semantic_f1,
        }
        for key in oracle_metric_keys:
            val = best_metrics.get(key)
            record[f"best_{key}"] = val
            if val is not None:
                metric_accumulators[key].append(val)
        best_per_query.append(record)

    oracle_metrics = {}
    for key, values in metric_accumulators.items():
        if values:
            oracle_metrics[key] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    oracle_metrics["strategy_distribution"] = dict(strategy_distribution)
    oracle_metrics["priority_order"] = strategies  # Record the priority order used for tie-breaking

    return {
        "oracle_metrics": oracle_metrics,
        "per_query_best": best_per_query,
    }


def compute_router_metrics_from_predictions(predictions: List[Dict[str, Any]], sample_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute routed average metrics from per-query prediction records."""
    metric_values = {key: [] for key in METRIC_KEYS}
    routed_records = []

    for record in predictions:
        sample_id = record["id"]
        predicted_strategy = record["predicted_strategy"]
        sample = sample_lookup.get(sample_id)
        if sample is None:
            continue
        metrics = sample.get("method_metrics", {}).get(predicted_strategy, {})
        routed_record = {
            "id": sample_id,
            "predicted_strategy": predicted_strategy,
        }
        for key in METRIC_KEYS:
            value = metrics.get(key)
            routed_record[key] = value
            if value is not None:
                metric_values[key].append(value)
        routed_records.append(routed_record)

    router_metrics = {}
    for key, values in metric_values.items():
        if values:
            router_metrics[key] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }

    return {
        "router_metrics": router_metrics,
        "routed_records": routed_records,
    }


def main():
    args = parse_args()

    aggregated, split, strategies, sample_lookup, test_samples = load_aggregated_and_split(
        args.dataset,
        args.result_model,
        args.split_name,
    )

    print(f"Aggregated data loaded for dataset={args.dataset}, result_model={args.result_model}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Strategies: {strategies}")

    if args.prediction_file is None:
        strategy_metrics = compute_average_metrics_for_strategies(test_samples, strategies)
        oracle = compute_oracle_metrics(test_samples, strategies)

        output_dir = RouterPathConfig.get_evaluation_dir("baseline_collection", args.dataset)
        RouterPathConfig.ensure_dir(output_dir)
        output_path = output_dir / f"rag_baseline_{args.result_model}_{args.split_name}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "dataset": args.dataset,
                        "result_model": args.result_model,
                        "split_name": args.split_name,
                        "test_size": len(test_samples),
                        "strategies": strategies,
                    },
                    "baseline_performance": strategy_metrics,
                    "oracle_performance": oracle["oracle_metrics"],
                    "per_query_best": oracle["per_query_best"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("Collected baseline + oracle performance on test split")
        print(f"Results saved to: {output_path}")
        return

    prediction_path = Path(args.prediction_file)
    with prediction_path.open("r", encoding="utf-8") as f:
        prediction_payload = json.load(f)

    predictions = prediction_payload.get("predictions", [])
    router = compute_router_metrics_from_predictions(predictions, sample_lookup)

    output_dir = prediction_path.parent
    output_path = output_dir / f"router_offline_eval_{args.split_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "dataset": args.dataset,
                    "result_model": args.result_model,
                    "split_name": args.split_name,
                    "prediction_file": str(prediction_path),
                    "test_size": len(test_samples),
                },
                "router_performance": router["router_metrics"],
                "per_query_routed": router["routed_records"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Computed router routed performance on test split")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
