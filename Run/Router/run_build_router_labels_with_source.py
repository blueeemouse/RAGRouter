"""CLI entrypoint for building router label files with custom aggregated data source.

This script extends run_build_router_labels.py to support specifying which aggregated
file to use as the label source.

Usage:
    python Run/Router/run_build_router_labels_with_source.py \
        --dataset musique \
        --result-model llama-3.1-8b-awq-int4 \
        --aggregated-name query_metrics_v4_3class_with_tokens \
        --label-name hard_llm_correct_rule_v4_3class
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RouterCore.RouterPathConfig import RouterPathConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build router label files from a specific aggregated router data file"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument(
        "--result-model",
        type=str,
        required=True,
        help="Result model name used by upstream benchmark evaluation files",
    )
    parser.add_argument(
        "--aggregated-name",
        type=str,
        required=True,
        help="Aggregated file name (without .json), e.g. query_metrics_v4_3class_with_tokens",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help="Output label file name prefix, e.g. hard_llm_correct_rule_v4_3class",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build label data but do not save it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load aggregated data from custom file
    aggregated_path = RouterPathConfig.get_aggregated_dir(args.dataset, args.result_model) / f"{args.aggregated_name}.json"
    print(f"Loading aggregated data from: {aggregated_path}")

    if not aggregated_path.exists():
        print(f"Error: Aggregated file not found: {aggregated_path}")
        return 1

    with aggregated_path.open("r", encoding="utf-8") as f:
        aggregated = json.load(f)

    strategies = aggregated.get("metadata", {}).get("strategies", [])
    print(f"Loaded {len(aggregated['samples'])} samples with strategies: {strategies}")

    # Build labels using HardLabelBuilder logic
    # We need to apply the same logic as HardLabelBuilder but with our aggregated data
    label_samples = []
    for sample in aggregated["samples"]:
        sample_id = sample.get("id")
        method_metrics = sample.get("method_metrics", {})

        # Apply v3a label rule: primary=llm_judge_correct, fallback=token_f1->bleu1->semantic_f1
        candidate_best_strategies, decision_source = select_candidate_strategies_v3a(
            sample_id, method_metrics, strategies
        )
        optimal_strategy = apply_priority_order(candidate_best_strategies, strategies)
        label_index = strategies.index(optimal_strategy)

        label_samples.append({
            "id": sample_id,
            "optimal_strategy": optimal_strategy,
            "label_index": label_index,
            "candidate_best_strategies": candidate_best_strategies,
            "decision_source": decision_source,
        })

    labels = {
        "metadata": {
            "dataset": args.dataset,
            "result_model": args.result_model,
            "label_name": args.label_name,
            "source_aggregated_file": f"{args.aggregated_name}.json",
            "strategies": strategies,
            "primary_metric": "llm_judge_correct",
            "fallback_metrics": ["token_f1", "bleu1", "semantic_f1"],
            "note": "Built with v3a label rule from custom aggregated source",
        },
        "samples": label_samples,
    }

    if args.dry_run:
        print(json.dumps(labels["metadata"], ensure_ascii=False, indent=2))
        print(f"dry-run: built {len(labels['samples'])} label samples")

        # Show label distribution
        strategy_counts = {}
        for s in labels["samples"]:
            strategy = s["optimal_strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        print("\nLabel distribution:")
        for strategy, count in sorted(strategy_counts.items()):
            print(f"  {strategy}: {count} ({count/len(labels['samples'])*100:.1f}%)")
        return 0

    # Save labels
    output_path = RouterPathConfig.get_hard_label_path(
        dataset_name=args.dataset,
        result_model=args.result_model,
        label_name=args.label_name,
    )
    RouterPathConfig.ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"saved router labels to: {output_path}")
    print(f"total samples: {len(labels['samples'])}")

    # Show label distribution
    strategy_counts = {}
    for s in labels["samples"]:
        strategy = s["optimal_strategy"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    print("\nLabel distribution:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count} ({count/len(labels['samples'])*100:.1f}%)")

    return 0


def select_candidate_strategies_v3a(
    sample_id: str,
    method_metrics: dict,
    strategies: list,
) -> tuple:
    """Select pre-tie-break candidate strategies using v3a rule.

    v3a rule: primary=llm_judge_correct, fallback=token_f1->bleu1->semantic_f1
    """
    # Step 1: llm_judge_correct = 1
    llm_correct_candidates = [
        s for s in strategies
        if method_metrics.get(s, {}).get("llm_judge_correct") == 1
    ]
    if llm_correct_candidates:
        return llm_correct_candidates, "llm_judge_correct"

    # Step 2: token_f1
    token_f1_values = {s: method_metrics.get(s, {}).get("token_f1") for s in strategies}
    if any(v is None for v in token_f1_values.values()):
        missing = [s for s, v in token_f1_values.items() if v is None]
        raise ValueError(f"Sample '{sample_id}' missing token_f1 for: {missing}")

    max_token_f1 = max(token_f1_values.values())
    token_f1_candidates = [s for s, v in token_f1_values.items() if v == max_token_f1]
    if token_f1_candidates:
        return token_f1_candidates, "token_f1"

    # Step 3: bleu1
    bleu1_values = {s: method_metrics.get(s, {}).get("bleu1") for s in strategies}
    if any(v is None for v in bleu1_values.values()):
        missing = [s for s, v in bleu1_values.items() if v is None]
        raise ValueError(f"Sample '{sample_id}' missing bleu1 for: {missing}")

    max_bleu1 = max(bleu1_values.values())
    bleu1_candidates = [s for s, v in bleu1_values.items() if v == max_bleu1]
    if bleu1_candidates:
        return bleu1_candidates, "bleu1"

    # Step 4: semantic_f1 (fallback of last resort)
    semantic_f1_values = {s: method_metrics.get(s, {}).get("semantic_f1") for s in strategies}
    if any(v is None for v in semantic_f1_values.values()):
        missing = [s for s, v in semantic_f1_values.items() if v is None]
        raise ValueError(f"Sample '{sample_id}' missing semantic_f1 for: {missing}")

    max_semantic_f1 = max(semantic_f1_values.values())
    semantic_f1_candidates = [s for s, v in semantic_f1_values.items() if v == max_semantic_f1]
    return semantic_f1_candidates, "semantic_f1"


def apply_priority_order(candidate_best_strategies: list, strategies: list) -> str:
    """Collapse candidate best strategies into one hard label using priority order."""
    priority_order = ["llm_direct", "naive_rag", "graph_rag"]  # Only 3 strategies
    candidate_set = set(candidate_best_strategies)
    for strategy_name in priority_order:
        if strategy_name in candidate_set:
            return strategy_name
    # Fallback to first candidate if none match priority order
    return candidate_best_strategies[0]


if __name__ == "__main__":
    raise SystemExit(main())
