"""CLI entrypoint for aggregating router training data with complete token counts.

This script aggregates evaluation metrics with token counts from RetrievalResultData,
providing complete input_tokens and output_tokens for all strategies.

Usage:
    python Run/Router/run_aggregate_router_data_with_tokens.py \
        --dataset musique \
        --result-model llama-3.1-8b-awq-int4 \
        --save-name query_metrics_v4_with_tokens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RouterCore.Data.EvaluationAggregatorWithTokens import EvaluationAggregatorWithTokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate query-level evaluation with token counts from RetrievalResultData"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument("--result-model", type=str, required=True, help="Result model name, e.g. llama-3.1-8b-awq-int4")
    parser.add_argument("--dry-run", action="store_true", help="Build aggregated data but do not save it")
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional output filename (without .json), e.g. query_metrics_v4_with_tokens",
    )
    parser.add_argument(
        "--eval-file-suffix",
        type=str,
        default="",
        help="Optional suffix for evaluation files, e.g., '_full' to look for 'graph_rag_full.json'",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="List of strategies to process, e.g., llm_direct naive_rag graph_rag",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Aggregating data with tokens for dataset={args.dataset}, result_model={args.result_model}")
    aggregator = EvaluationAggregatorWithTokens(
        eval_file_suffix=args.eval_file_suffix,
        strategies=args.strategies,
    )

    try:
        aggregated = aggregator.build(dataset_name=args.dataset, result_model=args.result_model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.dry_run:
        print(json.dumps(aggregated["metadata"], ensure_ascii=False, indent=2))
        print(f"dry-run: built {len(aggregated['samples'])} aggregated samples")

        # Show token coverage stats
        samples_with_tokens = 0
        strategy_token_counts = {s: 0 for s in ["llm_direct", "naive_rag", "graph_rag"]}
        for sample in aggregated["samples"]:
            for strategy in strategy_token_counts:
                if sample["method_metrics"][strategy].get("input_tokens") is not None:
                    strategy_token_counts[strategy] += 1
            if any(
                sample["method_metrics"][s].get("input_tokens") is not None
                for s in ["llm_direct", "naive_rag", "graph_rag"]
            ):
                samples_with_tokens += 1

        print(f"\nToken coverage (samples with non-None input_tokens):")
        print(f"  Total samples: {len(aggregated['samples'])}")
        print(f"  Samples with at least one strategy: {samples_with_tokens}")
        for strategy, count in strategy_token_counts.items():
            print(f"  {strategy}: {count}/{len(aggregated['samples'])}")

        return 0

    output_path = aggregator.save(
        aggregated,
        dataset_name=args.dataset,
        result_model=args.result_model,
        save_name=args.save_name,
    )
    print(f"saved aggregated router data to: {output_path}")
    print(f"total samples: {len(aggregated['samples'])}")

    # Verify token counts
    samples_with_tokens = 0
    for sample in aggregated["samples"]:
        for strategy in ["llm_direct", "naive_rag", "graph_rag"]:
            if sample["method_metrics"][strategy].get("input_tokens") is not None:
                samples_with_tokens += 1
                break
    print(f"samples with at least one strategy having token counts: {samples_with_tokens}/{len(aggregated['samples'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
