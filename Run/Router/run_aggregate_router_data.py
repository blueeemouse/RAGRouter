"""CLI entrypoint for aggregating router training data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RouterCore.Data.EvaluationAggregator import EvaluationAggregator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate query-level evaluation into router training data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument("--result-model", type=str, required=True, help="Result model name, e.g. llama-3.1-8b-awq-int4")
    parser.add_argument("--dry-run", action="store_true", help="Build aggregated data but do not save it")
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional output filename (without .json), e.g. query_metrics_v3a_textmetrics_nooverwrite",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    aggregator = EvaluationAggregator()
    aggregated = aggregator.build(dataset_name=args.dataset, result_model=args.result_model)

    if args.dry_run:
        print(json.dumps(aggregated["metadata"], ensure_ascii=False, indent=2))
        print(f"dry-run: built {len(aggregated['samples'])} aggregated samples")
        return 0

    output_path = aggregator.save(aggregated, dataset_name=args.dataset, result_model=args.result_model)
    if args.save_name:
        default_output_path = output_path
        custom_output_path = default_output_path.with_name(f"{args.save_name}.json")
        with custom_output_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        print(f"saved aggregated router data to: {custom_output_path}")
    else:
        print(f"saved aggregated router data to: {output_path}")
    print(f"total samples: {len(aggregated['samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
