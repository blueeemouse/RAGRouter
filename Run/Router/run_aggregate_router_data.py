"""CLI entrypoint for aggregating router training data."""

from __future__ import annotations

import argparse
import json

from RouterCore.Data.EvaluationAggregator import EvaluationAggregator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate query-level evaluation into router training data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument("--result-model", type=str, required=True, help="Result model name, e.g. llama-3.1-8b-awq-int4")
    parser.add_argument("--dry-run", action="store_true", help="Build aggregated data but do not save it")
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
    print(f"saved aggregated router data to: {output_path}")
    print(f"total samples: {len(aggregated['samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
