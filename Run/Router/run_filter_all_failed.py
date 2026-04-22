"""CLI entrypoint for filtering all-failed samples from aggregated router data.

A sample is considered "all_failed" when ALL strategies have:
- llm_judge_correct == 0 AND token_f1 == 0

This script filters out such samples and creates a new aggregated file.

Usage:
    python Run/Router/run_filter_all_failed.py \
        --dataset musique \
        --result-model llama-3.1-8b-awq-int4 \
        --aggregated-name query_metrics_v5 \
        --save-name query_metrics_v5_filtered
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RouterCore.RouterPathConfig import RouterPathConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter all-failed samples from aggregated router data"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument(
        "--result-model",
        type=str,
        required=True,
        help="Result model name, e.g. llama-3.1-8b-awq-int4",
    )
    parser.add_argument(
        "--aggregated-name",
        type=str,
        required=True,
        help="Input aggregated file name (without .json), e.g. query_metrics_v5",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        required=True,
        help="Output filtered file name (without .json), e.g. query_metrics_v5_filtered",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show filtering stats without saving")
    return parser.parse_args()


def is_sample_all_failed(sample: Dict[str, Any], strategies: List[str]) -> bool:
    """Check if all strategies failed for this sample.

    A strategy is considered failed when:
    - llm_judge_correct == 0 (or missing)
    - AND token_f1 == 0 (or missing)

    Returns True if ALL strategies are failed.
    """
    method_metrics = sample.get("method_metrics", {})

    for strategy in strategies:
        metrics = method_metrics.get(strategy, {})
        llm_correct = metrics.get("llm_judge_correct", 0)
        token_f1 = metrics.get("token_f1", 0.0)

        # If at least one strategy is not failed, sample is not all_failed
        if llm_correct != 0 or token_f1 != 0:
            return False

    return True


def main() -> int:
    args = parse_args()

    # Load aggregated data
    aggregated_dir = RouterPathConfig.get_aggregated_dir(args.dataset, args.result_model)
    input_path = aggregated_dir / f"{args.aggregated_name}.json"

    if not input_path.exists():
        print(f"Error: Aggregated file not found: {input_path}")
        return 1

    print(f"Loading aggregated data from: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    strategies = metadata.get("strategies", [])
    samples = data.get("samples", [])

    if not strategies:
        print("Error: No strategies found in metadata")
        return 1

    print(f"Loaded {len(samples)} samples with strategies: {strategies}")

    # Filter samples
    kept_samples = []
    removed_ids = []

    for sample in samples:
        if is_sample_all_failed(sample, strategies):
            removed_ids.append(sample.get("id"))
        else:
            kept_samples.append(sample)

    # Stats
    print(f"\n=== Filtering Results ===")
    print(f"Total samples: {len(samples)}")
    print(f"Kept samples: {len(kept_samples)}")
    print(f"Removed samples (all_failed): {len(removed_ids)}")
    print(f"Removal rate: {len(removed_ids)/len(samples)*100:.1f}%")

    if removed_ids and len(removed_ids) <= 10:
        print(f"Removed IDs: {removed_ids}")
    elif removed_ids:
        print(f"First 10 removed IDs: {removed_ids[:10]}...")

    if args.dry_run:
        print("\n[dry-run] No files saved")
        return 0

    # Build output data
    output_data = {
        "metadata": {
            **metadata,
            "filtered": True,
            "filter_rule": "all_failed: llm_judge_correct=0 AND token_f1=0 for ALL strategies",
            "original_file": f"{args.aggregated_name}.json",
            "original_samples": len(samples),
            "filtered_samples": len(kept_samples),
            "removed_samples": len(removed_ids),
        },
        "samples": kept_samples,
    }

    # Save
    output_path = aggregated_dir / f"{args.save_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved filtered data to: {output_path}")

    # Save removed IDs for reference
    removed_path = aggregated_dir / f"{args.save_name}_removed_ids.json"
    with removed_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "source_file": f"{args.aggregated_name}.json",
                "filter_rule": "all_failed",
                "count": len(removed_ids),
            },
            "removed_ids": removed_ids,
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved removed IDs to: {removed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
