"""CLI entrypoint for building router train/val/test splits."""

from __future__ import annotations

import argparse
import json

from RouterCore.Data.SplitBuilder import SplitBuilder


DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_SPLIT_NAME = "split_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build router train/val/test splits from hard labels")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument(
        "--result-model",
        type=str,
        required=True,
        help="Result model name used by upstream hard-label router data",
    )
    parser.add_argument("--split-name", type=str, default=DEFAULT_SPLIT_NAME, help="Split file name prefix")
    parser.add_argument(
        "--label-name",
        type=str,
        default="hard_llm_correct_rule_v1",
        help="Hard-label file name prefix used for stratified split",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic splitting")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--dry-run", action="store_true", help="Build split data but do not save it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    builder = SplitBuilder(
        split_name=args.split_name,
        label_name=args.label_name,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    split_data = builder.build(dataset_name=args.dataset, result_model=args.result_model)

    if args.dry_run:
        print(json.dumps(split_data["metadata"], ensure_ascii=False, indent=2))
        for split_name, split_ids in split_data["splits"].items():
            print(f"{split_name}: {len(split_ids)} samples")
        return 0

    output_path = builder.save(split_data, dataset_name=args.dataset)
    print(f"saved router split data to: {output_path}")
    for split_name, split_ids in split_data["splits"].items():
        print(f"{split_name}: {len(split_ids)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
