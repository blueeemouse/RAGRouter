"""CLI entrypoint for building router label files."""

from __future__ import annotations

import argparse
import json

from RouterCore.Data.HardLabelBuilder import HardLabelBuilder


SUPPORTED_LABEL_TYPES = ["hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build router label files from aggregated router data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument(
        "--result-model",
        type=str,
        required=True,
        help="Result model name used by upstream benchmark evaluation files",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        required=True,
        choices=SUPPORTED_LABEL_TYPES,
        help="Label type to build",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build label data but do not save it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.label_type == "hard":
        builder = HardLabelBuilder()
    else:
        raise ValueError(f"Unsupported label type: {args.label_type}")

    labels = builder.build(dataset_name=args.dataset, result_model=args.result_model)

    if args.dry_run:
        print(json.dumps(labels["metadata"], ensure_ascii=False, indent=2))
        print(f"dry-run: built {len(labels['samples'])} {args.label_type} label samples")
        return 0

    output_path = builder.save(labels, dataset_name=args.dataset, result_model=args.result_model)
    print(f"saved {args.label_type} router labels to: {output_path}")
    print(f"total samples: {len(labels['samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
