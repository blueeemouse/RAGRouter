"""Build train/val/test splits for router datasets."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from RouterCore.Data.DatasetSchema import STRATEGY_NAMES, validate_sample_id, validate_strategy_names
from RouterCore.RouterPathConfig import RouterPathConfig


class SplitBuilder:
    """Construct dataset splits for router training data from hard labels."""

    DEFAULT_SPLIT_NAME = "split_v1"
    DEFAULT_SEED = 42
    DEFAULT_TRAIN_RATIO = 0.8
    DEFAULT_VAL_RATIO = 0.1
    SPLIT_STRATEGY = "stratified_by_hard_label"

    def __init__(
        self,
        split_name: str | None = None,
        seed: int = DEFAULT_SEED,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
    ):
        self.split_name = split_name or self.DEFAULT_SPLIT_NAME
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.validate_ratios(train_ratio, val_ratio)

    def validate_ratios(self, train_ratio: float, val_ratio: float) -> None:
        """Validate split ratios.

        We require train_ratio + val_ratio < 1 so test gets an explicit remainder
        instead of becoming only whatever rounding error happens to leave behind.
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1 to leave room for test")

    def load_hard_labels(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Load hard-label router data from RouterTrainingData/Labels."""
        hard_label_path = RouterPathConfig.get_hard_label_path(dataset_name, result_model)
        if not hard_label_path.exists():
            raise FileNotFoundError(f"Missing hard-label router data file: {hard_label_path}")

        with hard_label_path.open("r", encoding="utf-8") as f:
            hard_labels = json.load(f)

        metadata = hard_labels.get("metadata", {})
        validate_strategy_names(metadata.get("strategies", []))
        samples = hard_labels.get("samples")
        if not isinstance(samples, list):
            raise ValueError("Hard-label router data must contain a 'samples' list")
        return hard_labels

    def build(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Build split metadata for a dataset/model pair."""
        hard_labels = self.load_hard_labels(dataset_name, result_model)
        grouped_ids = self.group_sample_ids_by_label(hard_labels["samples"])

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        for label_name in STRATEGY_NAMES:
            group_train, group_val, group_test = self.split_group(grouped_ids[label_name], label_name)
            train_ids.extend(group_train)
            val_ids.extend(group_val)
            test_ids.extend(group_test)

        train_ids = self.shuffle_ids(train_ids, offset=0)
        val_ids = self.shuffle_ids(val_ids, offset=1)
        test_ids = self.shuffle_ids(test_ids, offset=2)

        self.validate_final_split(grouped_ids, train_ids, val_ids, test_ids)

        return {
            "metadata": {
                "dataset": dataset_name,
                "result_model": result_model,
                "split_name": self.split_name,
                "seed": self.seed,
                "strategy": self.SPLIT_STRATEGY,
                "source_hard_label_file": RouterPathConfig.get_hard_label_path(dataset_name, result_model).name,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
            },
            "splits": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
        }

    def group_sample_ids_by_label(self, hard_label_samples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group sample ids by optimal hard-label strategy."""
        grouped_ids: Dict[str, List[str]] = {strategy_name: [] for strategy_name in STRATEGY_NAMES}

        for sample in hard_label_samples:
            sample_id = sample.get("id")
            validate_sample_id(sample_id)

            optimal_strategy = sample.get("optimal_strategy")
            if optimal_strategy not in grouped_ids:
                raise ValueError(
                    f"Sample '{sample_id}' has unknown optimal_strategy: {optimal_strategy}"
                )
            grouped_ids[optimal_strategy].append(sample_id)

        return grouped_ids

    def split_group(self, sample_ids: List[str], label_name: str) -> Tuple[List[str], List[str], List[str]]:
        """Split one hard-label group into train/val/test subsets."""
        shuffled_ids = self.shuffle_ids(sample_ids, offset=hash(label_name) % 1000)
        group_size = len(shuffled_ids)

        train_count = math.floor(group_size * self.train_ratio)
        val_count = math.floor(group_size * self.val_ratio)
        test_count = group_size - train_count - val_count

        if test_count < 0:
            raise ValueError(
                f"Invalid split counts for label '{label_name}': "
                f"train={train_count}, val={val_count}, test={test_count}"
            )

        train_ids = shuffled_ids[:train_count]
        val_ids = shuffled_ids[train_count:train_count + val_count]
        test_ids = shuffled_ids[train_count + val_count:]
        return train_ids, val_ids, test_ids

    def shuffle_ids(self, sample_ids: List[str], offset: int = 0) -> List[str]:
        """Return a deterministically shuffled copy of sample ids."""
        copied = list(sample_ids)
        rng = random.Random(self.seed + offset)
        rng.shuffle(copied)
        return copied

    def validate_final_split(
        self,
        grouped_ids: Dict[str, List[str]],
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
    ) -> None:
        """Validate that final split ids are disjoint and cover the full sample set."""
        full_sample_ids = set()
        for group_ids in grouped_ids.values():
            full_sample_ids.update(group_ids)

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        if train_set & val_set:
            raise ValueError("Train and val splits overlap")
        if train_set & test_set:
            raise ValueError("Train and test splits overlap")
        if val_set & test_set:
            raise ValueError("Val and test splits overlap")

        combined = train_set | val_set | test_set
        if combined != full_sample_ids:
            missing = sorted(full_sample_ids - combined)
            extra = sorted(combined - full_sample_ids)
            raise ValueError(
                f"Final split ids do not match full sample ids; missing={missing}, extra={extra}"
            )

    def save(self, split_data: Dict[str, Any], dataset_name: str) -> Path:
        """Save split metadata under RouterTrainingData/Splits."""
        output_path = RouterPathConfig.get_split_path(
            dataset_name=dataset_name,
            split_name=self.split_name,
        )
        RouterPathConfig.ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        return output_path
