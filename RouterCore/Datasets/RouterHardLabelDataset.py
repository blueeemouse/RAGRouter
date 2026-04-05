"""Router datasets."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from torch.utils.data import Dataset

from RouterCore.Data.DatasetSchema import validate_sample_id
from RouterCore.RouterPathConfig import RouterPathConfig


class RouterHardLabelDataset(Dataset):
    """Dataset for hard-label router training.

    First-stage minimal sample schema:
    - id
    - question
    - label_index
    """

    def __init__(
        self,
        dataset_name: str,
        result_model: str,
        split: str,
        split_name: str = "split_v1",
        label_name: str = "hard_llm_correct_rule_v1",
    ):
        self.dataset_name = dataset_name
        self.result_model = result_model
        self.split = split
        self.split_name = split_name
        self.label_name = label_name

        aggregated = self.load_aggregated(dataset_name, result_model)
        hard_labels = self.load_hard_labels(dataset_name, result_model, label_name)
        split_data = self.load_split(dataset_name, split_name)

        self.samples = self.build_samples(aggregated, hard_labels, split_data, split)

    def load_aggregated(self, dataset_name: str, result_model: str) -> Dict[str, Any]:
        """Load aggregated router data."""
        aggregated_path = RouterPathConfig.get_aggregated_path(dataset_name, result_model)
        if not aggregated_path.exists():
            raise FileNotFoundError(f"Missing aggregated router data file: {aggregated_path}")
        with aggregated_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_hard_labels(self, dataset_name: str, result_model: str, label_name: str) -> Dict[str, Any]:
        """Load hard-label router data."""
        hard_label_path = RouterPathConfig.get_hard_label_path(dataset_name, result_model, label_name)
        if not hard_label_path.exists():
            raise FileNotFoundError(f"Missing hard-label router data file: {hard_label_path}")
        with hard_label_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_split(self, dataset_name: str, split_name: str) -> Dict[str, Any]:
        """Load router split data."""
        split_path = RouterPathConfig.get_split_path(dataset_name, split_name)
        if not split_path.exists():
            raise FileNotFoundError(f"Missing router split data file: {split_path}")
        with split_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def build_samples(
        self,
        aggregated: Dict[str, Any],
        hard_labels: Dict[str, Any],
        split_data: Dict[str, Any],
        split: str,
    ) -> List[Dict[str, Any]]:
        """Join aggregated data, hard labels, and split ids into dataset samples."""
        split_ids = split_data.get("splits", {}).get(split)
        if split_ids is None:
            raise ValueError(f"Unknown split '{split}' in split file")

        aggregated_by_id = self.index_samples_by_id(aggregated.get("samples"), source_name="aggregated")
        labels_by_id = self.index_samples_by_id(hard_labels.get("samples"), source_name="hard_labels")

        dataset_samples: List[Dict[str, Any]] = []
        for sample_id in split_ids:
            validate_sample_id(sample_id)
            aggregated_sample = aggregated_by_id.get(sample_id)
            if aggregated_sample is None:
                raise KeyError(f"Split sample id '{sample_id}' not found in aggregated data")

            label_sample = labels_by_id.get(sample_id)
            if label_sample is None:
                raise KeyError(f"Split sample id '{sample_id}' not found in hard-label data")

            dataset_samples.append(
                {
                    "id": sample_id,
                    "question": aggregated_sample.get("question", ""),
                    "label_index": label_sample.get("label_index"),
                }
            )
        return dataset_samples

    def index_samples_by_id(self, samples: Any, source_name: str) -> Dict[str, Dict[str, Any]]:
        """Index a sample list by string sample id."""
        if not isinstance(samples, list):
            raise ValueError(f"{source_name} must contain a 'samples' list")

        indexed: Dict[str, Dict[str, Any]] = {}
        for sample in samples:
            sample_id = sample.get("id")
            validate_sample_id(sample_id)
            indexed[sample_id] = sample
        return indexed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
