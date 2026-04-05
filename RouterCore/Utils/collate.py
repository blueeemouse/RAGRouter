"""Unified collate utilities for router training batches."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class RouterBatchCollator:
    """Configurable router batch collator.

    Current supported direction:
    - text input via `question`

    Future extension direction:
    - feature-based input via precomputed representations
    - hybrid text + feature input

    The collator keeps a unified high-level batch protocol while allowing
    modality-specific fields to appear only when enabled.
    """

    def __init__(
        self,
        tokenizer=None,
        max_length: int = 512,
        use_text: bool = True,
        use_features: bool = False,
        return_questions: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_text = use_text
        self.use_features = use_features
        self.return_questions = return_questions
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate collator modality configuration."""
        if not self.use_text and not self.use_features:
            raise ValueError("RouterBatchCollator requires at least one enabled input modality")
        if self.use_text and self.tokenizer is None:
            raise ValueError("RouterBatchCollator requires a tokenizer when use_text=True")
        if self.use_features:
            raise NotImplementedError(
                "Feature collation is reserved for a later hidden-states integration step"
            )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("RouterBatchCollator received an empty batch")

        ids = [sample["id"] for sample in batch]
        labels = [sample["label_index"] for sample in batch]

        collated: Dict[str, Any] = {
            "ids": ids,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.use_text:
            questions = [sample["question"] for sample in batch]
            tokenized = self.tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            collated["input_ids"] = tokenized["input_ids"]
            collated["attention_mask"] = tokenized["attention_mask"]
            if self.return_questions:
                collated["questions"] = questions

        return collated


# Backward-compatible alias kept only during the migration transition.
RouterCollator = RouterBatchCollator
