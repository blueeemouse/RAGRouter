"""Unified collate utilities for router training batches."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class RouterCollator:
    """Minimal router collator for hard-label text routing.

    First-stage behavior:
    - tokenize batch questions with a provided tokenizer
    - rely on tokenizer defaults for attention_mask generation
    - return ids, optional raw questions, tokenized tensors, and labels
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        return_questions: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_questions = return_questions

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("RouterCollator received an empty batch")

        ids = [sample["id"] for sample in batch]
        questions = [sample["question"] for sample in batch]
        labels = [sample["label_index"] for sample in batch]

        tokenized = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        collated: Dict[str, Any] = {
            "ids": ids,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.return_questions:
            collated["questions"] = questions
        return collated


def router_collate_fn(batch):
    """Legacy placeholder kept only to make the migration stage explicit."""
    raise NotImplementedError(
        "Use RouterCollator(tokenizer=..., max_length=...) instead of router_collate_fn"
    )
