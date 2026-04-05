"""Text-based router model backed by a configurable sentence encoder."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoModel

from RouterCore.Models.base_model import BaseRouterModel


class TextRouterModel(BaseRouterModel):
    """Question-text router baseline model.

    Current first-stage goal:
    - use query text as the only input
    - keep backbone configurable via config.model.backbone_name
    - output multi-class routing logits over strategy_names
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        backbone_name = config.model.backbone_name
        if not backbone_name:
            raise ValueError("TextRouterModel requires config.model.backbone_name")

        self.encoder = AutoModel.from_pretrained(backbone_name)
        if config.model.freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Encoder config must expose hidden_size for TextRouterModel")

        self.pooling_type = config.model.pooling_type
        if self.pooling_type != "mean":
            raise ValueError(
                f"TextRouterModel currently only supports pooling_type='mean', got: {self.pooling_type}"
            )

        num_labels = len(config.model.strategy_names)
        self.dropout = nn.Dropout(config.model.dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def masked_mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token representations with attention-mask weighting."""
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked_hidden = last_hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode a text batch and return router logits."""
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise KeyError("TextRouterModel requires batch['input_ids'] and batch['attention_mask']")

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.masked_mean_pooling(encoder_outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}
