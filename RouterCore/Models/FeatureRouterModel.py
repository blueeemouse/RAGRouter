"""Feature-based router model over precomputed hidden-state representations."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from RouterCore.Models.base_model import BaseRouterModel


class FeatureRouterModel(BaseRouterModel):
    """Router model for precomputed internal representations.

    Current first-stage feature baseline assumptions:
    - input field is `features`
    - feature source is configured externally (e.g. `mean_hidden`)
    - current pooling over layer dimension is configured by `feature_pooling_type`
    - classifier backbone is a stronger deep MLP feature encoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_pooling_type = config.model.feature_pooling_type
        self.hidden_size = config.model.hidden_state_hidden_size
        self.num_layers = config.model.num_hidden_layers_used
        self.dropout_prob = config.model.dropout

        feature_dim = self._infer_feature_dim()
        hidden_dim = config.model.feature_hidden_dim
        mlp_hidden_dim = config.model.feature_mlp_hidden_dim
        projection_dim = config.model.feature_projection_dim
        num_labels = len(config.model.strategy_names)

        self.input_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(mlp_hidden_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(projection_dim, num_labels),
        )

    def _infer_feature_dim(self) -> int:
        """Infer final feature dimension from configured feature pooling strategy."""
        if self.feature_pooling_type == "flatten":
            return self.num_layers * self.hidden_size
        if self.feature_pooling_type == "layer_mean":
            return self.hidden_size
        raise ValueError(
            f"Unsupported feature_pooling_type for FeatureRouterModel: {self.feature_pooling_type}"
        )

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Transform raw feature tensor into classifier input vectors."""
        if features.dim() != 3:
            raise ValueError(
                f"FeatureRouterModel expects features with shape [batch, num_layers, hidden_size], got {tuple(features.shape)}"
            )

        if self.feature_pooling_type == "flatten":
            batch_size = features.size(0)
            return features.reshape(batch_size, -1)
        if self.feature_pooling_type == "layer_mean":
            return features.mean(dim=1)
        raise ValueError(
            f"Unsupported feature_pooling_type for FeatureRouterModel: {self.feature_pooling_type}"
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Read feature batch and return router logits."""
        features = batch.get("features")
        if features is None:
            raise KeyError("FeatureRouterModel requires batch['features']")

        projected = self.transform_features(features)
        projected = self.input_norm(projected)
        logits = self.feature_mlp(projected)
        return {"logits": logits}
