"""Feature-Semantic Fusion Router Model.

Fuses precomputed hidden-state features (HS_qonly) with semantic text embeddings.
Supports fusion_type: concat | gated

This is a NEW model class - does NOT modify the existing FeatureRouterModel.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from RouterCore.Models.base_model import BaseRouterModel


class FeatureSemanticFusionRouterModel(BaseRouterModel):
    """
    Fuse precomputed hidden-state features with semantic text embeddings.

    Architecture:
    - HS branch: hidden_states -> pooling -> MLP -> projection
    - Semantic branch: query text -> MiniLM (frozen) -> MLP -> projection
    - Fusion: concat or gated
    - Classifier head -> logits

    Fusion types:
    - concat: [h_hs; h_sem] -> fusion_mlp -> classifier
    - gated: g = sigmoid(W[h_hs; h_sem]); h = g*h_hs + (1-g)*h_sem -> classifier
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        mcfg = config.model

        # Feature pooling config (same as FeatureRouterModel)
        self.feature_pooling_type = mcfg.feature_pooling_type
        self.hidden_size = mcfg.hidden_state_hidden_size
        self.num_layers = mcfg.num_hidden_layers_used
        self.dropout_prob = mcfg.dropout

        # Fusion-specific config
        self.semantic_backbone_name = getattr(mcfg, "semantic_backbone_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.semantic_embedding_dim = getattr(mcfg, "semantic_embedding_dim", 384)
        self.fusion_type = getattr(mcfg, "fusion_type", "concat")
        self.fusion_hidden_dim = getattr(mcfg, "fusion_hidden_dim", 512)
        self.feature_projection_dim = mcfg.feature_projection_dim
        self.num_labels = len(mcfg.strategy_names)

        feature_dim = self._infer_feature_dim()

        # HS branch (mirrors FeatureRouterModel structure)
        self.hs_norm = nn.LayerNorm(feature_dim)
        self.hs_mlp = nn.Sequential(
            nn.Linear(feature_dim, mcfg.feature_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(mcfg.feature_hidden_dim, mcfg.feature_mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(mcfg.feature_mlp_hidden_dim, self.feature_projection_dim),
        )

        # Semantic branch
        self.sem_mlp = nn.Sequential(
            nn.Linear(self.semantic_embedding_dim, mcfg.feature_mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(mcfg.feature_mlp_hidden_dim, self.feature_projection_dim),
        )

        # Fusion head
        if self.fusion_type == "concat":
            self.fusion_head = nn.Sequential(
                nn.Linear(self.feature_projection_dim * 2, self.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.fusion_hidden_dim, self.num_labels),
            )
        elif self.fusion_type == "gated":
            self.gate = nn.Linear(self.feature_projection_dim * 2, self.feature_projection_dim)
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_projection_dim, self.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.fusion_hidden_dim, self.num_labels),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        # Semantic encoder (frozen MiniLM)
        self.semantic_encoder = SentenceTransformer(self.semantic_backbone_name)
        self.semantic_encoder.eval()
        for p in self.semantic_encoder.parameters():
            p.requires_grad = False

    def _infer_feature_dim(self) -> int:
        """Infer feature dimension from pooling strategy."""
        if self.feature_pooling_type == "flatten":
            return self.num_layers * self.hidden_size
        if self.feature_pooling_type == "layer_mean":
            return self.hidden_size
        raise ValueError(f"Unsupported feature_pooling_type: {self.feature_pooling_type}")

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Transform raw hidden-state tensor into feature vectors."""
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape [batch, num_layers, hidden_size], got {tuple(features.shape)}"
            )

        if self.feature_pooling_type == "flatten":
            return features.reshape(features.size(0), -1)
        if self.feature_pooling_type == "layer_mean":
            return features.mean(dim=1)
        raise ValueError(f"Unsupported feature_pooling_type: {self.feature_pooling_type}")

    @torch.no_grad()
    def encode_questions(self, questions: List[str], device: torch.device) -> torch.Tensor:
        """Encode questions using frozen semantic encoder."""
        emb = self.semantic_encoder.encode(
            questions,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )
        # Clone to avoid inference tensor issues with autograd
        return emb.clone().to(device=device, dtype=torch.float32)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass: fuse HS and semantic features, return logits."""
        features = batch.get("features")
        questions = batch.get("questions")

        if features is None:
            raise KeyError("FeatureSemanticFusionRouterModel requires batch['features']")
        if questions is None:
            raise KeyError("FeatureSemanticFusionRouterModel requires batch['questions']")

        # HS branch
        x_hs = self.transform_features(features)
        x_hs = self.hs_norm(x_hs)
        h_hs = self.hs_mlp(x_hs)

        # Semantic branch
        sem = self.encode_questions(questions, device=features.device)
        h_sem = self.sem_mlp(sem)

        # Fusion
        if self.fusion_type == "concat":
            fused = torch.cat([h_hs, h_sem], dim=-1)
            logits = self.fusion_head(fused)
        else:  # gated
            gate_input = torch.cat([h_hs, h_sem], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            fused = gate * h_hs + (1.0 - gate) * h_sem
            logits = self.classifier(fused)

        return {"logits": logits}
