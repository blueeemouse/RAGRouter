"""Router training configuration objects.

Concrete fields will be refined incrementally as the first-stage pipeline is implemented.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from RouterCore.Data.DatasetSchema import STRATEGY_NAMES


@dataclass
class RouterModelConfig:
    model_type: str = "text_router"
    backbone_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    pooling_type: str = "mean"
    dropout: float = 0.1
    freeze_backbone: bool = False
    strategy_names: List[str] = field(default_factory=lambda: STRATEGY_NAMES.copy())

    hidden_state_feature_name: str = "mean_hidden"
    feature_pooling_type: str = "layer_mean"
    hidden_state_hidden_size: int = 4096
    num_hidden_layers_used: int = 4
    feature_hidden_dim: int = 2048
    feature_mlp_hidden_dim: int = 1024
    feature_projection_dim: int = 256

    # Fusion-specific config (for feature_semantic_fusion_router)
    semantic_backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_embedding_dim: int = 384
    fusion_type: str = "concat"  # "concat" or "gated"
    fusion_hidden_dim: int = 512


@dataclass
class RouterTrainingConfig:
    trainer_type: str = "hard_classification"
    batch_size: int = 8
    learning_rate: float = 1.0e-4
    epochs: int = 1
    class_weights: Optional[List[float]] = None


@dataclass
class RouterDataConfig:
    dataset_name: str = ""
    result_model: str = ""
    split_name: str = "split_v1"
    hard_label_name: str = "hard_llm_correct_rule_v1"
    soft_label_name: str = "soft_llm_correct_v1"


@dataclass
class RouterConfig:
    model: RouterModelConfig = field(default_factory=RouterModelConfig)
    training: RouterTrainingConfig = field(default_factory=RouterTrainingConfig)
    data: RouterDataConfig = field(default_factory=RouterDataConfig)
