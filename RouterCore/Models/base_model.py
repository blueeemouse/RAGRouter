"""Base class for router models.

Concrete model implementations will be added incrementally.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn


class BaseRouterModel(nn.Module, ABC):
    """Abstract base class for router models.

    Router model class names should primarily express model structure / input
    modality (for example text / feature / hybrid), while concrete backbone
    choices should usually live in config rather than in the class name.

    Router models should consume a dict-based batch payload rather than custom
    positional argument signatures. The batch protocol is unified at a high
    level, while modality-specific fields may appear only when enabled.

    Current common high-level fields:
    - `ids`
    - supervision field(s), e.g. `labels`

    Current text pipeline v1 fields:
    - `input_ids`
    - `attention_mask`
    - optional `questions`

    Future feature / hidden-states direction:
    - `features`

    Therefore the model contract is:
    - accept `forward(batch)`
    - read only the fields needed by the concrete model
    - do not assume every batch always contains every modality field
    - return at least a dict containing task-relevant outputs (e.g. `logits`)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict[str, Any]):
        """Run the model on a dict-based router batch payload."""
        raise NotImplementedError
