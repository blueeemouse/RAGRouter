"""Base class for router trainers.

Concrete trainer implementations will be added incrementally.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseTrainer(ABC):
    """Abstract base class for router trainers.

    Current role:
    - hold common trainer state
    - define the trainer/model interaction contract
    - provide lightweight shared helpers that are stable across trainer types
    """

    def __init__(self, model, config, device: str | None = None):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def move_batch_to_device(self, batch: Dict[str, Any]) -> None:
        """Move tensor fields to the trainer device in place.

        Non-tensor fields such as ids or raw questions are kept unchanged.
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

    @abstractmethod
    def train(self, train_dataloader, val_dataloader=None):
        """Run the training loop for a concrete supervision type."""
        raise NotImplementedError
