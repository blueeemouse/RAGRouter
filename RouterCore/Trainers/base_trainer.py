"""Base class for router trainers.

Concrete trainer implementations will be added incrementally.
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract base class for router trainers."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    @abstractmethod
    def train(self, train_dataloader, val_dataloader=None):
        raise NotImplementedError
