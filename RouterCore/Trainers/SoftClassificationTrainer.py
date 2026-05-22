"""Soft-label classification trainer.

Implementation will be added incrementally in later steps.
"""

from .base_trainer import BaseTrainer


class SoftClassificationTrainer(BaseTrainer):
    """Train router models with soft-label supervision."""

    def train(self, train_dataloader, val_dataloader=None):
        raise NotImplementedError("SoftClassificationTrainer.train is not implemented yet")
