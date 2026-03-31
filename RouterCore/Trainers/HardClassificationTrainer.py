"""Hard-label classification trainer.

Implementation will be added incrementally in later steps.
"""

from .base_trainer import BaseTrainer


class HardClassificationTrainer(BaseTrainer):
    """Train router models with hard-label classification supervision."""

    def train(self, train_dataloader, val_dataloader=None):
        raise NotImplementedError("HardClassificationTrainer.train is not implemented yet")
