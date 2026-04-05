"""Hard-label classification trainer.

Implementation is kept intentionally minimal in the first stage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer


class HardClassificationTrainer(BaseTrainer):
    """Train router models with hard-label classification supervision."""

    def train(self, train_dataloader, val_dataloader=None):
        """Run a minimal hard-label training loop.

        Current assumptions:
        - model.forward(batch) returns a dict containing `logits`
        - batch contains `labels`
        - optimizer is attached to config.training.optimizer externally if used
        """
        optimizer = getattr(self.config.training, "optimizer", None)
        if optimizer is None:
            raise ValueError(
                "HardClassificationTrainer currently expects config.training.optimizer to be set externally"
            )

        self.model.train()
        epoch_losses = []

        for batch in train_dataloader:
            self.move_batch_to_device(batch)
            outputs = self.model.forward(batch)
            logits = outputs.get("logits")
            if logits is None:
                raise KeyError("model.forward(batch) must return a dict containing 'logits'")

            labels = batch.get("labels")
            if labels is None:
                raise KeyError("HardClassificationTrainer requires batch['labels']")

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().item())

        average_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        return {
            "train_loss": average_loss,
            "num_batches": len(epoch_losses),
        }
