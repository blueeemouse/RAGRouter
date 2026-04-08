"""Hard-label classification trainer.

Implementation is kept intentionally minimal in the first stage.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from RouterCore.Data.DatasetSchema import get_strategy_name_from_list

from .base_trainer import BaseTrainer


class HardClassificationTrainer(BaseTrainer):
    """Train router models with hard-label classification supervision."""

    def compute_loss(self, batch, outputs) -> torch.Tensor:
        """Compute hard-label classification loss from logits and labels."""
        logits = outputs.get("logits")
        if logits is None:
            raise KeyError("model.forward(batch) must return a dict containing 'logits'")

        labels = batch.get("labels")
        if labels is None:
            raise KeyError("HardClassificationTrainer requires batch['labels']")

        return F.cross_entropy(logits, labels)

    def evaluate(self, dataloader, return_predictions: bool = False) -> Dict[str, Any]:
        """Run evaluation and optionally return per-query predictions.

        When return_predictions=True, the returned dict also contains:
        - predictions: List[Dict[str, Any]]
        """
        self.model.eval()
        losses = []
        total_examples = 0
        total_correct = 0
        predictions_output: List[Dict[str, Any]] = []
        strategy_names = list(self.model.config.model.strategy_names)

        with torch.no_grad():
            for batch in dataloader:
                raw_ids = list(batch.get("ids", []))
                raw_questions = list(batch.get("questions", [])) if "questions" in batch else None

                self.move_batch_to_device(batch)
                outputs = self.model.forward(batch)
                loss = self.compute_loss(batch, outputs)
                losses.append(loss.detach().item())

                logits = outputs["logits"]
                labels = batch["labels"]
                predicted_indices = logits.argmax(dim=-1)
                total_examples += labels.size(0)
                total_correct += (predicted_indices == labels).sum().item()

                if return_predictions:
                    predicted_indices_list = predicted_indices.detach().cpu().tolist()
                    true_indices_list = labels.detach().cpu().tolist()
                    for idx, sample_id in enumerate(raw_ids):
                        predicted_index = int(predicted_indices_list[idx])
                        true_index = int(true_indices_list[idx])
                        record = {
                            "id": sample_id,
                            "predicted_index": predicted_index,
                            "predicted_strategy": get_strategy_name_from_list(predicted_index, strategy_names),
                            "true_index": true_index,
                            "true_strategy": get_strategy_name_from_list(true_index, strategy_names),
                            "correct": predicted_index == true_index,
                        }
                        if raw_questions is not None and idx < len(raw_questions):
                            record["question"] = raw_questions[idx]
                        predictions_output.append(record)

        average_loss = sum(losses) / len(losses) if losses else 0.0
        accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        result: Dict[str, Any] = {
            "loss": average_loss,
            "accuracy": accuracy,
            "num_batches": float(len(losses)),
            "num_examples": float(total_examples),
        }
        if return_predictions:
            result["predictions"] = predictions_output
        return result

    def train_epoch(self, train_dataloader) -> Dict[str, float]:
        """Run one training epoch and return aggregate training metrics."""
        self.model.train()
        epoch_losses = []

        for batch in train_dataloader:
            self.move_batch_to_device(batch)
            outputs = self.model.forward(batch)
            loss = self.compute_loss(batch, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.detach().item())

        average_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        return {
            "loss": average_loss,
            "num_batches": float(len(epoch_losses)),
        }

    def train(self, train_dataloader, val_dataloader=None):
        """Run a minimal epoch-based hard-label training loop.

        Returns training metrics together with the best validation summary and a
        cpu-side best_state_dict when validation is enabled.
        """
        num_epochs = getattr(self.config.training, "epochs", 1)
        best_val_accuracy = float("-inf")
        best_val_metrics = None
        best_state_dict = None
        final_train_metrics = None

        for epoch_idx in range(num_epochs):
            train_metrics = self.train_epoch(train_dataloader)
            final_train_metrics = train_metrics

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                print(
                    f"[Epoch {epoch_idx + 1}/{num_epochs}] "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"val_loss={val_metrics['loss']:.6f} "
                    f"val_accuracy={val_metrics['accuracy']:.6f}"
                )
                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    best_val_metrics = dict(val_metrics)
                    best_val_metrics["epoch"] = float(epoch_idx + 1)
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    print(
                        f"  -> updated best checkpoint at epoch {epoch_idx + 1} "
                        f"(val_accuracy={val_metrics['accuracy']:.6f})"
                    )
            else:
                print(
                    f"[Epoch {epoch_idx + 1}/{num_epochs}] "
                    f"train_loss={train_metrics['loss']:.6f}"
                )

        result = {
            "train_loss": final_train_metrics["loss"] if final_train_metrics is not None else 0.0,
            "train_num_batches": final_train_metrics["num_batches"] if final_train_metrics is not None else 0.0,
            "best_state_dict": best_state_dict,
        }
        if best_val_metrics is not None:
            result["best_val_loss"] = best_val_metrics["loss"]
            result["best_val_accuracy"] = best_val_metrics["accuracy"]
            result["best_val_epoch"] = best_val_metrics["epoch"]
        return result
