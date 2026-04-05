"""CLI entrypoint for training router models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Config.RouterConfig import RouterConfig
from RouterCore.Datasets.RouterHardLabelDataset import RouterHardLabelTextDataset
from RouterCore.Models.TextRouterModel import TextRouterModel
from RouterCore.RouterPathConfig import RouterPathConfig
from RouterCore.Trainers.HardClassificationTrainer import HardClassificationTrainer
from RouterCore.Utils.collate import RouterBatchCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train router models")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument(
        "--result-model",
        type=str,
        required=True,
        help="Result model name used by router training data, e.g. llama-3.1-8b-awq-int4",
    )
    parser.add_argument("--split-name", type=str, default="split_v1", help="Split file name prefix")
    parser.add_argument(
        "--label-name",
        type=str,
        default="hard_llm_correct_rule_v1",
        help="Hard-label file name prefix",
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder backbone name",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4, help="Optimizer learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length")
    parser.add_argument("--device", type=str, default=None, help="Optional explicit device, e.g. cuda or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument(
        "--save-name",
        type=str,
        default="text_router_baseline_v1",
        help="Directory name for saved training artifacts",
    )
    parser.add_argument("--dry-run", action="store_true", help="Construct the full training stack but do not train")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RouterConfig:
    config = RouterConfig()
    config.model.model_type = "text_router"
    config.model.backbone_name = args.backbone_name
    config.model.max_length = args.max_length

    config.training.trainer_type = "hard_classification"
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.epochs = args.epochs

    config.data.dataset_name = args.dataset
    config.data.result_model = args.result_model
    config.data.split_name = args.split_name
    config.data.hard_label_name = args.label_name
    return config


def build_dataloader(config: RouterConfig, args: argparse.Namespace, split: str, tokenizer):
    dataset = RouterHardLabelTextDataset(
        dataset_name=config.data.dataset_name,
        result_model=config.data.result_model,
        split=split,
        split_name=config.data.split_name,
        label_name=config.data.hard_label_name,
    )
    collator = RouterBatchCollator(
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        use_text=True,
        use_features=False,
        return_questions=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    return dataset, dataloader


def build_training_stack(config: RouterConfig, args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name)
    train_dataset, train_dataloader = build_dataloader(config, args, split="train", tokenizer=tokenizer)
    val_dataset, val_dataloader = build_dataloader(config, args, split="val", tokenizer=tokenizer)
    test_dataset, test_dataloader = build_dataloader(config, args, split="test", tokenizer=tokenizer)

    model = TextRouterModel(config)
    trainer = HardClassificationTrainer(model=model, config=config, device=args.device)
    return {
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "train_dataloader": train_dataloader,
        "val_dataset": val_dataset,
        "val_dataloader": val_dataloader,
        "test_dataset": test_dataset,
        "test_dataloader": test_dataloader,
        "model": model,
        "trainer": trainer,
    }


def save_training_artifacts(config: RouterConfig, state_dict, save_name: str, train_metrics: dict, test_metrics: dict) -> Path:
    """Save minimal model/config/metrics artifacts for the current training run."""
    output_dir = RouterPathConfig.get_model_dir(save_name, config.data.dataset_name)
    RouterPathConfig.ensure_dir(output_dir)

    model_path = output_dir / "best_model.pt"
    torch.save(state_dict, model_path)

    config_path = output_dir / "train_config.json"
    config_payload = {
        "model": {
            "model_type": config.model.model_type,
            "backbone_name": config.model.backbone_name,
            "max_length": config.model.max_length,
            "pooling_type": config.model.pooling_type,
            "dropout": config.model.dropout,
            "freeze_backbone": config.model.freeze_backbone,
            "strategy_names": config.model.strategy_names,
        },
        "training": {
            "trainer_type": config.training.trainer_type,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
        },
        "data": {
            "dataset_name": config.data.dataset_name,
            "result_model": config.data.result_model,
            "split_name": config.data.split_name,
            "hard_label_name": config.data.hard_label_name,
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"train": train_metrics, "test": test_metrics}, f, ensure_ascii=False, indent=2)

    return output_dir


def main() -> int:
    args = parse_args()
    config = build_config(args)
    stack = build_training_stack(config, args)

    if args.dry_run:
        print("dry-run: constructed training stack successfully")
        print(f"dataset: {config.data.dataset_name}")
        print(f"result_model: {config.data.result_model}")
        print(f"backbone: {config.model.backbone_name}")
        print(f"train samples: {len(stack['train_dataset'])}")
        print(f"val samples: {len(stack['val_dataset'])}")
        print(f"test samples: {len(stack['test_dataset'])}")
        print(f"train batches per epoch: {len(stack['train_dataloader'])}")
        print(f"device: {stack['trainer'].device}")
        print(f"model class: {stack['model'].__class__.__name__}")
        return 0

    train_metrics = stack["trainer"].train(stack["train_dataloader"], val_dataloader=stack["val_dataloader"])

    best_state_dict = train_metrics.pop("best_state_dict", None)
    if best_state_dict is not None:
        stack["model"].load_state_dict(best_state_dict)

    test_metrics = stack["trainer"].evaluate(stack["test_dataloader"])
    saved_state = best_state_dict if best_state_dict is not None else stack["model"].state_dict()
    output_dir = save_training_artifacts(
        config=config,
        state_dict=saved_state,
        save_name=args.save_name,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    print("training finished")
    for key, value in train_metrics.items():
        print(f"train/{key}: {value}")
    for key, value in test_metrics.items():
        print(f"test/{key}: {value}")
    print(f"saved training artifacts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
