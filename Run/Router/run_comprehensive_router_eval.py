"""Comprehensive offline router evaluation script.

This script evaluates trained router models with:
1. Router performance metrics (llm_correct, semantic_f1, token_f1, etc.)
2. Token overhead statistics (average in-token, out-token)
3. Single baseline comparison (always pick one method)
4. Oracle comparison (best per query)

Usage:
    python Run/Router/run_comprehensive_router_eval.py \
        --dataset musique \
        --result-model llama-3.1-8b-awq-int4 \
        --model-name router_high_quality_3class_text \
        --model-type text_router \
        --split-name filtered_v3a_3class_split_v1_token0_failed_removed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Config.RouterConfig import RouterConfig
from RouterCore.Datasets.RouterHardLabelDataset import (
    RouterHardLabelFeatureDataset,
    RouterHardLabelTextDataset,
)
from RouterCore.Models.FeatureRouterModel import FeatureRouterModel
from RouterCore.Models.TextRouterModel import TextRouterModel
from RouterCore.RouterPathConfig import RouterPathConfig
from RouterCore.Utils.collate import RouterBatchCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive router evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument(
        "--result-model",
        type=str,
        required=True,
        help="Result model name, e.g. llama-3.1-8b-awq-int4",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model save name for evaluation",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["text_router", "feature_router"],
        help="Router model type",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="filtered_v3a_3class_split_v1_token0_failed_removed",
        help="Split file name",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="hard_llm_correct_rule_v3a_3class_only_llm_naive_graph_filtered_token0_failed_removed",
        help="Hard label file name",
    )
    parser.add_argument(
        "--feature-name",
        type=str,
        default="last_hidden",
        help="Hidden-state feature field name (for feature_router)",
    )
    parser.add_argument(
        "--feature-pooling-type",
        type=str,
        default="layer_mean",
        choices=["flatten", "layer_mean"],
        help="Feature pooling strategy",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda or cpu")
    parser.add_argument("--save-name", type=str, default=None, help="Save name for evaluation output")
    parser.add_argument(
        "--aggregated-name",
        type=str,
        default="query_metrics_v1.json",
        help="Aggregated metrics file name (default: query_metrics_v1.json)",
    )
    return parser.parse_args()


def load_router_model(model_name: str, dataset_name: str) -> tuple:
    """Load trained router model and config."""
    model_dir = RouterPathConfig.get_model_dir(model_name, dataset_name)
    config_path = model_dir / "train_config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Build config
    config = RouterConfig()
    config.model.model_type = config_dict["model"]["model_type"]
    config.model.backbone_name = config_dict["model"].get("backbone_name")
    config.model.strategy_names = config_dict["model"]["strategy_names"]
    config.model.hidden_state_feature_name = config_dict["model"].get(
        "hidden_state_feature_name", "mean_hidden"
    )
    config.model.feature_pooling_type = config_dict["model"].get(
        "feature_pooling_type", "layer_mean"
    )
    config.model.hidden_state_hidden_size = config_dict["model"].get(
        "hidden_state_hidden_size", 4096
    )
    config.model.num_hidden_layers_used = config_dict["model"].get(
        "num_hidden_layers_used", 4
    )
    config.model.feature_hidden_dim = config_dict["model"].get("feature_hidden_dim", 2048)
    config.model.feature_mlp_hidden_dim = config_dict["model"].get(
        "feature_mlp_hidden_dim", 1024
    )
    config.model.feature_projection_dim = config_dict["model"].get(
        "feature_projection_dim", 256
    )
    config.model.dropout = config_dict["model"].get("dropout", 0.1)

    # Load model state
    model_path = model_dir / "best_model.pt"
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Initialize model
    model_type = config_dict["model"]["model_type"]
    if model_type == "text_router":
        model = TextRouterModel(config)
    elif model_type == "feature_router":
        model = FeatureRouterModel(config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(state_dict)

    return model, config


def build_dataloader(config: RouterConfig, args: argparse.Namespace, split: str, tokenizer):
    """Build dataloader for inference."""
    if config.model.model_type == "text_router":
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
    elif config.model.model_type == "feature_router":
        dataset = RouterHardLabelFeatureDataset(
            dataset_name=config.data.dataset_name,
            result_model=config.data.result_model,
            split=split,
            split_name=config.data.split_name,
            label_name=config.data.hard_label_name,
            feature_name=config.model.hidden_state_feature_name,
        )
        collator = RouterBatchCollator(
            tokenizer=None,
            max_length=config.model.max_length,
            use_text=False,
            use_features=True,
            return_questions=False,
        )
    else:
        raise ValueError(f"Unsupported model_type: {config.model.model_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    return dataset, dataloader


def run_inference(model, dataloader, device: str) -> tuple:
    """Run inference and return predictions."""
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    all_predictions = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            outputs = model(batch)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)

            # Collect results
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_ids.extend(batch["ids"])

    return all_predictions, all_labels, all_ids


def load_aggregated_metrics(dataset_name: str, result_model: str, aggregated_name: str = "query_metrics_v1.json") -> Dict[str, Any]:
    """Load aggregated metrics with token information."""
    aggregated_dir = RouterPathConfig.get_aggregated_dir(dataset_name, result_model)
    aggregated_path = aggregated_dir / aggregated_name
    with aggregated_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Index by sample id
    samples_by_id = {s["id"]: s for s in data["samples"]}
    return samples_by_id


METRIC_NAMES = [
    "llm_judge_correct",
    "semantic_f1",
    "token_f1",
    "bleu1",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "meteor",
    "coverage",
    "input_tokens",
    "output_tokens",
]


def _compute_mean_metrics(
    strategy_names: List[str],
    samples_by_id: Dict[str, Any],
    sample_ids: List[str],
    selector_fn,  # function(sample_id, samples_by_id, strategy_names) -> str
) -> Dict[str, Dict[str, float]]:
    """Compute mean metrics for a given strategy selector function."""
    metric_sums = {k: 0.0 for k in METRIC_NAMES}
    metric_counts = {k: 0 for k in METRIC_NAMES}

    for sample_id in sample_ids:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue

        method_metrics = sample.get("method_metrics", {})
        selected_strategy = selector_fn(sample_id, samples_by_id, strategy_names)

        if selected_strategy not in method_metrics:
            continue

        pm = method_metrics[selected_strategy]

        for metric_name in METRIC_NAMES:
            value = pm.get(metric_name)
            if value is not None:
                metric_sums[metric_name] += value
                metric_counts[metric_name] += 1

    result = {}
    for metric_name in METRIC_NAMES:
        if metric_counts[metric_name] > 0:
            result[metric_name] = {
                "mean": metric_sums[metric_name] / metric_counts[metric_name],
                "count": metric_counts[metric_name],
            }
        else:
            result[metric_name] = {"mean": 0.0, "count": 0}
    return result


def compute_router_metrics(
    predictions: List[int],
    labels: List[int],
    strategy_names: List[str],
    samples_by_id: Dict[str, Any],
    sample_ids: List[str],
) -> tuple:
    """Compute router performance metrics and per-query predictions."""
    correct = 0
    per_query = []

    for pred, label, sample_id in zip(predictions, labels, sample_ids):
        is_correct = pred == label
        if is_correct:
            correct += 1
        per_query.append({
            "id": sample_id,
            "predicted_index": pred,
            "predicted_strategy": strategy_names[pred] if pred < len(strategy_names) else "unknown",
            "true_index": label,
            "true_strategy": strategy_names[label] if label < len(strategy_names) else "unknown",
            "correct": is_correct,
        })

    accuracy = correct / len(predictions) if predictions else 0.0

    # Router always uses its predicted strategy
    def router_selector(sample_id, samples_by_id, strategy_names):
        pred_idx = predictions[sample_ids.index(sample_id)]
        return strategy_names[pred_idx] if pred_idx < len(strategy_names) else strategy_names[0]

    metrics = _compute_mean_metrics(strategy_names, samples_by_id, sample_ids, router_selector)
    metrics["accuracy"] = {"mean": accuracy, "count": len(predictions)}

    return metrics, per_query


def compute_single_baseline_metrics(
    strategy_names: List[str],
    samples_by_id: Dict[str, Any],
    sample_ids: List[str],
    baseline_strategy: str,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for a single baseline strategy (always pick one method)."""
    def baseline_selector(sample_id, samples_by_id, strategy_names):
        return baseline_strategy

    return _compute_mean_metrics(strategy_names, samples_by_id, sample_ids, baseline_selector)


def compute_oracle_metrics(
    strategy_names: List[str],
    samples_by_id: Dict[str, Any],
    sample_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute Oracle metrics (best per query based on llm_judge_correct)."""
    def oracle_selector(sample_id, samples_by_id, strategy_names):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            return strategy_names[0]

        method_metrics = sample.get("method_metrics", {})

        # Find best strategy for this query based on llm_judge_correct
        best_metric = -1
        best_strategy = strategy_names[0]
        for strategy in strategy_names:
            if strategy in method_metrics:
                pm = method_metrics[strategy]
                val = pm.get("llm_judge_correct", 0) or 0
                if val > best_metric:
                    best_metric = val
                    best_strategy = strategy
        return best_strategy

    return _compute_mean_metrics(strategy_names, samples_by_id, sample_ids, oracle_selector)


def build_comparison(
    router_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
    oracle_metrics: Dict[str, Any],
    strategy_names: List[str],
) -> Dict[str, Any]:
    """Build comparison summary between router, baselines, and oracle."""
    comparison = {
        "router_vs_best_baseline": {},
        "router_vs_oracle": {},
    }

    for metric_name in METRIC_NAMES + ["accuracy"]:
        router_val = router_metrics.get(metric_name, {}).get("mean", 0)

        # Best single baseline
        best_baseline_val = max(
            baseline_metrics[s].get(metric_name, {}).get("mean", 0)
            for s in strategy_names
        )
        best_baseline_name = max(
            strategy_names,
            key=lambda s: baseline_metrics[s].get(metric_name, {}).get("mean", 0),
        )

        # Oracle
        oracle_val = oracle_metrics.get(metric_name, {}).get("mean", 0)

        comparison["router_vs_best_baseline"][metric_name] = {
            "router": router_val,
            "best_baseline": best_baseline_val,
            "best_baseline_name": best_baseline_name,
            "gain": router_val - best_baseline_val,
            "gain_pct": (router_val - best_baseline_val) / best_baseline_val * 100 if best_baseline_val != 0 else 0,
        }

        comparison["router_vs_oracle"][metric_name] = {
            "router": router_val,
            "oracle": oracle_val,
            "regret": oracle_val - router_val,
            "efficiency": router_val / oracle_val if oracle_val > 0 else 0,
        }

    return comparison


def main() -> int:
    args = parse_args()

    # Load router model
    print(f"Loading model: {args.model_name}")
    model, config = load_router_model(args.model_name, args.dataset)
    config.data.split_name = args.split_name
    config.data.hard_label_name = args.label_name
    config.data.dataset_name = args.dataset
    config.data.result_model = args.result_model

    strategy_names = config.model.strategy_names
    print(f"Strategy names: {strategy_names}")

    # Get tokenizer for text router
    tokenizer = None
    if config.model.model_type == "text_router":
        tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name)

    # Build test dataloader
    print("Building test dataloader...")
    test_dataset, test_dataloader = build_dataloader(config, args, split="test", tokenizer=tokenizer)
    print(f"Test set size: {len(test_dataset)}")

    # Run inference
    print("Running inference...")
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    predictions, labels, sample_ids = run_inference(model, test_dataloader, device)

    # Load aggregated metrics
    print("Loading aggregated metrics...")
    samples_by_id = load_aggregated_metrics(args.dataset, args.result_model, args.aggregated_name)

    # Compute router metrics
    print("Computing router metrics...")
    router_metrics, per_query_predictions = compute_router_metrics(
        predictions,
        labels,
        strategy_names,
        samples_by_id,
        sample_ids,
    )

    # Compute single baseline metrics
    print("Computing single baseline metrics...")
    baseline_metrics = {}
    for baseline_strategy in strategy_names:
        print(f"  - {baseline_strategy}")
        baseline_metrics[baseline_strategy] = compute_single_baseline_metrics(
            strategy_names,
            samples_by_id,
            sample_ids,
            baseline_strategy,
        )

    # Compute oracle metrics
    print("Computing Oracle metrics...")
    oracle_metrics = compute_oracle_metrics(
        strategy_names,
        samples_by_id,
        sample_ids,
    )

    # Build comparisons
    print("Building comparisons...")
    comparison = build_comparison(
        router_metrics,
        baseline_metrics,
        oracle_metrics,
        strategy_names,
    )

    # Build result
    result = {
        "metadata": {
            "dataset": args.dataset,
            "result_model": args.result_model,
            "model_name": args.model_name,
            "model_type": config.model.model_type,
            "split_name": args.split_name,
            "label_name": args.label_name,
            "test_size": len(test_dataset),
            "strategy_names": strategy_names,
        },
        "router_performance": router_metrics,
        "single_baseline": baseline_metrics,
        "oracle": oracle_metrics,
        "comparison": comparison,
    }

    # Save result
    save_name = args.save_name or f"{args.model_name}_comprehensive_eval"
    eval_dir = RouterPathConfig.get_evaluation_dir(save_name, args.dataset)
    RouterPathConfig.ensure_dir(eval_dir)
    eval_path = eval_dir / f"{args.dataset}_comprehensive_eval.json"

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation saved to: {eval_path}")

    # Print summary
    _print_summary(result)

    return 0


def _print_summary(result: Dict[str, Any]) -> None:
    """Print evaluation summary."""
    metadata = result["metadata"]
    router = result["router_performance"]
    baseline = result["single_baseline"]
    oracle = result["oracle"]
    comparison = result["comparison"]

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nRouter: {metadata['model_name']} ({metadata['model_type']})")
    print(f"Dataset: {metadata['dataset']}, Test size: {metadata['test_size']}")
    print(f"Strategies: {metadata['strategy_names']}")

    print("\n--- Router Performance ---")
    for metric_name in ["llm_judge_correct", "semantic_f1", "token_f1", "accuracy"]:
        val = router.get(metric_name, {}).get("mean", 0)
        print(f"  {metric_name}: {val:.4f}")

    print("\n--- Token Overhead (Router) ---")
    for metric_name in ["input_tokens", "output_tokens"]:
        val = router.get(metric_name, {}).get("mean", 0)
        count = router.get(metric_name, {}).get("count", 0)
        print(f"  {metric_name} (avg over {count} samples): {val:.1f}")

    print("\n--- Single Baseline Comparison ---")
    header = f"  {'Strategy':<15} {'llm_correct':>12} {'sem_f1':>10} {'tok_f1':>10} {'in_tok':>10} {'out_tok':>10}"
    print(header)
    print("  " + "-" * 65)
    for strategy in metadata["strategy_names"]:
        s = baseline[strategy]
        llm = s.get("llm_judge_correct", {}).get("mean", 0)
        sem = s.get("semantic_f1", {}).get("mean", 0)
        tok = s.get("token_f1", {}).get("mean", 0)
        inp = s.get("input_tokens", {}).get("mean", 0)
        out = s.get("output_tokens", {}).get("mean", 0)
        print(f"  {strategy:<15} {llm:>12.4f} {sem:>10.4f} {tok:>10.4f} {inp:>10.1f} {out:>10.1f}")

    print("\n--- Oracle (Best per Query) ---")
    for metric_name in ["llm_judge_correct", "semantic_f1", "token_f1"]:
        val = oracle.get(metric_name, {}).get("mean", 0)
        print(f"  {metric_name}: {val:.4f}")

    print("\n--- Token Overhead (Oracle) ---")
    for metric_name in ["input_tokens", "output_tokens"]:
        val = oracle.get(metric_name, {}).get("mean", 0)
        count = oracle.get(metric_name, {}).get("count", 0)
        print(f"  {metric_name} (avg over {count} samples): {val:.1f}")

    print("\n--- Router vs Best Baseline (Gain) ---")
    comp = comparison["router_vs_best_baseline"]
    for metric_name in ["llm_judge_correct", "semantic_f1", "token_f1", "accuracy"]:
        c = comp.get(metric_name, {})
        gain = c.get("gain", 0)
        gain_pct = c.get("gain_pct", 0)
        best_name = c.get("best_baseline_name", "N/A")
        print(f"  {metric_name}: router={c.get('router', 0):.4f}, best_baseline({best_name})={c.get('best_baseline', 0):.4f}, gain={gain:+.4f} ({gain_pct:+.1f}%)")

    print("\n--- Router vs Oracle (Regret/Efficiency) ---")
    comp_oracle = comparison["router_vs_oracle"]
    for metric_name in ["llm_judge_correct", "semantic_f1", "token_f1"]:
        c = comp_oracle.get(metric_name, {})
        regret = c.get("regret", 0)
        efficiency = c.get("efficiency", 0)
        print(f"  {metric_name}: router={c.get('router', 0):.4f}, oracle={c.get('oracle', 0):.4f}, regret={regret:.4f}, efficiency={efficiency:.2%}")

    print("\n--- Token Efficiency ---")
    router_in = router.get("input_tokens", {}).get("mean", 0)
    router_out = router.get("output_tokens", {}).get("mean", 0)
    oracle_in = oracle.get("input_tokens", {}).get("mean", 0)
    oracle_out = oracle.get("output_tokens", {}).get("mean", 0)
    print(f"  Router: in={router_in:.1f}, out={router_out:.1f}, total={router_in+router_out:.1f}")
    print(f"  Oracle: in={oracle_in:.1f}, out={oracle_out:.1f}, total={oracle_in+oracle_out:.1f}")
    if oracle_in > 0:
        in_eff = router_in / oracle_in
        out_eff = router_out / oracle_out
        print(f"  Efficiency: in={in_eff:.2%}, out={out_eff:.2%}")

    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
