"""Run L2 inference on queries that L1 predicts as complex_rag.

This fills in the missing L2 predictions for complete hierarchical evaluation.

Usage:
python Run/Router/run_l2_inference.py \
    --l1-prediction-file Dataset/RouterTrainingData/Evaluation/hierarchical_l1_mean_hidden/musique/musique_test_predictions.json \
    --l2-model-dir Dataset/RouterTrainingData/Models/hierarchical_l2_mean_hidden/musique/ \
    --output-file Dataset/RouterTrainingData/Evaluation/hierarchical_l2_mean_hidden/musique/l2_predictions_for_l1_complex.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RouterCore.RouterPathConfig import RouterPathConfig
from RouterCore.Models.FeatureRouterModel import FeatureRouterModel


L2_STRATEGIES = ["hybrid_rag", "iterative_rag_naive", "iterative_rag_graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run L2 inference on L1-complex queries")
    parser.add_argument(
        "--l1-prediction-file",
        type=str,
        required=True,
        help="Path to L1 prediction JSON file",
    )
    parser.add_argument(
        "--l2-model-dir",
        type=str,
        required=True,
        help="Path to L2 model directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="musique",
        help="Dataset name",
    )
    parser.add_argument(
        "--result-model",
        type=str,
        default="llama-3.1-8b-awq-int4",
        help="Result model name",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output path for L2 predictions on L1-complex queries",
    )
    return parser.parse_args()


def load_l2_model(model_dir: Path, device: str = "cuda"):
    """Load L2 feature router model."""
    from Config.RouterConfig import RouterConfig, RouterModelConfig, RouterTrainingConfig, RouterDataConfig

    # Load config
    config_path = model_dir / "train_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # Build nested config object
    model_cfg = RouterModelConfig(**config_dict["model"])
    training_cfg = RouterTrainingConfig(**config_dict.get("training", {}))
    data_cfg = RouterDataConfig(**config_dict.get("data", {}))
    config = RouterConfig(model=model_cfg, training=training_cfg, data=data_cfg)

    # Create model
    model = FeatureRouterModel(config)
    model.to(device)

    # Load weights
    weights_path = model_dir / "best_model.pt"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def get_hidden_state_root(dataset: str, result_model: str) -> Path:
    """Get hidden state directory path."""
    return RouterPathConfig.DATASET_ROOT / "HiddenStates" / dataset / result_model


def load_hidden_states(query_ids: List[str], hidden_root: Path, feature_name: str = "mean_hidden") -> Dict[str, torch.Tensor]:
    """Load hidden states for given query IDs."""
    features = {}
    for qid in query_ids:
        fp = hidden_root / f"{qid}.safetensors"
        if fp.exists():
            tensors = load_file(str(fp))
            features[qid] = tensors[feature_name].to(torch.float32)
        else:
            print(f"Warning: Hidden state file not found for {qid}")
    return features


def run_l2_inference(model, hidden_states: Dict[str, torch.Tensor], device: str = "cuda") -> List[Dict[str, Any]]:
    """Run L2 inference on hidden states."""
    predictions = []

    with torch.no_grad():
        for qid, features in hidden_states.items():
            # features shape: [num_layers, hidden_size] e.g. [4, 4096]
            # Add batch dimension
            features_batch = features.unsqueeze(0).to(device)  # [1, num_layers, hidden_size]

            # Forward
            outputs = model({"features": features_batch})
            logits = outputs["logits"]

            # Get prediction
            pred_idx = logits.argmax(dim=1).item()
            pred_strategy = L2_STRATEGIES[pred_idx]

            predictions.append({
                "id": qid,
                "predicted_index": pred_idx,
                "predicted_strategy": pred_strategy,
            })

    return predictions


def main():
    args = parse_args()

    # Load L1 predictions
    with open(args.l1_prediction_file) as f:
        l1_data = json.load(f)

    l1_preds = l1_data["predictions"]

    # Find queries where L1 predicted complex_rag
    l1_complex_ids = [p["id"] for p in l1_preds if p["predicted_strategy"] == "complex_rag"]
    print(f"Found {len(l1_complex_ids)} queries where L1 predicted complex_rag")

    # Load L2 model
    l2_model_dir = Path(args.l2_model_dir)
    print(f"Loading L2 model from {l2_model_dir}")
    model, config = load_l2_model(l2_model_dir)
    print(f"L2 model loaded. Feature: {config.model.hidden_state_feature_name}, Pooling: {config.model.feature_pooling_type}")

    # Load hidden states for L1-complex queries
    hidden_root = get_hidden_state_root(args.dataset, args.result_model)
    print(f"Loading hidden states from {hidden_root}")
    hidden_states = load_hidden_states(l1_complex_ids, hidden_root, config.model.hidden_state_feature_name)
    print(f"Loaded hidden states for {len(hidden_states)} queries")

    # Run L2 inference
    print("Running L2 inference...")
    l2_predictions = run_l2_inference(model, hidden_states)

    # Print some stats
    strategy_counts = {}
    for p in l2_predictions:
        s = p["predicted_strategy"]
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    print(f"L2 prediction distribution: {strategy_counts}")

    # Save predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "l1_prediction_file": args.l1_prediction_file,
            "l2_model_dir": str(l2_model_dir),
            "num_l1_complex_queries": len(l1_complex_ids),
            "strategies": L2_STRATEGIES,
        },
        "predictions": l2_predictions,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"L2 predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
