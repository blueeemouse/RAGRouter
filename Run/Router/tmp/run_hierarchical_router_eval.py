"""Hierarchical Router Offline Evaluation.

整合 L1 和 L2 预测，计算 hierarchical router 的端到端性能。

Usage:
python Run/Router/run_hierarchical_router_eval.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --l1-prediction-file Dataset/RouterTrainingData/Evaluation/hierarchical_l1_mean_hidden/musique/musique_test_predictions.json \
    --l2-prediction-file Dataset/RouterTrainingData/Evaluation/hierarchical_l2_mean_hidden/musique/musique_test_predictions.json \
    --l2-prediction-file-l1-complex Dataset/RouterTrainingData/Evaluation/hierarchical_l2_mean_hidden/musique/l2_predictions_for_l1_complex.json \
    --output-dir Dataset/RouterTrainingData/Evaluation/hierarchical_mean_hidden
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RouterCore.RouterPathConfig import RouterPathConfig


# L1 的 5 类策略
L1_STRATEGIES = ["llm_direct", "naive_rag", "graph_rag", "complex_rag", "all_failed"]

# L2 的 3 类策略（complex_rag 的子策略）
L2_STRATEGIES = ["hybrid_rag", "iterative_rag_naive", "iterative_rag_graph"]

METRIC_KEYS = [
    "llm_judge_correct",
    "semantic_f1",
    "coverage",
    "faithfulness_hard",
    "faithfulness_soft",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical Router Offline Evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. musique")
    parser.add_argument("--result-model", type=str, required=True, help="Result model name")
    parser.add_argument("--split-name", type=str, default="split_v2_hierarchical", help="Split name")
    parser.add_argument(
        "--l1-prediction-file",
        type=str,
        required=True,
        help="Path to L1 prediction JSON file",
    )
    parser.add_argument(
        "--l2-prediction-file",
        type=str,
        required=True,
        help="Path to L2 prediction JSON file (original L2 test predictions)",
    )
    parser.add_argument(
        "--l2-prediction-file-l1-complex",
        type=str,
        default=None,
        help="Path to L2 prediction JSON file for L1-complex queries (from run_l2_inference.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    return parser.parse_args()


def load_predictions(prediction_file: Path) -> Dict[str, Any]:
    with prediction_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_aggregated(dataset: str, result_model: str) -> Dict[str, Any]:
    aggregated_path = RouterPathConfig.get_aggregated_path(dataset, result_model)
    with aggregated_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_hierarchical_metrics(
    l1_predictions: List[Dict[str, Any]],
    l2_predictions: List[Dict[str, Any]],
    aggregated: Dict[str, Any],
) -> Dict[str, Any]:
    """整合 L1 和 L2 预测，计算端到端性能。

    逻辑：
    1. 构建 L2 预测 lookup（key: query_id）
    2. 对于每个 L1 预测：
       - 如果 L1 预测为 complex_rag，查找 L2 预测作为最终策略
       - 否则直接使用 L1 预测
    3. 从 aggregated data 中获取每个 query 的真实指标
    4. 计算 routed average performance
    """
    # 构建 L2 lookup
    l2_lookup: Dict[str, Dict[str, Any]] = {}
    for pred in l2_predictions:
        l2_lookup[pred["id"]] = pred

    # 构建 aggregated sample lookup
    sample_lookup = {sample["id"]: sample for sample in aggregated["samples"]}

    # 统计
    metric_values = {key: [] for key in METRIC_KEYS}
    routed_records = []
    complex_rag_routed_to_l2 = 0
    complex_rag_no_l2_match = 0

    for pred in l1_predictions:
        query_id = pred["id"]
        l1_predicted_strategy = pred["predicted_strategy"]

        # 确定最终策略
        if l1_predicted_strategy == "complex_rag":
            # 需要 L2 决策
            l2_pred = l2_lookup.get(query_id)
            if l2_pred is not None:
                final_strategy = l2_pred["predicted_strategy"]
                complex_rag_routed_to_l2 += 1
            else:
                # L1 预测 complex_rag 但 L2 没有对应预测（理论上不应该发生）
                final_strategy = "hybrid_rag"  # 默认 fallback
                complex_rag_no_l2_match += 1
        else:
            final_strategy = l1_predicted_strategy

        # 获取真实标签（用于最终正确性判断）
        true_strategy = pred["true_strategy"]

        # 构建路由记录
        routed_record = {
            "id": query_id,
            "l1_predicted": l1_predicted_strategy,
            "l2_predicted": l2_lookup.get(query_id, {}).get("predicted_strategy") if l1_predicted_strategy == "complex_rag" else None,
            "final_predicted_strategy": final_strategy,
            "true_strategy": true_strategy,
        }

        # 从 aggregated 获取该 query 的指标
        sample = sample_lookup.get(query_id)
        if sample is None:
            routed_records.append(routed_record)
            continue

        # 如果 L1 预测为 all_failed，指标记为 0
        if l1_predicted_strategy == "all_failed":
            routed_record["llm_judge_correct"] = 0.0
            routed_record["semantic_f1"] = 0.0
            routed_record["coverage"] = None
            routed_record["faithfulness_hard"] = None
            routed_record["faithfulness_soft"] = None
            routed_record["is_abstain"] = True
            routed_record["action"] = "abstain"
            for key in METRIC_KEYS:
                metric_values[key].append(0.0 if key in ["llm_judge_correct", "semantic_f1"] else None)
            routed_records.append(routed_record)
            continue

        # 否则使用最终策略的指标
        routed_record["is_abstain"] = False
        routed_record["action"] = "route"

        metrics = sample.get("method_metrics", {}).get(final_strategy, {})
        for key in METRIC_KEYS:
            value = metrics.get(key)
            routed_record[key] = value
            if value is not None:
                metric_values[key].append(value)

        routed_records.append(routed_record)

    # 计算平均指标
    router_metrics = {}
    for key, values in metric_values.items():
        non_none_values = [v for v in values if v is not None]
        if non_none_values:
            router_metrics[key] = {
                "mean": sum(non_none_values) / len(non_none_values),
                "count": len(non_none_values),
            }

    # Abstain 统计（all_failed）
    abstain_count = sum(1 for r in routed_records if r.get("is_abstain", False))
    total_predictions = len(routed_records)

    return {
        "router_metrics": router_metrics,
        "routed_records": routed_records,
        "abstain_count": abstain_count,
        "abstain_rate": abstain_count / total_predictions if total_predictions > 0 else 0.0,
        "num_routed": total_predictions - abstain_count,
        "num_total": total_predictions,
        "complex_rag_routed_to_l2": complex_rag_routed_to_l2,
        "complex_rag_no_l2_match": complex_rag_no_l2_match,
    }


def main():
    args = parse_args()

    # 加载 L1 预测
    l1_pred_path = Path(args.l1_prediction_file)
    l1_data = load_predictions(l1_pred_path)
    l1_predictions = l1_data["predictions"]
    print(f"L1 predictions: {len(l1_predictions)}")

    # 加载 L2 预测（原始 L2 test predictions）
    l2_pred_path = Path(args.l2_prediction_file)
    l2_data = load_predictions(l2_pred_path)
    l2_predictions = l2_data["predictions"]
    print(f"L2 predictions (original): {len(l2_predictions)}")

    # 如果提供了 L1-complex 的 L2 predictions，合并到 l2_lookup
    l2_predictions_combined = l2_predictions
    if args.l2_prediction_file_l1_complex:
        l2_l1_complex_path = Path(args.l2_prediction_file_l1_complex)
        l2_l1_complex_data = load_predictions(l2_l1_complex_path)
        l2_l1_complex_predictions = l2_l1_complex_data["predictions"]
        print(f"L2 predictions (L1-complex): {len(l2_l1_complex_predictions)}")

        # 合并：用 L1-complex predictions 覆盖/补充原有的 L2 predictions
        l2_lookup_original = {p["id"]: p for p in l2_predictions}
        for p in l2_l1_complex_predictions:
            l2_lookup_original[p["id"]] = p
        l2_predictions_combined = list(l2_lookup_original.values())
        print(f"L2 predictions (combined): {len(l2_predictions_combined)}")

    # 加载 aggregated data
    aggregated = load_aggregated(args.dataset, args.result_model)
    print(f"Aggregated samples: {len(aggregated['samples'])}")

    # 计算 hierarchical metrics
    results = compute_hierarchical_metrics(l1_predictions, l2_predictions_combined, aggregated)

    # 输出结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hierarchical_router_offline_eval_{args.split_name}.json"

    output = {
        "metadata": {
            "dataset": args.dataset,
            "result_model": args.result_model,
            "split_name": args.split_name,
            "l1_prediction_file": str(l1_pred_path),
            "l2_prediction_file": str(l2_pred_path),
            "l2_prediction_file_l1_complex": args.l2_prediction_file_l1_complex,
            "l1_test_size": len(l1_predictions),
            "l2_test_size": len(l2_predictions_combined),
        },
        "router_performance": results["router_metrics"],
        "abstain_count": results["abstain_count"],
        "abstain_rate": results["abstain_rate"],
        "num_routed": results["num_routed"],
        "num_total": results["num_total"],
        "complex_rag_routed_to_l2": results["complex_rag_routed_to_l2"],
        "complex_rag_no_l2_match": results["complex_rag_no_l2_match"],
        "per_query_routed": results["routed_records"],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Hierarchical Router Offline Evaluation Results ===")
    print(f"Total queries: {results['num_total']}")
    print(f"Abstain (all_failed): {results['abstain_count']} ({results['abstain_rate']:.1%})")
    print(f"Routed to L2 (complex_rag): {results['complex_rag_routed_to_l2']}")
    print(f"\nRouter Performance:")
    for key, val in results["router_metrics"].items():
        print(f"  {key}: {val['mean']:.4f} (n={val['count']})")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
