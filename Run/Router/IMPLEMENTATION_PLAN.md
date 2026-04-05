# Router 离线评估功能实现规划

## 背景

当前 `HardClassificationTrainer.evaluate()` 只返回聚合指标（loss/accuracy），无法获取每个 query 的路由决策结果。需要扩展以支持离线评估。

---

## 1. 修改 HardClassificationTrainer.evaluate()

**文件**: `RouterCore/Trainers/HardClassificationTrainer.py`

**改动**: 添加 `return_predictions=False` 参数，当开启时返回每个 query 的预测详情。

```python
def evaluate(self, dataloader, return_predictions=False) -> Dict[str, Any]:
    """
    当 return_predictions=True 时，返回:
    {
        "loss": float,
        "accuracy": float,
        "num_batches": float,
        "num_examples": float,
        "predictions": [
            {
                "id": str,
                "question": str,
                "predicted_index": int,
                "predicted_strategy": str,
                "true_index": int,
                "true_strategy": str,
                "correct": bool
            },
            ...
        ]
    }

    注意：当 return_predictions=True 时返回 Dict[str, Any]，否则返回 Dict[str, float]。
    """
```

**实现细节**:
- `id` 从 `batch["ids"]` 获取
- `question` 从 `batch["questions"]` 获取（collator 已设置 `return_questions=True`）
- `predicted_strategy` / `true_strategy` 通过 `self.model.config.model.strategy_names` 映射
- `logits` **不保存**，避免结果文件过大

---

## 2. 修改 run_train_router.py

**文件**: `Run/Router/run_train_router.py`

**改动**:

1. 训练完成后，用 best_val_model 在 test 上做预测
2. 添加 `--save-predictions` 参数控制是否保存预测结果（默认 True）
3. 预测结果保存到 `Dataset/RouterTrainingData/Evaluation/{save_name}/{dataset}_test_predictions.json`

**预测结果格式**:
```json
{
  "metadata": {
    "dataset": "musique",
    "result_model": "llama-3.1-8b-awq-int4",
    "split_name": "split_v1",
    "model_save_name": "text_router_baseline_v1",
    "test_size": 340
  },
  "predictions": [
    {
      "id": "musique_0302",
      "question": "Who is the spouse of the Green performer?",
      "predicted_index": 0,
      "predicted_strategy": "llm_direct",
      "true_index": 2,
      "true_strategy": "graph_rag",
      "correct": false
    },
    ...
  ],
  "aggregated_metrics": {
    "accuracy": 0.538,
    "num_correct": 183,
    "num_total": 340
  }
}
```

---

## 3. 新增离线评估脚本

**文件**: `Run/Router/run_offline_router_eval.py` (新建)

**功能**:
1. 加载训练好的模型预测结果 (`test_predictions.json`)
2. 结合 aggregated data (`query_metrics_v1.json`)，计算端到端性能
3. 对比各 baseline（所有单个 RAG 策略的性能）

**输出格式**:
```json
{
  "metadata": {
    "dataset": "musique",
    "result_model": "llama-3.1-8b-awq-int4",
    "split_name": "split_v1",
    "model_save_name": "text_router_baseline_v1",
    "test_size": 340
  },
  "router_performance": {
    "accuracy": 0.538,
    "avg_semantic_f1": 0.45,
    "avg_llm_correct_rate": 0.22,
    "avg_coverage": 0.28,
    "avg_faithfulness_hard": 0.15
  },
  "baseline_performance": {
    "llm_direct": {
      "avg_semantic_f1": 0.43,
      "avg_llm_correct_rate": 0.04,
      "avg_coverage": 0.20,
      "avg_faithfulness_hard": 0.00
    },
    "naive_rag": {...},
    "graph_rag": {...},
    "hybrid_rag": {...},
    "iterative_rag_naive": {...},
    "iterative_rag_graph": {...}
  },
  "oracle_performance": {
    "avg_semantic_f1": 0.62,
    "avg_llm_correct_rate": 0.28
  },
  "comparison": {
    "router_vs_best_strategy": {
      "best_strategy": "hybrid_rag",
      "router_vs_best": {
        "avg_semantic_f1": {"router": 0.45, "best": 0.40, "diff": +0.05},
        "avg_llm_correct_rate": {"router": 0.22, "best": 0.21, "diff": +0.01}
      }
    },
    "router_vs_all_strategies": {
      "llm_direct": {"diff_semantic_f1": +0.02, "diff_llm_correct_rate": +0.18},
      "naive_rag": {...},
      ...
    }
  }
}
```

---

## 4. 离线评估计算逻辑

### 4.1 端到端性能计算

对于 test 集中的每个 query：
1. 从预测结果获取 router 选择的策略
2. 从 aggregated data 获取该 query 该策略的性能指标
3. 汇总计算 router 的平均性能

**容错处理**：
- 如果某 query 在某策略上没有评估结果（超时/失败），对应指标记为 `null`，不计入平均
- 计算平均时只对非 null 值求平均（与 `run_collect_rag_baseline.py` 保持一致）

### 4.2 Baseline 对比

计算以下 baseline：
- **各策略单独使用**: 简单平均所有 query 在该策略上的性能
- **Oracle**: 每个 query 都选择最优策略（仅作为参考上限）

### 4.3 对比方式

**router_vs_best_strategy**: 比较 router 和单独表现最好的策略

**router_vs_all_strategies**: 分别比较 router 和每个策略，格式：
```json
{
  "llm_direct": {"diff_semantic_f1": +0.02, "diff_llm_correct_rate": +0.18},
  "naive_rag": {"diff_semantic_f1": +0.15, "diff_llm_correct_rate": +0.05},
  ...
}
```

### 4.4 评估指标

- `semantic_f1`: 语义相似度
- `llm_judge_correct`: LLM 评判正确率
- `coverage`: 覆盖度
- `faithfulness_hard`: 忠诚度（硬指标）

---

## 5. 实施顺序

1. **修改 HardClassificationTrainer.evaluate()** - 支持返回 per-query 预测（含 question）
2. **修改 run_train_router.py** - 训练后保存预测结果
3. **创建 run_offline_router_eval.py** - 离线评估脚本，对比所有策略
4. **测试完整流程** - 用 filtered split 训练并评估

---

## 6. 待确认

- [x] 不保存 logits，避免结果文件过大
- [ ] 是否需要处理 feature_router 模型（目前只实现 text_router）
- [ ] 评估时是否需要记录推理延迟？

---

## 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `RouterCore/Trainers/HardClassificationTrainer.py` | 修改 | evaluate() 返回类型改为 Dict[str, Any]，支持 return_predictions，predictions 包含 question |
| `Run/Router/run_train_router.py` | 修改 | 训练后保存预测结果，添加 --save-predictions 参数 |
| `Run/Router/run_offline_router_eval.py` | 新增 | 离线评估脚本，对比所有策略 |
