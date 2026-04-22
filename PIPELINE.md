# Router Training Pipeline

本文档描述从数据收集到模型评估的完整流程。

---

## 快速开始

```bash
# 完整流程（从评估开始，跳过 Step 1）
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5

# 跳过评估，从聚合开始
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5 --skip-eval

# 同时训练 text 和 feature router
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5 --skip-eval --train-feat

# 从指定步骤开始
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5 --start-step 5
```

---

## 脚本目录

```
scripts/
├── step2_eval.sh       # 评估 RAG 答案
├── step3_aggregate.sh  # 聚合评估数据
├── step4_filter.sh     # 过滤 all-failed 样本
├── step5_labels.sh     # 生成训练标签
├── step6_split.sh      # 划分数据集
├── step7_train.sh      # 训练模型
├── step8_eval.sh       # 端到端评估
└── run_all.sh          # 完整流程
```

---

## 目录结构

```
Dataset/
├── RawData/                          # 原始数据集 (Question.json, Corpus.json)
├── ProcessedData/                    # 预处理数据 (Index, Graph, Embedding)
├── RetrievalResultData/              # Step 1: RAG 执行结果
│   ├── LLMDirect/{model}/{dataset}/
│   │   └── answer.jsonl
│   ├── NaiveRAG/{model}/{dataset}/
│   │   └── answer.jsonl
│   └── GraphRAG/{model}/{dataset}/
│       └── answer.jsonl
├── EvaluationData/                   # Step 2: 评估结果
│   └── ResultEvaluation/{model}/{dataset}/{method}/
│       └── result.jsonl
├── HiddenStates/                     # Step 1.5: Hidden States (可选)
│   └── {model}/{dataset}/{method}/
│       └── hidden_states.safetensors
└── RouterTrainingData/               # Step 3-7: Router 训练相关
    ├── Aggregated/{dataset}/{model}/
    │   └── query_metrics_*.json
    ├── Labels/{dataset}/{model}/
    │   └── hard_llm_correct_rule_*.json
    ├── Splits/{dataset}/
    │   └── split_*.json
    ├── Models/{model_name}/{dataset}/
    │   ├── best_model.pt
    │   ├── train_config.json
    │   └── metrics.json
    └── Evaluation/{model_name}/{dataset}/
        └── *_comprehensive_eval.json
```

---

## Step 1: 执行 RAG 获取答案

运行各种 RAG 方法，生成 `answer.jsonl` 文件。

### 脚本

```bash
# LLM Direct (无检索)
python Run/Retrieval/run_llm_direct.py \
    --dataset musique \
    --model-name llama-3.1-8b-awq-int4

# Naive RAG (向量检索)
python Run/Retrieval/run_naive_rag.py \
    --dataset musique \
    --model-name llama-3.1-8b-awq-int4

# Graph RAG (图检索)
python Run/Retrieval/run_graph_rag.py \
    --dataset musique \
    --model-name llama-3.1-8b-awq-int4
```

### 输出

- `Dataset/RetrievalResultData/{Method}/{model}/{dataset}/answer.jsonl`

### answer.jsonl 格式

```json
{"id": "musique_0001", "question": "...", "answer": "...", "token_usage": {"total": {"in_tokens": 123, "out_tokens": 45}}}
```

**注意**: 温度=0 时，`out_tokens` 可以准确统计。

---

## Step 1.5: 提取 Hidden States (可选，用于 feature_router)

如果需要训练 feature_router，需要提取 LLM 的 hidden states。

### 脚本

```bash
# 提取指定方法的 hidden states
python Run/HiddenStates/run_extract_hidden_states.py \
    --dataset musique \
    --model-name llama-3.1-8b-awq-int4 \
    --method llm_direct
```

### 输出

- `Dataset/HiddenStates/{model}/{dataset}/{method}/hidden_states.safetensors`

### safetensors 格式

包含以下特征：
- `token_ids`: token IDs
- `mean_hidden`: 所有层的平均 hidden state
- `last_hidden`: 最后一层的 hidden state
- `last_5_mean`: 最后5层的平均 hidden state

---

## Step 2: 评估答案质量

对每个方法的答案进行多维度评估。

### 脚本

```bash
# 评估单个方法
python Run/Evaluation/run_result_eval.py \
    --dataset musique \
    --method llm_direct \
    --result-model llama-3.1-8b-awq-int4

python Run/Evaluation/run_result_eval.py \
    --dataset musique \
    --method naive_rag \
    --result-model llama-3.1-8b-awq-int4

python Run/Evaluation/run_result_eval.py \
    --dataset musique \
    --method graph_rag \
    --result-model llama-3.1-8b-awq-int4
```

### 输出

- `Dataset/EvaluationData/ResultEvaluation/{model}/{dataset}/{method}/result.jsonl`

### result.jsonl 格式

```json
{
  "id": "musique_0001",
  "llm_judge_correct": 1,
  "semantic_f1": 0.85,
  "token_f1": 0.72,
  "bleu1": 0.65,
  "rouge1_f": 0.70,
  "coverage": 0.80,
  ...
}
```

---

## Step 3: 聚合评估数据

将所有方法的评估结果合并到单个文件，并整合 token 使用信息。

### 脚本

```bash
# 聚合评估数据 (含 token 信息)
python Run/Router/run_aggregate_router_data_with_tokens.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --strategies llm_direct naive_rag graph_rag \
    --save-name query_metrics_v5
```

### 输出

- `Dataset/RouterTrainingData/Aggregated/{dataset}/{model}/query_metrics_v5.json`

### aggregated 文件格式

```json
{
  "metadata": {
    "dataset": "musique",
    "result_model": "llama-3.1-8b-awq-int4",
    "strategies": ["llm_direct", "naive_rag", "graph_rag"],
    "total_samples": 3356
  },
  "samples": [
    {
      "id": "musique_0001",
      "question": "...",
      "ground_truth": "...",
      "features": {
        "mean_hidden": [...],
        "last_hidden": [...],
        "last_5_mean": [...]
      },
      "method_metrics": {
        "llm_direct": {
          "llm_judge_correct": 1,
          "semantic_f1": 0.85,
          "token_f1": 0.72,
          "input_tokens": 123,
          "output_tokens": 45
        },
        "naive_rag": {...},
        "graph_rag": {...}
      }
    }
  ]
}
```

---

## Step 4: 过滤无效样本

过滤掉所有方法都失败的样本（all_failed）。

### 过滤条件

- 所有方法的 `llm_judge_correct == 0` AND `token_f1 == 0`

### 脚本

```bash
python Run/Router/run_filter_all_failed.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --aggregated-name query_metrics_v5 \
    --save-name query_metrics_v5_filtered
```

### 输出

- `Dataset/RouterTrainingData/Aggregated/{dataset}/{model}/query_metrics_v5_filtered.json`
- `Dataset/RouterTrainingData/Aggregated/{dataset}/{model}/query_metrics_v5_filtered_removed_ids.json` (被过滤的 sample IDs)

### 选项

- `--dry-run`: 只显示过滤统计，不保存文件

---

## Step 5: 生成训练标签

根据评估结果生成每个样本的最优策略标签。

### 标签规则 (v3a)

1. 主指标: `llm_judge_correct`
2. 平局决胜: `token_f1` → `bleu1` → `semantic_f1`
3. 策略优先级: `llm_direct` > `naive_rag` > `graph_rag` (当多策略并列最优时)

### 脚本

```bash
python Run/Router/run_build_router_labels_with_source.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --aggregated-name query_metrics_v5_filtered \
    --label-name hard_llm_correct_rule_v5
```

### 输出

- `Dataset/RouterTrainingData/Labels/{dataset}/{model}/hard_llm_correct_rule_v5.json`

### labels 文件格式

```json
{
  "metadata": {
    "dataset": "musique",
    "result_model": "llama-3.1-8b-awq-int4",
    "label_name": "hard_llm_correct_rule_v5",
    "strategies": ["llm_direct", "naive_rag", "graph_rag"],
    "primary_metric": "llm_judge_correct"
  },
  "samples": [
    {
      "id": "musique_0001",
      "optimal_strategy": "graph_rag",
      "label_index": 2,
      "candidate_best_strategies": ["graph_rag"],
      "decision_source": "llm_judge_correct"
    }
  ]
}
```

---

## Step 6: 划分训练/验证/测试集

分层抽样，保持各策略比例。

### 脚本

```bash
python Run/Router/run_build_router_split.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --label-name hard_llm_correct_rule_v5 \
    --split-name split_v5_8_1_1 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42
```

### 输出

- `Dataset/RouterTrainingData/Splits/{dataset}/split_v5_8_1_1.json`

### split 文件格式

```json
{
  "metadata": {
    "dataset": "musique",
    "split_name": "split_v5_8_1_1",
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1
  },
  "splits": {
    "train": ["musique_0001", ...],
    "val": ["musique_0100", ...],
    "test": ["musique_0200", ...]
  }
}
```

---

## Step 7: 训练 Router 模型

支持 text_router 和 feature_router 两种类型。

### Text Router

```bash
python Run/Router/run_train_router.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --model-type text_router \
    --backbone-name sentence-transformers/all-MiniLM-L6-v2 \
    --split-name split_v5_8_1_1 \
    --label-name hard_llm_correct_rule_v5 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --epochs 10 \
    --save-name router_v5_text
```

### Feature Router

```bash
# mean_hidden
python Run/Router/run_train_router.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --model-type feature_router \
    --feature-name mean_hidden \
    --split-name split_v5_8_1_1 \
    --label-name hard_llm_correct_rule_v5 \
    --batch-size 128 \
    --learning-rate 1e-4 \
    --epochs 10 \
    --save-name router_v5_feat_mean

# last_hidden
python Run/Router/run_train_router.py \
    ... \
    --feature-name last_hidden \
    --save-name router_v5_feat_last

# last_5_mean
python Run/Router/run_train_router.py \
    ... \
    --feature-name last_5_mean \
    --save-name router_v5_feat_last5
```

### 输出

- `Dataset/RouterTrainingData/Models/{model_name}/{dataset}/`
  - `best_model.pt` - 模型权重
  - `train_config.json` - 训练配置
  - `metrics.json` - 训练指标

---

## Step 8: 端到端离线评估

评估 Router 的实际效果，包含多维度指标和 token 开销。

### 脚本

```bash
# Text Router
python Run/Router/run_comprehensive_router_eval.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --model-name router_v5_text \
    --model-type text_router \
    --split-name split_v5_8_1_1 \
    --label-name hard_llm_correct_rule_v5 \
    --aggregated-name query_metrics_v5_filtered

# Feature Router
python Run/Router/run_comprehensive_router_eval.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --model-name router_v5_feat_mean \
    --model-type feature_router \
    --feature-name mean_hidden \
    --split-name split_v5_8_1_1 \
    --label-name hard_llm_correct_rule_v5 \
    --aggregated-name query_metrics_v5_filtered
```

### 输出

- `Dataset/RouterTrainingData/Evaluation/{model_name}_comprehensive_eval/{dataset}/musique_comprehensive_eval.json`

### 评估报告内容

```json
{
  "metadata": {...},
  "router_metrics": {
    "llm_judge_correct": 0.75,
    "semantic_f1": 0.88,
    "token_f1": 0.65,
    "accuracy": 0.82
  },
  "token_overhead": {
    "input_tokens_avg": 4500,
    "output_tokens_avg": 100
  },
  "baseline_metrics": {
    "llm_direct": {...},
    "naive_rag": {...},
    "graph_rag": {...}
  },
  "oracle_metrics": {...},
  "comparison": {
    "router_vs_best_baseline": {...},
    "router_vs_oracle": {...}
  }
}
```

---

## 完整 Pipeline 示例

```bash
# === Step 1: 执行 RAG ===
python Run/Retrieval/run_llm_direct.py --dataset musique --model-name llama-3.1-8b-awq-int4
python Run/Retrieval/run_naive_rag.py --dataset musique --model-name llama-3.1-8b-awq-int4
python Run/Retrieval/run_graph_rag.py --dataset musique --model-name llama-3.1-8b-awq-int4

# === Step 2: 评估 ===
python Run/Evaluation/run_result_eval.py --dataset musique --method llm_direct --result-model llama-3.1-8b-awq-int4
python Run/Evaluation/run_result_eval.py --dataset musique --method naive_rag --result-model llama-3.1-8b-awq-int4
python Run/Evaluation/run_result_eval.py --dataset musique --method graph_rag --result-model llama-3.1-8b-awq-int4

# === Step 3: 聚合 ===
python Run/Router/run_aggregate_router_data_with_tokens.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --strategies llm_direct naive_rag graph_rag \
    --save-name query_metrics_v5

# === Step 4: 过滤 all_failed ===
python Run/Router/run_filter_all_failed.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --aggregated-name query_metrics_v5 \
    --save-name query_metrics_v5_filtered

# === Step 5: 生成标签 ===
python Run/Router/run_build_router_labels_with_source.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --aggregated-name query_metrics_v5_filtered \
    --label-name hard_llm_correct_rule_v5

# === Step 6: 划分数据集 ===
python Run/Router/run_build_router_split.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --label-name hard_llm_correct_rule_v5 \
    --split-name split_v5_8_1_1

# === Step 7: 训练模型 ===
python Run/Router/run_train_router.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --model-type text_router \
    --split-name split_v5_8_1_1 \
    --label-name hard_llm_correct_rule_v5 \
    --save-name router_v5_text

# === Step 8: 评估 ===
python Run/Router/run_comprehensive_router_eval.py \
    --dataset musique \
    --result-model llama-3.1-8b-awq-int4 \
    --model-name router_v5_text \
    --model-type text_router \
    --split-name split_v5_8_1_1 \
    --label-name hard_llm_correct_rule_v5 \
    --aggregated-name query_metrics_v5_filtered
```

---

---

## 各脚本详细说明

### step2_eval.sh - 评估 RAG 答案

```bash
./scripts/step2_eval.sh <dataset> <result_model>

# 示例
./scripts/step2_eval.sh musique llama-3.1-8b-awq-int4
```

评估 llm_direct, naive_rag, graph_rag 三种方法的答案质量。

---

### step3_aggregate.sh - 聚合评估数据

```bash
./scripts/step3_aggregate.sh <dataset> <result_model> <version>

# 示例
./scripts/step3_aggregate.sh musique llama-3.1-8b-awq-int4 v5
```

输出：`query_metrics_v5.json`

---

### step4_filter.sh - 过滤样本

```bash
./scripts/step4_filter.sh <dataset> <result_model> <version>

# 示例
./scripts/step4_filter.sh musique llama-3.1-8b-awq-int4 v5
```

过滤条件：所有方法的 `llm_judge_correct=0 AND token_f1=0`

输出：`query_metrics_v5_filtered.json`

---

### step5_labels.sh - 生成标签

```bash
./scripts/step5_labels.sh <dataset> <result_model> <version>

# 示例
./scripts/step5_labels.sh musique llama-3.1-8b-awq-int4 v5
```

标签规则：`llm_judge_correct` → `token_f1` → `bleu1` → `semantic_f1`

输出：`hard_llm_correct_rule_v5.json`

---

### step6_split.sh - 划分数据集

```bash
./scripts/step6_split.sh <dataset> <result_model> <version> [train_ratio] [val_ratio] [seed]

# 示例
./scripts/step6_split.sh musique llama-3.1-8b-awq-int4 v5 0.8 0.1 42
```

输出：`split_v5_8_1_1.json`

---

### step7_train.sh - 训练模型

```bash
./scripts/step7_train.sh <dataset> <result_model> <version> [options]

# 只训练 text router（默认）
./scripts/step7_train.sh musique llama-3.1-8b-awq-int4 v5

# 只训练 feature router
./scripts/step7_train.sh musique llama-3.1-8b-awq-int4 v5 --feat-only

# 同时训练
./scripts/step7_train.sh musique llama-3.1-8b-awq-int4 v5 --all

# 指定设备
./scripts/step7_train.sh musique llama-3.1-8b-awq-int4 v5 --device cuda
```

选项：
- `--text-only`: 只训练 text router（默认）
- `--feat-only`: 只训练 feature router
- `--all`: 同时训练
- `--batch-size N`: 批大小
- `--epochs N`: 训练轮数
- `--device DEVICE`: 设备

输出：`router_v5_text/`, `router_v5_feat_mean/` 等

---

### step8_eval.sh - 端到端评估

```bash
./scripts/step8_eval.sh <dataset> <result_model> <version> [options]

# 只评估 text router（默认）
./scripts/step8_eval.sh musique llama-3.1-8b-awq-int4 v5

# 同时评估
./scripts/step8_eval.sh musique llama-3.1-8b-awq-int4 v5 --all
```

输出：`router_v5_text_comprehensive_eval/` 等

---

### run_all.sh - 完整流程

```bash
./scripts/run_all.sh <dataset> <result_model> <version> [options]

# 完整流程
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5

# 跳过评估
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5 --skip-eval

# 训练 feature router
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5 --skip-eval --train-feat

# 从指定步骤开始
./scripts/run_all.sh musique llama-3.1-8b-awq-int4 v5 --start-step 5
```

---

## 版本命名约定

| 文件类型 | 命名规则 | 示例 |
|---------|---------|------|
| Aggregated | `query_metrics_v{版本}_{描述}` | `query_metrics_v5_3class_with_tokens` |
| Labels | `hard_llm_correct_rule_v{版本}_{描述}` | `hard_llm_correct_rule_v5_filtered` |
| Splits | `split_v{版本}_{比例}` | `split_v5_8_1_1` |
| Models | `router_v{版本}_{类型}[_{特征}]` | `router_v5_text`, `router_v5_feat_mean` |

---

## 需要补充的脚本

目前缺失的脚本：

1. ~~**`run_filter_all_failed.py`** - 过滤所有方法都失败的样本~~ ✅ 已创建
2. **Hidden states 提取** - 如果需要 feature_router，需要从 temperature=0 的数据重新提取
   - 现有的 hidden states 在 `Dataset/HiddenStates/{dataset}/{model}/`
   - 这些是之前 temperature≠0 时提取的
   - 需要用 temperature=0 的 answer.jsonl 重新提取

## Features 加载方式

**现状**：
- aggregated 文件**不存储** features，只存储评估指标
- 训练 feature_router 时，`RouterHardLabelFeatureDataset` 动态从 safetensors 加载
- 路径：`HiddenStates/{dataset}/{model}/{sample_id}.safetensors`
- 每个 safetensors 包含：`mean_hidden`, `last_hidden`, `last_5_mean`

**优点**：
- aggregated 文件小
- 可灵活选择特征类型
- 不重复存储

**流程**：
```
训练时:
  Dataset.__init__() 
    → 遍历 split_ids
    → load_file("{sample_id}.safetensors")
    → 提取 tensors[feature_name]
    → 放入 sample_dict["features"]
```
