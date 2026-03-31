# 开发笔记

## feat/hidden-states-extraction 分支

**创建时间**：2026-03-31
**基于分支**：dev

### 功能说明

离线提取 LLM (Llama-3.1-8B-AWQ-INT4) 在 prefill 阶段的隐藏状态表征，用于 Router 训练的特征输入。

### 动机

- 表征包含 query 语义 + LLM 对 query 的了解程度
- 可用于更合理的 RAG 路由决策
- 现有 RAG routing 工作未考虑这一点

### 技术方案

使用 vLLM 原生的 `speculative_config` + `kv_transfer_config` 提取隐藏状态：
- 基于 PR #33736 实现的 `extract_hidden_states` 方法
- 利用 speculative decoding 基础设施
- 仅做 prefill（max_tokens=1），提取 prompt 阶段的隐藏状态

### 提取配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 模型 | Llama-3.1-8B-AWQ-INT4 | 与在线服务一致 |
| 层选择 | [8, 16, 24, 31] | 浅、中、深中、最后层 |
| max_model_len | 8192 | 与在线服务一致 |
| dtype | float16 | 与模型运行时一致 |
| seed | 42 | 确保可复现 |
| temperature | 0.0 | 贪婪解码 |
| System Prompt | QA_SYSTEM | LLMDirect 场景 |

### 聚合策略

为减少存储压力，保存聚合结果而非原始 hidden states：
- `mean_hidden`: 所有 token 的均值
- `last_hidden`: 最后一个 token
- `last_5_mean`: 最后 5 个 token 的均值

### 新增文件

```
RAGRouter/
├── Config/
│   └── HiddenStatesConfig.py         # 配置管理
├── HiddenStatesExtraction/
│   ├── __init__.py
│   └── extractor.py                  # 核心提取器
└── Run/
    └── HiddenStates/
        └── extract_hidden_states.py  # 运行入口
```

### 数据存储

```
Dataset/
└── HiddenStates/
    └── {dataset}/
        └── {model}/
            ├── {sample_id}.safetensors  # 单个样本的隐藏状态
            ├── metadata.json             # 元数据
            └── progress.json             # 进度追踪（用于断点续传）
```

### 使用方式

```bash
# 提取单个数据集
python Run/HiddenStates/extract_hidden_states.py --dataset musique

# 不续传，重新开始
python Run/HiddenStates/extract_hidden_states.py --dataset musique --no-resume

# 自定义层选择
python Run/HiddenStates/extract_hidden_states.py --dataset musique --layer-ids 8 16 24 31

# 测试模式（限制样本数）
python Run/HiddenStates/extract_hidden_states.py --dataset musique --max-samples 10
```

### 数据格式

每个 `.safetensors` 文件包含：
```python
{
    "token_ids": Tensor[seq_len],           # int64
    "mean_hidden": Tensor[4, 4096],         # float16
    "last_hidden": Tensor[4, 4096],         # float16
    "last_5_mean": Tensor[4, 4096],         # float16
}
```

### 后续工作

- [ ] 运行提取并评估存储空间
- [ ] 与 Router 训练流程集成
- [ ] 验证表征对路由决策的有效性

---

## dev 分支当前功能

### Token 计数功能（部分完成）
- 新增 `RAGCore/Utils/TokenTracker.py`，记录每次 LLM 调用的 token 用量（in_tokens, out_tokens），附带 phase/function/round 元数据
- 已集成到 LLMDirect 范式，运行验证通过
- 待扩展：NaiveRAG、GraphRAG、HybridRAG、IterativeRAG

#### 设计要点
- 每次处理问题时局部创建 TokenTracker 实例，避免异步并发问题
- token_usage 数据随结果 JSON 持久化，细粒度记录每次调用，事后可按任意维度汇总
- ctx_len 作为独立指标记录上下文长度，不参与 token 总量求和
### 知识图谱三元组提取 - 失败重试机制改进
- **问题**：原版 `retry_failed_chunks` 在重试后直接清空 `failed_chunks.json`（`clear_failed_chunks`），导致重试仍然失败的 chunk 被永久丢弃，下次 resume 无法再次处理
- **改动**（`RAGCore/Graph/GraphDo.py`）：
  - `retry_failed_chunks` 返回值从 `Dict` 改为 `Tuple[Dict, List]`，第二个元素为仍然失败的 chunk 列表
  - `_retry_failed_chunks_async` 中重试结束后，仍失败的 chunk 通过 `GraphSaver.save_failed_chunks()` 重新写回 `failed_chunks.json`（而非直接丢弃）
  - `process()` 方法中检测 `still_failed_chunks`，如有仍失败的 chunk 则抛出 `RuntimeError` 阻止构建图，提示用户处理
  - 去掉了原来无条件调用 `clear_failed_chunks` 的逻辑，仅在全部重试成功时才清除



## feat/router-migration-phase1 分支当前进展

### 已完成：Router 第一阶段目录与骨架搭建
- 新增顶层 `RouterCore/` 子系统目录，明确 router 作为独立模块存在，不继续往原 `main.py` 和全局 `Config/PathConfig.py` 塞 router 专属逻辑
- 新增目录：
  - `RouterCore/Data/`
  - `RouterCore/Datasets/`
  - `RouterCore/Models/`
  - `RouterCore/Trainers/`
  - `RouterCore/Utils/`
  - `RouterCore/Evaluation/`
  - `Run/Router/`
  - `Config/router/`
- 新增骨架文件：
  - `RouterCore/RouterPathConfig.py`
  - `RouterCore/Data/DatasetSchema.py`
  - `RouterCore/Data/EvaluationAggregator.py`
  - `RouterCore/Data/HardLabelBuilder.py`
  - `RouterCore/Data/SoftLabelBuilder.py`
  - `RouterCore/Data/SplitBuilder.py`
  - `RouterCore/Datasets/RouterHardLabelDataset.py`
  - `RouterCore/Datasets/RouterSoftLabelDataset.py`
  - `RouterCore/Models/base_model.py`
  - `RouterCore/Trainers/base_trainer.py`
  - `RouterCore/Trainers/HardClassificationTrainer.py`
  - `RouterCore/Trainers/SoftClassificationTrainer.py`
  - `RouterCore/Utils/collate.py`
  - `RouterCore/Utils/factory.py`
  - `RouterCore/Evaluation/RouterEvaluator.py`
  - `Config/RouterConfig.py`
  - `Config/router/train_hard_label.yaml`
  - `Config/router/train_soft_label.yaml`
  - `Run/Router/run_aggregate_router_data.py`
  - `Run/Router/run_build_router_labels.py`
  - `Run/Router/run_build_router_split.py`
  - `Run/Router/run_train_router.py`
  - `Run/Router/run_eval_router.py`

### 已完成：Step 2 的基础做实
- `RouterCore/RouterPathConfig.py`
  - 已具备 router 专属数据路径管理能力
  - 负责 `Dataset/RouterTrainingData/` 下 Aggregated / Labels / Splits / Models / Evaluation 的路径组织
  - 与全局 `Config/PathConfig.py` 独立，不交叉 import
- `RouterCore/Data/DatasetSchema.py`
  - 已定义第一阶段 6 类策略空间：
    - `llm_direct`
    - `naive_rag`
    - `graph_rag`
    - `hybrid_rag`
    - `iterative_rag_naive`
    - `iterative_rag_graph`
  - 已实现策略名标准化与 index 映射辅助函数
  - 已提供 per-method metrics 的空骨架生成函数

### 当前阶段说明
- 目前大多数 router 文件仍是 skeleton，核心逻辑还未开始实现
- 这样做是为了先把目录、命名、模块边界、路径职责定稳，避免后续迁移旧 model / trainer 时把旧项目的混乱结构直接带进来
- 第一阶段仍然只打算先支持：
  - hard label
  - soft label
  - 两个 dataset
  - 两个 trainer

### 设计口径（当前已明确）
- `main.py` 不扩 router 入口
- router 入口放在 `Run/Router/` 下，并且当前阶段先拆成多个小入口文件，而不是一开始就做一个统一大入口
- router 的路径管理放在 `RouterCore/RouterPathConfig.py` 中，而不是扩展全局 `Config/PathConfig.py`
- router 数据产物仍然落在 `Dataset/RouterTrainingData/` 下
- schema 中的 sample `id` 统一按 **字符串** 处理（例如 `musique_0002`），不再默认使用整数 id
- `token_usage` 未来可能会进入 aggregated data 扩展字段，但当前第一阶段 hard/soft label 逻辑暂时不依赖它

### 下一步计划
- 实现 `EvaluationAggregator.py`
- 让它能够读取 query-level result evaluation 并生成 `query_metrics_v1.json`
- 之后再逐步实现：
  - `HardLabelBuilder.py`
  - `SoftLabelBuilder.py`
  - `SplitBuilder.py`


### 三元组提取偶尔卡住
- **现象**：vLLM 日志显示 Running requests 从 10 逐渐降到 1，生成速度从 ~20 tokens/s 降到 ~2 tokens/s，进度条停滞
- **原因**：某些复杂文档导致 LLM 陷入重复生成（repetition loop），未输出 EOS token。`_extract_single_triplet_async` 未设置 `max_tokens`，vLLM 默认使用 `max_model_len` 作为上限，以 2 tokens/s 生成需要极长时间
- **现有兜底**：`REQUEST_TIMEOUT=600s`（10 分钟），超时后走重试逻辑（最多 4 次尝试，最坏约 40 分钟），最终记为 failed chunk
- **可选改进**：在 `chat.completions.create()` 调用中加入 `max_tokens=2048`（三元组输出通常不超过此长度），可避免此类卡顿。当前未修改以保持与上游项目一致
