# All-Failed 第7类扩展实施说明（可直接执行）

目标：在不破坏现有 6 类实验（v1）的前提下，新增 7 类实验（v2），第 7 类为 `all_failed`。

---

## 0. 协议约定（先固定）

- 新类名：`all_failed`
- 触发条件（按 query）：
  - 对全部 6 个 RAG 策略都满足：
    - `llm_judge_correct != 1`
    - `semantic_f1 == 0`
- 新标签文件名：`hard_llm_correct_rule_v2_all_failed_class`
- 新切分文件名：`split_v2_all_failed_class`

兼容原则：
- `v1` 保持原样，可继续复现实验。
- `v2` 独立新增，不覆盖 v1 文件。

---

## 1. 数据层改造

### 1.1 HardLabelBuilder 增加 v2 逻辑

**文件**：`RouterCore/Data/HardLabelBuilder.py`

#### 要做
1. 保留当前 v1 逻辑不变。
2. 支持通过 `label_name` 或显式开关走 v2：
   - 先判断 query 是否满足 all_failed 条件；若满足：
     - `optimal_strategy = "all_failed"`
     - `decision_source = "all_failed_gate"`
     - `label_index` 按当前 strategies 列表计算。
   - 若不满足，沿用现有 `llm_judge_correct -> semantic_f1 -> priority_order` 逻辑。
3. v2 输出 metadata：
   - `strategies` 包含 7 类：`[六类..., "all_failed"]`
   - 建议新增字段：`all_failed_rule` 说明判断规则。

#### 验收
- 生成 `hard_llm_correct_rule_v2_all_failed_class.json` 成功。
- musique 全量 `all_failed` 计数应与报告一致（预期 975）。

---

### 1.2 SplitBuilder 改为按 label metadata.strategies 分层

**文件**：`RouterCore/Data/SplitBuilder.py`

#### 要做
1. 不再固定使用 `STRATEGY_NAMES` 做分组与循环。
2. 从 hard label 文件读取 `metadata.strategies`，作为当前切分类别空间。
3. 保持现有校验：train/val/test 不重叠且全集覆盖。

#### 验收
- 能同时对 v1(6类) 与 v2(7类) label 成功切分。
- 输出 `split_v2_all_failed_class.json` 正常。

---

## 2. 训练链路兼容 6/7 类

### 2.1 DatasetSchema 增加“按传入策略表映射”工具

**文件**：`RouterCore/Data/DatasetSchema.py`

#### 要做
1. 保留现有默认 6 类常量（兼容旧代码）。
2. 新增通用映射函数（示例命名）：
   - `get_strategy_index_from_list(strategy_name, strategies)`
   - `get_strategy_name_from_list(index, strategies)`
3. 新增/调整策略校验函数，允许非固定 6 类（至少非空、无重复、字符串）。

---

### 2.2 RouterHardLabel*Dataset 暴露 strategy_names

**文件**：`RouterCore/Datasets/RouterHardLabelDataset.py`

#### 要做
1. 在读取 hard label 时，保存 `metadata.strategies` 到 `self.strategy_names`。
2. 若 metadata 缺失，可 fallback 到默认 6 类（仅兼容旧文件）。

---

### 2.3 run_train_router 动态覆盖 config.model.strategy_names

**文件**：`Run/Router/run_train_router.py`

#### 要做
1. 构建 train dataset 后，读取其 `strategy_names`。
2. 设置：`config.model.strategy_names = train_dataset.strategy_names`。
3. 再实例化 model，确保 `num_labels = len(config.model.strategy_names)` 自动对应 6/7 类。

说明：`TextRouterModel` 与 `FeatureRouterModel` 已按 `len(config.model.strategy_names)` 建头，天然兼容。

---

### 2.4 HardClassificationTrainer 的标签名映射改为 config 驱动

**文件**：`RouterCore/Trainers/HardClassificationTrainer.py`

#### 要做
1. 预测输出里不要再用全局固定映射函数。
2. 直接从 `self.model.config.model.strategy_names[index]` 得到 `predicted_strategy/true_strategy`。

#### 验收
- 7 类训练时，预测文件可正确出现 `all_failed`。
- 6 类训练行为与之前一致。

---

## 3. 离线评估支持 all_failed（拒答类）

### 3.1 run_collect_rag_baseline 处理 `predicted_strategy == all_failed`

**文件**：`Run/Router/run_collect_rag_baseline.py`

#### 要做
1. 在 `compute_router_metrics_from_predictions` 中分支处理：
   - 若为 `all_failed`：
     - 记 routed record：`action="abstain"`（或 `is_abstain=true`）
     - 不从 `method_metrics[predicted_strategy]` 取指标。
2. 新增输出统计：
   - `abstain_count`
   - `abstain_rate`
3. 建议同时输出：
   - 全量 router metrics（含 abstain 的口径说明）
   - 非 abstain 子集 metrics（便于公平对比）。

#### 验收
- prediction 中有 `all_failed` 时脚本不报错。
- 输出包含 abstain 统计字段。

---

## 4. CLI / 运行流程建议

### 4.1 构建 v2 标签

```bash
python Run/Router/run_build_router_labels.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --label-type hard \
  --dry-run
```

> 需支持指定新 `label_name`（若当前 CLI 无该参数，请补 `--label-name`）。

### 4.2 构建 v2 split

```bash
python Run/Router/run_build_router_split.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --split-name split_v2_all_failed_class \
  --dry-run
```

### 4.3 训练 7 类 router（text/feature 各跑）

- `--label-name hard_llm_correct_rule_v2_all_failed_class`
- `--split-name split_v2_all_failed_class`
- `--save-predictions`

### 4.4 离线评估

沿用 `run_collect_rag_baseline.py --prediction-file ...`，确认有 abstain 统计。

---

## 5. 对比与决策口径（必须）

比较至少四组：
1. 旧 6 类 text
2. 新 7 类 text
3. 旧 6 类 feature
4. 新 7 类 feature

重点看：
- `abstain_rate`（省 token 的直接代理）
- routed `llm_judge_correct`
- routed `semantic_f1`
- （可选）非 abstain 子集的上述指标

决策建议：
- 若 abstain_rate 有明显收益且核心质量不明显下降，可保留 7 类路线。
- 若 abstain 过高并显著伤害质量，考虑提高 all_failed 判定门槛（例如引入软阈值而非 strict=0）。

---

## 6. 代码改动清单（最小集合）

- `RouterCore/Data/HardLabelBuilder.py`
- `RouterCore/Data/SplitBuilder.py`
- `RouterCore/Data/DatasetSchema.py`
- `RouterCore/Datasets/RouterHardLabelDataset.py`
- `Run/Router/run_train_router.py`
- `RouterCore/Trainers/HardClassificationTrainer.py`
- `Run/Router/run_collect_rag_baseline.py`
- （可选）`Run/Router/run_build_router_labels.py`（若需加 `--label-name`）

---

## 7. 回归检查（提交前）

1. v1 6类流程可完整 dry-run + train + eval。
2. v2 7类流程可完整 dry-run + train + eval。
3. 预测文件中 `predicted_index` 与 `predicted_strategy` 一致。
4. offline eval 在出现 `all_failed` 预测时不崩溃，并有 abstain 统计。

---

## 8. 评估口径更新（all_failed：性能=0，检索类指标=None）

### 8.1 背景与目标

`all_failed` 本质是“拒答/不调用RAG”的动作。离线评估上应体现两点：
- 拒答在回答质量上有代价（记 0 分）；
- 检索相关指标在未调用RAG时不适用（记 None）。

### 8.2 指标口径（必须统一）

当 `predicted_strategy == "all_failed"` 时：

- `llm_judge_correct = 0.0`
- `semantic_f1 = 0.0`
- `coverage = None`
- `faithfulness_hard = None`
- `faithfulness_soft = None`
- `is_abstain = true`
- `action = "abstain"`

解释：
- 质量类指标（correct/f1）按 0 进入全量评估，体现拒答机会成本；
- 检索类指标（coverage/faithfulness）不适用，不参与均值。

### 8.3 统计规则

主报告指标：`router_performance`（全量口径）
- `llm_judge_correct` / `semantic_f1`：abstain 样本按 0.0 参与均值；
- `coverage` / `faithfulness_*`：abstain 样本为 None，不参与均值。

同时保留：
- `abstain_count`
- `abstain_rate`
- `num_routed`
- `num_total`

可选诊断：`router_performance_non_abstain`
- 仅用于看“被回答子集”的质量；
- 主结论仍应以 `router_performance` 为准。

### 8.4 代码改动点

**文件**：`Run/Router/run_collect_rag_baseline.py`  
**函数**：`compute_router_metrics_from_predictions(...)`

需要保证 abstain 分支行为：
1. 对 `llm_judge_correct`、`semantic_f1` 赋 `0.0`；
2. 这两个值进入 `metric_values` 的累计；
3. 对 `coverage`、`faithfulness_hard`、`faithfulness_soft` 赋 `None`；
4. 这三个值不进入均值累计；
5. routed record 写入 `is_abstain/action`。

### 8.5 验收标准

1. `per_query_routed` 中 abstain 样本：
   - `llm_judge_correct`、`semantic_f1` 为 `0.0`；
   - `coverage`、`faithfulness_hard`、`faithfulness_soft` 为 `null`。
2. `router_performance` 中：
   - `llm_judge_correct.count` 与 `semantic_f1.count` 应接近 `num_total`（除非存在其他缺失样本）；
   - `coverage/faithfulness` 的 count 应小于等于 `num_routed`。
3. `abstain_rate` 计算正确，且 `num_routed + abstain_count == num_total`。
4. 与旧口径相比，出现 abstain 时全量 quality 均值会下降（符合“拒答有代价”预期）。

---

## 9. 七分类训练与评估执行步骤（可直接运行）

### 9.1 构建 7 类标签与 split

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_build_router_labels.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --label-type hard \
  --label-name hard_llm_correct_rule_v2_all_failed_class
```

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_build_router_split.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --label-name hard_llm_correct_rule_v2_all_failed_class \
  --split-name split_v2_all_failed_class
```

---

### 9.2 训练 7 类 text router（保存预测）

```bash
CUDA_VISIBLE_DEVICES=3 python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --model-type text_router \
  --backbone-name sentence-transformers/all-MiniLM-L6-v2 \
  --label-name hard_llm_correct_rule_v2_all_failed_class \
  --split-name split_v2_all_failed_class \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name text_router_7cls_v1_bs64 \
  --save-predictions
```

---

### 9.3 训练 7 类 feature router（保存预测）

```bash
CUDA_VISIBLE_DEVICES=3 python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --model-type feature_router \
  --feature-name mean_hidden \
  --feature-pooling-type layer_mean \
  --label-name hard_llm_correct_rule_v2_all_failed_class \
  --split-name split_v2_all_failed_class \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name feature_router_7cls_mean_hidden_layer_mean_v1_bs128 \
  --save-predictions
```

---

### 9.4 基于预测文件计算 7 类离线 routed performance

#### text router
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_collect_rag_baseline.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --split-name split_v2_all_failed_class \
  --prediction-file /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Dataset/RouterTrainingData/Evaluation/text_router_7cls_v1_bs64/musique/musique_test_predictions.json
```

#### feature router
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_collect_rag_baseline.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --split-name split_v2_all_failed_class \
  --prediction-file /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Dataset/RouterTrainingData/Evaluation/feature_router_7cls_mean_hidden_layer_mean_v1_bs128/musique/musique_test_predictions.json
```

---

### 9.5 结果解读口径（必须按第8节）

优先看：
1. `router_performance.llm_judge_correct`（全量，abstain=0）
2. `router_performance.semantic_f1`（全量，abstain=0）
3. `abstain_rate`
4. `coverage / faithfulness_*`（abstain=None，不参与均值）

---

### 9.6 横向比较建议

至少比较四组：
- 6类 text vs 7类 text
- 6类 feature vs 7类 feature

并判断：
- abstain_rate 是否换来了可接受的质量代价；
- 在主目标（端到端 routed 性能）上是否值得保留 7 类路线。
