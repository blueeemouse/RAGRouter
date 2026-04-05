# Router 实验运行与离线评估操作说明

这个文件用于记录：
1. 如何运行 router 训练实验
2. 如何保存 per-query prediction
3. 如何整理 router 在 test split 上的 routed average performance
4. 如何与 baseline / oracle 对比

建议后续如果由另一个 agent 持续尝试实验，统一参考本文件。

---

## 一、基础约定

当前工作目录：
- `/home/lhz/code/RAGRouter-b-feat-router-migration-phase1`

当前默认环境：
- `ragBench`

基础运行方式：
- 直接执行 `python Run/Router/run_train_router.py ...`
- 当前训练入口内部已自动补项目根路径，不需要手工 export `PYTHONPATH`

当前支持两类模型：
- `text_router`
- `feature_router`

---

## 二、训练前提

在开始训练前，应确保以下数据已经准备完成：

1. aggregated data
2. hard label
3. split

对应目录通常为：
- `Dataset/RouterTrainingData/Aggregated/{dataset}/{result_model}/query_metrics_v1.json`
- `Dataset/RouterTrainingData/Labels/{dataset}/{result_model}/hard_llm_correct_rule_v1.json`
- `Dataset/RouterTrainingData/Splits/{dataset}/split_v1.json`

如果要用 hidden states / 内部表征路线，还需确保：
- `Dataset/HiddenStates/{dataset}/{result_model}/` 下的 `.safetensors` 文件 shape 一致
- 对于 musique / llama-3.1-8b-awq-int4，目前已确认四个数据集的 hidden-state shape 是一致的

---

## 三、text router 训练命令

### 1. dry-run
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --model-type text_router \
  --backbone-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name text_router_baseline_v2_bs64 \
  --dry-run
```

### 2. 正式训练（保存 per-query prediction）
```bash
CUDA_VISIBLE_DEVICES=3 python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --model-type text_router \
  --backbone-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name text_router_baseline_v2_bs64 \
  --save-predictions
```

---

## 四、feature router 训练命令

### 当前 feature baseline 约定
- feature source: `mean_hidden` / `last_hidden` / `last_5_mean`
- 当前优先实验输入：4 个 layer 做 `layer_mean`
- 当前骨干：更深的 MLP（不是一层线性头）

### 1. dry-run
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --model-type feature_router \
  --feature-name mean_hidden \
  --feature-pooling-type layer_mean \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name feature_router_mean_hidden_layer_mean_v2_bs128 \
  --dry-run
```

### 2. 正式训练（保存 per-query prediction）
```bash
CUDA_VISIBLE_DEVICES=3 python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --model-type feature_router \
  --feature-name mean_hidden \
  --feature-pooling-type layer_mean \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name feature_router_mean_hidden_layer_mean_v2_bs128 \
  --save-predictions
```

### 3. 可替换的 feature source
如果要继续实验，可将 `--feature-name` 替换为：
- `mean_hidden`
- `last_hidden`
- `last_5_mean`

例如：
```bash
--feature-name last_hidden
```

---

## 五、训练产物说明

每次训练后，会保存：

### 模型与配置
目录：
- `Dataset/RouterTrainingData/Models/{save_name}/{dataset}/`

其中包含：
- `best_model.pt`
- `train_config.json`
- `metrics.json`

### test split 的 per-query prediction
目录：
- `Dataset/RouterTrainingData/Evaluation/{save_name}/{dataset}/`

其中包含：
- `{dataset}_test_predictions.json`

prediction 文件里保存：
- `id`
- `predicted_index`
- `predicted_strategy`
- `true_index`
- `true_strategy`
- `correct`
- `question`（如果当前 batch 中有）

---

## 六、baseline / oracle 的离线结果

baseline 与 oracle 只需要跑一次。

命令：
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_collect_rag_baseline.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4
```

输出文件：
- `Dataset/RouterTrainingData/Evaluation/baseline_collection/musique/rag_baseline_llama-3.1-8b-awq-int4_split_v1.json`

当前主参考文件还包括：
- `Dataset/RouterTrainingData/Evaluation/musique/llama-3.1-8b-awq-int4/rag_baseline_split_v1.json`
- `Dataset/RouterTrainingData/Evaluation/musique/llama-3.1-8b-awq-int4/RAG_BASELINE_REPORT.md`

---

## 七、基于 prediction 计算 router routed performance

当训练完成并保存了 prediction 文件后，可以单独计算该 router 在 test split 上的 routed average performance。

### text router 示例
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_collect_rag_baseline.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --prediction-file /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Dataset/RouterTrainingData/Evaluation/text_router_baseline_v2_bs64/musique/musique_test_predictions.json
```

### feature router 示例
```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_collect_rag_baseline.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --prediction-file /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Dataset/RouterTrainingData/Evaluation/feature_router_mean_hidden_layer_mean_v2_bs128/musique/musique_test_predictions.json
```

输出文件会保存到 prediction 文件所在目录：
- `router_offline_eval_split_v1.json`

其中当前包含：
- `router_performance`
- `per_query_routed`

---

## 八、当前 offline eval 的使用原则

### 不传 `--prediction-file`
表示：
- 只收集 baseline + oracle
- 这部分通常只需要跑一次

### 传入 `--prediction-file`
表示：
- 只收集当前 router 的 routed performance
- 这部分会随着训练实验变化而多次运行

---

## 九、当前重点比较方式

当前真正要比较的不是单纯分类 accuracy，而是：

1. 各单一 RAG 策略在 **test split** 上的平均性能
2. router 在 **test split** 上 routed average performance
3. oracle 上限

重点指标当前包括：
- `llm_judge_correct`
- `semantic_f1`
- `coverage`
- `faithfulness_hard`
- `faithfulness_soft`

---

## 十、推荐实验顺序

### 先做 text baseline
1. dry-run
2. 正式训练并保存 predictions
3. 跑 router offline eval

### 再做 feature baseline
建议优先顺序：
1. `mean_hidden + layer_mean`
2. `last_hidden + layer_mean`
3. `last_5_mean + layer_mean`

每个实验都重复：
- 训练
- 保存 predictions
- 跑 router offline eval

---

## 十一、当前 feature 路线注意事项

1. hidden-state 数据 shape 需要一致；目前四个数据集已经做过全量检查，shape 一致。
2. 旧异常样本（如 musique_0001 / musique_0002）如果再次出现，应优先怀疑 hidden-state 提取残留问题，而不是 trainer/model 本身。
3. feature router 当前默认使用更深的 MLP 骨干，而不再使用早期的一层线性头。

---

## 十二、当前文件职责

- `Run/Router/run_train_router.py`
  - 训练入口
  - 支持 text_router / feature_router
  - 支持 `--save-predictions`

- `Run/Router/run_collect_rag_baseline.py`
  - 不传 prediction：baseline + oracle
  - 传 prediction：router routed performance

- `Dataset/RouterTrainingData/Evaluation/.../RAG_BASELINE_REPORT.md`
  - 汇总结果报告

---

## 十三、建议另一个 agent 每次实验后记录

建议每次实验至少记录：
- split 名
- model_type
- feature_name（如果是 feature_router）
- feature_pooling_type（如果是 feature_router）
- batch_size
- learning_rate
- epochs
- save_name
- test routed metrics
- 与 best single strategy / oracle 的对比结论
