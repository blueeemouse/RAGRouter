# Router 训练前数据准备步骤（musique / llama-3.1-8b-awq-int4）

当前 six-strategy 的 query-level evaluation 文件已位于：

- `/home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Dataset/EvaluationData/ResultEvaluation/llama-3.1-8b-awq-int4/musique`

在开始第一版 hard-label text baseline 训练前，建议按下面顺序准备数据。

---

## 1. 构造 aggregated router data

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_aggregate_router_data.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4
```

如果想先只检查是否能成功读取六种结果文件，而不落盘：

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_aggregate_router_data.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --dry-run
```

预期产物：
- `Dataset/RouterTrainingData/Aggregated/musique/llama-3.1-8b-awq-int4/query_metrics_v1.json`

---

## 2. 构造 hard label

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_build_router_labels.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --label-type hard
```

如果想先只检查 metadata 和样本数，不落盘：

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_build_router_labels.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --label-type hard \
  --dry-run
```

预期产物：
- `Dataset/RouterTrainingData/Labels/musique/llama-3.1-8b-awq-int4/hard_llm_correct_rule_v1.json`

当前 hard label 规则：
- 先看 `llm_judge_correct`
- 若没有 winner，再退回 `semantic_f1`
- 并列时按预设 priority 选最终策略

详细规则请查看：
- `RouterCore/Data/HardLabelBuilder.py`

---

## 3. 构造 split

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_build_router_split.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4
```

如果想先只检查 split metadata 和各部分样本数：

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_build_router_split.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --dry-run
```

可选参数：
- `--split-name`
- `--seed`
- `--train-ratio`
- `--val-ratio`

当前默认逻辑：
- 先按 hard label 分组
- 每组内部打乱并切分
- `train` / `val` 按比例取
- 剩余全部归 `test`

预期产物：
- `Dataset/RouterTrainingData/Splits/musique/split_v1.json`

---

## 4. 训练前最小 sanity check（建议）

在正式训练前，建议先跑一次 training stack dry-run：

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --dry-run
```

它会检查并构造：
- tokenizer
- hard-label text dataset
- dataloader
- text router model
- optimizer
- hard trainer

并打印：
- train sample 数
- 每 epoch batch 数
- 使用设备
- model class

---

## 5. 第一版 text baseline 训练命令

当前训练入口已经补成：
- train / val / test 三个 split 都会被构造
- 每个 epoch 后会在 val 上评估
- 当前按 val 结果选最优模型（第一版只保留 top-1）
- 训练结束后会在 test 上跑一次最小评估，并保存训练产物

```bash
python /home/lhz/code/RAGRouter-b-feat-router-migration-phase1/Run/Router/run_train_router.py \
  --dataset musique \
  --result-model llama-3.1-8b-awq-int4 \
  --backbone-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --epochs 10 \
  --save-name text_router_baseline_v1
```

说明：
- 当前 trainer 默认：
  - `model.forward(batch)` 返回 `{"logits": ...}`
  - batch 中提供 `labels`
  - loss 用 cross entropy
- 当前评估频率先按 **epoch** 进行，而不是按 step。
- 当前会保存：
  - `best_model.pt`
  - `train_config.json`
  - `metrics.json`

默认保存目录：
- `Dataset/RouterTrainingData/Models/text_router_baseline_v1/musique/`

---

## 6. 当前训练路径的边界

当前已经接通的是：
- hard-label text dataset
- text batch collator
- `TextRouterModel`
- `HardClassificationTrainer`
- `run_train_router.py`

当前还没有正式接通的包括：
- soft-label trainer path
- hidden-states / feature-based router path
- checkpoint 保存与恢复
- 训练后模型持久化
- eval / val loop 的正式实现

所以这一步的目标是：
- 尽快验证 first-stage text baseline 能否正式跑通
- 为后续 hidden-states router 提供对照 baseline
