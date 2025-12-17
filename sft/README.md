# SFT / RM / RLHF(PPO) 训练工程（Transformers + DeepSpeed + TRL）

本目录是一个**最小可运行**的三阶段训练工程，根目录固定为：

- ` /home/liufeng/sdk-ragflow/sft `

包含：

- **阶段一：SFT**（监督微调，Transformers Trainer + DeepSpeed）
- **阶段二：RM**（奖励模型，TRL RewardTrainer）
- **阶段三：RLHF**（PPO，TRL PPOTrainer）

---

## 1. 安装依赖

在该目录下任选一种方式即可：

### 1.1 使用 conda（推荐）

```bash
cd /home/liufeng/sdk-ragflow/sft

conda create -n sft python=3.10 -y
conda activate sft

pip install -U pip
pip install -r requirements.txt
```

### 1.2 使用 venv

```bash
cd /home/liufeng/sdk-ragflow/sft
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2. 数据格式

### 2.1 SFT 数据（jsonl）

每行一个样本：

```json
{"prompt":"用户输入","completion":"期望的回应"}
```

示例：`data/examples/sft_sample.jsonl`

### 2.2 RM 偏好数据（jsonl）

每行一个样本：

```json
{"prompt":"用户输入","chosen":"更好的回应","rejected":"较差的回应"}
```

示例：`data/examples/rm_sample.jsonl`

---

## 3. 快速运行（单机/单卡）

### 3.1 SFT（建议 DeepSpeed）

如果你想最“标准”的方式启用 DeepSpeed（推荐），用它的 launcher 启动：

```bash
cd /home/liufeng/sdk-ragflow/sft

deepspeed --num_gpus=1 -m training.sft \
  model.name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_file=data/examples/sft_sample.jsonl \
  output_dir=outputs/sft-qwen05b \
  train.learning_rate=8e-6 \
  train.per_device_train_batch_size=1 \
  train.gradient_accumulation_steps=8 \
  train.max_steps=50 \
  train.deepspeed_config=configs/deepspeed/zero2.json
```

你也可以继续使用 `python -m training.sft ...`（本工程已在单进程场景下自动避免 DeepSpeed 的 MPI 依赖）。

```bash
cd /home/liufeng/sdk-ragflow/sft

python -m training.sft \
  model.name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_file=data/examples/sft_sample.jsonl \
  output_dir=outputs/sft-qwen05b \
  train.learning_rate=8e-6 \
  train.per_device_train_batch_size=1 \
  train.gradient_accumulation_steps=8 \
  train.max_steps=50 \
  train.deepspeed_config=configs/deepspeed/zero2.json
```

如果遇到 **CUDA OOM（显存不足）**，在单机单卡场景下更推荐开启 **LoRA**（只训练少量 adapter 参数），并关闭 deepspeed：

```bash
cd /home/liufeng/sdk-ragflow/sft

python -m training.sft \
  model.name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_file=data/examples/sft_sample.jsonl \
  output_dir=outputs/sft-qwen05b-lora \
  peft.enabled=true \
  train.deepspeed_config=null \
  train.per_device_train_batch_size=1 \
  train.gradient_accumulation_steps=8 \
  train.max_steps=50
```

> 训练稳定性经验：学习率建议 **≤ 8e-6**（你提供的知识库发现：高于 `8.05e-6` 易发散）。

### 3.2 RM（奖励模型训练）

```bash
cd /home/liufeng/sdk-ragflow/sft

python -m training.reward \
  model.name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_file=data/examples/rm_sample.jsonl \
  output_dir=outputs/rm-qwen05b \
  train.learning_rate=8e-6 \
  train.per_device_train_batch_size=1 \
  train.gradient_accumulation_steps=8 \
  train.max_steps=50
```

### 3.3 RLHF（PPO）

```bash
cd /home/liufeng/sdk-ragflow/sft

python -m training.rlhf \
  policy.name_or_path=outputs/sft-qwen05b \
  reward.name_or_path=outputs/rm-qwen05b \
  data.prompts_file=data/examples/sft_sample.jsonl \
  output_dir=outputs/ppo-qwen05b \
  ppo.kl_coef=0.015 \
  ppo.total_episodes=30
```

> KL 系数建议范围：**0.01–0.02**（你提供的知识库发现）。

---

## 4. 目录结构

```text
sft/
├── configs/
│   ├── sft.yaml
│   ├── reward.yaml
│   ├── rlhf_ppo.yaml
│   └── deepspeed/
│       ├── zero2.json
│       └── zero3.json
├── data/
│   ├── examples/
│   │   ├── sft_sample.jsonl
│   │   └── rm_sample.jsonl
│   └── loaders.py
├── training/
│   ├── sft.py
│   ├── reward.py
│   └── rlhf.py
└── utils/
    ├── seed.py
    └── text.py
```

---

## 5. 常见问题

### 5.1 想用多卡/集群？

- **SFT**：直接用 `deepspeed` 启动或 `torchrun`，`TrainingArguments.deepspeed` 已支持 ZeRO。
- **PPO**：TRL 通常走 `accelerate`；本工程先给出最小可跑实现，后续可按需要升级为 accelerate + deepspeed 配置。


