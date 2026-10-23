# 关于蒸馏脚本

| 文件 | 知识来源与区别 |
| --- | --- |
| [train_distill_mi_ma.py](train_distill_mi_ma.py) | （主实验）原始单层离线蒸馏；支持监督分类、logit、feature、attention 损失，并保存 macro/micro 对应模型 |
| [train_distill_mi_ma_cached_topk.py](train_distill_mi_ma_cached_topk.py) | （level-token的top-k探测实验脚本）从紧凑 FP32 缓存构造层级 token Top-K 目标；`--attention_topk` 支持正整数或 `all` |
| [train_distill_mi_ma_feature_ablation.py](train_distill_mi_ma_feature_ablation.py) | （feature-mean消融）`level_token` 与 `text_mean` 特征载体消融；分类仍使用层级 token |
| [train_distill_mi_ma_fullatt.py](train_distill_mi_ma_fullatt.py) | 单个 Transformer 层的完整有效 token 对 attention 蒸馏 |
| [train_distill_mi_ma_fullatt_ltot.py](train_distill_mi_ma_fullatt_ltot.py) | （level-token的fullatt蒸馏）`--full_attention_scope full/level_to_text`，默认 `full` |
| [train_distill_mi_ma_fullatt_scope.py](train_distill_mi_ma_fullatt_scope.py) | （当时主要为了做正文 token → 正文 token 的全量注意力蒸馏）`--full_attention_scope full/text_to_text`，默认 `full` |
| [train_distill_mi_ma_text_topk.py](train_distill_mi_ma_text_topk.py) | （这个是fp16保存了完整的att，然后选择的top-k，之后的top-k的选择可能会有点差别）增加 `matched_text_topk` 对照：使用末尾文本查询，教师 Top-K 加 `other` 桶 |
| [train_distill_mi_ma_text_topk1.py](train_distill_mi_ma_text_topk1.py) | 同类对照的另一实现：FP32 缓存/选取 → FP16 教师概率桶 → FP32 KL；保留为独立分支 |
| [teacher_bank_ops.py](teacher_bank_ops.py) | 多层分支共用的选层、配对与特征/attention 融合工具，不是独立训练入口 |
| [train_distill_mi_ma_multilayer.py](train_distill_mi_ma_multilayer.py) | `legacy/bank` 两种模式，支持单层、平均、多层一对一匹配 |
| [train_distill_mi_ma_multilayer_sampling.py](train_distill_mi_ma_multilayer_sampling.py) | 在多层蒸馏上增加固定、均匀及逐 epoch 随机选层 |

# 关于训练与评估

| 文件 | 用途 |
| --- | --- |
| [train_baseline.py](train_baseline.py) | 固定长度 padding 的分类基线，同时定义共用的数据集、模型与指标工具 |
| [train_baseline_collator.py](train_baseline_collator.py) | （我的教师学生主要用的这个教你）带独立 collator 的分类基线；`--padding_mode fixed/dynamic`，默认 `dynamic` |
| [eval_localconcat.py](eval_localconcat.py) | （评估脚本）对兼容的 LoRA 分类 checkpoint 进行独立评估，基于 `train_baseline` 的模型结构 |
| [train_generative_sft.py](train_generative_sft.py) | （正在跑的SFT脚本）生成式 HTC LoRA SFT，复用 collator 基线的数据/指标约定，生成按层组织的标签 JSON |

# AGT Plus：层级多标签文本分类与离线知识蒸馏

本项目使用 Qwen、Llama 等 decoder 模型进行层级多标签文本分类（HTC）。分类分支在文本后追加层级 token，训练局部分类头与全局分类头；离线蒸馏分支提取教师的 logits、特征和 attention，供学生训练使用。另提供生成式 LoRA SFT，作为不同建模方式的对照。

本次代码来自 **2026-10-23** 的服务器 `/data1/lwq/agt_plus`：包含指定的 24 个文件，以及多层蒸馏必需的 `teacher_bank_ops.py`。保留原有相对路径和源码内容。模型、数据集、教师缓存、checkpoint 和训练日志需自行准备。

## 1. 文件与流程

```text
原始语料与标签树
  └─ 数据预处理 → train/dev/test JSONL + 标签元信息
       ├─ 分类教师训练 → checkpoint
       │    └─ 教师知识提取 → logits / features / attention
       │         └─ 学生蒸馏训练 → 验证集选模型 → 独立测试集评估
       └─ 生成式 LoRA SFT → 生成层级标签 JSON → F1 评估
```

### 数据处理

| 文件 | 用途 |
| --- | --- |
| [data/preprocess_trainer_multi_plus.py](data/preprocess_trainer_multi_plus.py) | 处理 WoS、RCV1、NYT，建立层级标签空间并写出训练格式及 `.meta.json` |
| [data/bgc/preprocess_bgc_trainer_multi.py](data/bgc/preprocess_bgc_trainer_multi.py) | 根据 BGC taxonomy 处理 train/dev/test，统一各层标签编号 |
| [data/bgc/data_bgc.py](data/bgc/data_bgc.py) | BGC 的独立旧式处理工具，生成标签/层级文件及 token、one-hot 文本文件；不直接生成 HTC trainer 格式 |

### 分类训练与评估

| 文件 | 用途 |
| --- | --- |
| [train_baseline.py](train_baseline.py) | 固定长度 padding 的分类基线，同时定义共用的数据集、模型与指标工具 |
| [train_baseline_collator.py](train_baseline_collator.py) | 带独立 collator 的分类基线；`--padding_mode fixed/dynamic`，默认 `dynamic` |
| [eval_localconcat.py](eval_localconcat.py) | 对兼容的 LoRA 分类 checkpoint 进行独立评估，基于 `train_baseline` 的模型结构 |
| [train_generative_sft.py](train_generative_sft.py) | 生成式 HTC LoRA SFT，复用 collator 基线的数据/指标约定，生成按层组织的标签 JSON |

### 教师知识提取、转换与检查

| 文件 | 用途 |
| --- | --- |
| [extract_teacher.py](extract_teacher.py) | 原始单层知识提取：logits、层级 token 特征、层级 token 到文本的 Top-K attention |
| [extract_teacher_multilayer.py](extract_teacher_multilayer.py) | 按 Transformer 层保存知识库；`--layers original` 使用原始单层流程 |
| [extract_teacher_full_attention.py](extract_teacher_full_attention.py) | 独立保存所有有效 token 对的 post-softmax、head-mean attention |
| [extract_teacher_text_features.py](extract_teacher_text_features.py) | 提取有效文本 token 的均值特征，供特征载体消融使用 |
| [check_full_attention_tokens.py](check_full_attention_tokens.py) | 在 CPU 上核查 full-attention 缓存保存的 token IDs，使用提取时的 tokenizer |
| [convert_full_attention_to_level_text.py](convert_full_attention_to_level_text.py) | 将 FP32 full-attention 缓存转换为紧凑的层级 token→文本缓存，不做 Top-K、归一化或量化 |
| [diagnose_attention_topk.py](diagnose_attention_topk.py) | CPU 上统计 Top-K attention 覆盖率与文本长度关系，输出 JSON/CSV 报告 |

### 学生训练分支

| 文件 | 知识来源与区别 |
| --- | --- |
| [train_distill_mi_ma.py](train_distill_mi_ma.py) | 原始单层离线蒸馏；支持监督分类、logit、feature、attention 损失，并保存 macro/micro 对应模型 |
| [train_distill_mi_ma_cached_topk.py](train_distill_mi_ma_cached_topk.py) | 从紧凑 FP32 缓存构造层级 token Top-K 目标；`--attention_topk` 支持正整数或 `all` |
| [train_distill_mi_ma_feature_ablation.py](train_distill_mi_ma_feature_ablation.py) | `level_token` 与 `text_mean` 特征载体消融；分类仍使用层级 token |
| [train_distill_mi_ma_fullatt.py](train_distill_mi_ma_fullatt.py) | 单个 Transformer 层的完整有效 token 对 attention 蒸馏 |
| [train_distill_mi_ma_fullatt_ltot.py](train_distill_mi_ma_fullatt_ltot.py) | `--full_attention_scope full/level_to_text`，默认 `full` |
| [train_distill_mi_ma_fullatt_scope.py](train_distill_mi_ma_fullatt_scope.py) | `--full_attention_scope full/text_to_text`，默认 `full` |
| [train_distill_mi_ma_multilayer.py](train_distill_mi_ma_multilayer.py) | `legacy/bank` 两种模式，支持单层、平均、多层一对一匹配 |
| [train_distill_mi_ma_multilayer_sampling.py](train_distill_mi_ma_multilayer_sampling.py) | 在多层蒸馏上增加固定、均匀及逐 epoch 随机选层 |
| [train_distill_mi_ma_text_topk.py](train_distill_mi_ma_text_topk.py) | 增加 `matched_text_topk` 对照：使用末尾文本查询，教师 Top-K 加 `other` 桶 |
| [train_distill_mi_ma_text_topk1.py](train_distill_mi_ma_text_topk1.py) | 同类对照的另一实现：FP32 缓存/选取 → FP16 教师概率桶 → FP32 KL；保留为独立分支 |
| [teacher_bank_ops.py](teacher_bank_ops.py) | 多层分支共用的选层、配对与特征/attention 融合工具，不是独立训练入口 |

“标签层级”与“Transformer decoder 层”是两个维度。多层知识库中的层号指 decoder block，使用从 0 开始的索引。

## 2. 环境与路径

使用 Linux 与现有可运行的训练环境。源码主要依赖 `torch`、`transformers`、`peft`、`numpy`、`scikit-learn`、`safetensors`、`tqdm`；Trainer 通常还需配套 `accelerate`。SwanLab 用于实验记录，训练入口可用 `--no_swanlab` 关闭。

当前快照未包含验证过的依赖版本锁定文件。主流程的 LoRA 配置使用 `trainable_token_indices`，需使用支持该参数的 PEFT 环境；Transformers 也必须支持实际使用的模型结构。教师提取需要 CUDA，多进程提取使用 NCCL。转换与 Top-K 诊断主要依赖 NumPy、safetensors，可在 CPU 上运行。

代码默认模型目录为 `/data1/lwq/model/`，数据目录为 `/data1/lwq/agt_plus/data/<dataset>`。跨机器运行时，通过 `--model_path`、`--data_dir` 和 `--output_dir` 显式指定路径。以下 Bash 示例从项目根目录执行，路径和超参数均需按实际实验调整。

## 3. 数据准备

### WoS、RCV1、NYT

`preprocess_trainer_multi_plus.py` 从 `data/<dataset>/` 读取原始划分、value dict 和标签树。先处理 `train`，再处理 `dev/test`；后两者复用训练集元信息中的标签编号。

```bash
PROJECT=/data1/lwq/agt_plus
python data/preprocess_trainer_multi_plus.py --dataset wos --split train --base-dir "$PROJECT"
python data/preprocess_trainer_multi_plus.py --dataset wos --split dev --base-dir "$PROJECT"
python data/preprocess_trainer_multi_plus.py --dataset wos --split test --base-dir "$PROJECT"
```

默认 `--label-space-source value_dict` 结合 value dict 与树建立标签空间；`train` 模式使用训练集中出现的标签。`--tree-path` 可指定树文件，`--num-levels` 可指定层数。**RCV1 dev 当前固定读取 `rcv1_dev_ctr_com_0603.json`**，其他常规输入为 `<dataset>_<split>.json`，准备数据时注意这一差异。

### BGC

需要 `bgc.taxonomy`、`train_data.jsonl`、`dev_data.jsonl`、`test_data.jsonl`：

```bash
python data/bgc/preprocess_bgc_trainer_multi.py \
  --data-dir /data1/lwq/agt_plus/data/bgc
```

脚本默认路径使用大写 `data/BGC`，Linux 区分大小写；示例显式指定小写路径。它根据 taxonomy 为所有划分建立同一标签空间，发现未知标签会报错。

`data_bgc.py` 是另一条处理路线：需从包含上述原始文件的目录运行，加载 `bert-base-uncased` 和 `roberta-base` tokenizer，写出 `bert_value_dict.pt`、`slot.pt`、`tok.txt`、`Y.txt`。它不是运行 `preprocess_bgc_trainer_multi.py` 的必需前置步骤。

### 训练格式

```text
data/wos/
├── wos_train_trainer_multi.json
├── wos_train_trainer_multi.meta.json
├── wos_dev_trainer_multi.json
├── wos_dev_trainer_multi.meta.json
├── wos_test_trainer_multi.json
└── wos_test_trainer_multi.meta.json
```

数据文件虽以 `.json` 结尾，实际按 JSONL 逐行读取，每行类似：

```json
{"query":"example text","label":[[0,2],[1]],"label_mask":[1,1]}
```

`label` 存放每层的局部标签编号；`label_mask` 表示该层是否参与损失/指标，不能直接由该层标签列表是否为空推断。`.meta.json` 包含 `num_levels`、`level_classes`、`layer_label2id` 等标签结构。训练、知识提取与评估必须共享一致的映射；提取缓存后改变文本、顺序、截断方式或标签映射，需要重新核对缓存。

## 4. 分类教师 → 单层蒸馏 → 测试

### 4.1 设置路径

```bash
DATA=/data1/lwq/agt_plus/data/wos
TEACHER_RUN=/data1/lwq/agt_plus/output_plus/wos_teacher_example
TEACHER_CKPT="$TEACHER_RUN/best_global_macro"
KNOWLEDGE="$TEACHER_CKPT/teacher_knowledge_train"
STUDENT_RUN=/data1/lwq/agt_plus/output_distill/wos_student_example
```

### 4.2 教师训练

```bash
python train_baseline_collator.py \
--dataset wos \
--model_key qwen8b \
--lr 1e-4 \
--early_stop_target global  \
--output_dir "$TEACHER_CKPT" \
--global_loss_weight 1  \
--local_loss_weight 0  \
--train_batch 1   \
--eval_batch 1  \
--grad_acc 32  \
--epochs 7
```

训练入口的 `--gpu` 默认值是 `1`，请显式选择分配给你的 GPU。需要动态 padding 时，可将入口改为 `train_baseline_collator.py`，增加 `--padding_mode dynamic`。教师 checkpoint 应保留 adapter、tokenizer、`classifier_head.pt` 与 `teacher_config.json`，以及其引用的基础模型。

### 4.3 提取教师知识

```bash
CUDA_VISIBLE_DEVICES=0 python extract_teacher.py \
  --dataset wos \
  --checkpoint_dir "$TEACHER_CKPT"\
  --knowledge_types logits,features,att
```

原始单层库保存 manifest 与分片，支持相同配置下续提；用于训练前应达到 `status: complete`。多 GPU 可使用 `torchrun --nproc_per_node=N`。`--overwrite_shards` 会重写分片，普通续提无需开启。

### 4.4 学生蒸馏

```bash
python train_distill_mi_ma.py \
  --dataset wos \
  --knowledge_dir "$KNOWLEDGE" \
  --model_key qwen0.6b \
  --output_dir "$STUDENT_RUN" \
  --global_supervised_bce_weight 1 \
  --local_supervised_bce_weight 0 \
  --global_logit_kd_weight 1 \
  --local_logit_kd_weight 0 \
  --feature_kd_weight 0.02 \
  --attention_kd_weight 0.25 \
  --attention_mass_kd_weight 0 \
  --early_stopping_target global_macro \
  --epochs 10 \
  --lr 2e-4
```

主要目标：监督 BCE、带温度的教师 logit 蒸馏、特征 MSE/cosine，以及教师 Top-K attention 与剩余概率桶的 KL。对应权重设为 0 可关闭该项；启用的知识类型必须存在。不同 attention 范围的损失不能仅凭相同权重视为强度相等。

### 4.5 独立测试

```bash
CUDA_VISIBLE_DEVICES=0 python eval_localconcat.py \
  --checkpoint_dir "$STUDENT_RUN/best_global_macro" \
  --dataset wos --data_dir "$DATA" --split test --eval_batch 16 \
  --output_json "$STUDENT_RUN/test_metrics.json"
```

该评估入口适用于其模型结构兼容的 LoRA 分类 checkpoint，不用于生成式 SFT，也不能假定支持所有 full tuning 产物。先用验证集选模型和超参数，再使用测试集报告结果。训练结束时的 `final_eval_metrics.json` 不应直接当作独立测试结果。

## 5. Attention 缓存与对照实验

### 5.1 提取完整 attention

```bash
FULL_BANK="$TEACHER_CKPT/teacher_full_attention_fp32"
CUDA_VISIBLE_DEVICES=0 python extract_teacher_full_attention.py \
  --dataset wos --checkpoint_dir "$TEACHER_CKPT" --data_dir "$DATA" \
  --output_dir "$FULL_BANK" --layers last:1 \
  --batch_size 1 --shard_size 4 --storage_dtype float32
```

此库保存所有有效 token 对的 head-mean attention，存储开销随序列长度平方增长。可在独立目录先使用 `--max_samples 8` 做小样本检查；正式全量任务使用另一目录，避免混用部分缓存。

```bash
python check_full_attention_tokens.py --knowledge_dir "$FULL_BANK"
python diagnose_attention_topk.py \
  --knowledge_dir "$FULL_BANK" --output_dir ./attention_diagnostics/wos \
  --max_samples 100 --topks 16,32,64,128
```

Token 检查默认使用 manifest 中教师 checkpoint 的 tokenizer，可通过 `--tokenizer_dir` 指定同一个提取 tokenizer。诊断只读取层级 token 查询行；去掉 `--max_samples` 才统计整个库，报告目录应放在缓存外。

### 5.2 紧凑层级 token→文本缓存

```bash
python convert_full_attention_to_level_text.py \
  --full_attention_dir /data1/lwq/agt_plus/output_plus/wos_qwen8b_local0_global1_collator-epoch7/best_global_macro/teacher_knowledge_full_attention_last1_float32 \
  --output_dir /data1/lwq/agt_plus/output_plus/wos_qwen8b_local0_global1_collator-epoch7/best_global_macro/teacher_knowledge_level_text_last1_float32 \
  --layer -1
python train_distill_mi_ma_cached_topk.py \
  --dataset wos \
  --knowledge_dir "$KNOWLEDGE" \
  --attention_cache_dir "$COMPACT_BANK" \
  --attention_topk 32 \
  --model_key qwen0.6b \
  --output_dir ./output_distill/cached_topk_example \
  --global_supervised_bce_weight 1 \
  --local_supervised_bce_weight 0 \
  --global_logit_kd_weight 1 \
  --local_logit_kd_weight 0 \
  --feature_kd_weight 0.1 \
  --attention_kd_weight 2 \
  --attention_mass_kd_weight 0 \
  --early_stopping_target global_macro \
  --epochs 10 \
  --lr 2e-4 \
  --train_batch 4 \
  --eval_batch 4 \
  --grad_acc 8
```

转换器要求 FP32 full-attention 源库，输出到新目录。紧凑缓存保留层级 token 查询到文本的原始概率；训练时再进行 Top-K 选择。`--attention_topk all` 使用全部文本键。原始 `--knowledge_dir` 仍负责样本对齐与其他知识类型。

### 5.3 完整/文本查询范围

以下 full-attention 分支在启用 attention KD 时都需要 `--knowledge_dir` 与 `--full_attention_dir`；设置 `--attention_kd_weight`，并保持 `--attention_mass_kd_weight 0`。范围需显式指定：

| 实验 | 入口与参数 |
| --- | --- |
| 所有有效 token 对 | `train_distill_mi_ma_fullatt.py` |
| 层级 token 查询 → 全部有效文本键 | `train_distill_mi_ma_fullatt_ltot.py --full_attention_scope level_to_text` |
| 全部有效文本查询 → 文本键 | `train_distill_mi_ma_fullatt_scope.py --full_attention_scope text_to_text` |
| 末尾文本查询与教师 Top-K + other | `train_distill_mi_ma_text_topk.py --full_attention_scope matched_text_topk --text_attention_topk 32` |
| 上述对照的 FP16 教师桶版本 | `train_distill_mi_ma_text_topk1.py --full_attention_scope matched_text_topk --text_attention_topk 32` |

`ltot`、`scope`、`text_topk*` 默认范围均为 `full`，仅换文件名不会自动开启对应对照。`text_topk1` 将查询数限制为有效层级数与文本长度的较小值，并使用 FP16 教师桶；它与 `text_topk` 的数值路径不同，不应混为同一实验。

## 6. 多层蒸馏与特征载体消融

### 多层知识库

```bash
MULTI_BANK="$TEACHER_CKPT/teacher_multilayer"
CUDA_VISIBLE_DEVICES=0 python extract_teacher_multilayer.py \
  --dataset wos --checkpoint_dir "$TEACHER_CKPT" --data_dir "$DATA" \
  --output_dir "$MULTI_BANK" --layers all --knowledge_types logits,features,attention
```

`train_distill_mi_ma_multilayer.py` 默认 `--knowledge_mode legacy`，转到多层库时需使用 `--knowledge_mode bank`。Feature 与 attention 各自有 `--feature_alignment` / `--attention_alignment`，取值为 `single`、`mean`、`one_to_one`，并分别设置 `--teacher_feature_layers` / `--student_feature_layers` 与 `--teacher_attention_layers` / `--student_attention_layers`。

`train_distill_mi_ma_multilayer_sampling.py` 提供 `--layer_selection fixed/uniform/random_epoch`。均匀或逐 epoch 随机选层需要 bank 模式及全层教师库；启用的知识项使用 `one_to_one`，教师/学生层参数设为 `all`。使用 `--layer_selection_seed` 记录选层随机性，`random_epoch` 还要求 `--num_workers 0`。`teacher_bank_ops.py` 必须与这些训练脚本一起保留。

### 文本均值特征

```bash
TEXT_FEATURES="$TEACHER_CKPT/teacher_text_features"
CUDA_VISIBLE_DEVICES=0 python extract_teacher_text_features.py \
  --dataset wos --checkpoint_dir "$TEACHER_CKPT" --data_dir "$DATA" \
  --output_dir "$TEXT_FEATURES" --batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_distill_mi_ma_feature_ablation.py \
  --dataset wos --model_key qwen0.6b --data_dir "$DATA" \
  --knowledge_dir "$KNOWLEDGE" --feature_representation text_mean \
  --generic_feature_dir "$TEXT_FEATURES" --feature_kd_weight 0.1 \
  --output_dir ./output_distill/text_mean_example --no_swanlab
```

该分支仅改变 feature KD 的载体，分类仍使用层级 token。`text_mean` 需要与原单层库匹配的教师、tokenizer、文本/截断设置、特征层及样本 UID；不能直接把多层 bank 目录当作这里的原始单层库。

## 7. 生成式 SFT

```bash
python train_generative_sft.py \
  --dataset wos \
  --model_key qwen0.6b \
  --output_dir /data1/lwq/agt_plus/output_sft/wos_Qwen3-0.6B-generative-seed42 \
  --max_length 2048 \
  --truncate_mode head \
  --max_new_tokens 512 \
  --epochs 10 \
  --lr 2e-4 \
  --train_batch 8 \
  --eval_batch 8 \
  --grad_acc 4 \
  --eval_steps 200 \
  --save_steps 200 \
  --early_stopping_patience 5 \
  --seed 42 \
  --do_test
```

生成式分支不使用分类头、附加层级 token 或教师知识库，使用答案 token 的交叉熵训练。输出格式是按层组织的标签 JSON 数组；基于验证集 Macro-F1 选模型，`--do_test` 在所选模型上执行测试。

本次服务器版本使用贪心解码，并在检测到完整层级 JSON 数组后停止；评估使用相同 JSON 边界。输出中记录 `complete_json_rate` 等诊断指标；完整 JSON 不等于标签一定正确，仍需标签解析与 F1 评估。`max_new_tokens` 是生成上限，不是强制生成长度。该模型的评估流程由 `train_generative_sft.py` 管理，不使用 `eval_localconcat.py`。

## 8. 复现与常见问题

- **路径不存在**：本仓库保存代码，运行前准备真实数据、基础模型和教师 checkpoint，检查 Linux 大小写与 `--data_dir`。
- **缓存不匹配**：原始单层、多层、full-attention、紧凑层级→文本和文本特征缓存有不同格式。检查 manifest、样本 UID、tokenizer、标签映射、截断方式与所选层。
- **显存/磁盘不足**：降低 batch 或提取 shard 大小；完整 attention 的开销尤其大。不要通过跳过完整性检查继续训练。
- **续训**：原始学生入口在完整原命令上增加 `--resume_from_checkpoint latest` 或指定 `checkpoint-*`，沿用原输出目录与知识库。恢复优化器等状态使用训练 checkpoint，不以 `best_global_macro` 替代。
- **结果比较**：固定数据/标签映射、教师、缓存版本、层选择、padding、截断长度、batch/梯度累积、种子及评估阈值；不同 attention 范围和精度路径单独记录。

本次发布核对了文件传输哈希、Python 语法、内部依赖及 README 中的入口/参数。未执行 GPU 训练或端到端评估，示例不代表已验证的性能结果。部分源码注释引用的独立实验文档不在本次文件集合中，使用时以实际参数和缓存校验为准。
