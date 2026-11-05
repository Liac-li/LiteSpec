# Qwen3.5 DLite：图文训练与视频评测

记录日期：2026-11-05。原始源码基准为 `191fbce`。随后在
`dev/qwen35-v-layer` 上加入了三阶段共享 context KV，本文已同步当前配置。
第 5 节仍记录历史 checkpoint；它们不会因修改训练 YAML 而变成新架构。

## 1. 实验范围与证据

当前链路是：**用 Qwen3.5-4B 生成的图文数据训练 DLite，再把图文训练得到的
draft 用于视频推测解码评测。** 当前 video 入口只做推理评测，没有视频数据训练。

```text
FineVision 单图与首轮问题
        ↓ Qwen3.5-4B / SGLang 重新生成回答
图文训练 JSONL
        ↓ 冻结 HF target，在线提取 hidden states 和监督分布
DLite direct SFT
        ↓ 保存 draft checkpoint
VideoDetailCaption / Video-MME 视频描述评测
        ↓
与同一 GPU 上的 target-only AR、原始 DLite、图文 DFlash 比较
```

本文区分以下证据：

- **源码与配置已确认**：模型构造、训练目标、数据处理逻辑、默认 checkpoint
  路径、评测协议、环境锁定版本。
- **历史实验组由代码引用**：服务器 run 路径及其名称所记录的 epoch、micro
  batch、block；最终配置仍须核对该 run 的 `effective_config.json`。
- **本地未验证**：服务器 checkpoint 是否完整、实际有效数据条数、完整训练是否
  收敛，以及视频评测的实测 TPS、接受长度、加速比和显存。

本次本地检查未找到这些服务器 checkpoint、训练日志或视频结果文件。README 中
的 Qwen3-8B 训练曲线也不属于本次 Qwen3.5-4B 视频实验。

## 2. 模型结构

### 2.1 冻结的 target

Target 使用完整的 `Qwen3_5ForConditionalGeneration`，视觉编码器和语言模型均
冻结。默认服务器模型目录为：

```text
$HOME/LMM/lmm_model/Qwen3.5-4B
```

按 [Qwen3.5-4B 官方配置](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)：

| 项目 | 设置 |
| --- | --- |
| 语言模型层数 | 32 |
| 语言层类型 | 每三层 linear attention 加一层 full attention，共 24 / 8 层 |
| Hidden size / MLP intermediate size | 2560 / 9216 |
| Full-attention Q head / KV head / head dim | 16 / 4 / 256 |
| 视觉编码器层数 / hidden size | 24 / 1024 |
| 视觉输出宽度 | 2560 |
| Spatial patch / merge / temporal patch | 16 / 2 / 2 |

实际训练和评测从指定本地模型目录加载配置；服务器模型文件是最终依据。

### 2.2 自行训练的 DLite student

```text
图像 / 视频 + 问题 + 已生成文本
                ↓
冻结的 Qwen3.5-4B target
                ↓
同一个 pivot 位置的 12 层 hidden states（CHS）
                ↓
5 层 DLite Transformer，结合 anchor token 和 MASK
                ↓
rank 320 的 RNN sequential head
                ↓
7 个候选 token
                ↓
target 验证并提交接受的前缀
```

| 项目 | 当前图文训练配置 |
| --- | --- |
| 架构 / 版本 / 角色 | `DLiteDraftModel` / `dlite_v2` / `student` |
| Draft backbone | 5 层 Qwen3 风格 Transformer，全部 full attention |
| 宽度、MLP、attention heads | 从 target 文本配置继承 |
| CHS | 同一个 `anchor-1` 位置，选取 target 的 12 层 hidden |
| 当前 target 层号，零起始 | `[1,3,7,11,15,18,19,22,23,26,27,31]` |
| CHS 处理 | 加入 depth 与 stage embedding，经各 stage 独立 RMSNorm |
| Context KV | 三个 stage 独立投影；每次 forward 计算一次，供五层 draft 共享 |
| Sequential head | 低秩 RNN，rank 320 |
| Block size | 8，包含 1 个已知 anchor 和 7 个候选 token |
| Draft 位置编码 | 局部位置；CHS 位于 0，anchor 从 1 开始 |
| Draft 历史 | student 每轮使用当前 CHS，不维护历史 draft KV |
| 初始化 | 新建 draft 随机初始化；anchor/MASK 输入使用冻结 target embedding |

**5 层 draft、12 层 target hidden、1 个时间位置是三个独立维度。** 视频帧先由
target 处理，DLite 读取包含上下文信息的 CHS；DLite 本身没有独立的视频编码器。
“不维护历史 draft KV”也不意味着 target 不维护自己的 hybrid cache。

原始 DLite checkpoint 按自身 `config.json` 加载，不能直接套用上述自行训练
student 的结构参数。

源码：[模型构造](../litespec/vlm/train.py)、
[DLite backbone 与 RNN](../litespec/modeling/draft/dlite.py)、
[冻结 target 的加载](../litespec/vlm/runtime.py)。

### 2.3 三阶段共享 context KV

用户确认的三组为：

| Stage | Target 深度区间 | 选取层 | CHS 数量 |
| --- | --- | --- | ---: |
| 0 | 0–6 | `[1,3]` | 2 |
| 1 | 7–17 | `[7,11,15]` | 3 |
| 2 | 18–31 | `[18,19,22,23,26,27,31]` | 7 |

当前图文 YAML 显式设置：

```yaml
target_layer_ids: [1, 3, 7, 11, 15, 18, 19, 22, 23, 26, 27, 31]
shared_stage_kv: true
chs_stage_ids: [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
```

对来自 target 层 `l`、属于 stage `s` 的 CHS：

```text
z_l = RMSNorm_s(h_l + depth_embedding[l] + stage_embedding[s])
K_s = context_KNorm(WK_s(z_s))，再应用 CHS 位置的 RoPE
V_s = WV_s(z_s)
Kctx / Vctx = 三组沿 CHS 位置维拼接
```

各 stage 都投影出完整的四个 KV heads。最终单个 block 的共享 context K/V
形状为 `[batch, 4, 12, 256]`。多 anchor 训练保留 anchor-major 的位置顺序，
沿用原有 block 隔离 mask。

每层 draft 的 Q 与 draft-token K/V 仍由该层独立计算，再与共享 context KV
拼接做一次联合 attention。所有 heads 都能读取全部 12 个 CHS；没有按 stage
屏蔽 head。CHS 只作为 K/V 来源，单个 block 的 query 长度为 8，KV 长度为 20。
共享 KV 保留梯度，只复用于本次 forward；下一轮 pivot 改变后重新生成。

`Q:KV=4:1` 表示每四个 Q heads 共用一个 K head 和一个 V head：

| Q heads | 共用的 KV head |
| --- | --- |
| 0–3 | 0 |
| 4–7 | 1 |
| 8–11 | 2 |
| 12–15 | 3 |

每个 Q head 仍独立计算注意力分数和权重。这个比例不表示 token 数量，也不
表示三个 stage 如何分配；stage 是特征来源的深度分组，GQA 是 head 的共享方式。

4B 下 stage embedding 增加 7680 个参数；三组额外 context K/V 投影增加
15,728,640 个参数。原有各层 K/V 投影仍用于 draft tokens。该结构需要训练，
不会把旧 checkpoint 的五套投影自动转换为共享权重。

缺少 `shared_stage_kv` 的旧配置继续使用原有计算和权重布局。若要训练旧结构
对照，应同时关闭此开关并恢复旧 taps
`[0,1,4,7,11,14,17,20,24,27,30,31]`。

实现后的本地检查使用小尺寸随机模型和 CPU：六个 stage K/V 投影各调用一次，
五层收到同一份 context KV；新增参数及 depth embedding 均获得有限非零梯度；
三 anchor 打包与逐块计算最大误差约 `4.8e-7`，保存重载误差为 0，旧模式与
修改前 attention 输出误差为 0。未新增测试文件。
本地 Torch 2.14.0 / Transformers 5.17.0 与锁定的训练环境不同，也没有 CUDA，
且完整 trainer 导入缺少 `yunchang`；服务器训练、实际 4B 权重和性能尚未验证。
开发契约和检查记录见 [intent](../dev-doc/intent.md)、[spec](../dev-doc/spec.md)、
[plan](../dev-doc/plan.md)。

## 3. 正式训练配置

### 3.1 配置继承与执行路径

```text
scripts/dlite/run_training_vlm.sh
  → torchrun -m litespec.vlm.train_aligned
  → scripts/config/vlm/sft_student_aligned.yaml
  → 继承 scripts/config/dlite_sft.yaml 的 arguments
  → 叠加图文配置和命令行参数
```

当前图文路径执行 direct SFT，每个 rank 都运行完整、冻结的 HF target，在线
提取 hidden states 和监督分布。CUDA 使用 FSDP1 `SHARD_GRAD_OP`、
`use_orig_params=True` 训练 draft；target 不是八卡 tensor parallel。

基础 YAML 中继承来的 SGLang、离线缓存和 disaggregate 字段，不能直接理解为
当前 HF 图文适配器的实际执行方式。仓库另有 teacher/two-stage 入口，本路径
使用的是 direct-SFT student。

### 3.2 基础配方

| 参数 | Aligned 基础设置 |
| --- | --- |
| Epoch | 6，无默认训练步数上限 |
| 每卡 micro batch / 梯度累积 | 1 / 4 |
| 8 卡 global batch | 32 |
| 最大序列长度 | 4096，包含视觉 token、问题和回答 |
| 每条样本最多 anchor 数 | 512 |
| 精度 | BF16，优化器使用 FP32 master 参数 |
| 优化器 | Muon + AdamW |
| Muon | 峰值 LR `1e-3`，momentum `0.95`，weight decay `0` |
| AdamW | 峰值 LR `1e-3`，weight decay `0` |
| 两组 LR 曲线 | 从 `1e-4` 起步，前 `5%` optimizer steps 线性升到 `1e-3`，之后 cosine 降到 `3e-4` |
| Gradient clipping | 全局 norm `1.0` |
| Final CE / 概率 L1 / base LM CE 权重 | `1.0 / 1.0 / 0.0` |
| 位置衰减 | `gamma=4` |
| Seed | 模型 seed 42；DistributedSampler seed 0 |
| 图文配置日志 / 保存间隔 | 每个 optimizer step 记录；每 0.5 epoch 保存 |
| 日志后端 | 图文 YAML 默认 SwanLab；项目 `qwen35-dlite-vlm` |

本次训练六轮全部使用纯 SFT 的 `1.0 CE + 1.0 L1`，从随机初始化的 student
开始，不包含 DFlash teacher 或 KL warm-up。全局有效 batch 32 对应八卡、
每卡 micro batch 1、梯度累积 4。

`sft_student.yaml` 中的两层、rank 64、100 steps 属于旧 smoke 配方，正式入口
使用的是 `sft_student_aligned.yaml`。

### 3.3 监督和数据资格

损失只覆盖 assistant 回答位置：

```text
L = weighted_mean[1.0 × CE(argmax p_target, q_draft)
                  + ||p_target - q_draft||_1]
```

- CE 标签取当前 HF target 的 argmax。
- RNN teacher forcing 的前序 token 来自保存的回答序列。
- L1 比较完整词表概率分布。
- 同一 block 内的候选按位置指数衰减，`gamma=4`，越靠后的权重越小。
- 训练前过滤超过 4096 token 的记录，以及不足 `2 × block_size` 个回答标签、
  没有完整 block anchor 的记录；不会截断视觉块来凑长度。
- 实际有效样本数记录在 `effective_config.json` 的 `eligible_samples`；
  排除记录写入 `skipped_training.jsonl`。

输入资格检查使用现有训练 ranks 并行处理：八卡时，每个 rank 在 CPU 上检查
约 1/8 记录，最后按原始行号汇总，保持与串行检查相同的有效样本集合和顺序。
该阶段涉及图片解码和 processor 编码，GPU 尚未开始训练。无需增加 worker
或修改环境。已有完整校验结果仍可用 `REUSE_VALIDATION_FROM` 复用；正在运行
的旧进程不会热加载并行实现，需要同步代码后重新启动。

源码与说明：[基础 YAML](../scripts/config/dlite_sft.yaml)、
[图文覆盖配置](../scripts/config/vlm/sft_student_aligned.yaml)、
[训练适配器](../litespec/vlm/train_aligned.py)、
[损失实现](../litespec/core/dlite.py)、
[训练说明](../docs/VLM_ALIGNED_TRAINING.md)。

## 4. 训练数据来源

### 4.1 FineVision 与回答重新生成

默认原始数据路径：

```text
$HOME/LMM/liyt/datasets/FineVision/FineVision_subset/finevision_sharegpt.jsonl
```

处理规则：

1. 保留单图记录；问题取 `conversations[0].content`，仅使用第一轮 user turn，
   移除单个 `<image>` 标记。
2. 提取并去重图像与问题，用 Qwen3.5-4B 经 SGLang 重新生成回答。
3. 使用 `temperature=0`、关闭 thinking、输出上限 256 token；默认根据剩余
   context 空间自适应降低回答预算，最终序列上限为 4096。
4. 新的限额图像数据最多保留 1024 个合并后的视觉 token。对 patch 16、merge 2
   的配置，面积上限约为 `1024×1024`，保持长宽比并对齐 patch/merge。
5. 带图像限额的新数据按源图像内容哈希划分约 95% train、5% val；旧的未重写
   图像路径使用路径哈希划分，复用旧数据时须核对生成元数据。
6. 保存 `id / image / question / answer / answer_target` 等字段。

训练接口的最小记录形式：

```json
{"id":"example-1","image":"/absolute/path/image.png","question":"Describe the image.","answer":"A target-generated response."}
```

当前大数据组合训练脚本默认读取：

```text
$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl
```

基础单独训练入口仍默认读取较早的数据：

```text
$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl
```

**10w/100k 是候选规模，不能直接当作训练有效条数。** 生成失败、验证集划分和
训练资格过滤都会减少训练条数。采集阶段默认允许保留短回答，block 8 训练仍
要求至少 16 个回答标签。原始 FineVision 子集的具体组成比例没有本地统计。

### 4.2 生成入口与元数据

- [统一生成入口](../scripts/data/generate_finevision_8gpu.sh)：默认启动八个
  独立的 TP=1 SGLang replica，每个 replica 默认并发 8。
- [FineVision 字段映射与生成客户端](../litespec/vlm/serving_data.py)。
- [图像预算实现](../litespec/vlm/image_budget.py)。
- [生成链路说明](../docs/VLM_PIPELINE_AND_BASELINES.md)。

核对某份历史数据时，应查看其 `generation_config.json`、`last_summary.json`、
manifest 元数据以及 train/val JSONL，确认实际模型、预算、成功条数和排除原因。

## 5. 代码记录的实验组与 checkpoint

### 5.1 三组配置不要混用

| 实验入口或记录 | Epoch | Micro batch | 累积 | 8 卡 global batch | Block |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned 基础配方 | 6 | 1 | 4 | 32 | 8 |
| `train_dlite_dflash_5ep_mb4.sh` 默认 | 5 | 4 | 4 | 128 | 8 |
| 视频评测引用的历史实验组名称 | 3 | 2 | 需查有效配置 | 需查有效配置 | 8 |

历史组名称中的 `3ep_mb2_b8` 可确认命名记录，不能代替该 run 的
`effective_config.json`。按当前脚本显式指定三轮、micro batch 2、累积 4、8 卡
重新启动时，global batch 为 64。

### 5.2 视频入口默认权重

Target 为 `$HOME/LMM/lmm_model/Qwen3.5-4B`，比较以下方法：

| 标签 | 默认权重位置 |
| --- | --- |
| `ar` | 仅运行 target |
| `dlite_original` | `$HOME/LMM/lmm_model/Qwen3.5-4B-DLite` |
| `dlite_vlm_100k` | `$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dlite/last` |
| `dflash_vlm_100k` | 同一实验组的 `dflash/last` |

权重路径可以通过 `ORIGINAL_DLITE`、`DLITE_VLM`、`DFLASH_VLM` 覆盖。
直接运行五轮组合训练脚本不会自动复现这里引用的三轮权重。

DFlash 对照使用完整历史 draft KV、5 个 target 层、token CE、AdamW 和
`6e-4` 学习率。共同数据、block、batch 之外，各方法保留自身配方。

Checkpoint 的实际结构以其 `config.json` 为准，历史训练设置以 run 中的
`effective_config.json`、生成的 YAML 和日志为准。默认路径见
[视频评测入口](../tests/benchmark_video.py) 与
[通用矩阵 checkpoint 选择](../tests/run_vlm_matrix.py)。

### 5.3 保存与恢复

Aligned trainer 默认导出推理权重；`last` 是指向 `step_XXXXXXX` 的符号链接。
只有显式启用 `SAVE_TRAINING_STATE=1`，才同时保存各 rank 的优化器、FP32
master、scheduler、RNG 和训练 cursor 状态，支持完整恢复。

`train_dlite_dflash_5ep_mb4.sh` 固定设置 `SAVE_TRAINING_STATE=0`。需要完整恢复
时应使用直接训练入口并显式启用状态保存。权重导出本身不能用于完整 resume。

## 6. 视频评测协议

| 项目 | 默认设置 |
| --- | --- |
| 数据集 | VideoDetailCaption、Video-MME |
| 视频数据根目录 | `$HOME/LMM/liyt/datasets/eval_bench` |
| 样本 | 每个数据集 16 个不同视频，seed 42 |
| 帧数 | 均匀抽取 16 帧 |
| 每帧面积上限 | `448×448=200704` 像素，保持长宽比并按 patch/merge 对齐 |
| 解码 | Greedy、关闭 thinking、最多生成 256 token |
| Batch | B=1、16 |
| Attention | SDPA |
| GDN 恢复 | `per_token` |
| 执行方式 | 同一 GPU 依次运行 AR 和各 draft |
| Warmup | 每个模型/B 至少一次；GDN 内核在测量前预热 |

### 6.1 数据与输入

- VideoDetailCaption 使用原始 `question`，通过 `video_name` 定位 MP4/MKV。
- Video-MME 按唯一 `videoID` 取视频，按时长桶采样，使用固定的详细描述
  prompt；这里不报告官方 Video-MME 选择题准确率。
- 原始参考答案不进入 prompt。
- PyAV 解码，保留源 FPS、帧索引和时间戳，继续使用原生视频归一化、
  patchification 和 timestamp insertion。
- 输入尺寸先按每帧上限处理，再关闭 processor 的额外 resize；这里是显式
  每帧预算，不代表完全复现 qwen-vl-utils 的 smart-resize 启发式。

### 6.2 Batch 与状态恢复

**B=16 是同一个视频问题复制 16 份的静态 batch。** 每个数据集 16 个独立视频
对应 256 条复制序列，不是在线 continuous batching 的 16 个独立请求。

`per_token` 验证使用自定义 fused recurrent GDN kernel，保存每个候选位置的
FP32 recurrent state。接受部分前缀后选取相应状态，并裁剪 convolution history
和 full-attention KV。当前视频入口固定使用这一模式。

### 6.3 指标口径

```text
decode TPS = sum(output_tokens) / sum(decode_seconds)
model end-to-end TPS = sum(output_tokens) / sum(prefill_seconds + decode_seconds)
raw ratio = draft TPS / 同一 B、同一数据集的 AR TPS
accepted draft tokens = committed length - 1 个已知 anchor
```

模型计时排除视频解码、预处理、输入复制和 CPU→GPU 传输，预处理耗时单独保存。
因此这里的 model end-to-end latency 不是完整请求服务时延。

脚本记录 token match，但即使 draft 输出与 AR 不一致，也保留 raw ratio。
解读性能时必须同时看 token match；接受长度也不能代替实际 TPS 或加速比。

输出文件包括：

- `settings.json`、`samples.json`：模型指纹、环境、样本和设置。
- `*_inputs.json`：尺寸、帧索引、时间戳、grid、视觉 token 数和预处理时间。
- `cells/*.json`：各模型/B 的汇总与原始记录。
- `requests/*.jsonl`：逐视频 token、文本和 token match。
- `summary.json`、`comparison.csv`、`comparison.md`：汇总报告。

说明与实现：[视频协议](../tests/VIDEO_BENCHMARK.md)、
[视频选择](../tests/video_benchmark_data.py)、
[视频预处理](../tests/video_runtime.py)、
[GDN 状态事务](../litespec/vlm/gdn_state.py)。

## 7. 环境配置

### 7.1 HF 训练与评测环境

环境由用户自行创建和激活，启动脚本使用当前 PATH 中的 `python`。核心依赖参考
[环境定义](../environments/vlm/pyproject.toml) 和
[锁文件](../environments/vlm/uv.lock) 固定：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| PyTorch / torchvision | 2.9.1 / 0.24.1 |
| Transformers | 5.20.1 |
| yunchang | 0.6.4 |
| accelerated extra | causal-conv1d 1.7.0、flash-linear-attention 0.5.2 |

服务器需要 Linux、支持 BF16 的 NVIDIA GPU，以及与所选 CUDA build 兼容的
驱动。扩展编译还需要适配的 CUDA toolkit。Aligned CUDA Muon trainer 显式要求
`torch==2.9.1`。

当前服务器仓库目录是 `/home/jovyan/LMM/liyt/Projects/DLLM/LiteSpec`。
激活自己的训练环境后，可以检查当前解释器：

```bash
python -c 'import sys; print(sys.executable)'
```

按服务器驱动和编译工具链选择 `cu128` 或 `cu130`，二选一。视频 `per_token`
路径在 CUDA 上需要 causal-conv1d 和 Triton。`av/pandas/pyarrow/tensorboard`
目前未纳入这份 VLM 锁文件，若需固定全部环境，应另外保留实际安装版本。

训练脚本不安装依赖、不激活或切换虚拟环境，也不假定环境位于 `.venv-vlm`。

### 7.2 SGLang 回答生成环境

回答生成使用另一套支持 Qwen3.5 多模态的 SGLang 环境，通过 `SGLANG_PYTHON`
指定。统一生成管线默认指向：

```text
$HOME/.conda/envs/dflash/bin/python
```

仓库没有锁定这套服务器环境的实际版本，因此完整还原历史数据生成环境仍需
服务器记录。默认服务参数为 TP=1、BF16、context 4096、静态显存比例 0.80。
相关入口：[SGLang 启动脚本](../scripts/data/serve_qwen35_vlm.sh)。

## 8. 复现命令

以下命令使用用户提供的服务器 checkout。它们是启动说明，本次整理没有实际
启动生成、训练或 GPU 评测。

### 8.1 六轮纯 SFT、全局 batch 32、SwanLab

用户自行激活训练环境后执行：

```bash
cd /home/jovyan/LMM/liyt/Projects/DLLM/LiteSpec
bash train.sh
```

[`train.sh`](../train.sh) 根据自身路径定位仓库，不依赖 `DLITE_WORKSPACE`。
当前采用用户指定的 100k 候选数据路径：

```text
/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl
```

输出位于 `/home/jovyan/LMM/liyt/Projects/DLLM/LiteSpec/runs/$RUN_NAME`。
这条命令只训练 DLite，使用当前 YAML 及上游 direct-SFT 配方：六轮、micro
batch 1、累积 4、block 8。总优化步数为
`ceil(6 * floor(ceil(N_eligible / 8) / 1) / 4)`，实际值保存在有效配置中。

SwanLab 复用现有 tracker，只由 rank 0 初始化和上传，横轴为 optimizer step。
记录 `objective/loss`、`objective/final_ce`、`objective/l1_distance`、
`hard_label/token_accuracy`、`hard_label/block_accepted_length`、
`expected_accept_len`、`learning_rate/adamw`、`learning_rate/muon`、`grad_norm`、
`epoch_progress` 以及 `progress/*` 进度/ETA。训练 acceptance 指标来自 teacher
forcing blocks，不能当作视频推理的实测 acceptance 或加速比。运行配置上传时
不包含认证 key。SwanLab 的在线日志与本地 `swanlog/` 使用 SDK 的
[标准 init/log 接口](https://docs.swanlab.cn/en/api/py-init.html)。

当前 CE/L1 的 **系数**为 1:1，不代表原始数值等大。随机初始化时 CE 约 12.5、
L1 约 2，新总 loss 可能从旧配方的约 3.25 变为约 14.5，不能直接比较总值。
对于当前服务器的 10,634 个 optimizer steps，5% warmup 对应 531 步；两组
优化器均采用 `1e-4 → 1e-3 → 3e-4`。`learning_rate`、`muon_adamw_lr`、
`muon_lr` 配置的是峰值；`warmup_start_lr=1e-4` 配置起点，
`min_learning_rate=3e-4` 配置 cosine 终点。现有日志在每次更新后记录 scheduler
为下一步设置的 LR，因此第一条日志会略高于起始 LR。

### 8.1.1 分阶段耗时日志

`timing_steps: 10` 在前十个 optimizer window 开启同步计时。各项为这一整个
累积窗口的秒数（默认四个 microbatch），JSON 的 `timing.seconds_rank_max` 和
`timing.seconds_rank_mean` 汇总所有 ranks：

| 字段 | 范围 |
| --- | --- |
| `data` | 图片/文本准备、拼 batch 和传输到设备 |
| `target_prepare` | anchor 准备、target、logit gather 和训练 batch 构造 |
| `target_forward` | 上一项内的 target forward 子项 |
| `draft_forward_loss` | FSDP draft forward 与 loss 全过程 |
| `draft_backbone` | 上一项内的 backbone 子项 |
| `draft_head_loss` | 上一项内的 RNN、词表投影、CE/L1 与训练指标子项 |
| `backward` | 梯度回传，包含其通信等待 |
| `optimizer` | 梯度归一化、更新及 FP32 master 回写 |
| `optimizer_adamw` / `optimizer_muon` | optimizer 内的两个更新子项 |
| `window` | 整个 optimizer window，排除本次 tracking/checkpoint 回调 |

父项和子项有包含关系，不能相加。不同阶段的最大值也可能来自不同 rank。
`timing.ranks` 保留各 rank 的原始秒数、调用次数、输入长度、有效 anchors、监督
位置数、词表投影元素数和 CUDA 峰值分配；`slowest_rank` 指本窗口最慢 rank。
显存峰值是训练启动后的累计峰值，不是独立重置的每窗口峰值。
SwanLab 对应字段为 `timing/<stage>_max_seconds` 和
`timing/<stage>_mean_seconds`。现有 JSON 另有 `tracking_seconds`、`log_seconds`，
checkpoint 保存日志另有 `save_seconds`。
`tracking_seconds` 记录 tracker 调用返回的时间，不包含异步后台上传的全部耗时。

首用编译耗时计入相应阶段。诊断窗口会在 CUDA 边界同步，影响原来的重叠，
所以不能作为无插桩吞吐测量。第十步之后自动停止分阶段同步计时；设置
`timing_steps: 0` 可关闭。当前只通过了 CPU 小模型检查，尚不能确定服务器
瓶颈，需根据新的耗时日志定位。

变更 loss/LR 后使用新 run。可复用已完成的输入校验来避免重新检查图片：

```bash
REUSE_VALIDATION_FROM=/home/jovyan/LMM/liyt/Projects/DLLM/LiteSpec/runs/dlite-stage-sft-6ep-gb32-b8-data100k20261105_064721 \
bash train.sh
```

这里仅复用数据资格检查，不加载旧模型权重或优化器状态。

### 8.2 保存结果与节奏

```text
$OUTPUT_DIR/
├── train.log
├── effective_config.json
├── skipped_training.jsonl
├── swanlog/
├── step_XXXXXXX/
│   ├── config.json
│   ├── model.safetensors          # 大模型可能按 Hugging Face 规则分片
│   ├── vlm_run.json
│   ├── checkpoint.json
│   └── training_state_rank_*.pt
└── last -> step_XXXXXXX
```

`save_epoch_interval=0.5` 请求在 0.5、1.0、1.5、…、6.0 epoch 保存。触发点是
到达边界后的第一个完整 optimizer window，累积 4 时最多顺延 3 个 microbatch。
训练完成时也会请求保存；同一个 optimizer step 只导出一次。正常完整六轮约
得到 12 个 checkpoint，实际数量取决于有效 batch 数和优化步对齐。

文件夹编号是 optimizer step，不是 epoch。`checkpoint.json` 记录
`save_reason`（例如 `epoch_0.5`）、实际 `epoch_progress`、world size 和 cursor。
`save_interval=0` 关闭另一个固定 microstep 保存触发器。

当前 `train.sh` 保留用户设置的 `SAVE_TRAINING_STATE=1`。每个 checkpoint 除
draft 推理权重与配置之外，还保存八个
`training_state_rank_00000.pt` … `training_state_rank_00007.pt`，包含优化器、
FP32 master、scheduler、RNG 和数据 cursor。恢复时使用相同配置、数据和 world
size，并给内部训练命令追加 `--resume "$OUTPUT_DIR/last"`。如果只需要推理
权重，可以在启动前将脚本中的 `SAVE_TRAINING_STATE` 改为 0。

### 8.3 用默认已有权重做视频评测

```bash
export PYTHON_BIN="$PWD/.venv-vlm/bin/python"
export RESULT_DIR="$DLITE_WORKSPACE/runs/video_matrix_$(date +%Y%m%d_%H%M%S)"

EVAL_GPU=7 EVAL_SAMPLES=16 BATCH_SIZES="1 16" \
VIDEO_FRAMES=16 VIDEO_MAX_PIXELS=200704 EVAL_MAX_NEW_TOKENS=256 \
bash tests/run_video_matrix.sh

cat "$RESULT_DIR/comparison.md"
```

默认使用第 5 节列出的已有权重。使用新训练模型时，先显式设置：

```bash
export DLITE_VLM="/absolute/path/to/new/run/dlite/last"
export DFLASH_VLM="/absolute/path/to/new/run/dflash/last"
```

可选覆盖项包括 `TARGET_MODEL`、`ORIGINAL_DLITE`、`VIDEO_DATA_ROOT`、`TESTS`、
`MODELS`、`SAMPLE_SEED`、`WARMUP`。同一 GPU 顺序完成全部选定模型。

恢复未完成的评测时，设置 `RESUME=1`，并保留相同 `RESULT_DIR`、样本、模型、
参数和代码；完成的 cell 会跳过。结果文件中的实际 GPU、版本、checkpoint
指纹和输入预算应与报告一起保留。
