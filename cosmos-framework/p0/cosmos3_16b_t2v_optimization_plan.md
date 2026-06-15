# Cosmos3 16B 单卡 720p 5 秒 T2V 推理加速方案

## 1. 任务目标

在单张 GPU 上优化 Cosmos3 16B 的 Text-to-Video 推理，生成
`1280x720`、约 5 秒的视频。优化以免训练方法为主，可使用但不限于：

- Diffusion / Rectified Flow 采样加速
- Attention、GEMM、VAE 等 kernel 优化
- `torch.compile` 和 CUDA Graph
- 模型量化
- 特征缓存和计算复用

最终优化版本必须满足：

- 不出现相对提示词的主体、动作、数量、关系或镜头语义偏移
- 不出现明显的模糊、噪点、闪烁、结构崩坏或运动退化
- 使用完全相同的初始 latent noise 对比 baseline 和优化版本
- 以实际视频人工定性判断为最终质量标准
- 报告相对 baseline 的端到端加速和各环节时间 breakdown

## 2. 本地环境与代码路径

当前环境：

- GPU：NVIDIA H20，约 96 GB 显存
- PyTorch：`2.8.0+cu128`
- CUDA：12.8
- Triton：3.4
- 模型：`/root/autodl-tmp/Cosmos3-Nano`
- 官方文档仓库：`/root/autodl-tmp/cosmos`
- NVIDIA 推理框架：`/root/autodl-tmp/cosmos-framework`

`cosmos` 仓库以文档和示例 notebook 为主。实际高性能推理优化建议基于
NVIDIA `cosmos-framework`，因为它已经提供：

- FlashAttention-3 等 attention backend
- 按 Transformer block 执行 `torch.compile`
- CUDA Graph 支持
- Benchmark 和 profiler
- `guidance_interval`
- Cosmos3 原生 MoT、并行和序列打包实现

当前磁盘剩余空间约 18 GB。官方框架环境和依赖缓存可能需要超过 20 GB，
正式安装前应扩容或把 `uv`、PyTorch 等缓存移到其他磁盘。

## 3. 官方基准与本题基准的区别

官方基准见：

```text
/root/autodl-tmp/cosmos/inference_benchmarks.md
```

官方 H20 单卡 720p T2V 数据约为：

| Backend | 官方延迟 |
|---|---:|
| PyTorch | 931.39 s |
| vLLM-Omni | 929.81 s |
| Diffusers | 926.00 s |

该结果使用 189 帧、24 FPS、35 steps、BF16、batch size 1，视频时长约
7.9 秒，不能直接作为本题 5 秒 baseline。最终报告必须在本题统一配置下
重新测量。

## 4. 统一实验配置

建议固定以下配置：

| 参数 | 值 |
|---|---|
| Resolution | `1280x720` |
| Frames | 121 |
| FPS | 24 |
| Batch size | 1 |
| Precision | BF16 |
| Scheduler | UniPC，Rectified Flow |
| Steps | 35 |
| CFG scale | 6.0 |
| Flow shift | 10.0 |
| Sound generation | 关闭 |
| Prompt upsampling | 关闭或预先固定结果 |
| Guardrails | 主实验中关闭，baseline 和优化版保持一致 |
| CPU offload | 关闭 |

Cosmos3 VAE 的时间压缩率为 4，合法帧数满足 `4n+1`。121 帧在 24 FPS
下，首帧到末帧的时间跨度为：

```text
(121 - 1) / 24 = 5.0 seconds
```

如果按视频容器帧数除以 FPS 计算则约为 5.04 秒。最终报告中应明确采用
121 帧，避免把 120 帧自动对齐到 121 帧后造成口径混乱。

121 帧经过 VAE 时间压缩后得到 31 个 latent frames。根据 Framework
中的 token 计算方式：

```text
31 x (720 / 32) x (1280 / 32)
```

考虑有效整数尺寸后约为 27,280 个 generation tokens。

## 5. 推理链路与瓶颈

T2V 主要链路如下：

1. Prompt tokenize 和序列准备
2. 创建初始视频 latent noise
3. 文本与视频 token 打包，生成 mask 和 3D RoPE
4. 36 层 Cosmos3 MoT denoising
5. Conditional / unconditional CFG 合并
6. UniPC scheduler 更新 latent
7. VAE decode
8. GPU 到 CPU 传输及 MP4 编码

默认 CFG 每一步分别运行 conditional 和 unconditional forward：

```text
35 steps x 2 CFG branches x 36 Transformer blocks
= 2,520 Transformer block forwards
```

预计瓶颈顺序：

1. 长序列 generation full attention
2. MLP 和 QKV/O projection GEMM
3. VAE decode
4. GPU 到 CPU 传输和视频编码
5. Tokenization、噪声初始化和 scheduler

必须通过 profiler 验证实际占比，而不能只依赖理论判断。

## 6. Baseline 和计时体系

### 6.1 Baseline 原则

- 使用官方 `cosmos-framework` BF16 路径
- 固定 checkpoint、commit SHA、软件版本和 GPU 状态
- 固定 prompt、negative prompt 和 token IDs
- 保存并复用完全相同的初始 latent noise tensor
- baseline 和优化版使用相同 guardrail、编码及后处理设置
- 不允许只固定 seed，因为代码路径变化可能改变随机数消费顺序

建议将初始 noise 保存为 `safetensors`，并在 latent 准备阶段加入
dump/load 入口。

### 6.2 计时项目

使用 NVTX、CUDA Event 和 Framework profiler 分别统计：

| 阶段 | 计时内容 |
|---|---|
| Prepare | tokenize、pack、mask、RoPE、noise |
| Cond forward | 每步 conditional MoT |
| Uncond forward | 每步 unconditional MoT |
| Scheduler | CFG combine 和 UniPC update |
| VAE | latent 到 RGB 视频 |
| Transfer | GPU 到 CPU |
| Postprocess | dtype/layout 转换 |
| Encode | MP4 编码 |
| E2E | 输入 prompt 到视频文件写出完成 |

模型加载及首次 `torch.compile` 时间应单列，不计入稳态生成延迟。

建议：

- 1 次 compile/warmup
- 核心 anchor prompt 至少 3 次稳态测量
- 报告 median；有足够时间时进行 5 到 10 次并报告 p95
- 质量测试的高成本 prompt suite 每个配置每个 noise 生成一次

## 7. 分阶段优化路线

### P0：建立可复现的 Baseline

工作内容：

- 安装独立的官方 Framework 环境
- 固定 720p、121 frames、35 steps 的输入 profile
- 增加 noise dump/load
- 增加细粒度计时和 NVTX
- 获取 PyTorch profiler / Nsight Systems trace
- 记录各 kernel、CPU sync、内存分配和 graph break

P0 不追求加速，目标是建立后续所有实验共享的测量和质量基准。

#### P0 执行状态（2026-06-13）

P0 工程验收已完成，可以进入 P1。当前 canonical baseline 为：

- 输出：`cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/t2v/vision.mp4`
- Benchmark：`cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/benchmark.json`
- Breakdown：`cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/t2v/timing_breakdown.json`
- Exact noise：`cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors`
- Profiler：`cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/profile.json.gz`

已完成项目：

- 独立 Framework 推理环境和 H20 单卡运行链路
- 720p、121 帧、24 FPS、35 steps、CFG 6、shift 10 的固定 profile
- `safetensors` 初始 noise dump/load，格式为 `[B, N]`
- CUDA Event 细分 prepare、cond、uncond、CFG combine 和 UniPC scheduler
- Framework timer 细分 VAE、D2H transfer、postprocess、MP4 encode 和 E2E
- PyTorch profiler trace 与 NVTX 发射点
- FA3 实际 kernel 证据：trace 中 288 次 `flash_attn_3::_flash_attn_forward`
- 不同 CLI seed 下加载同一 noise，capture/replay MP4 字节级一致
- 三次成功稳态 B0：481.286 s、481.361 s、481.494 s，中位数 481.361 s

P0 剩余的非工程项：

- 人工完整播放 canonical MP4，并在质量表中签字确认；该项不能由 PSNR 或自动指标替代
- 在最终交付前扩展 12 到 20 个 prompt 的质量 suite；不阻塞 P1 工程消融

已知注意事项：仅为 profiler 证明而设置 `I4_ATTN_BACKENDS=flash3` 时，动态
`torch.compile` 会因 backend filter 的 debug logging 触发 Dynamo graph break。
Canonical compile baseline 使用默认 backend 顺序；FA3 由独立 profiler run 证明。

### P1：低风险工程优化

#### 7.1 FlashAttention-3

Framework 在 Hopper 架构上的优先 backend 包括 FA3、NATTEN 和 FA2。
需要通过日志和 profiler 确认实际选择了 FA3，没有静默回退到低效实现。

Cosmos3 使用 32 个 Q heads 和 8 个 KV heads。替换 backend 时必须保留
原生 GQA，不能简单把 KV 扩展成 32 heads，否则会增加约 4 倍 KV 流量。

#### 7.2 静态 `torch.compile`

针对固定的 720p、121 帧 profile：

- 设置 `dynamic=False`
- 对重复 Transformer block 和 head 分区编译
- 使用 `fullgraph=True`
- 消除 graph break
- 编译时间和稳态时间分开报告

#### 7.3 CUDA Graph

为固定 shape 的 conditional 和 unconditional forward 分别捕获 graph。
需要提前固定：

- Prompt padding 和序列长度桶
- Resolution 和 frame count
- 输入输出 tensor 地址
- timestep buffer
- mask、RoPE 和 scheduler tensor

应移出 graph 或消除的操作包括：

- `t.item()`
- 循环中的 `.cpu()`
- Python tensor 条件判断
- 每步临时 tensor 分配
- callback 和进度条同步

#### 7.4 循环外提和内存复用

将 timestep 不变的工作移出 denoising loop：

- 文本 token 模板
- packed sequence plan
- attention mask
- position IDs 和 RoPE
- 固定 shape buffer

预分配 CFG、scheduler 和 latent 更新所需 tensor，避免循环中重复分配和
layout 转换。

#### 7.5 Cosmos3 架构特异的文本缓存

Cosmos3 两路 attention 中，understanding/text 路径不依赖 noisy generation
tokens 和当前 timestep。可以对 conditional 和 unconditional 文本分别缓存：

- 每层 understanding hidden state
- 每层 text K/V
- 固定文本 mask 和 position 信息

Framework 已有 `gen_only` 和 `MemoryState` 基础设施，但当前 diffusion loop
没有完整利用。这一优化应接近数值等价，虽然文本 token 占比不高，预期收益
只有低个位数，但风险低且能体现对 Cosmos3 架构的深入理解。

#### 7.6 VAE 和视频写出

- VAE 保持 BF16
- 编译 VAE decoder
- 显存允许时不要启用 spatial tiling
- 预分配连续输出 tensor，避免逐帧 `cat`
- 使用 pinned host memory 和独立 copy stream
- 直接处理连续 `uint8 THWC`，避免创建大量 PIL 对象
- 将 MP4 encode 单独计时

P1 合理的端到端累计目标为 `1.3x` 到 `1.6x`，理想情况下约 `1.7x`。

### P2：采样和 CFG 加速

#### 7.7 Step sweep

保持官方 Rectified Flow 和 UniPC，依次测试：

```text
35 -> 30 -> 28 -> 24 -> 20 steps
```

优先沿官方 schedule 重新抽取 timestep，不要简单按线性间隔删除步骤。

Cosmos3 使用 flow prediction。不能未经验证直接替换为主要面向 VP diffusion
的 solver。比较 Euler、Heun、UniPC 等方法时必须报告实际 NFE，而不是只比较
名义 steps。

#### 7.8 CFG interval 和降频

优先实验：

1. 前中期执行完整 CFG，后期只执行 conditional forward
2. Unconditional 分支每两步更新一次，中间复用预测
3. 当 cond/uncond velocity cosine 连续达到阈值后停止 uncond

例如：

```text
Baseline:
35 conditional + 35 unconditional = 70 forwards

Candidate:
28 conditional + 18 unconditional = 46 forwards

Theoretical denoising speedup:
70 / 46 = 1.52x
```

CFG batching 可能降低 wall time，但不会减少 FLOPs。单卡下需要以 profiler
实测，不能假设 batch CFG 一定更快。

步骤减少和 CFG 降频必须先分别做消融，通过质量门槛后再组合。

### P3：选择性 FP8

H20 支持原生 FP8，建议只量化计算量大的 Linear：

- Q、K、V 和 output projection
- MLP gate/up projection
- MLP down projection

以下保持 BF16：

- Attention QK、softmax 和 PV
- RMSNorm、RoPE、residual
- Timestep embedding
- 输入和输出 projection
- VAE
- 首尾 2 到 4 层及敏感层

按 early、middle、late timestep 做层敏感度和量化误差校准。量化权重应预先
转换并缓存，不能在 denoising loop 中重复量化。

FP8 的预期额外端到端收益约为 `1.08x` 到 `1.25x`，实际收益受 attention
占比限制。

不建议主线采用：

- FP4：H20 上不是最合适的硬件路径，质量风险更高
- Weight-only INT8：大 token GEMM 通常计算受限，可能没有明显加速
- 全模型 FP8：误差可能在迭代采样中累积为闪烁和纹理退化

### P4：保守的特征缓存

在 P1 到 P3 稳定后，再研究 TeaCache、FasterCache、TaylorSeer 或 PAB 类
免训练缓存：

- 只对白名单中的中间 Transformer blocks 缓存 residual 或 attention 输出
- 首尾若干 steps 强制刷新
- CFG 开关和 guidance 模式变化时强制刷新
- 缓存误差超过阈值时刷新
- 输入层、输出层和强 timestep 依赖模块不缓存

Framework 中存在 TaylorSeer 相关工具和属性，但当前没有发现完整接入
diffusion forward 的现成路径，应视为研究基础设施，而非直接可启用功能。

Token pruning、token merge、local/sparse attention 和 attention broadcast
放在最后。它们最容易损伤：

- 快速运动
- 摄像机移动
- 小物体和文字
- 人脸和手
- 多物体遮挡及空间关系

## 8. 质量评测

### 8.1 同噪声原则

每个 baseline/optimized pair 必须共享：

- 完全相同的初始 latent noise tensor
- Prompt 和 negative prompt
- Token IDs
- Resolution、frames 和 FPS
- Checkpoint 和 VAE
- 除当前消融变量外的全部 scheduler 参数

最终材料应提供同步播放的左右并排 MP4。Difference video 可以作为辅助，
但不能替代人工语义和视觉判断。

### 8.2 Prompt suite

最终建议准备 12 到 20 个高风险 prompt，每个使用两个固定 noise：

- 主体数量和空间关系
- 人物、脸、手和人体动作
- 快速主体运动
- Camera pan、dolly、tracking、orbit
- 小物体、细线条和可读文字
- 多物体遮挡
- 低光和高动态范围
- 液体、碰撞和复杂物理交互
- 细密纹理和重复结构

开发早期可在 256p 或 480p 快速筛选明显失败的方案，但最终通过与否必须由
720p、121 帧结果决定。

### 8.3 人工质量闸门

出现以下任意情况应淘汰配置：

- 主体、动作、数量或空间关系变化
- 镜头运动偏离 prompt
- 新增模糊、噪点或明显闪烁
- 人脸、手、肢体或物体几何崩坏
- 运动幅度明显减弱或被冻结
- 颜色、曝光或细节稳定性明显下降

如果条件允许，使用 3 名评审进行盲测，评价：

- Prompt alignment
- Visual quality
- Temporal consistency
- Motion quality
- Baseline 和优化版总体偏好

VBench、CLIP、DINO、光流一致性和 flicker 指标只用于自动筛选和定位问题，
不作为最终接受标准。

## 9. 消融与结果表格

所有优化必须逐项加入，不能只报告最终 all-on 配置：

| ID | 配置 | Steps | Cond/Uncond NFE | Precision | Cache | E2E | Speedup | 质量结论 |
|---|---|---:|---:|---|---|---:|---:|---|
| B0 | Framework BF16 baseline | 35 | 35/35 | BF16 | Off | 481.361 s median | 1.00x | Reference |
| E1 | B0 + FA3/static compile/graph | 35 | 35/35 | BF16 | Off | TBD | TBD | TBD |
| E2 | E1 + hoist/text K/V cache | 35 | 35/35 | BF16 | Exact | TBD | TBD | TBD |
| E3 | E2 + reduced steps | 28 | 28/28 | BF16 | Exact | TBD | TBD | TBD |
| E4 | E3 + CFG interval | 28 | 28/18 | BF16 | Exact | TBD | TBD | TBD |
| E5 | E4 + selective FP8 | 28 | 28/18 | Mixed | Exact | TBD | TBD | TBD |
| E6 | E5 + conservative block cache | 28 | 28/18 | Mixed | Approx. | TBD | TBD | TBD |

Breakdown 表：

| 配置 | Prepare | Cond | Uncond | Scheduler | VAE | D2H/Post | Encode | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 5.963 s | 231.918 s | 220.726 s | 0.050 s | 18.071 s | 2.077 s | 1.383 s | 481.494 s |
| Optimized | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Speedup | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

同时报告：

- 实际执行的 Transformer forward 数
- 实际执行或跳过的 block/attention 数
- Peak GPU memory
- Cold-start load/compile latency
- Steady-state median 和 p95
- 质量测试通过率及失败 prompt

## 10. 预期目标

各项收益不能直接相乘，合理目标为：

| 方案 | 预期端到端加速 |
|---|---:|
| 低风险 kernel/compile/graph/VAE | `1.3x` 到 `1.6x` |
| 上述方案 + 保守减步和 CFG | `2.0x` 到 `2.8x` |
| 再加选择性 FP8 和验证后的缓存 | `2.5x` 到 `3.5x` |

在不进行蒸馏训练且严格保持视频质量的前提下，不应预先承诺超过 `4x`。
如果实验最终只能在 `2x` 左右稳定通过质量门槛，也比通过激进近似得到高数字
但产生语义和视觉退化更有说服力。

## 11. 推荐最终配置候选

第一版目标配置：

- NVIDIA Cosmos Framework
- 720p、121 frames、24 FPS
- FA3
- Static `torch.compile`
- CUDA Graph
- 预构建 mask/RoPE/sequence plan
- 文本 understanding K/V cache
- 28 steps UniPC
- 前中期 CFG、后期 conditional-only
- BF16 VAE
- 大型 generation Linear 选择性 FP8
- 不使用 token pruning 和激进 sparse attention

如果选择性 FP8 或特征缓存无法稳定通过同噪声盲测，应退回到 BF16 和无近似
缓存版本，把最终结果定位为质量优先的 `2.0x` 到 `2.8x` 工程加速。

## 12. 最终交付物

建议最终提交包含：

1. 可复现的 baseline 和 optimized 启动脚本
2. 环境、依赖、硬件和 commit SHA 记录
3. 初始 latent noise 文件及 prompt manifest
4. 细粒度 profiler 和 benchmark 输出
5. 每项优化的消融表
6. Breakdown 和端到端加速结果
7. 同噪声同步并排视频
8. 人工盲测表和失败案例
9. 最终推荐配置、适用范围和风险说明

这套材料的重点不是展示最多的优化名词，而是证明每一项加速都经过独立测量，
并且最终配置在严格的同噪声视频对比中保持了提示词语义和可接受的视觉质量。
