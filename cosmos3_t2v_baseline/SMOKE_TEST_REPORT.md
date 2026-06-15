# Cosmos3-Nano T2V Smoke Test Report

## 结论

Smoke test **通过**。本次运行成功完成了 Cosmos3-Nano 的文本到视频推理全链路，包括：模型配置解析、Qwen tokenizer 初始化、Cosmos3 权重加载、Wan2.2 原生 VAE 解码、UniPC 扩散采样、H.264 视频编码以及 benchmark 数据落盘。

本结果只证明当前环境和推理链路可用，**不代表正式 720p / 5 秒质量或性能 baseline**。Smoke test 使用低分辨率、25 帧和 4 个采样步，主要用于快速排除安装、权重、CUDA、VAE 与输出编码问题。

## 运行信息

- 运行目录：`outputs/smoke_20260613T060950Z`
- 模型视图：`/root/autodl-tmp/Cosmos3-Nano-framework`
- 原始模型：`/root/autodl-tmp/Cosmos3-Nano`
- 框架：`/root/autodl-tmp/cosmos-framework`
- GPU：NVIDIA H20，单卡
- 精度：BF16
- 模式：`text2video`
- 采样器：UniPC
- `torch.compile`：关闭（smoke test）
- CUDA Graph：关闭（smoke test）
- Guardrails：关闭
- 音频生成：关闭

## 实际执行指令

本次成功运行由 `/root/autodl-tmp` 目录下的以下命令启动：

```bash
cd /root/autodl-tmp
bash cosmos3_t2v_baseline/run_smoke.sh
```

脚本使用 UTC 时间自动生成 `RUN_ID`。本次生成的值为 `20260613T060950Z`，因此输出目录为：

```text
/root/autodl-tmp/cosmos3_t2v_baseline/outputs/smoke_20260613T060950Z
```

如需明确指定同一个输出目录名，对应入口命令为：

```bash
cd /root/autodl-tmp
RUN_ID=20260613T060950Z CUDA_VISIBLE_DEVICES=0 \
  bash cosmos3_t2v_baseline/run_smoke.sh
```

注意：再次使用相同 `RUN_ID` 会写入已有目录。做新的复现实验时应更换 `RUN_ID`，以免混合两次运行的日志和结果。

`run_smoke.sh` 在设置环境变量后，实际执行的推理命令展开如下：

```bash
cd /root/autodl-tmp/cosmos-framework

COSMOS_TRAINING=0 \
LD_LIBRARY_PATH= \
PATH="/root/autodl-tmp/.tools:$PATH" \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
HF_HOME=/tmp/cosmos3-hf-cache \
HF_HUB_OFFLINE=1 \
IMAGINAIRE_CACHE_DIR=/tmp/cosmos3-imaginaire-cache \
/tmp/cosmos-framework-venv/bin/python \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --checkpoint-path=/root/autodl-tmp/Cosmos3-Nano-framework \
  --no-guardrails \
  --no-use-torch-compile \
  --no-use-cuda-graphs \
  --benchmark \
  --seed=0 \
  --defaults-file=/root/autodl-tmp/cosmos3_t2v_baseline/configs/smoke_defaults.json \
  --resolution=256 \
  --aspect-ratio=16,9 \
  --fps=24 \
  --num-frames=25 \
  -i /root/autodl-tmp/cosmos-framework/inputs/omni/t2v.json \
  -o /root/autodl-tmp/cosmos3_t2v_baseline/outputs/smoke_20260613T060950Z \
  2>&1 | tee /root/autodl-tmp/cosmos3_t2v_baseline/outputs/smoke_20260613T060950Z/run.log
```

参数来源如下：

- Prompt 和 `model_mode=text2video`：`cosmos-framework/inputs/omni/t2v.json`
- `num_steps=4`、`guidance=6.0`、`shift=3.0`：`cosmos3_t2v_baseline/configs/smoke_defaults.json`
- `resolution=256`、`aspect_ratio=16,9`、`fps=24`、`num_frames=25` 和 `seed=0`：由命令行明确指定
- 最终合并后的完整参数：`outputs/smoke_20260613T060950Z/t2v/sample_args.json`

## 实际采样参数

参数来自生成后保存的 `t2v/sample_args.json`：

| 参数 | 值 |
|---|---:|
| seed | 0 |
| resolution 档位 | 256 |
| 实际视频尺寸 | 320 x 192 |
| FPS | 24 |
| 帧数 | 25 |
| 时长 | 1.0417 秒 |
| diffusion steps | 4 |
| CFG guidance | 6.0 |
| shift | 3.0 |
| sampler | UniPC |

注意：`resolution=256` 是 Framework 的内部测试档位，经过尺寸 bucket 对齐后实际编码尺寸为 `320x192`，不能把该结果称为 256p 或 720p baseline。

## 输出核验

生成文件：

- `t2v/vision.mp4`
- `t2v/sample_args.json`
- `t2v/sample_outputs.json`
- `benchmark.json`
- `run.log`
- `console.log`
- `debug.log`

视频容器检查结果：

| 项目 | 结果 |
|---|---|
| 编码 | H.264 |
| 尺寸 | 320 x 192 |
| FPS | 24 |
| 可解码帧数 | 25 |
| 文件大小 | 87,313 bytes |
| 输出状态 | `success` |

全部 25 帧均可解码。像素均值范围为 `39.61 - 66.28`，帧内像素标准差范围为 `65.64 - 79.33`，平均相邻帧绝对差为 `8.01`。这些数据可以排除全黑、全白、冻结帧和损坏视频等基本故障，但不能替代人工主观画质评价。

## 性能结果

数据来自 `benchmark.json`：

| 阶段 | 耗时 |
|---|---:|
| `OmniInference.generate_batch` | 3.5683 秒 |
| `OmniMoTModel.generate_samples_from_batch` | 2.8321 秒 |
| `OmniMoTModel.decode` | 0.3420 秒 |

以上数字仅适用于 320x192、25 帧、4 步、未启用 `torch.compile` 的 smoke 配置，不能与官方 720p benchmark 直接比较。

## 运行中发现并处理的问题

### 1. Framework 所需 VAE 格式不同

`Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors` 是 Diffusers 格式；当前 Cosmos Framework 的 tokenizer 路径直接使用 `torch.load()` 加载原生 `Wan2.2_VAE.pth`，二者不能直接互换。因此补充了 Framework 所需的原生 Wan2.2 VAE，并通过符号链接接入，未重复复制大文件。

### 2. 官方 T2V defaults 与参数校验冲突

官方 `text2video/sample_args.json` 包含 `negative_prompt_file`，但当前代码的 transfer 参数校验会把该字段误判为“请求 transfer inference”，继而要求 `edge/blur/depth/seg/wsm` 控制提示。输入文件本身已经明确设置 `model_mode: text2video`，不存在任务类型配置错误。

已对 `cosmos_framework/inference/args.py` 做最小兼容修复：`negative_prompt_file` 不再单独触发 transfer inference 校验。该修复不改变模型权重、采样器、随机种子或扩散计算。

### 3. 模型配置只读适配

建立了轻量的 `Cosmos3-Nano-framework` 模型视图：大权重仍通过符号链接复用原始目录，只调整 Framework 所需的本地 VAE 路径并关闭当前任务不需要的音频生成。原始 `/root/autodl-tmp/Cosmos3-Nano` 未被修改。

## 已知限制

- 本次只有 4 个 diffusion steps，画质不具备代表性。
- 本次时长约 1 秒，而正式目标为 5 秒、121 帧。
- 本次未启用 `torch.compile`，不代表官方推荐性能配置。
- 自动完整性检查通过；最终语义一致性、清晰度、时序稳定性仍需对正式 720p 视频进行人工定性检查。
- 日志中出现 FFmpeg 的重复 `-pix_fmt` 提示，但最终 H.264 文件完整可解码，不影响本次 smoke 结论。

## 下一阶段验收条件

正式 baseline 应固定相同 prompt 与 seed，并使用：单张 H20、1280x720、121 帧、24 FPS、UniPC 35 步、CFG 6.0、shift 10.0、BF16。需要分别记录首次编译/warmup 与稳定推理耗时，并保存端到端、扩散采样、VAE 解码等 breakdown，作为后续免训练加速的对照基线。
