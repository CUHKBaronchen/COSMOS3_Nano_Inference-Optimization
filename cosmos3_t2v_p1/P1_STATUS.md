# Cosmos3 T2V P1 Status

## 当前结论

截至 2026-06-14，P0 已完成，P1 已完成 FA3 证据固化、静态 `torch.compile`、CUDA Graph 和 Cosmos3 架构特异文本缓存消融。目前尚未得到可计入最终加速比的 P1 性能收益。

Canonical B0 为 `481.361 s`（三次稳态运行中位数），配置固定为单张 H20、1280x720、121 帧、24 FPS、UniPC 35 步、CFG 6、shift 10、BF16，并复用 SHA256 为 `44122c662d3bd10659534486aa1d455a22ffe6d0c24ffb0dbb5fe7f0553d4b61` 的 initial noise。

## P1 项目状态

| 项目 | 状态 | 结论 |
|---|---|---|
| FlashAttention-3 | 完成 | P0 profiler 中有 288 次 `flash_attn_3::_flash_attn_forward`，未静默回退 |
| 静态 `torch.compile` | 完成，拒绝合入 | E2E `481.142 s`，相对 B0 中位数仅 `1.00046x`，属于噪声 |
| CUDA Graph | 完成 shape smoke，拒绝完整运行 | 相对静态 shape smoke 慢 `0.23%`，峰值显存增加约 `3.1 GB` |
| 循环外提/内存复用 | 待实现 | Prepare 约 6 秒，只占 B0 的 1.24%，需限定投入 |
| 文本路径缓存 | 完成，拒绝合入 | 严格 A/B E2E 仅 `1.0037x`；逐帧画面哈希不同，收益不足以覆盖质量审查成本 |
| VAE/写出 | 待评估 | VAE 18.07 秒，占 B0 3.75%，只能提供有限 E2E 收益 |

## 静态 compile 正式消融

输出目录：

```text
/root/autodl-tmp/cosmos3_t2v_p1/outputs/static_baseline_20260614T073524Z
```

| 阶段 | B0 | Static | 变化 |
|---|---:|---:|---:|
| E2E | 481.494 s（该次 B0） | 481.142 s | -0.073% |
| Denoise | 458.682 s | 458.510 s | -0.038% |
| Cond forwards | 231.918 s | 231.827 s | -0.039% |
| Uncond forwards | 220.726 s | 220.641 s | -0.039% |
| VAE decode | 18.071 s | 18.069 s | -0.012% |

相对三次 B0 中位数 `481.361 s`，Static 的加速比只有 `1.00046x`。峰值显存均为 `45,871 MiB`。该结果不足以宣称加速，因此保留为负结果，不作为后续组合优化的基线。

静态编译改变了浮点计算路径。与 B0 使用同一 initial noise 时，输出 MP4 不同，121 帧解码后的平均绝对像素差为 `4.70/255`。这不是最终质量指标，只说明结果并非字节或像素等价，必须人工播放判断。

人工对比视频：

```text
/root/autodl-tmp/cosmos3_t2v_p1/B0_left_vs_P1_static_right_same_noise.mp4
```

左侧为 B0，右侧为 P1 Static。

## CUDA Graph shape smoke

输出目录：

```text
/root/autodl-tmp/cosmos3_t2v_p1/outputs/cudagraph_smoke_20260614T075230Z
```

固定真实 720p/121 帧形状，使用 2 steps、同一 initial noise：

| 指标 | Static smoke | CUDA Graph smoke | 变化 |
|---|---:|---:|---:|
| E2E | 54.555 s | 54.682 s | +0.233% |
| Denoise | 31.831 s | 31.920 s | +0.281% |
| Cond forwards | 13.249 s | 13.259 s | +0.076% |
| Uncond forwards | 12.612 s | 12.686 s | +0.589% |
| Peak VRAM | 45,621 MiB | 48,689 MiB | +3,068 MiB |

CUDA Graph 路径能够运行并执行固定长度 padding，但没有速度收益。由于 MoT 每步计算约 13 秒，Python/kernel launch 开销不是主要瓶颈，故不执行昂贵的 35-step 完整运行。

## Diffusion 文本路径缓存消融

实现内容：

- 新增默认关闭的 `cache_diffusion_text` 采样参数。
- conditional / unconditional 分支各自维护逐层 understanding K/V 缓存。
- 首次 forward 计算并保存文本 K/V，后续 timestep 使用已有 `gen_only` 路径，只计算 generation tower。
- 当前原型明确限制为单卡、`two_way` attention、无 control-CFG；不支持的组合直接报错。
- 多样本 packed attention 小张量测试中，cached generation attention 与完整 attention 在 BF16 下达到 `atol=0, rtol=0` 精确一致。

严格 A/B 仅改变 `cache_diffusion_text` 开关，均为 720p、121 帧、2 steps、CFG 6、seed 999，并复用同一个 initial noise：

| 指标 | Cache off | Cache on | 变化 |
|---|---:|---:|---:|
| E2E | 54.655 s | 54.453 s | `1.0037x` |
| Generate/denoise | 31.847 s | 31.679 s | `1.0053x` |
| Cond forwards | 13.258 s | 13.093 s | `1.0126x` |
| Uncond forwards | 12.616 s | 12.604 s | `1.0010x` |
| VAE decode | 18.070 s | 18.070 s | 无变化 |
| Peak VRAM | 45,621 MiB | 45,201 MiB | -420 MiB（采样噪声口径） |

输出目录：

```text
/root/autodl-tmp/cosmos3_t2v_p1/outputs/text_cache_control_20260614T082101Z
/root/autodl-tmp/cosmos3_t2v_p1/outputs/text_cache_smoke_20260614T081642Z
```

两份 MP4 的 121 个解码帧哈希全部不同，说明动态 compile 的完整路径因图形变化产生了非零浮点差异。单层 attention 等价测试证明缓存拼接和 sample offsets 正确，但整模型输出不能宣称像素等价。由于 E2E 收益仅 0.37%，低于稳态噪声和完整质量审查的合理门槛，不执行昂贵的 35-step 正式运行，不计入最终优化组合。

## 基础设施修复

Framework 原实现即使设置 `HF_HUB_OFFLINE=1`，仍通过 `uvx hf@1.16.4` 解析 tokenizer，可能因 uvx 临时依赖缺失而触网失败。已修改 `cosmos_framework/utils/checkpoint_db.py`：离线模式使用本机 `hf` CLI 校验并返回精确 revision 的本地 snapshot；在线模式仍保留官方固定版本的 uvx 路径。该修改不进入模型数值计算。

## 下一步

1. P1 文本缓存因收益不足已停止，保留为负结果。
2. 进入 P2 Step sweep，先运行 30-step UniPC 同噪声候选，再依次评估 28、24、20 steps。
3. 步数减少和 CFG interval 必须先分别消融，通过人工视频质量门槛后才能组合。
4. 任何候选优化都必须报告 E2E、阶段 breakdown、显存和同噪声人工视频对比。
