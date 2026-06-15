# Cosmos3 T2V P3 Status

## Selective FP8 + CFG interval

P3 在 P2 的 35-step CFG interval 主候选上叠加选择性 FP8：

- 保持 35-step UniPC、shift 10、CFG 6、720p、121 帧、24 FPS
- 保持 `guidance_interval=[800, 1000]`
- 使用与 B0、P2 完全相同的 initial noise
- 仅量化 generation pathway 第 4 到 31 层的大型 Linear
- 量化 `q_proj_moe_gen`、`o_proj_moe_gen` 和 MLP `gate/up/down`
- K/V、文本分支、首尾各 4 层、Norm、RoPE、Attention 与 VAE 保持 BF16

共量化 `28 layers x 5 Linear = 140` 个模块。TorchAO 使用动态 activation
FP8 E4M3 和预量化 FP8 E4M3 weight。

正式运行目录：

```text
/root/autodl-tmp/cosmos3_t2v_p3/outputs/fp8_steps_35_20260614T160455Z
```

## 性能结果

| 阶段 | B0 BF16 | P2 CFG BF16 | P3 CFG + FP8 |
|---|---:|---:|---:|
| E2E | 481.494 s | 418.109 s | 359.067 s |
| Generate/denoise | 458.682 s | 395.580 s | 336.448 s |
| Cond forwards | 231.918 s | 231.900 s | 197.415 s |
| Uncond forwards | 220.726 s | 157.650 s | 133.020 s |
| VAE decode | 18.071 s | 18.073 s | 18.070 s |
| Prepare | 5.963 s | 5.962 s | 5.944 s |

P3 相对 B0：

- E2E：`1.3410x`，节省 122.427 秒
- Generate/denoise：`1.3633x`，节省 122.234 秒

P3 相对 P2：

- E2E：`1.1644x`，节省 59.042 秒
- Generate/denoise：`1.1758x`，节省 59.132 秒

计时器确认 conditional/unconditional forward 数为 `35/25`，scheduler
更新 35 次。峰值显存为 41,233 MiB。

## 人工质量验收

同噪声对比材料：

```text
/root/autodl-tmp/cosmos3_t2v_p3/P2_BF16_left_vs_P3_FP8_right_same_noise.mp4
/root/autodl-tmp/cosmos3_t2v_p3/B0_BF16_left_vs_P3_FP8_CFG_right_same_noise.mp4
```

2026-06-15，用户完整查看两组 121 帧并排视频并确认质量合格：

- P2 BF16 vs P3 FP8：PASS
- B0 BF16 vs P3 CFG + FP8：PASS

当前 anchor prompt 未见不可接受的语义偏移、明显质量下降、模糊、噪点或
闪烁。该结论只覆盖当前 prompt 和 noise；P3 成为候选配置，但最终泛化结论
仍需代表性 prompt suite。

## 下一步

1. 使用主体关系、人物动作、镜头运动、细纹理等代表性 prompt 做质量验证。
2. 对至少一个代表性 prompt 重复生成，检查 FP8 质量稳定性。
3. 只有 P3 泛化质量通过后，才进入 P4 保守特征缓存消融。


## 三次正式性能复测

三次均使用独立进程，每次包含一次 warmup 和一次 formal；只统计 formal：

| Run | E2E | Generate/denoise | Cond | Uncond |
|---|---:|---:|---:|---:|
| Run 1 | 359.067 s | 336.448 s | 197.415 s | 133.020 s |
| Run 2 | 359.158 s | 336.415 s | 197.397 s | 133.009 s |
| Run 3 | 359.056 s | 336.445 s | 197.418 s | 133.016 s |
| Median | **359.067 s** | **336.445 s** | **197.415 s** | **133.016 s** |

E2E 范围为 0.102 秒，变异系数为 0.0127%。相对三次 B0 中位数
481.361 秒，P3 median E2E 加速比为 `1.3406x`。三次峰值显存为
41,233 到 41,235 MiB。

## 确定性说明

三次运行加载相同 initial noise，但解码后原始帧哈希不同。额外测试显式关闭
TorchAO FP8 fast accumulation，跨进程仍未恢复 bit-exact，说明差异来自 FP8
动态量化及并行 kernel 归约路径的组合，而非 MP4 编码或 fast accumulation
单一因素。

该现象不影响 E2E 性能重复性，但意味着 FP8 候选不能声称 bit-exact
reproducibility。最终质量稳定性必须覆盖多个 prompt，并抽查同 prompt 的重复
输出。三次 FP8 重复输出并排材料：

```text
/root/autodl-tmp/cosmos3_t2v_p3/P3_FP8_three_repeats_same_noise.mp4
```
