# Cosmos3-Nano 16B 单卡 720p 5s T2V 免训练推理加速报告

**最终结果：344.612 s；相对本机同协议 B0 为 1.3968×；端到端时延下降 28.41%；Anchor + 3/3 高风险 Prompt 通过。**

## 摘要

本项目在 NVIDIA Cosmos Framework 上，对 Cosmos3-Nano 16B 的文本生成视频路径进行单卡、720p、121 帧、24 FPS、35-step UniPC 推理优化。全流程不训练、不蒸馏、不更改 VAE。最终方案由后期 CFG 分支裁剪、generation pathway 选择性 FP8、动态阈值 conditional residual cache 组成。

## 固定协议

- GPU：1× NVIDIA H20 96GB
- 输出：1280×720，121 frames，24 FPS，首尾帧跨度 5.000s
- Sampler：UniPC 35 steps，shift 10，CFG 6
- Noise SHA-256：`44122c662d3bd10659534486aa1d455a22ffe6d0c24ffb0dbb5fe7f0553d4b61`
- 质量：同初始噪声完整视频人工审核，不以 PSNR/SSIM 替代

## 优化结论

1. P1：Static compile、CUDA Graph、text K/V cache 均无可计入收益，保留为负结果。
2. P2：最后 10 个低噪声 step 跳过 unconditional CFG，NFE 35/35→35/25，E2E 418.109s。
3. P3：选择性量化第 4–31 层 140 个 generation Linear，E2E median 359.067s。
4. P4：conditional 层 8–27 动态 residual reuse，候选 11–23，阈值 0.0545，E2E median 344.612s。

## 消融结果

|配置|Steps|Cond/Uncond|Precision|Cache|E2E|vs B0|质量|
|---|---:|---:|---|---|---:|---:|---|
|B0|35|35/35|BF16|Off|481.361s median|1.0000×|Reference|
|P2|35|35/25|BF16|Off|418.109s|1.1513×|Anchor PASS|
|P3|35|35/25|Selective FP8|Off|359.067s median|1.3406×|Anchor+3/3 PASS|
|P4 Balanced|35|35/25|Selective FP8|Dynamic residual|344.612s median|1.3968×|Anchor+3/3 PASS|
|P4 Aggressive|35|35/25|Selective FP8|Aggressive|335.881s|1.4331×|Reject: 边缘模糊|

## 官方参考口径

NVIDIA 当前 H20 PyTorch 720p/1 为 931.39s，但官方说明标准视频 workload 通常为 189 frames；本项目为 121 frames，且计时范围不同。因此官方值只作背景，不用于计算本项目加速比。

官方链接：https://github.com/NVIDIA/cosmos/blob/main/inference_benchmarks.md#text-to-video-t2v

## 绝对路径证据索引

- **Baseline video**: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/t2v/vision.mp4`
- **Baseline benchmark**: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/benchmark.json`
- **Exact initial noise**: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors`
- **Profiler trace**: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/profile.json.gz`
- **P2 output**: `/root/autodl-tmp/cosmos3_t2v_p2/outputs/cfg_interval_800_1000_20260614T130154Z/t2v/vision.mp4`
- **P3 output**: `/root/autodl-tmp/cosmos3_t2v_p3/outputs/fp8_steps_35_20260614T160455Z/t2v/vision.mp4`
- **P4 canonical output**: `/root/autodl-tmp/cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_20260615T0254Z/t2v/vision.mp4`
- **P4 repeat benchmark**: `/root/autodl-tmp/cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_steady_repeat3_20260615T1418Z/benchmark.json`
- **Quality manifest**: `/root/autodl-tmp/cosmos3_t2v_quality/QUALITY_MANIFEST.json`
- **P4 status**: `/root/autodl-tmp/cosmos3_t2v_p4/P4_STATUS.md`

## 同噪声完整视频

- **B0 vs P2 CFG**: `/root/autodl-tmp/cosmos3_t2v_p2/B0_35_left_vs_CFG800_1000_right_same_noise.mp4`
- **B0 vs P3 FP8+CFG**: `/root/autodl-tmp/cosmos3_t2v_p3/B0_BF16_left_vs_P3_FP8_CFG_right_same_noise.mp4`
- **P3 vs P4 Balanced**: `/root/autodl-tmp/cosmos3_t2v_p4/P3_left_vs_P4_balanced_right_same_noise.mp4`
- **human_handoff P3 vs P4**: `/root/autodl-tmp/cosmos3_t2v_quality/human_handoff_P3_left_vs_P4_right_same_noise.mp4`
- **rally_tracking P3 vs P4**: `/root/autodl-tmp/cosmos3_t2v_quality/rally_tracking_P3_left_vs_P4_right_same_noise.mp4`
- **tabletop_relations P3 vs P4**: `/root/autodl-tmp/cosmos3_t2v_quality/tabletop_relations_P3_left_vs_P4_right_same_noise.mp4`
- **Aggressive boundary**: `/root/autodl-tmp/cosmos3_t2v_quality/human_handoff_P4_balanced_left_vs_aggressive_right_same_noise.mp4`

## Commit

- Framework base: `a5ae92b7d7aab100e2f5e96c44788adfce26331c`
- NVIDIA cosmos reference: `1fe7e3be1687d797392b0e82ff6fe6296638b49f`

## 结论

最终 P4 Balanced 在保持 35-step UniPC 和严格同噪声质量门的前提下实现 1.3968× E2E 加速。报告公开负结果和 Aggressive 失败案例，保证结果可审计、可复现。
