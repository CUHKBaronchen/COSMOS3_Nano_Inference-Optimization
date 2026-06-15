# Cosmos3 T2V P2 Status

## 30-step UniPC 候选

P2 Step sweep 的第一个候选保持 canonical baseline 的模型、UniPC、shift 10、CFG 6、BF16、720p、121 帧、24 FPS、prompt、negative prompt 和 exact initial noise 不变，仅将 `num_steps` 从 35 改为 30。

运行目录：

```text
/root/autodl-tmp/cosmos3_t2v_p2/outputs/steps_30_20260614T082851Z
```

执行命令：

```bash
STEPS=30 WARMUP=1 bash /root/autodl-tmp/cosmos3_t2v_p2/run_step_sweep.sh
```

同噪声人工对比材料：

```text
/root/autodl-tmp/cosmos3_t2v_p2/B0_35_left_vs_P2_30_right_same_noise.mp4
/root/autodl-tmp/cosmos3_t2v_p2/B0_35_left_vs_P2_30_right_contact_sheet.png
```

并排视频左侧为 35-step canonical baseline，右侧为 30-step candidate；文件固定为 2560x720、24 FPS、121 帧。

## 性能结果

| 阶段 | 35-step B0 | 30-step | 加速比 | 节省 |
|---|---:|---:|---:|---:|
| E2E | 481.494 s | 416.589 s | `1.1558x` | 64.905 s |
| Generate/denoise | 458.682 s | 393.931 s | `1.1644x` | 64.752 s |
| Cond forwards | 231.918 s | 198.749 s | `1.1669x` | 33.168 s |
| Uncond forwards | 220.726 s | 189.160 s | `1.1669x` | 31.566 s |
| VAE decode | 18.071 s | 18.071 s | `1.0000x` | 0.001 s |
| Prepare | 5.963 s | 5.961 s | `1.0004x` | 0.003 s |

30-step 峰值显存为 45,911 MiB。denoise 加速与理论 `35/30 = 1.1667x` 基本一致，说明没有额外 kernel 回退或调度器开销。

相对三次 canonical B0 中位数 481.361 s，30-step E2E 加速比为 `1.1555x`。

## 质量状态

性能门槛已通过。质量门槛必须以同噪声并排视频人工定性判断；联系图用于筛查五个时间点，不能替代完整播放。五时间点联系图经独立定性审阅保守判定为 PASS：未见语义偏移、明显模糊、噪点或结构退化；差异主要是允许的火花、烟雾和高亮分布变化。联系图不能验证帧间闪烁，完整并排视频仍需最终人工播放签字。基于静态质量门槛通过，继续 28-step 候选。


## Step sweep 定位调整

30-step 和 28-step 分别得到 416.589 s 与 390.207 s 的 E2E，但它们改变了官方默认 35-step 去噪配置。根据项目最终口径，这两项仅保留为采样敏感度消融，不作为最终优化方案，也不继续 24/20-step sweep。

主线改为固定 35-step UniPC，通过 `guidance_interval` 仅减少后期 unconditional CFG forward。第一候选为 `[800, 1000]`：真实 shift=10 timestep 中前 25 步执行完整 CFG，最后 10 步只执行 conditional forward，总 forward 数从 70 降至 60。

## 35-step CFG interval 主候选

正式运行目录：

```text
/root/autodl-tmp/cosmos3_t2v_p2/outputs/cfg_interval_800_1000_20260614T130154Z
```

执行命令：

```bash
WARMUP=1 bash /root/autodl-tmp/cosmos3_t2v_p2/run_cfg_interval.sh
```

除 `guidance_interval=[800, 1000]` 外，配置与 canonical B0 相同：35-step
UniPC、shift 10、CFG 6、BF16、720p、121 帧、24 FPS，并加载完全相同的
initial noise。

计时器确认实际执行：

```text
conditional forwards:   35
unconditional forwards: 25
CFG combine:             25
scheduler updates:       35
```

因此该方案没有减少去噪步数，也没有改变 scheduler trajectory；它只在
最后 10 个低噪声 step 跳过 unconditional forward。

## 主候选性能结果

| 阶段 | 35-step B0 | 35-step CFG interval | 加速比 | 节省 |
|---|---:|---:|---:|---:|
| E2E | 481.494 s | 418.109 s | `1.1516x` | 63.385 s |
| Generate/denoise | 458.682 s | 395.580 s | `1.1595x` | 63.103 s |
| Cond forwards | 231.918 s | 231.900 s | `1.0001x` | 0.018 s |
| Uncond forwards | 220.726 s | 157.650 s | `1.4001x` | 63.077 s |
| VAE decode | 18.071 s | 18.073 s | `0.9999x` | -0.002 s |
| Prepare | 5.963 s | 5.962 s | `1.0002x` | 0.001 s |

相对三次 canonical B0 中位数 481.361 s，E2E 加速比为 `1.1513x`。峰值
显存为 45,831 MiB。35 个 conditional forward 耗时不变，全部收益都来自
少执行 10 个 unconditional forward，结果与优化机制一致。

同噪声人工对比材料：

```text
/root/autodl-tmp/cosmos3_t2v_p2/B0_35_left_vs_CFG800_1000_right_same_noise.mp4
/root/autodl-tmp/cosmos3_t2v_p2/B0_35_left_vs_CFG800_1000_right_contact_sheet.png
```

并排视频左侧为 canonical B0，右侧为 CFG interval candidate，固定为
2560x720、24 FPS、121 帧。用户已完成初步播放，判断没有明显质量差距；
最终提交前仍需在多 prompt suite 上复验语义、运动、细节和闪烁。

## P2 结论

`guidance_interval=[800, 1000]` 通过当前 anchor prompt 的性能和人工质量
门槛，作为 P2 主候选进入后续组合实验。30-step 和 28-step 仅作为采样
敏感度消融保留，不计入最终 all-on 配置。

