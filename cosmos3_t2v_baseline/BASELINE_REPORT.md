# Cosmos3 Nano 720p T2V Baseline Report

## Result

The single-GPU BF16 baseline completed successfully on one NVIDIA H20.

- Canonical output: `outputs/p0_baseline_20260613T115211Z/t2v/vision.mp4`
- Benchmark: `outputs/p0_baseline_20260613T115211Z/benchmark.json`
- Fine timing: `outputs/p0_baseline_20260613T115211Z/t2v/timing_breakdown.json`
- GPU samples: `outputs/p0_baseline_20260613T115211Z/gpu.csv`
- Final arguments: `outputs/p0_baseline_20260613T115211Z/t2v/sample_args.json`
- Exact initial noise: `outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors`
- Profiler trace: `outputs/p0_capture_20260613T114518Z/t2v/profile.json.gz`

## Fixed configuration

| Parameter | Value |
|---|---:|
| Resolution | 1280x720 |
| Frames | 121 |
| FPS | 24 |
| First-to-last-frame duration | 5.000 s |
| Sampler | UniPC rectified flow |
| Steps | 35 |
| CFG | 6.0 |
| Shift | 10.0 |
| Precision | BF16 |
| CLI seed | 999, intentionally different during replay |
| Initial noise | Fixed safetensors tensor, `[1, 5356800]`, FP32 |
| `torch.compile` | Enabled |
| CUDA Graph | Disabled |
| Compile dynamic | Enabled |
| Guardrails | Disabled |
| Sound | Disabled |

## Timing

| Stage | Warmup | Measured |
|---|---:|---:|
| End-to-end generation | 481.155 s | 481.494 s |
| Denoising/sample generation | 463.073 s | 458.682 s |
| Prepare | 6.316 s | 5.963 s |
| Conditional MoT forwards | 235.980 s | 231.918 s |
| Unconditional MoT forwards | 220.898 s | 220.726 s |
| CFG combine | 0.003 s | 0.003 s |
| UniPC scheduler | 0.093 s | 0.047 s |
| VAE decode | 18.081 s | 18.071 s |
| GPU to CPU transfer | Not saved during warmup | 1.846 s |
| Tensor postprocess | Not saved during warmup | 0.231 s |
| MP4 encode | Not saved during warmup | 1.383 s |

Measured stage shares:

- Conditional forwards: 48.17% of E2E
- Unconditional forwards: 45.84% of E2E
- Prepare: 1.24% of E2E
- VAE decode: 3.75% of E2E
- Transfer + postprocess + encode: 0.72% of E2E
- CFG combine + scheduler: less than 0.02% of E2E

The measured denoising loop averaged 13.105 seconds per UniPC step including
loop-level overhead. Each step spent on average 6.626 seconds in the conditional
forward and 6.306 seconds in the unconditional forward. The three successful
steady B0 measurements are 481.286 s, 481.361 s, and 481.494 s; their median is
481.361 s.

## GPU

| Metric | Value |
|---|---:|
| Peak observed GPU memory | 45,871 MiB |
| Peak GPU utilization | 100% |
| Peak power | 335.61 W |

The H20 has 97,871 MiB total memory, leaving substantial headroom for static
buffers, CUDA Graph capture, or selected precision experiments.

## Output validation

| Property | Value |
|---|---:|
| Codec | H.264 |
| Width | 1280 |
| Height | 720 |
| FPS | 24 |
| Decodable frames | 121 |
| First-to-last-frame span | 5.000 s |
| File size | 3,370,211 bytes |

The file is structurally valid and all frames decode. A five-frame montage was
generated at `/tmp/cosmos3_p0_baseline_montage.png`. Final semantic and visual
quality acceptance still requires human playback of the MP4.

## Exact-noise and profiler validation

The 2-step 720p capture and replay runs are stored under
`outputs/p0_capture_20260613T114518Z` and `outputs/p0_replay_20260613T114750Z`.

- Noise SHA256: `44122c662d3bd10659534486aa1d455a22ffe6d0c24ffb0dbb5fe7f0553d4b61`
- Capture MP4 SHA256: `2fbb0144ebbfc11a016e6f7669cf45166a8bed5e4c674739f84c0b54510d7bc6`
- Replay MP4 SHA256: `2fbb0144ebbfc11a016e6f7669cf45166a8bed5e4c674739f84c0b54510d7bc6`

Changing the replay CLI seed from 0 to 999 while loading the captured exact
initial noise produced a byte-identical MP4 and identical decoded-frame hash.
This proves the comparison path is independent of random-number consumption.

The profiler trace contains 288 `flash_attn_3::_flash_attn_forward` calls, so
the real 720p shape is using FlashAttention-3 rather than silently falling back.

## Interpretation

The measurement confirms that denoising is the dominant optimization target.
VAE-only work cannot provide a large end-to-end speedup because decode accounts
for less than 4% of measured latency.

This run is the B0 performance and visual reference. The canonical noise tensor,
fine timers, profiler trace, GPU telemetry, and byte-identical replay are all
available for subsequent ablations.

## Next work

1. Run the P1 static-compile ablation with `compile_dynamic=False`.
2. Test CUDA Graph capture for the fixed 720p/121-frame shape.
3. Hoist loop-invariant sequence-plan, mask, and RoPE work where profiler evidence supports it.
4. Investigate exact understanding/text K/V reuse before any reduced-step or approximate method.
