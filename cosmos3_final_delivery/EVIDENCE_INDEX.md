# Evidence Index

所有路径均为服务器绝对路径。

- `Baseline video`: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/t2v/vision.mp4`
- `Baseline benchmark`: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_baseline_20260613T115211Z/benchmark.json`
- `Exact initial noise`: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors`
- `Profiler trace`: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/profile.json.gz`
- `P2 output`: `/root/autodl-tmp/cosmos3_t2v_p2/outputs/cfg_interval_800_1000_20260614T130154Z/t2v/vision.mp4`
- `P3 output`: `/root/autodl-tmp/cosmos3_t2v_p3/outputs/fp8_steps_35_20260614T160455Z/t2v/vision.mp4`
- `P4 canonical output`: `/root/autodl-tmp/cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_20260615T0254Z/t2v/vision.mp4`
- `P4 repeat benchmark`: `/root/autodl-tmp/cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_steady_repeat3_20260615T1418Z/benchmark.json`
- `Quality manifest`: `/root/autodl-tmp/cosmos3_t2v_quality/QUALITY_MANIFEST.json`
- `P4 status`: `/root/autodl-tmp/cosmos3_t2v_p4/P4_STATUS.md`

## Videos

- `B0 vs P2 CFG`: `/root/autodl-tmp/cosmos3_t2v_p2/B0_35_left_vs_CFG800_1000_right_same_noise.mp4`
- `B0 vs P3 FP8+CFG`: `/root/autodl-tmp/cosmos3_t2v_p3/B0_BF16_left_vs_P3_FP8_CFG_right_same_noise.mp4`
- `P3 vs P4 Balanced`: `/root/autodl-tmp/cosmos3_t2v_p4/P3_left_vs_P4_balanced_right_same_noise.mp4`
- `human_handoff P3 vs P4`: `/root/autodl-tmp/cosmos3_t2v_quality/human_handoff_P3_left_vs_P4_right_same_noise.mp4`
- `rally_tracking P3 vs P4`: `/root/autodl-tmp/cosmos3_t2v_quality/rally_tracking_P3_left_vs_P4_right_same_noise.mp4`
- `tabletop_relations P3 vs P4`: `/root/autodl-tmp/cosmos3_t2v_quality/tabletop_relations_P3_left_vs_P4_right_same_noise.mp4`
- `Aggressive boundary`: `/root/autodl-tmp/cosmos3_t2v_quality/human_handoff_P4_balanced_left_vs_aggressive_right_same_noise.mp4`
