# P4 Conservative Residual Cache Status

## Status

P4 engineering, performance stability validation, and quality validation are complete.
Anchor-prompt same-noise human review is **PASS**. The three-prompt high-risk P4 suite
passed human review 3/3. The user judged P4 at least equal and subjectively slightly
better overall. Balanced is the final selected P4 configuration.

## Fixed protocol

- Model: `/root/autodl-tmp/Cosmos3-Nano-framework`
- Resolution / duration: 1280x720, 121 frames, 24 FPS (5.0 s frame span)
- Sampler: UniPC, 35 steps, shift 10
- CFG: guidance 6, interval `[800, 1000]` (35 conditional / 25 unconditional forwards)
- Precision: P3 selective generation FP8, 140 Linear modules in layers 4-31
- Exact initial noise: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors`
- Noise SHA-256: `44122c662d3bd10659534486aa1d455a22ffe6d0c24ffb0dbb5fe7f0553d4b61`

P3 and P4 `sample_args.json` differ only in `output_dir`; prompt, noise path, scheduler, steps, CFG, resolution, frames, and FPS are identical.

## Implementation

The cache is disabled unless `COSMOS3_RESIDUAL_CACHE=1`.

- Conditional branch only; unconditional CFG remains exact.
- Reuses decoder residuals only for a layer whitelist (default layers 8-27).
- Preserves all 35 scheduler steps and all conditional model calls.
- Reuse steps must be non-adjacent, so every reuse is followed by a full refresh.
- First/last steps and explicit CFG-transition steps are protected.
- At the first cached layer, current and previous generation hidden states are compared with relative L1. A threshold miss automatically runs the full blocks.
- Input/output projections, first/last model layers, VAE, scheduler, attention kernels, and FP8 policy are unchanged.

Primary files:

- `cosmos-framework/cosmos_framework/model/vfm/utils/residual_cache.py`
- `cosmos-framework/cosmos_framework/model/vfm/mot/unified_mot.py`
- `cosmos-framework/cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`
- `cosmos-framework/cosmos_framework/model/vfm/omni_mot_model.py`
- `cosmos3_t2v_p4/run_residual_cache.sh`

## Validation

- Python compile checks passed for the full parameter path.
- `git diff --check` passed.
- CPU assertions passed for cache acceptance, threshold rejection, protected-step rejection, and adjacent-step rejection.
- 2-step 720p CUDA smoke accepted step 1 and skipped layers 8-27; MP4 and benchmark were produced successfully.

## 35-step calibration

Calibration candidate steps were `7,9,11,13,15,17,19,21,23,27,29` with threshold `1e-12`, forcing every candidate to refresh while recording the input change:

| Step | Relative L1 |
|---:|---:|
| 7 | 0.069336 |
| 9 | 0.059814 |
| 11 | 0.054443 |
| 13 | 0.050537 |
| 15 | 0.050293 |
| 17 | 0.050537 |
| 19 | 0.052734 |
| 21 | 0.053711 |
| 23 | 0.055176 |
| 27 | 0.064941 |
| 29 | 0.079590 |

The curve is U-shaped. Late CFG-free steps are not automatically safer: change increases after the CFG transition. The balanced candidate therefore excludes the transition and late steps.

Calibration output:

`cosmos3_t2v_p4/outputs/residual_steps_35_reuse_7_9_11_13_15_17_19_21_23_27_29_20260614T184626Z`

## Balanced formal candidate

Configuration:

- Candidate steps: `11,13,15,17,19,21,23`
- Relative-L1 threshold: `0.0545`
- Layers: `8-27`
- Protected first/last steps: 5 / 5
- Explicit protected CFG transition: step 25
- Warmup: one complete 35-step generation

Within the original process, warmup and measured runs made identical decisions:

- Accepted: `11,13,15,17,21`
- Refreshed by threshold: `19,23`
- Skipped block executions: 5 steps x 20 layers = 100 conditional decoder blocks

Formal output:

`cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_20260615T0254Z`

## Performance

| Metric | B0 BF16 | P3 FP8 + CFG | P4 balanced representative | P4 vs B0 | P4 vs P3 |
|---|---:|---:|---:|---:|---:|
| E2E | 481.361 s median | 359.067 s | 344.612 s median | 1.3968x | 1.0420x |
| Generate / denoise | 458.682 s | 336.448 s | 321.809 s | 1.4253x | 1.0455x |
| Conditional forward | 231.918 s | 197.415 s | 182.772 s | 1.2689x | 1.0801x |
| Unconditional forward | 220.726 s | 133.020 s | 133.025 s | 1.6593x | 1.0000x |
| VAE decode | 18.071 s | 18.070 s | 18.065 s | 1.0003x | 1.0002x |
| Peak GPU memory | n/a | 41,233 MiB | 41,233 MiB | n/a | unchanged |

The detailed component values above are from the original representative measured run.
The unchanged unconditional, scheduler, and VAE timings support attribution of the P4
gain to conditional decoder-block reuse.

## Performance stability retest

Three independent processes used the exact formal protocol: one complete 35-step warmup,
then one measured 35-step generation.

| Run | Measured E2E | Conditional | Unconditional | Peak GPU memory | Accepted steps |
|---|---:|---:|---:|---:|---|
| Original | 344.400 s | 182.772 s | 133.025 s | 41,233 MiB | 11,13,15,17,21 |
| Steady repeat 2 | 344.662 s | 182.790 s | 133.031 s | 41,233 MiB | 13,15,17,19,23 |
| Steady repeat 3 | 344.612 s | 182.797 s | 133.046 s | 41,233 MiB | 11,13,15,17,21 |

- Median E2E: **344.612 s**
- Mean E2E: **344.558 s**
- Min / max: **344.400 / 344.662 s**
- Range: **0.262 s** (0.076% of the median)
- Population CV: approximately **0.033%**
- Median speedup: **1.3968x vs B0**, **1.0420x vs P3**

The threshold is close to quantized relative-L1 values, so FP8 process-level numerical
variation can change which boundary candidates are accepted. However, all three processes
accepted exactly five candidates, warmup and measured decisions matched within each process,
and E2E/VRAM were effectively invariant. This is a recorded implementation characteristic,
not hidden as deterministic cache-step selection.

Two additional `WARMUP=0` cold-process runs measured 353.741 s and 354.725 s. They are
retained as cold-start observations and are deliberately excluded from the formal
steady-state median.

Stability outputs:

- `cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_steady_repeat2_20260615T1406Z`
- `cosmos3_t2v_p4/outputs/residual_steps_35_reuse_11_13_15_17_19_21_23_balanced_steady_repeat3_20260615T1418Z`

## Quality gate

Review this file with **left = P3** and **right = P4 balanced**:

`cosmos3_t2v_p4/P3_left_vs_P4_balanced_right_same_noise.mp4`

Static helper:

`cosmos3_t2v_p4/P3_left_vs_P4_balanced_right_contact_sheet.png`

Check prompt semantics, crucible/mold relationship, molten stream continuity, sparks, fine industrial detail, temporal flicker, motion amplitude, blur, and noise. The user reviewed the complete anchor pair on 2026-06-15 and marked P4 quality PASS. The next gate is P3-vs-P4 comparison on the three existing high-risk prompts.

## Reproduction

```bash
env \
  WARMUP=1 \
  STEPS=35 \
  RESIDUAL_CACHE_STEPS=11,13,15,17,19,21,23 \
  RESIDUAL_CACHE_REL_L1_THRESHOLD=0.0545 \
  RESIDUAL_CACHE_PROTECTED_FIRST_STEPS=5 \
  RESIDUAL_CACHE_PROTECTED_LAST_STEPS=5 \
  RESIDUAL_CACHE_PROTECTED_STEPS=25 \
  bash /root/autodl-tmp/cosmos3_t2v_p4/run_residual_cache.sh
```


## Multi-prompt P4 execution

| Prompt | P4 E2E | Accepted / candidates | Quality status |
|---|---:|---:|---|
| human_handoff | 343.507 s | 6/7 | PASS; P4 slightly preferred |
| rally_tracking | 354.685 s | 1/7 | PASS; P4 slightly preferred |
| tabletop_relations | 337.070 s | 7/7 | PASS; P4 slightly preferred |

Comparison files are under `cosmos3_t2v_quality/` with names
`<prompt>_P3_left_vs_P4_right_same_noise.mp4`.


## Aggressive anchor quality note

The aggressive anchor (`steps=9,11,13,15,17,19,21,23`, threshold `0.0600`) reached
335.881 s E2E. Human review judged it basically acceptable with no obvious blur or noise,
but background detail was less fully rendered than balanced. It is not promoted; three
additional balanced-vs-aggressive prompt pairs are required to test whether this is systematic.


## Aggressive multi-prompt background-detail check

Execution and human review complete. Aggressive is rejected for the final configuration because human_handoff shows perceptible edge blur.

| Prompt | Aggressive E2E | Accepted / candidates |
|---|---:|---:|
| human_handoff | 337.092 s | 8/8 |
| rally_tracking | 347.891 s | 5/8 |
| tabletop_relations | 341.616 s | 8/8 |

Balanced-left/aggressive-right comparisons are stored under `cosmos3_t2v_quality/`.


## Final P4 decision

- **Selected:** Balanced, 344.612 s three-process median E2E, 1.3968x over B0.
- **Rejected for final:** Aggressive, 335.881 s formal E2E.
- **Reason:** Aggressive is only 2.54% faster than balanced and causes perceptible edge blur in
  `human_handoff`. `rally_tracking` trades line/color contrast for more realistic physics;
  `tabletop_relations` remains high quality and more physically plausible. These prompt-specific
  benefits do not override the strict no-material-degradation gate.
- **Use of aggressive result:** ablation demonstrating the measured quality boundary.
