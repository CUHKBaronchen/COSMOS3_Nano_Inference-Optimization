# Cosmos3 16B T2V Multi-Prompt Quality Gates

## Status

- Experiment execution: complete (3/3 same-noise B0/P3 pairs).
- Human review: 3/3 passed.
- Quality gate: complete.

## Controlled Variables

- GPU: single NVIDIA H20.
- Resolution: 1280x720.
- Duration: 121 frames at 24 fps (5.04 seconds in the encoded files).
- Sampler: UniPC, 35 denoising steps, shift 10.
- Seed: 0.
- Initial noise: the same captured `initial_noise.safetensors` for every B0/P3 pair.
- Initial-noise SHA256: `44122c662d3bd10659534486aa1d455a22ffe6d0c24ffb0dbb5fe7f0553d4b61`.
- B0: BF16, full CFG for all 35 steps.
- P3: selective FP8 on 140 generation-branch Linear modules in layers 4-31, plus `guidance_interval=[800,1000]`.
- VAE, text branch, K/V projections, first/last four transformer layers, attention, norms, and RoPE remain BF16.

## Results

| Prompt | Risk covered | B0 E2E (s) | P3 E2E (s) | Paired speedup | Human quality |
|---|---|---:|---:|---:|---|
| `human_handoff` | Hands, subject count, transfer action | 475.155 | 394.202 | 1.205x | PASS: user reports approximately equal quality |
| `rally_tracking` | Fast motion, tracking camera, parallax | 474.629 | 357.909 | 1.326x | PASS: P3 judged slightly better |
| `tabletop_relations` | Object count, color, spatial relation, manipulation | 474.323 | 359.231 | 1.320x | PASS: approximately equal quality |

The first P3 pair includes a large one-time TorchInductor/Triton FP8 compilation cost in its first denoising step (about 55.9 seconds). The later pairs reuse the compile cache. Therefore these quality-suite E2E values are supporting observations, not the formal performance result. The formal warmed P3 result remains the three-run median of 359.067 seconds, or 1.341x versus the B0 median.

All three same-noise pairs passed human review. The `rally_tracking` result is recorded as a prompt-specific subjective improvement, not as evidence that selective FP8 or interval CFG improves quality on average. Approximate arithmetic and the changed CFG schedule perturb the denoising trajectory; for an individual seed and prompt that perturbation may land on a visually preferable sample even though neither B0 nor P3 is a ground-truth target.

## Comparison Files

Left is B0; right is P3 in every comparison.

- `human_handoff_B0_left_vs_P3_right_same_noise.mp4`
- `rally_tracking_B0_left_vs_P3_right_same_noise.mp4`
- `tabletop_relations_B0_left_vs_P3_right_same_noise.mp4`

Five-timepoint contact sheets:

- `human_handoff_B0_left_vs_P3_right_contact_sheet.png`
- `rally_tracking_B0_left_vs_P3_right_contact_sheet.png`
- `tabletop_relations_B0_left_vs_P3_right_contact_sheet.png`

## Review Criteria

For each pair, inspect the complete video rather than only the contact sheet:

1. Prompt semantics and requested subject/object count.
2. Spatial relations, colors, and requested action.
3. Temporal continuity, motion smoothness, and camera behavior.
4. Faces/hands or rigid object geometry where applicable.
5. Blur, noise, flicker, duplicated objects, sudden topology changes, or other new artifacts.

Do not mark the quality gate complete merely because the P3 result looks plausible in isolation. It must be no materially worse than the paired B0 result under the same initial noise.


## P4 Balanced Residual Cache Gate

Execution and human review are complete for 3/3 P3-vs-P4 same-noise pairs. All three passed; the user judged P4 at least equal and subjectively slightly better overall.
P4 keeps the P3 configuration and adds conditional residual reuse for layers 8-27,
candidate steps `11,13,15,17,19,21,23`, and relative-L1 threshold `0.0545`.
Every candidate is decided dynamically; rejected steps execute the full model.

| Prompt | P3 E2E (s) | P4 E2E (s) | P4 vs P3 | Accepted cache steps | Human quality |
|---|---:|---:|---:|---|---|
| `human_handoff` | 394.202 | 343.507 | 1.148x* | 11,13,15,17,19,21 (6/7) | PASS; P4 slightly preferred |
| `rally_tracking` | 357.909 | 354.685 | 1.009x | 19 (1/7) | PASS; P4 slightly preferred |
| `tabletop_relations` | 359.231 | 337.070 | 1.066x | 11,13,15,17,19,21,23 (7/7) | PASS; P4 slightly preferred |

`*` The earlier `human_handoff` P3 run included a one-time FP8 compile cost, so that pairwise
speedup is not a formal steady-state result. The formal three-process warmed median is 344.612 s,
`1.0420x` over P3 and `1.3968x` over B0.

The prompt-dependent hit rate is intentional. `rally_tracking` has fast motion and camera
tracking; six of seven candidate steps exceeded the threshold and automatically refreshed.
The static tabletop scene accepted all seven. This trades potential speedup for protection
of high-motion trajectories.

Review files, with P3 on the left and P4 on the right:

- `human_handoff_P3_left_vs_P4_right_same_noise.mp4`
- `rally_tracking_P3_left_vs_P4_right_same_noise.mp4`
- `tabletop_relations_P3_left_vs_P4_right_same_noise.mp4`

Matching five-timepoint contact sheets use the same base names ending in
`_contact_sheet.png`. Full-video review remains required for flicker, motion amplitude,
camera behavior, hands, object count, color, and spatial relations.


## P4 Aggressive Background-Detail Check

The anchor aggressive result was basically acceptable but showed weaker background rendering
than balanced. Three same-noise balanced-vs-aggressive pairs were reviewed. Aggressive is
rejected for the final configuration because `human_handoff` has perceptibly blurrier edges.

| Prompt | Balanced E2E | Aggressive E2E | Balanced hits | Aggressive hits | Review |
|---|---:|---:|---:|---:|---|
| human_handoff | 343.507 s | 337.092 s | 6/7 | 8/8 | FAIL: edges visibly softer/blurrier |
| rally_tracking | 354.685 s | 347.891 s | 1/7 | 5/8 | PASS with tradeoff: weaker lines/contrast, more realistic physics |
| tabletop_relations | 337.070 s | 341.616 s | 7/7 | 8/8 | PASS/preferred: both strong, aggressive more physically plausible |

These quality runs use `WARMUP=0`; their E2E values are supporting observations, not formal
steady-state rankings. The formal aggressive anchor result is 335.881 s.

Review files have balanced on the left and aggressive on the right:

- `human_handoff_P4_balanced_left_vs_aggressive_right_same_noise.mp4`
- `rally_tracking_P4_balanced_left_vs_aggressive_right_same_noise.mp4`
- `tabletop_relations_P4_balanced_left_vs_aggressive_right_same_noise.mp4`

Focus on background texture density, distant-scene layering, small-object edges, hand/object
boundaries, motion detail, and whether any simplification is consistent across prompts.


### P4 final selection

Aggressive is retained only as a speed/quality-boundary ablation. Its formal anchor E2E is
335.881 s, only 2.54% faster than balanced, while one of three high-risk prompts develops
perceptible edge blur. Under the assignment quality rule, that trade is not acceptable.

**Final P4 configuration: Balanced** (`threshold=0.0545`, candidates
`11,13,15,17,19,21,23`, layers `8-27`). It passed the anchor and all three high-risk
same-noise comparisons without material degradation and was subjectively slightly preferred
over P3 overall.

Formal performance stability was verified in three independent warmed processes: 344.400,
344.662, and 344.612 seconds E2E. The median is 344.612 seconds and the range is only
0.262 seconds (0.076%). Each process accepted five cache candidates, although FP8 numerical
variation changed which threshold-boundary steps were selected. Full evidence is recorded in
`cosmos3_t2v_p4/P4_STATUS.md`.
