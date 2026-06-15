# Cosmos3 Nano T2V Baseline

This directory contains reproducible single-GPU inference runs for the
Cosmos3 16B optimization project.

## Fixed paths

- Framework: `/root/autodl-tmp/cosmos-framework`
- Checkpoint: `/root/autodl-tmp/Cosmos3-Nano`
- Python environment: `/tmp/cosmos-framework-venv`
- Outputs: `/root/autodl-tmp/cosmos3_t2v_baseline/outputs`

The runs use NVIDIA's structured T2V prompt from
`cosmos-framework/inputs/omni/t2v.json`.

## Run order

```bash
bash /root/autodl-tmp/cosmos3_t2v_baseline/verify_env.sh
bash /root/autodl-tmp/cosmos3_t2v_baseline/run_smoke.sh
bash /root/autodl-tmp/cosmos3_t2v_baseline/run_baseline.sh
```

## Baseline definition

- Resolution: 1280x720
- Frames: 121
- FPS: 24
- First-to-last-frame duration: 5.0 seconds
- Sampler: UniPC rectified flow
- Steps: 35
- CFG: 6.0
- Shift: 10.0
- Precision: BF16
- Seed: 0
- Sound generation: disabled by the input/defaults
- Guardrails: disabled for baseline and future optimized runs
- `torch.compile`: enabled, matching the Framework default
- CUDA Graph: disabled, matching the Omni Framework default
- Dynamic compile: enabled, matching the Framework default

The baseline defaults to one same-shape warmup generation. Framework records
warmup timing separately, so the main timing excludes first-use compilation.
Set `WARMUP=0` to record cold-start latency:

```bash
WARMUP=0 RUN_ID=cold \
  bash /root/autodl-tmp/cosmos3_t2v_baseline/run_baseline.sh
```

Every run writes to a new directory unless `RUN_ID` is explicitly supplied.
Keep `benchmark.json`, `console.log`, `debug.log`, `run.log`,
`sample_args.json`, and `vision.mp4` together as one experiment record.
