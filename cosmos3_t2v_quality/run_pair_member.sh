#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
FRAMEWORK_ROOT="$ROOT/cosmos-framework"
MODEL_ROOT="$ROOT/Cosmos3-Nano-framework"
QUALITY_ROOT="$ROOT/cosmos3_t2v_quality"
PYTHON=/tmp/cosmos-framework-venv/bin/python
PROMPT_ID="${PROMPT_ID:?set PROMPT_ID}"
MODE="${MODE:?set MODE to b0, p3, p4, or p4_aggressive}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP="${WARMUP:-0}"
INPUT_FILE="$QUALITY_ROOT/inputs/$PROMPT_ID.json"
NOISE="$ROOT/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors"
case "$MODE" in
  b0)
    DEFAULTS="$ROOT/cosmos3_t2v_baseline/configs/baseline_defaults.json"
    CACHE=/tmp/cosmos3-torchinductor-cache
    unset COSMOS3_SELECTIVE_FP8 || true
    ;;
  p3)
    DEFAULTS="$ROOT/cosmos3_t2v_p3/configs/fp8_steps_35_defaults.json"
    CACHE=/tmp/cosmos3-fp8-torchinductor-cache
    export COSMOS3_SELECTIVE_FP8=1
    unset COSMOS3_RESIDUAL_CACHE || true
    ;;
  p4)
    DEFAULTS="$ROOT/cosmos3_t2v_p4/configs/residual_steps_35_defaults.json"
    CACHE=/tmp/cosmos3-p4-torchinductor-cache
    export COSMOS3_SELECTIVE_FP8=1
    export COSMOS3_RESIDUAL_CACHE=1
    export COSMOS3_RESIDUAL_CACHE_STEPS=11,13,15,17,19,21,23
    export COSMOS3_RESIDUAL_CACHE_FIRST_LAYER=8
    export COSMOS3_RESIDUAL_CACHE_LAST_LAYER=27
    export COSMOS3_RESIDUAL_CACHE_REL_L1_THRESHOLD=0.0545
    export COSMOS3_RESIDUAL_CACHE_PROTECTED_FIRST_STEPS=5
    export COSMOS3_RESIDUAL_CACHE_PROTECTED_LAST_STEPS=5
    export COSMOS3_RESIDUAL_CACHE_PROTECTED_STEPS=25
    ;;
  p4_aggressive)
    DEFAULTS="$ROOT/cosmos3_t2v_p4/configs/residual_steps_35_defaults.json"
    CACHE=/tmp/cosmos3-p4-torchinductor-cache
    export COSMOS3_SELECTIVE_FP8=1
    export COSMOS3_RESIDUAL_CACHE=1
    export COSMOS3_RESIDUAL_CACHE_STEPS=9,11,13,15,17,19,21,23
    export COSMOS3_RESIDUAL_CACHE_FIRST_LAYER=8
    export COSMOS3_RESIDUAL_CACHE_LAST_LAYER=27
    export COSMOS3_RESIDUAL_CACHE_REL_L1_THRESHOLD=0.0600
    export COSMOS3_RESIDUAL_CACHE_PROTECTED_FIRST_STEPS=5
    export COSMOS3_RESIDUAL_CACHE_PROTECTED_LAST_STEPS=5
    export COSMOS3_RESIDUAL_CACHE_PROTECTED_STEPS=25
    ;;
  *) echo "MODE must be b0, p3, p4, or p4_aggressive" >&2; exit 2 ;;
esac
OUT="$QUALITY_ROOT/outputs/${PROMPT_ID}_${MODE}_$RUN_ID"
mkdir -p "$OUT" "$CACHE" /tmp/cosmos3-triton-cache
export COSMOS_TRAINING=0
export LD_LIBRARY_PATH=
export PATH="$ROOT/.tools:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/cosmos3-hf-cache
export HF_HUB_OFFLINE=1
export IMAGINAIRE_CACHE_DIR=/tmp/cosmos3-imaginaire-cache
export TORCHINDUCTOR_CACHE_DIR="$CACHE"
export TRITON_CACHE_DIR=/tmp/cosmos3-triton-cache
cd "$FRAMEWORK_ROOT"
"$PYTHON" -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --checkpoint-path="$MODEL_ROOT" \
  --no-guardrails \
  --use-torch-compile \
  --no-use-cuda-graphs \
  --compile-dynamic \
  --compiled-region=all \
  --benchmark \
  --initial-noise-file="$NOISE" \
  --warmup="$WARMUP" \
  --seed=0 \
  --defaults-file="$DEFAULTS" \
  --resolution=720 \
  --aspect-ratio=16,9 \
  --fps=24 \
  --num-frames=121 \
  -i "$INPUT_FILE" \
  -o "$OUT" \
  2>&1 | tee "$OUT/run.log"
printf "OUTPUT_DIR=%s\n" "$OUT"
