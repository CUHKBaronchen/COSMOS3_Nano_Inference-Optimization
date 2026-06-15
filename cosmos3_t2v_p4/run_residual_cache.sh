#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
FRAMEWORK_ROOT="$ROOT/cosmos-framework"
MODEL_ROOT="$ROOT/Cosmos3-Nano-framework"
EXPERIMENT_ROOT="$ROOT/cosmos3_t2v_p4"
PYTHON=/tmp/cosmos-framework-venv/bin/python
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP="${WARMUP:-1}"
STEPS="${STEPS:-35}"
INITIAL_NOISE_FILE="${INITIAL_NOISE_FILE:-$ROOT/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors}"
DEFAULTS_FILE="$EXPERIMENT_ROOT/configs/residual_steps_${STEPS}_defaults.json"
CACHE_TAG="$(printf %s "${RESIDUAL_CACHE_STEPS:-unset}" | tr , _)"
OUTPUT_DIR="$EXPERIMENT_ROOT/outputs/residual_steps_${STEPS}_reuse_${CACHE_TAG}_$RUN_ID"
GPU_LOG="$OUTPUT_DIR/gpu.csv"

if [[ ! -x "$PYTHON" || ! -f "$INITIAL_NOISE_FILE" || ! -f "$DEFAULTS_FILE" ]]; then
    echo "Missing environment, exact noise, or defaults file" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" /tmp/cosmos3-p4-torchinductor-cache /tmp/cosmos3-p4-triton-cache
export COSMOS_TRAINING=0
export COSMOS3_SELECTIVE_FP8=1
export COSMOS3_RESIDUAL_CACHE=1
export COSMOS3_RESIDUAL_CACHE_STEPS="${RESIDUAL_CACHE_STEPS:?set comma-separated zero-based steps}"
export COSMOS3_RESIDUAL_CACHE_FIRST_LAYER="${RESIDUAL_CACHE_FIRST_LAYER:-8}"
export COSMOS3_RESIDUAL_CACHE_LAST_LAYER="${RESIDUAL_CACHE_LAST_LAYER:-27}"
export COSMOS3_RESIDUAL_CACHE_REL_L1_THRESHOLD="${RESIDUAL_CACHE_REL_L1_THRESHOLD:-0.05}"
export COSMOS3_RESIDUAL_CACHE_PROTECTED_FIRST_STEPS="${RESIDUAL_CACHE_PROTECTED_FIRST_STEPS:-5}"
export COSMOS3_RESIDUAL_CACHE_PROTECTED_LAST_STEPS="${RESIDUAL_CACHE_PROTECTED_LAST_STEPS:-5}"
export COSMOS3_RESIDUAL_CACHE_PROTECTED_STEPS="${RESIDUAL_CACHE_PROTECTED_STEPS:-25}"
export LD_LIBRARY_PATH=
export PATH="$ROOT/.tools:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/cosmos3-hf-cache
export HF_HUB_OFFLINE=1
export IMAGINAIRE_CACHE_DIR=/tmp/cosmos3-imaginaire-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/cosmos3-p4-torchinductor-cache
export TRITON_CACHE_DIR=/tmp/cosmos3-p4-triton-cache

cleanup() {
    kill "${GPU_MONITOR_PID:-}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits --loop=1 >"$GPU_LOG" &
GPU_MONITOR_PID=$!

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
    --initial-noise-file="$INITIAL_NOISE_FILE" \
    --warmup="$WARMUP" \
    --seed=0 \
    --defaults-file="$DEFAULTS_FILE" \
    --resolution=720 \
    --aspect-ratio=16,9 \
    --fps=24 \
    --num-frames=121 \
    -i "$FRAMEWORK_ROOT/inputs/omni/t2v.json" \
    -o "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

printf "OUTPUT_DIR=%s\n" "$OUTPUT_DIR"
