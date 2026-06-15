#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
FRAMEWORK_ROOT="$ROOT/cosmos-framework"
MODEL_ROOT="$ROOT/Cosmos3-Nano-framework"
PYTHON=/tmp/cosmos-framework-venv/bin/python
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODE="${MODE:-smoke}"
WARMUP="${WARMUP:-1}"
INITIAL_NOISE_FILE="${INITIAL_NOISE_FILE:-$ROOT/cosmos3_t2v_baseline/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/cosmos3_t2v_p1/outputs}"

case "$MODE" in
    smoke)
        DEFAULTS_FILE="$FRAMEWORK_ROOT/p1/configs/shape_smoke_defaults.json"
        ;;
    baseline)
        DEFAULTS_FILE="$FRAMEWORK_ROOT/p1/configs/baseline_defaults.json"
        ;;
    *)
        echo "MODE must be 'smoke' or 'baseline', got: $MODE" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/static_${MODE}_$RUN_ID"
GPU_LOG="$OUTPUT_DIR/gpu.csv"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing inference environment: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$INITIAL_NOISE_FILE" ]]; then
    echo "Missing exact initial noise: $INITIAL_NOISE_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" /tmp/cosmos3-p1-static-inductor /tmp/cosmos3-p1-static-triton

export COSMOS_TRAINING=0
export LD_LIBRARY_PATH=
export PATH="$ROOT/.tools:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/cosmos3-hf-cache
export HF_HUB_OFFLINE=1
export IMAGINAIRE_CACHE_DIR=/tmp/cosmos3-imaginaire-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/cosmos3-p1-static-inductor
export TRITON_CACHE_DIR=/tmp/cosmos3-p1-static-triton

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
    --no-compile-dynamic \
    --compiled-region=all \
    --benchmark \
    --warmup="$WARMUP" \
    --seed=999 \
    --initial-noise-file="$INITIAL_NOISE_FILE" \
    --defaults-file="$DEFAULTS_FILE" \
    --resolution=720 \
    --aspect-ratio=16,9 \
    --fps=24 \
    --num-frames=121 \
    -o "$OUTPUT_DIR" \
    -i "$FRAMEWORK_ROOT/inputs/omni/t2v.json" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
