#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
FRAMEWORK_ROOT="$ROOT/cosmos-framework"
MODEL_ROOT="$ROOT/Cosmos3-Nano-framework"
EXPERIMENT_ROOT="$ROOT/cosmos3_t2v_baseline"
PYTHON=/tmp/cosmos-framework-venv/bin/python
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="$EXPERIMENT_ROOT/outputs/shape_smoke_$RUN_ID"
GPU_LOG="$OUTPUT_DIR/gpu.csv"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing inference environment: $PYTHON" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

export COSMOS_TRAINING=0
export LD_LIBRARY_PATH=
export PATH="$ROOT/.tools:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/cosmos3-hf-cache
export HF_HUB_OFFLINE=1
export IMAGINAIRE_CACHE_DIR=/tmp/cosmos3-imaginaire-cache

cleanup() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

nvidia-smi \
    --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    --loop=1 >"$GPU_LOG" &
GPU_MONITOR_PID=$!

cd "$FRAMEWORK_ROOT"

"$PYTHON" -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    --checkpoint-path="$MODEL_ROOT" \
    --no-guardrails \
    --no-use-torch-compile \
    --no-use-cuda-graphs \
    --benchmark \
    --seed=0 \
    --defaults-file="$EXPERIMENT_ROOT/configs/shape_smoke_defaults.json" \
    --resolution=720 \
    --aspect-ratio=16,9 \
    --fps=24 \
    --num-frames=121 \
    -i "$FRAMEWORK_ROOT/inputs/omni/t2v.json" \
    -o "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"
