#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
FRAMEWORK_ROOT="$ROOT/cosmos-framework"
MODEL_ROOT="$ROOT/Cosmos3-Nano-framework"
EXPERIMENT_ROOT="$ROOT/cosmos3_t2v_baseline"
PYTHON=/tmp/cosmos-framework-venv/bin/python
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP="${WARMUP:-1}"
INITIAL_NOISE_FILE="${INITIAL_NOISE_FILE:-$EXPERIMENT_ROOT/outputs/p0_capture_20260613T114518Z/t2v/initial_noise.safetensors}"

OUTPUT_DIR="$EXPERIMENT_ROOT/outputs/baseline_$RUN_ID"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing inference environment: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$INITIAL_NOISE_FILE" ]]; then
    echo "Missing exact initial noise: $INITIAL_NOISE_FILE" >&2
    exit 1
fi


mkdir -p "$OUTPUT_DIR"
mkdir -p /tmp/cosmos3-torchinductor-cache
mkdir -p /tmp/cosmos3-triton-cache

export COSMOS_TRAINING=0
export LD_LIBRARY_PATH=
export PATH="$ROOT/.tools:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/cosmos3-hf-cache
export HF_HUB_OFFLINE=1
export IMAGINAIRE_CACHE_DIR=/tmp/cosmos3-imaginaire-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/cosmos3-torchinductor-cache
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
    --initial-noise-file="$INITIAL_NOISE_FILE" \
    --warmup="$WARMUP" \
    --seed=0 \
    --defaults-file="$EXPERIMENT_ROOT/configs/baseline_defaults.json" \
    --resolution=720 \
    --aspect-ratio=16,9 \
    --fps=24 \
    --num-frames=121 \
    -i "$FRAMEWORK_ROOT/inputs/omni/t2v.json" \
    -o "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"
