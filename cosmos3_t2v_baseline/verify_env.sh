#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
FRAMEWORK_ROOT="$ROOT/cosmos-framework"
PYTHON=/tmp/cosmos-framework-venv/bin/python

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing inference environment: $PYTHON" >&2
    exit 1
fi

export COSMOS_TRAINING=0
export LD_LIBRARY_PATH=
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$FRAMEWORK_ROOT"

"$PYTHON" - <<'PY'
import torch
import flash_attn_3_nv
import natten

from cosmos_framework.model.attention.backends import get_backend_list
from cosmos_framework.model.attention.flash3 import FLASH3_SUPPORTED

props = torch.cuda.get_device_properties(0)

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu={props.name}")
print(f"compute_capability={props.major}.{props.minor}")
print(f"vram_gib={props.total_memory / 2**30:.1f}")
print(f"flash3_supported={FLASH3_SUPPORTED}")
print(f"attention_backends={get_backend_list(props.major * 10 + props.minor)}")
print(f"natten={natten.__version__}")
print(f"flash_attn_3_nv={flash_attn_3_nv.__file__}")

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(f"bf16_gemm={tuple(y.shape)}:{y.dtype}")
PY

"$PYTHON" -m cosmos_framework.scripts.inference --help >/dev/null
echo "Cosmos Framework inference CLI: OK"
