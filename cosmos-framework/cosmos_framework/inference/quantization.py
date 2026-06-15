import re

import torch

from cosmos_framework.utils import log


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_FP8_GENERATION_LINEAR_SUFFIXES = (
    ".self_attn.q_proj_moe_gen",
    ".self_attn.o_proj_moe_gen",
    ".mlp_moe_gen.gate_proj",
    ".mlp_moe_gen.up_proj",
    ".mlp_moe_gen.down_proj",
)


def is_selective_generation_linear(
    module: torch.nn.Module,
    fqn: str,
    *,
    first_layer: int = 4,
    last_layer: int = 31,
) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    match = _LAYER_PATTERN.search(fqn)
    if match is None:
        return False
    layer_idx = int(match.group(1))
    return first_layer <= layer_idx <= last_layer and fqn.endswith(_FP8_GENERATION_LINEAR_SUFFIXES)


def apply_selective_generation_fp8(
    model: torch.nn.Module,
    *,
    first_layer: int = 4,
    last_layer: int = 31,
) -> int:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

    selected = [
        fqn
        for fqn, module in model.named_modules()
        if is_selective_generation_linear(
            module,
            fqn,
            first_layer=first_layer,
            last_layer=last_layer,
        )
    ]
    if not selected:
        raise RuntimeError("Selective FP8 filter did not match any generation Linear modules")

    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(),
        filter_fn=lambda module, fqn: is_selective_generation_linear(
            module,
            fqn,
            first_layer=first_layer,
            last_layer=last_layer,
        ),
    )
    log.info(
        f"Applied selective generation FP8 to {len(selected)} Linear modules "
        f"in layers {first_layer}-{last_layer}"
    )
    return len(selected)
