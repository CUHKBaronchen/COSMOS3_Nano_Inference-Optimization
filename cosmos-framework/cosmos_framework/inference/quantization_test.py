import torch

from cosmos_framework.inference.quantization import is_selective_generation_linear


def test_selective_generation_linear_filter():
    linear = torch.nn.Linear(16, 16)

    assert is_selective_generation_linear(
        linear,
        "language_model.model.layers.4.self_attn.q_proj_moe_gen",
    )
    assert is_selective_generation_linear(
        linear,
        "language_model.model.layers.31.mlp_moe_gen.down_proj",
    )

    assert not is_selective_generation_linear(
        linear,
        "language_model.model.layers.3.self_attn.q_proj_moe_gen",
    )
    assert not is_selective_generation_linear(
        linear,
        "language_model.model.layers.32.mlp_moe_gen.gate_proj",
    )
    assert not is_selective_generation_linear(
        linear,
        "language_model.model.layers.10.self_attn.k_proj_moe_gen",
    )
    assert not is_selective_generation_linear(
        linear,
        "language_model.model.layers.10.mlp.gate_proj",
    )
    assert not is_selective_generation_linear(
        torch.nn.Identity(),
        "language_model.model.layers.10.mlp_moe_gen.gate_proj",
    )
