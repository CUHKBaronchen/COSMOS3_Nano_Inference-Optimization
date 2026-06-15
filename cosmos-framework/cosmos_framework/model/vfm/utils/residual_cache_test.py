import pytest
import torch

from cosmos_framework.model.vfm.utils.residual_cache import DiffusionResidualCache, ResidualCacheConfig


def _pack(und: float, gen: float) -> dict:
    return {
        "causal_seq": torch.full((2, 3), und),
        "full_only_seq": torch.full((4, 3), gen),
    }


def test_residual_cache_reuses_selected_non_adjacent_step():
    cache = DiffusionResidualCache(
        ResidualCacheConfig(
            num_steps=12,
            first_layer=1,
            last_layer=2,
            reuse_steps=frozenset({5}),
            input_rel_l1_threshold=0.2,
            protected_first_steps=2,
            protected_last_steps=2,
        )
    )
    for step in range(6):
        cache.begin_forward()
        hidden = _pack(1.0 + step * 0.01, 2.0 + step * 0.01)
        for layer in range(4):
            reused = cache.apply_if_available(layer, hidden)
            if reused is not None:
                hidden = reused
                continue
            output = _pack(
                float(hidden["causal_seq"][0, 0] + 1.0),
                float(hidden["full_only_seq"][0, 0] + 2.0),
            )
            cache.update(layer, hidden, output)
            hidden = output

    assert cache.accepted_steps == [5]
    assert torch.allclose(hidden["causal_seq"], torch.full((2, 3), 5.05))
    assert torch.allclose(hidden["full_only_seq"], torch.full((4, 3), 10.05))


def test_residual_cache_rejects_large_input_change():
    cache = DiffusionResidualCache(
        ResidualCacheConfig(
            num_steps=8,
            first_layer=0,
            last_layer=0,
            reuse_steps=frozenset({3}),
            input_rel_l1_threshold=0.01,
            protected_first_steps=1,
            protected_last_steps=1,
        )
    )
    for gen in (1.0, 1.01, 1.02, 2.0):
        cache.begin_forward()
        hidden = _pack(1.0, gen)
        reused = cache.apply_if_available(0, hidden)
        if reused is None:
            cache.update(0, hidden, _pack(2.0, gen + 1.0))

    assert cache.accepted_steps == []
    assert cache.rejected_steps[0][0] == 3


def test_residual_cache_rejects_protected_or_adjacent_steps():
    with pytest.raises(ValueError, match="protected"):
        ResidualCacheConfig(
            num_steps=10,
            first_layer=1,
            last_layer=2,
            reuse_steps=frozenset({1}),
            input_rel_l1_threshold=0.1,
            protected_first_steps=2,
            protected_last_steps=2,
        )
    with pytest.raises(ValueError, match="separated"):
        ResidualCacheConfig(
            num_steps=10,
            first_layer=1,
            last_layer=2,
            reuse_steps=frozenset({3, 4}),
            input_rel_l1_threshold=0.1,
            protected_first_steps=2,
            protected_last_steps=2,
        )
