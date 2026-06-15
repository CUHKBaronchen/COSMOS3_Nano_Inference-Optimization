# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.vfm.inference_utils import resolve_initial_noise


@pytest.mark.L0
def test_resolve_initial_noise_uses_generated_noise_when_not_provided():
    generated = [torch.randn(8)]

    assert resolve_initial_noise(generated, None) is generated


@pytest.mark.L0
def test_resolve_initial_noise_accepts_one_dimensional_tensor():
    generated = [torch.randn(8, dtype=torch.bfloat16)]
    provided = torch.arange(8, dtype=torch.bfloat16)

    resolved = resolve_initial_noise(generated, provided)

    assert len(resolved) == 1
    torch.testing.assert_close(resolved[0], provided)


@pytest.mark.L0
def test_resolve_initial_noise_accepts_batched_tensor():
    generated = [torch.randn(4), torch.randn(4)]
    provided = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    resolved = resolve_initial_noise(generated, provided)

    assert len(resolved) == 2
    torch.testing.assert_close(torch.stack(resolved), provided)


@pytest.mark.L0
def test_resolve_initial_noise_rejects_invalid_rank():
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        resolve_initial_noise([torch.randn(4)], torch.randn(1, 1, 4))


@pytest.mark.L0
def test_resolve_initial_noise_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="batch size"):
        resolve_initial_noise([torch.randn(4), torch.randn(4)], torch.randn(4))


@pytest.mark.L0
def test_resolve_initial_noise_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="expected 4"):
        resolve_initial_noise([torch.randn(4)], torch.randn(5))


@pytest.mark.L0
def test_resolve_initial_noise_rejects_dtype_mismatch():
    with pytest.raises(ValueError, match="dtype"):
        resolve_initial_noise([torch.randn(4, dtype=torch.bfloat16)], torch.randn(4, dtype=torch.float32))
