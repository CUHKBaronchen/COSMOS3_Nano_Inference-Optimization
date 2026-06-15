# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from contextlib import contextmanager

import pytest
import torch

from cosmos_framework.model.vfm.diffusion.samplers import unipc


class _FakeScheduler:
    def __init__(self, **_kwargs):
        self.timesteps = torch.empty(0)

    def set_timesteps(self, num_steps, *, device, shift):
        del shift
        self.timesteps = torch.arange(num_steps, 0, -1, device=device)

    def step(self, *, model_output, timestep, sample, return_dict, generator):
        del model_output, timestep, return_dict, generator
        return (sample,)


@pytest.mark.L0
def test_timing_context_wraps_each_scheduler_step(monkeypatch):
    monkeypatch.setattr(unipc, "FlowUniPCMultistepScheduler", _FakeScheduler)
    sampler = unipc.UniPCSampler(tensor_kwargs={"device": "cpu"})
    noise = torch.randn(8)
    regions = []

    @contextmanager
    def timing_context(name):
        regions.append(("enter", name))
        try:
            yield
        finally:
            regions.append(("exit", name))

    result = sampler(
        lambda latent, _timestep: torch.zeros_like(latent),
        noise,
        num_steps=3,
        shift=1.0,
        seed=123,
        timing_context=timing_context,
    )

    torch.testing.assert_close(result, noise)
    assert regions == [
        ("enter", "scheduler"),
        ("exit", "scheduler"),
    ] * 3
