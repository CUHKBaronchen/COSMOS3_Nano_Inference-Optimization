# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch


def resolve_initial_noise(
    generated_noise: list[torch.Tensor], provided_noise: torch.Tensor | list[torch.Tensor] | None
) -> list[torch.Tensor]:
    """Validate and normalize externally supplied flattened initial noise."""
    if provided_noise is None:
        return generated_noise
    if isinstance(provided_noise, torch.Tensor):
        if provided_noise.ndim == 1:
            provided = [provided_noise]
        elif provided_noise.ndim == 2:
            provided = list(torch.unbind(provided_noise, dim=0))
        else:
            raise ValueError(f"initial_noise must be 1D or 2D, got shape {tuple(provided_noise.shape)}")
    else:
        provided = provided_noise
    if len(provided) != len(generated_noise):
        raise ValueError(
            f"initial_noise batch size {len(provided)} does not match generated batch size {len(generated_noise)}"
        )
    resolved: list[torch.Tensor] = []
    for index, (loaded, generated) in enumerate(zip(provided, generated_noise, strict=True)):
        loaded = loaded.reshape(-1)
        if loaded.numel() != generated.numel():
            raise ValueError(
                f"initial_noise[{index}] has {loaded.numel()} values, expected {generated.numel()}"
            )
        if loaded.dtype != generated.dtype:
            raise ValueError(f"initial_noise[{index}] has dtype {loaded.dtype}, expected {generated.dtype}")
        resolved.append(loaded.to(device=generated.device, non_blocking=True).contiguous())
    return resolved
