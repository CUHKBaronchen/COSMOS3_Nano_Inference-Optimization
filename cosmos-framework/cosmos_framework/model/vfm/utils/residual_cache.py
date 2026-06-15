# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Conservative cross-timestep residual reuse for diffusion inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.data.vfm.sequence_packing import (
    FactoredSequencePack,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
)
from cosmos_framework.utils import log


@dataclass(frozen=True)
class ResidualCacheConfig:
    num_steps: int
    first_layer: int
    last_layer: int
    reuse_steps: frozenset[int]
    input_rel_l1_threshold: float
    protected_first_steps: int = 5
    protected_last_steps: int = 5
    protected_steps: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if not 0 <= self.first_layer <= self.last_layer:
            raise ValueError(f"Invalid residual-cache layer range: {self.first_layer}-{self.last_layer}")
        if self.input_rel_l1_threshold <= 0:
            raise ValueError("Residual-cache input_rel_l1_threshold must be positive")
        protected_last_start = self.num_steps - self.protected_last_steps
        invalid = sorted(
            step
            for step in self.reuse_steps
            if step < self.protected_first_steps
            or step >= protected_last_start
            or step in self.protected_steps
        )
        if invalid:
            raise ValueError(
                "Residual-cache reuse steps overlap protected first/last steps: "
                f"{invalid}; protected=[0,{self.protected_first_steps}) and "
                f"[{protected_last_start},{self.num_steps}), explicit={sorted(self.protected_steps)}"
            )
        adjacent = sorted(step for step in self.reuse_steps if step - 1 in self.reuse_steps)
        if adjacent:
            raise ValueError(f"Residual-cache reuse steps must be separated by refresh steps: {adjacent}")


class DiffusionResidualCache:
    """Reuse cached decoder-layer residuals on explicitly selected steps."""

    def __init__(self, config: ResidualCacheConfig):
        self.config = config
        self._forward_index = -1
        self._requested = False
        self._active = False
        self._decision_made = False
        self._previous_probe: torch.Tensor | None = None
        self._und_delta: dict[int, torch.Tensor] = {}
        self._gen_delta: dict[int, torch.Tensor] = {}
        self.accepted_steps: list[int] = []
        self.rejected_steps: list[tuple[int, float]] = []

    def begin_forward(self) -> None:
        self._forward_index += 1
        self._requested = self._forward_index in self.config.reuse_steps
        self._active = False
        self._decision_made = False

    def _decide(self, hidden_states: FactoredSequencePack) -> None:
        current_probe = get_gen_seq(hidden_states).detach()
        has_all_layers = all(
            layer in self._und_delta and layer in self._gen_delta
            for layer in range(self.config.first_layer, self.config.last_layer + 1)
        )
        if not self._requested or self._previous_probe is None or not has_all_layers:
            self._active = False
        else:
            denominator = self._previous_probe.abs().mean().clamp_min(1e-6)
            relative_l1 = float(((current_probe - self._previous_probe).abs().mean() / denominator).item())
            self._active = relative_l1 <= self.config.input_rel_l1_threshold
            if self._active:
                self.accepted_steps.append(self._forward_index)
                log.info(
                    f"Residual cache accepted step {self._forward_index}: "
                    f"relative_l1={relative_l1:.6f}"
                )
            else:
                self.rejected_steps.append((self._forward_index, relative_l1))
                log.info(
                    f"Residual cache refreshed step {self._forward_index}: "
                    f"relative_l1={relative_l1:.6f} exceeds "
                    f"{self.config.input_rel_l1_threshold:.6f}"
                )
        self._previous_probe = current_probe.clone()
        self._decision_made = True

    def apply_if_available(
        self,
        layer_idx: int,
        hidden_states: FactoredSequencePack,
    ) -> FactoredSequencePack | None:
        if layer_idx == self.config.first_layer and not self._decision_made:
            self._decide(hidden_states)
        if not self._active or not self.config.first_layer <= layer_idx <= self.config.last_layer:
            return None
        output = dict(hidden_states)
        if "packed_sequence" in output:
            output["packed_sequence"] = output["packed_sequence"].clone()
        set_und_seq(output, get_und_seq(hidden_states) + self._und_delta[layer_idx])
        set_gen_seq(output, get_gen_seq(hidden_states) + self._gen_delta[layer_idx])
        return output

    def update(
        self,
        layer_idx: int,
        layer_input: FactoredSequencePack,
        layer_output: FactoredSequencePack,
    ) -> None:
        if self._active or not self.config.first_layer <= layer_idx <= self.config.last_layer:
            return
        self._und_delta[layer_idx] = (get_und_seq(layer_output) - get_und_seq(layer_input)).detach()
        self._gen_delta[layer_idx] = (get_gen_seq(layer_output) - get_gen_seq(layer_input)).detach()

    def summary(self) -> str:
        rejected = ",".join(f"{step}:{value:.6f}" for step, value in self.rejected_steps) or "none"
        return (
            f"accepted_steps={self.accepted_steps}, rejected_steps={rejected}, "
            f"layers={self.config.first_layer}-{self.config.last_layer}"
        )
