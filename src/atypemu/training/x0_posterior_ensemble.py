"""Differentiable x0-support posterior ensemble components.

This module keeps the scientific target explicit: a BioEmu x0 support is first
interpreted as a finite empirical measure, then NMR evidence changes the
population weights over that support.  The conformer-level chemical-shift
decoder predicts observables per support point; supervised BMRB losses are
applied to the weighted ensemble observable, not to an unconstrained direct
predictor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


ATOM_FAMILY_NAMES: tuple[str, ...] = ("HN", "N", "CA", "CB", "C'")
ATOM_FAMILY_PPM_CENTERS: tuple[float, ...] = (8.0, 117.5, 52.5, 45.0, 170.0)
ATOM_FAMILY_PPM_SCALES: tuple[float, ...] = (2.0, 18.75, 16.25, 22.5, 10.0)
COMPACT_X2D_EDGE_CLASSES: tuple[str, ...] = (
    "sequence",
    "peptide_plane",
    "spatial_contact",
    "hbond",
    "ring",
    "electrostatic",
)


@dataclass(frozen=True)
class X0PosteriorRegularizationConfig:
    """Collapse guards for a posterior measure over a fixed x0 support."""

    ess_floor: float = 0.0
    entropy_floor: float = 0.0
    top_mass_cap: float = 1.0
    prior_kl_weight: float = 0.0
    diversity_floor: float = 0.0
    diversity_weight: float = 0.0


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    depth: int = 2,
    dropout: float = 0.0,
    zero_init_output: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(float(dropout)))
    for _ in range(max(int(depth) - 1, 0)):
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(hidden_dim, output_dim))
    network = nn.Sequential(*layers)
    if zero_init_output:
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
    return network


def _has_trainable_parameters(module: nn.Module | None) -> bool:
    """Return whether a module has parameters that need autograd activations."""

    return module is not None and any(parameter.requires_grad for parameter in module.parameters())


def _module_or_inputs_require_grad(module: nn.Module | None, *inputs: object) -> bool:
    """Return whether checkpointing can save activations for params or inputs."""

    if _has_trainable_parameters(module):
        return True
    return any(torch.is_tensor(value) and value.requires_grad for value in inputs)


def _support_chunk_axis(features: torch.Tensor) -> int:
    """Return the support axis for adapter inputs shaped ``[K,L,D]`` or ``[B,K,L,D]``."""

    if features.ndim >= 4:
        return features.ndim - 3
    return 0


def _iter_support_chunks(
    features: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[int, list[torch.Tensor]]:
    support_axis = _support_chunk_axis(features)
    support_size = int(features.shape[support_axis])
    if chunk_size <= 0 or support_size <= int(chunk_size):
        return support_axis, []
    chunks = [
        features.narrow(support_axis, start, min(int(chunk_size), support_size - start))
        for start in range(0, support_size, int(chunk_size))
    ]
    return support_axis, chunks


def _iter_contiguous_index_runs(
    indices: torch.Tensor,
    *,
    chunk_size: int,
) -> list[tuple[int, int, int]]:
    """Return ``(offset, start, width)`` runs suitable for ``Tensor.narrow``."""

    limit = max(int(chunk_size), 1)
    values = [int(value) for value in indices.detach().cpu().tolist()]
    runs: list[tuple[int, int, int]] = []
    offset = 0
    while offset < len(values):
        start = values[offset]
        width = 1
        while (
            width < limit
            and offset + width < len(values)
            and values[offset + width] == start + width
        ):
            width += 1
        runs.append((offset, start, width))
        offset += width
    return runs


def _compact_diagnostic_weights(weights: torch.Tensor) -> torch.Tensor:
    """Keep only small gate diagnostics instead of full ``K x L`` expert maps."""

    if weights.ndim <= 1:
        return weights.detach()
    reduce_dims = tuple(range(weights.ndim - 1))
    return weights.detach().mean(dim=reduce_dims, keepdim=True)


class GatedCapacityAdapter(nn.Module):
    """Mixture-of-experts adapter for conformer-level CS support shifts."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        depth: int = 3,
        expert_count: int = 2,
        dropout: float = 0.0,
        family_gated: bool = False,
        output_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.expert_count = max(int(expert_count), 1)
        self.output_dim = max(int(output_dim), 1)
        self.family_gated = bool(family_gated)
        init_std = max(float(output_init_std), 0.0)
        self.experts = nn.ModuleList(
            []
        )
        for _ in range(self.expert_count):
            expert = _make_mlp(
                input_dim,
                hidden_dim,
                self.output_dim,
                depth=depth,
                dropout=dropout,
                zero_init_output=init_std <= 0.0,
            )
            if init_std > 0.0:
                nn.init.normal_(expert[-1].weight, mean=0.0, std=init_std)
                nn.init.zeros_(expert[-1].bias)
            self.experts.append(expert)
        gate_hidden_dim = max(8, min(int(hidden_dim), 512))
        self.gate = _make_mlp(
            input_dim,
            gate_hidden_dim,
            self.expert_count * self.output_dim
            if self.family_gated
            else self.expert_count,
            depth=1,
            dropout=dropout,
            zero_init_output=True,
        )

    @staticmethod
    def _run_module(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
        use_checkpoint = (
            torch.is_grad_enabled()
            and _module_or_inputs_require_grad(module, features)
        )
        if use_checkpoint:
            return checkpoint(module, features, use_reentrant=False)
        return module(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta, _ = self.forward_with_weights(features)
        return delta

    def forward_with_weights(
        self,
        features: torch.Tensor,
        *,
        chunk_size: int = 0,
        compact_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_axis, feature_chunks = _iter_support_chunks(
            features,
            chunk_size=int(chunk_size),
        )
        if feature_chunks:
            delta_chunks: list[torch.Tensor] = []
            weight_chunks: list[torch.Tensor] = []
            for chunk in feature_chunks:
                chunk_delta, chunk_weights = self.forward_with_weights(
                    chunk,
                    chunk_size=0,
                    compact_weights=compact_weights,
                )
                delta_chunks.append(chunk_delta)
                weight_chunks.append(chunk_weights)
            return torch.cat(delta_chunks, dim=support_axis), torch.cat(
                weight_chunks,
                dim=0 if compact_weights else support_axis,
            )
        if self.expert_count == 1:
            if self.family_gated:
                weights = features.new_ones(*features.shape[:-1], self.output_dim, 1)
            else:
                weights = features.new_ones(*features.shape[:-1], 1)
            if compact_weights:
                weights = _compact_diagnostic_weights(weights)
            return self._run_module(self.experts[0], features), weights
        gate_logits = self._run_module(self.gate, features)
        if self.family_gated:
            gate_logits = gate_logits.view(
                *features.shape[:-1],
                self.output_dim,
                self.expert_count,
            )
        weights = torch.softmax(gate_logits, dim=-1)
        weighted_output: torch.Tensor | None = None
        for expert_index, expert in enumerate(self.experts):
            expert_output = self._run_module(expert, features)
            if self.family_gated:
                expert_weight = weights[..., expert_index]
            else:
                expert_weight = weights[..., expert_index : expert_index + 1]
            weighted_expert = expert_output * expert_weight
            weighted_output = (
                weighted_expert
                if weighted_output is None
                else weighted_output + weighted_expert
            )
        if weighted_output is None:
            weighted_output = features.new_zeros(*features.shape[:-1], 0)
        # The returned weights are diagnostics-only.  Detaching them avoids
        # retaining a second copy of the gate graph for very wide adapters.
        if compact_weights:
            return weighted_output, _compact_diagnostic_weights(weights)
        return weighted_output, weights.detach()


class FamilySpecificCapacityAdapter(nn.Module):
    """Separate expert adapters per atom family to avoid shared-direction leaks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        depth: int = 3,
        expert_count: int = 2,
        dropout: float = 0.0,
        output_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = max(int(output_dim), 1)
        self.expert_count = max(int(expert_count), 1)
        self.family_adapters = nn.ModuleList(
            [
                GatedCapacityAdapter(
                    input_dim,
                    hidden_dim,
                    1,
                    depth=depth,
                    expert_count=self.expert_count,
                    dropout=dropout,
                    output_init_std=output_init_std,
                )
                for _ in range(self.output_dim)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta, _ = self.forward_with_weights(features)
        return delta

    def forward_with_weights(
        self,
        features: torch.Tensor,
        *,
        chunk_size: int = 0,
        compact_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        for adapter in self.family_adapters:
            family_delta, family_weights = adapter.forward_with_weights(
                features,
                chunk_size=chunk_size,
                compact_weights=compact_weights,
            )
            delta_chunks.append(family_delta)
            weight_chunks.append(family_weights)
        return torch.cat(delta_chunks, dim=-1), torch.stack(weight_chunks, dim=-2)


class FamilyHeadedCapacityAdapter(nn.Module):
    """Large shared trunk with family-specific expert heads.

    This keeps most of the extra capacity in one shared representation pass, then
    lets each atom family choose its own expert mixture.  It is a middle path
    between fast shared experts and expensive fully family-specific MLP banks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        depth: int = 3,
        expert_count: int = 2,
        dropout: float = 0.0,
        output_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = max(int(output_dim), 1)
        self.expert_count = max(int(expert_count), 1)
        self.trunk = _make_mlp(
            input_dim,
            hidden_dim,
            hidden_dim,
            depth=max(int(depth), 1),
            dropout=dropout,
        )
        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.output_dim) for _ in range(self.expert_count)]
        )
        init_std = max(float(output_init_std), 0.0)
        for head in self.expert_heads:
            if init_std > 0.0:
                nn.init.normal_(head.weight, mean=0.0, std=init_std)
            else:
                nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        gate_hidden_dim = max(8, min(int(hidden_dim), 512))
        self.gate = _make_mlp(
            hidden_dim,
            gate_hidden_dim,
            self.output_dim * self.expert_count,
            depth=1,
            dropout=dropout,
            zero_init_output=True,
        )

    @staticmethod
    def _run_module(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
        use_checkpoint = (
            torch.is_grad_enabled()
            and _module_or_inputs_require_grad(module, features)
        )
        if use_checkpoint:
            return checkpoint(module, features, use_reentrant=False)
        return module(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta, _ = self.forward_with_weights(features)
        return delta

    def forward_with_weights(
        self,
        features: torch.Tensor,
        *,
        chunk_size: int = 0,
        compact_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_axis, feature_chunks = _iter_support_chunks(
            features,
            chunk_size=int(chunk_size),
        )
        if feature_chunks:
            delta_chunks: list[torch.Tensor] = []
            weight_chunks: list[torch.Tensor] = []
            for chunk in feature_chunks:
                chunk_delta, chunk_weights = self.forward_with_weights(
                    chunk,
                    chunk_size=0,
                    compact_weights=compact_weights,
                )
                delta_chunks.append(chunk_delta)
                weight_chunks.append(chunk_weights)
            return torch.cat(delta_chunks, dim=support_axis), torch.cat(
                weight_chunks,
                dim=0 if compact_weights else support_axis,
            )
        hidden = self._run_module(self.trunk, features)
        if self.expert_count == 1:
            weights = features.new_ones(*features.shape[:-1], self.output_dim, 1)
            if compact_weights:
                weights = _compact_diagnostic_weights(weights)
            return self._run_module(self.expert_heads[0], hidden), weights
        gate_logits = self._run_module(self.gate, hidden).view(
            *features.shape[:-1],
            self.output_dim,
            self.expert_count,
        )
        weights = torch.softmax(gate_logits, dim=-1)
        weighted_output: torch.Tensor | None = None
        for expert_index, head in enumerate(self.expert_heads):
            expert_output = self._run_module(head, hidden)
            weighted_expert = expert_output * weights[..., expert_index]
            weighted_output = (
                weighted_expert
                if weighted_output is None
                else weighted_output + weighted_expert
            )
        if weighted_output is None:
            weighted_output = features.new_zeros(*features.shape[:-1], 0)
        # The returned weights are diagnostics-only.  Detaching them avoids
        # retaining a second copy of the gate graph for very wide adapters.
        if compact_weights:
            return weighted_output, _compact_diagnostic_weights(weights)
        return weighted_output, weights.detach()


class FactorizedFamilyHeadedCapacityAdapter(nn.Module):
    """Very-wide family-headed adapter with low-rank residual trunk blocks.

    A dense 16k-wide MLP would make the hidden-to-hidden layers quadratic in the
    requested width.  This adapter keeps the large representation width, but
    routes each refinement block through a much smaller bottleneck so the
    capacity increase is feasible on a single L40S.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        factorized_rank: int,
        depth: int = 3,
        expert_count: int = 2,
        dropout: float = 0.0,
        output_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = max(int(output_dim), 1)
        self.expert_count = max(int(expert_count), 1)
        self.factorized_rank = max(int(factorized_rank), 1)
        input_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            input_layers.append(nn.Dropout(float(dropout)))
        self.input_projection = nn.Sequential(*input_layers)
        self.blocks = nn.ModuleList()
        for _ in range(max(int(depth), 1)):
            block_layers: list[nn.Module] = [
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.factorized_rank),
                nn.GELU(),
            ]
            if dropout > 0.0:
                block_layers.append(nn.Dropout(float(dropout)))
            up_projection = nn.Linear(self.factorized_rank, hidden_dim)
            nn.init.zeros_(up_projection.weight)
            nn.init.zeros_(up_projection.bias)
            block_layers.append(up_projection)
            self.blocks.append(nn.Sequential(*block_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.output_dim) for _ in range(self.expert_count)]
        )
        init_std = max(float(output_init_std), 0.0)
        for head in self.expert_heads:
            if init_std > 0.0:
                nn.init.normal_(head.weight, mean=0.0, std=init_std)
            else:
                nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        gate_hidden_dim = max(8, min(int(hidden_dim), 512))
        self.gate = _make_mlp(
            hidden_dim,
            gate_hidden_dim,
            self.output_dim * self.expert_count,
            depth=1,
            dropout=dropout,
            zero_init_output=True,
        )

    @staticmethod
    def _run_module(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
        use_checkpoint = (
            torch.is_grad_enabled()
            and _module_or_inputs_require_grad(module, features)
        )
        if use_checkpoint:
            return checkpoint(module, features, use_reentrant=False)
        return module(features)

    def _trunk(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self._run_module(self.input_projection, features)
        for block in self.blocks:
            hidden = hidden + self._run_module(block, hidden)
        return self._run_module(self.output_norm, hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta, _ = self.forward_with_weights(features)
        return delta

    def forward_with_weights(
        self,
        features: torch.Tensor,
        *,
        chunk_size: int = 0,
        compact_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_axis, feature_chunks = _iter_support_chunks(
            features,
            chunk_size=int(chunk_size),
        )
        if feature_chunks:
            delta_chunks: list[torch.Tensor] = []
            weight_chunks: list[torch.Tensor] = []
            for chunk in feature_chunks:
                chunk_delta, chunk_weights = self.forward_with_weights(
                    chunk,
                    chunk_size=0,
                    compact_weights=compact_weights,
                )
                delta_chunks.append(chunk_delta)
                weight_chunks.append(chunk_weights)
            return torch.cat(delta_chunks, dim=support_axis), torch.cat(
                weight_chunks,
                dim=0 if compact_weights else support_axis,
            )
        hidden = self._trunk(features)
        if self.expert_count == 1:
            weights = features.new_ones(*features.shape[:-1], self.output_dim, 1)
            if compact_weights:
                weights = _compact_diagnostic_weights(weights)
            return self._run_module(self.expert_heads[0], hidden), weights
        gate_logits = self._run_module(self.gate, hidden).view(
            *features.shape[:-1],
            self.output_dim,
            self.expert_count,
        )
        weights = torch.softmax(gate_logits, dim=-1)
        weighted_output: torch.Tensor | None = None
        for expert_index, head in enumerate(self.expert_heads):
            expert_output = self._run_module(head, hidden)
            weighted_expert = expert_output * weights[..., expert_index]
            weighted_output = (
                weighted_expert
                if weighted_output is None
                else weighted_output + weighted_expert
            )
        if weighted_output is None:
            weighted_output = features.new_zeros(*features.shape[:-1], 0)
        # The returned weights are diagnostics-only.  Detaching them avoids
        # retaining a second copy of the gate graph for very wide adapters.
        if compact_weights:
            return weighted_output, _compact_diagnostic_weights(weights)
        return weighted_output, weights.detach()


class FactorizedEnergyExtraHead(nn.Module):
    """Wide posterior-energy residual head with low-rank refinement blocks.

    Dense very-wide MLPs become quadratic in the requested hidden dimension.
    This head lets us increase representational width while keeping the
    hidden-to-hidden path low-rank enough for single-L40S experiments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        factorized_rank: int,
        depth: int = 2,
        dropout: float = 0.0,
        output_init_std: float = 0.0,
        shared_block_count: int = 0,
        chunk_size: int = 0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.factorized_rank = max(int(factorized_rank), 1)
        self.virtual_depth = max(int(depth), 1)
        self.shared_block_count = max(int(shared_block_count), 0)
        self.chunk_size = max(int(chunk_size), 0)
        self.use_checkpoint = bool(use_checkpoint)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        self.input_projection = nn.Sequential(*layers)
        self.blocks = nn.ModuleList()
        unique_block_count = self.virtual_depth
        if self.shared_block_count > 0:
            unique_block_count = min(self.shared_block_count, self.virtual_depth)
        self.actual_block_count = unique_block_count
        for _ in range(unique_block_count):
            block_layers: list[nn.Module] = [
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.factorized_rank),
                nn.GELU(),
            ]
            if dropout > 0.0:
                block_layers.append(nn.Dropout(float(dropout)))
            up_projection = nn.Linear(self.factorized_rank, hidden_dim)
            nn.init.zeros_(up_projection.weight)
            nn.init.zeros_(up_projection.bias)
            block_layers.append(up_projection)
            self.blocks.append(nn.Sequential(*block_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1)
        init_std = max(float(output_init_std), 0.0)
        if init_std > 0.0:
            nn.init.normal_(self.output_head.weight, mean=0.0, std=init_std)
        else:
            nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def _run_module(self, module: nn.Module, features: torch.Tensor) -> torch.Tensor:
        use_checkpoint = (
            self.use_checkpoint
            and
            torch.is_grad_enabled()
            and _module_or_inputs_require_grad(module, features)
        )
        if use_checkpoint:
            return checkpoint(module, features, use_reentrant=False)
        return module(features)

    def _forward_no_chunk(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self._run_module(self.input_projection, features)
        for block_index in range(self.virtual_depth):
            block = self.blocks[block_index % self.actual_block_count]
            hidden = hidden + self._run_module(block, hidden)
        hidden = self._run_module(self.output_norm, hidden)
        return self._run_module(self.output_head, hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.chunk_size)
        if chunk_size > 0 and int(features.shape[0]) > chunk_size:
            chunks = [
                self._forward_no_chunk(features[start : start + chunk_size])
                for start in range(0, int(features.shape[0]), chunk_size)
            ]
            return torch.cat(chunks, dim=0)
        return self._forward_no_chunk(features)


def _resolve_attention_heads(hidden_dim: int, requested_heads: int) -> int:
    hidden = max(int(hidden_dim), 1)
    requested = max(int(requested_heads), 1)
    for heads in range(min(hidden, requested), 0, -1):
        if hidden % heads == 0:
            return heads
    return 1


class SupportSetEnergyAttentionHead(nn.Module):
    """Permutation-equivariant support attention for conformer energy ranking.

    The head reads the full fixed x0 support at once and returns one residual
    energy per conformer.  It never changes the exported population contract:
    downstream still receives a single ``softmax(log prior - E / T)`` shared q.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        head_count: int = 4,
        depth: int = 1,
        dropout: float = 0.0,
        output_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = max(int(hidden_dim), 1)
        heads = _resolve_attention_heads(hidden, int(head_count))
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.input_projection = nn.Linear(int(input_dim), hidden)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn_norm": nn.LayerNorm(hidden),
                        "attn": nn.MultiheadAttention(
                            hidden,
                            heads,
                            dropout=float(dropout),
                            batch_first=True,
                        ),
                        "ffn": _make_mlp(
                            hidden,
                            max(hidden * 2, 1),
                            hidden,
                            depth=1,
                            dropout=dropout,
                            zero_init_output=False,
                        ),
                    }
                )
                for _ in range(max(int(depth), 1))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, 1)
        init_std = max(float(output_init_std), 0.0)
        if init_std > 0.0:
            nn.init.normal_(self.output_head.weight, mean=0.0, std=init_std)
        else:
            nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self.input_norm(features))
        for block in self.blocks:
            attn_input = block["attn_norm"](hidden).unsqueeze(0)
            attn_output, _ = block["attn"](
                attn_input,
                attn_input,
                attn_input,
                need_weights=False,
            )
            hidden = hidden + attn_output.squeeze(0)
            hidden = hidden + block["ffn"](hidden)
        return self.output_head(self.output_norm(hidden))


def resolve_factorized_capacity_dims(
    *,
    requested_hidden_dim: int,
    requested_rank: int,
    depth: int,
    effective_hidden_dim: int = 0,
    effective_rank: int = 0,
    max_factorized_params: int = 450_000_000,
) -> tuple[int, int]:
    """Resolve feasible dimensions for very-wide low-rank adapter blocks."""

    requested_hidden = max(int(requested_hidden_dim), 1)
    requested_factor_rank = max(int(requested_rank), 1)
    explicit_hidden = max(int(effective_hidden_dim), 0)
    explicit_rank = max(int(effective_rank), 0)
    block_depth = max(int(depth), 1)
    product_budget = max(int(max_factorized_params) // max(2 * block_depth + 2, 1), 1)
    if explicit_hidden > 0 or explicit_rank > 0:
        hidden = explicit_hidden if explicit_hidden > 0 else requested_hidden
        rank = explicit_rank if explicit_rank > 0 else requested_factor_rank
        hidden = max(int(hidden), 1)
        rank = max(int(rank), 1)
        if hidden * rank <= product_budget:
            return hidden, rank
        requested_hidden = hidden
        requested_factor_rank = rank

    if requested_hidden * requested_factor_rank <= product_budget:
        return requested_hidden, requested_factor_rank

    rank_ratio = requested_factor_rank / max(float(requested_hidden), 1.0)
    hidden = int((product_budget / max(rank_ratio, 1.0e-9)) ** 0.5)
    hidden = max(256, (hidden // 64) * 64)
    rank = max(16, int(hidden * rank_ratio) // 16 * 16)
    while hidden * rank > product_budget and hidden > 256 and rank > 16:
        hidden = max(256, hidden - 64)
        rank = max(16, int(hidden * rank_ratio) // 16 * 16)
    return min(hidden, requested_hidden), min(rank, requested_factor_rank)


def posterior_weights(
    prior_log_probs: torch.Tensor | None,
    energy: torch.Tensor,
    *,
    temperature: float = 1.0,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``softmax(log p_BioEmu - E_theta / temperature)`` over support.

    ``energy`` may be shaped ``[K]`` or ``[B, K]``.  If no prior logits are
    provided, a uniform BioEmu prior is assumed.
    """

    temperature_value = max(float(temperature), 1.0e-6)
    safe_energy = torch.nan_to_num(
        energy,
        nan=0.0,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    logits = -safe_energy / temperature_value
    if prior_log_probs is not None:
        prior_logits = prior_log_probs.to(device=energy.device, dtype=energy.dtype)
        prior_logits = torch.nan_to_num(
            prior_logits,
            nan=0.0,
            posinf=80.0,
            neginf=-80.0,
        )
        logits = logits + prior_logits
    logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
    logits = logits.clamp(min=-80.0, max=80.0)
    if support_mask is not None:
        mask = support_mask.to(device=energy.device, dtype=torch.bool)
        logits = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    return torch.softmax(logits, dim=-1)


def weighted_ensemble_observable(
    conformer_observable: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute ``sum_k q_k phi(X_k)`` for a support-first observable tensor."""

    if weights.ndim == 1 and conformer_observable.shape[0] == weights.shape[0]:
        view_shape = (weights.shape[0],) + (1,) * (conformer_observable.ndim - 1)
        return torch.sum(conformer_observable * weights.reshape(view_shape), dim=0)
    if (
        weights.ndim == 2
        and conformer_observable.ndim >= 3
        and conformer_observable.shape[:2] == weights.shape
    ):
        view_shape = weights.shape + (1,) * (conformer_observable.ndim - 2)
        return torch.sum(conformer_observable * weights.reshape(view_shape), dim=1)
    raise ValueError(
        "weights must match the support dimension of conformer_observable "
        f"(got observable={tuple(conformer_observable.shape)}, "
        f"weights={tuple(weights.shape)})"
    )


def posterior_diagnostics(
    weights: torch.Tensor,
    *,
    prior_weights: torch.Tensor | None = None,
    support_distances: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return ESS, entropy, top mass, KL-to-prior, and optional diversity."""

    safe_weights = weights.clamp_min(1.0e-12)
    entropy = -torch.sum(safe_weights * torch.log(safe_weights), dim=-1)
    ess = 1.0 / torch.sum(weights.square(), dim=-1).clamp_min(1.0e-12)
    top_mass = torch.amax(weights, dim=-1)
    if prior_weights is None:
        prior = torch.full_like(weights, 1.0 / max(int(weights.shape[-1]), 1))
    else:
        prior = prior_weights.to(device=weights.device, dtype=weights.dtype)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    prior_kl = torch.sum(safe_weights * (torch.log(safe_weights) - torch.log(prior.clamp_min(1.0e-12))), dim=-1)
    diagnostics: dict[str, torch.Tensor] = {
        "ess": ess,
        "entropy": entropy,
        "top_mass": top_mass,
        "prior_kl": prior_kl,
    }
    if support_distances is not None:
        distances = support_distances.to(device=weights.device, dtype=weights.dtype)
        diagnostics["diversity"] = torch.einsum("...i,ij,...j->...", weights, distances, weights)
    return diagnostics


def posterior_regularization_terms(
    weights: torch.Tensor,
    *,
    config: X0PosteriorRegularizationConfig,
    prior_weights: torch.Tensor | None = None,
    support_distances: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build differentiable anti-collapse penalties for posterior weights."""

    diagnostics = posterior_diagnostics(
        weights,
        prior_weights=prior_weights,
        support_distances=support_distances,
    )
    zero = weights.new_tensor(0.0)
    terms: dict[str, torch.Tensor] = dict(diagnostics)
    terms["ess_floor_loss"] = torch.relu(
        weights.new_tensor(float(config.ess_floor)) - diagnostics["ess"]
    ).mean()
    terms["entropy_floor_loss"] = torch.relu(
        weights.new_tensor(float(config.entropy_floor)) - diagnostics["entropy"]
    ).mean()
    terms["top_mass_cap_loss"] = torch.relu(
        diagnostics["top_mass"] - weights.new_tensor(float(config.top_mass_cap))
    ).mean()
    terms["prior_kl_loss"] = diagnostics["prior_kl"].mean() * float(config.prior_kl_weight)
    if "diversity" in diagnostics:
        terms["diversity_floor_loss"] = (
            torch.relu(
                weights.new_tensor(float(config.diversity_floor))
                - diagnostics["diversity"]
            ).mean()
            * float(config.diversity_weight)
        )
    else:
        terms["diversity_floor_loss"] = zero
    terms["collapse_guard_loss"] = (
        terms["ess_floor_loss"]
        + terms["entropy_floor_loss"]
        + terms["top_mass_cap_loss"]
        + terms["prior_kl_loss"]
        + terms["diversity_floor_loss"]
    )
    return terms


class CompactX2DEdgeEncoder(nn.Module):
    """Pool compact x2d/geometry edges into residue-local mechanism context.

    The input is an edge list, not a full ``K x L x L x d`` tensor.  This is the
    implementation boundary that keeps sequence-local and global-topology
    neighbors available without materializing full pair trajectories.
    """

    def __init__(
        self,
        edge_feature_dim: int,
        hidden_dim: int,
        *,
        edge_class_count: int = len(COMPACT_X2D_EDGE_CLASSES),
        dropout: float = 0.0,
        preserve_class_channels: bool = False,
        class_channel_dim: int | None = None,
        sample_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_class_count = max(int(edge_class_count), 1)
        self.preserve_class_channels = bool(preserve_class_channels)
        self.sample_chunk_size = max(int(sample_chunk_size), 0)
        self.edge_projection = _make_mlp(
            max(int(edge_feature_dim), 1),
            hidden_dim,
            hidden_dim,
            depth=2,
            dropout=dropout,
        )
        self.edge_class_embedding = nn.Embedding(self.edge_class_count, hidden_dim)
        self.class_channel_dim = 0
        self.class_projection: nn.Module | None = None
        if self.preserve_class_channels:
            default_class_dim = max(16, min(64, int(hidden_dim) // 4))
            self.class_channel_dim = max(
                int(class_channel_dim if class_channel_dim is not None else default_class_dim),
                1,
            )
            self.class_projection = nn.Linear(hidden_dim, self.class_channel_dim)
        self.output_dim = self.hidden_dim + self.edge_class_count * self.class_channel_dim

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_target_indices: torch.Tensor,
        edge_class_indices: torch.Tensor,
        *,
        residue_count: int,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled context with shape ``[K, L, hidden_dim]``."""

        if edge_features.ndim != 3:
            raise ValueError("edge_features must have shape [K, E, edge_feature_dim]")
        sample_count, edge_count, _ = edge_features.shape
        residue_count = max(int(residue_count), 1)
        if edge_count == 0:
            return edge_features.new_zeros(sample_count, residue_count, self.output_dim)
        targets = edge_target_indices.to(device=edge_features.device).long()
        targets = targets.clamp(min=0, max=residue_count - 1)
        classes = edge_class_indices.to(device=edge_features.device).long()
        classes = classes.clamp(min=0, max=self.edge_class_embedding.num_embeddings - 1)
        edge_weights = (
            edge_features.new_ones(edge_count)
            if edge_mask is None
            else edge_mask.to(device=edge_features.device, dtype=edge_features.dtype)
        )
        counts = edge_features.new_zeros(residue_count)
        counts.index_add_(0, targets, edge_weights)
        chunk_size = int(self.sample_chunk_size)
        if chunk_size > 0 and sample_count > chunk_size:
            return self._forward_streamed_by_sample_chunk(
                edge_features,
                targets,
                classes,
                edge_weights,
                counts,
                residue_count=residue_count,
                chunk_size=chunk_size,
            )

        messages = self.edge_projection(edge_features) + self.edge_class_embedding(
            classes
        ).unsqueeze(0)
        if edge_mask is not None:
            messages = messages * edge_weights.reshape(1, edge_count, 1)
        context = edge_features.new_zeros(sample_count, residue_count, self.hidden_dim)
        context.index_add_(1, targets, messages)
        context = context / counts.clamp_min(1.0).reshape(1, residue_count, 1)
        if not self.preserve_class_channels or self.class_projection is None:
            return context

        class_messages = self.class_projection(messages)
        class_contexts: list[torch.Tensor] = []
        for class_index in range(self.edge_class_count):
            class_selector = (classes == class_index).to(dtype=edge_features.dtype)
            if edge_mask is not None:
                class_selector = class_selector * edge_mask.to(
                    device=edge_features.device,
                    dtype=edge_features.dtype,
                )
            class_context = edge_features.new_zeros(
                sample_count,
                residue_count,
                self.class_channel_dim,
            )
            class_context.index_add_(
                1,
                targets,
                class_messages * class_selector.reshape(1, edge_count, 1),
            )
            class_counts = edge_features.new_zeros(residue_count)
            class_counts.index_add_(0, targets, class_selector)
            class_contexts.append(
                class_context / class_counts.clamp_min(1.0).reshape(1, residue_count, 1)
            )
        return torch.cat([context, *class_contexts], dim=-1)

    def _forward_streamed_by_sample_chunk(
        self,
        edge_features: torch.Tensor,
        targets: torch.Tensor,
        classes: torch.Tensor,
        edge_weights: torch.Tensor,
        counts: torch.Tensor,
        *,
        residue_count: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Pool edge messages in support chunks to avoid a full ``[K,E,H]`` tensor."""

        class_bias = self.edge_class_embedding(classes).unsqueeze(0)
        context_chunks: list[torch.Tensor] = []
        class_context_chunks: list[list[torch.Tensor]] = [
            [] for _ in range(self.edge_class_count)
        ]
        count_scale = counts.clamp_min(1.0).reshape(1, residue_count, 1)
        class_selectors: list[torch.Tensor] = []
        class_counts: list[torch.Tensor] = []
        if self.preserve_class_channels and self.class_projection is not None:
            for class_index in range(self.edge_class_count):
                selector = (classes == class_index).to(dtype=edge_features.dtype)
                selector = selector * edge_weights
                selector_counts = edge_features.new_zeros(residue_count)
                selector_counts.index_add_(0, targets, selector)
                class_selectors.append(selector)
                class_counts.append(selector_counts.clamp_min(1.0).reshape(1, residue_count, 1))

        for start in range(0, int(edge_features.shape[0]), int(chunk_size)):
            edge_chunk = edge_features[start : start + int(chunk_size)]
            messages = self.edge_projection(edge_chunk) + class_bias
            messages = messages * edge_weights.reshape(1, int(edge_features.shape[1]), 1)
            context_chunk = edge_features.new_zeros(
                int(edge_chunk.shape[0]),
                residue_count,
                self.hidden_dim,
            )
            context_chunk.index_add_(1, targets, messages)
            context_chunks.append(context_chunk / count_scale)
            if not self.preserve_class_channels or self.class_projection is None:
                continue
            class_messages = self.class_projection(messages)
            for class_index, selector in enumerate(class_selectors):
                class_context = edge_features.new_zeros(
                    int(edge_chunk.shape[0]),
                    residue_count,
                    self.class_channel_dim,
                )
                class_context.index_add_(
                    1,
                    targets,
                    class_messages * selector.reshape(1, int(edge_features.shape[1]), 1),
                )
                class_context_chunks[class_index].append(
                    class_context / class_counts[class_index]
                )

        context = torch.cat(context_chunks, dim=0)
        if not self.preserve_class_channels or self.class_projection is None:
            return context
        return torch.cat(
            [
                context,
                *[torch.cat(chunks, dim=0) for chunks in class_context_chunks],
            ],
            dim=-1,
        )


class ChemicalShiftConformerDecoder(nn.Module):
    """Predict per-conformer, per-residue atom-family chemical shifts."""

    def __init__(
        self,
        residue_feature_dim: int,
        hidden_dim: int,
        *,
        atom_family_count: int = len(ATOM_FAMILY_NAMES),
        mechanism_feature_dim: int = 0,
        dropout: float = 0.0,
        min_sigma: float = 1.0e-3,
        mechanism_support_basis_count: int = 0,
        mechanism_support_basis_cap: float = 0.0,
        hn_signed_support_basis_count: int = 0,
        hn_signed_support_basis_cap: float = 0.0,
        hn_direct_support_spread_scale_ppm: float = 0.0,
        hn_direct_support_spread_trainable_scale_cap_ppm: float = 0.0,
        hn_direct_support_spread_trainable_scale_init_ppm: float = 0.0,
        family_direct_support_spread_scale_ppm_by_family: dict[str, float]
        | None = None,
        family_direct_support_spread_trainable_scale_cap_ppm_by_family: dict[
            str, float
        ]
        | None = None,
        family_direct_support_spread_trainable_scale_init_ppm_by_family: dict[
            str, float
        ]
        | None = None,
        cprime_isolated_support_basis_count: int = 0,
        cprime_isolated_support_basis_cap: float = 0.0,
        n_ca_cb_support_deviation_enabled: bool = True,
        emit_support_basis_diagnostics: bool = True,
        use_checkpoint: bool = False,
        support_head_chunk_size: int = 0,
        capacity_adapter_hidden_dim: int = 0,
        capacity_adapter_depth: int = 3,
        capacity_adapter_cap_ppm: float = 0.0,
        capacity_adapter_scale: float = 1.0,
        capacity_adapter_trainable_scale_cap: float = 0.0,
        capacity_adapter_trainable_scale_init: float = 0.0,
        capacity_adapter_extra_hidden_dim: int = 0,
        capacity_adapter_extra_depth: int = 3,
        capacity_adapter_extra_cap_ppm: float = 0.0,
        capacity_adapter_extra_scale: float = 1.0,
        capacity_adapter_extra_expert_count: int = 1,
        capacity_adapter_extra_factorized_rank: int = 0,
        capacity_adapter_extra_effective_hidden_dim: int = 0,
        capacity_adapter_extra_effective_factorized_rank: int = 0,
        capacity_adapter_extra_max_factorized_params: int = 450_000_000,
        capacity_adapter_extra_output_init_std: float = 0.0,
        capacity_adapter_extra_family_scales: dict[str, float] | None = None,
        capacity_adapter_extra_family_gated: bool = False,
        capacity_adapter_extra_family_specific: bool = False,
        capacity_adapter_extra_family_headed: bool = False,
        capacity_adapter_aux_hidden_dim: int = 0,
        capacity_adapter_aux_depth: int = 2,
        capacity_adapter_aux_cap_ppm: float = 0.0,
        capacity_adapter_aux_scale: float = 1.0,
        capacity_adapter_aux_expert_count: int = 1,
        capacity_adapter_aux_output_init_std: float = 0.0,
        capacity_adapter_aux_family_scales: dict[str, float] | None = None,
        capacity_adapter_aux_family_gated: bool = False,
        capacity_adapter_aux_family_specific: bool = False,
        capacity_adapter_aux_family_headed: bool = False,
        family_affine_scale_cap: float = 0.0,
        family_affine_shift_cap_ppm: float = 0.0,
        family_affine_family_scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.use_checkpoint = bool(use_checkpoint)
        self.support_head_chunk_size = max(int(support_head_chunk_size), 0)
        self.capacity_adapter_extra_support_head_chunk_size = (
            self.support_head_chunk_size
        )
        self.capacity_adapter_aux_support_head_chunk_size = (
            self.support_head_chunk_size
        )
        self.capacity_adapter_cap_ppm = max(float(capacity_adapter_cap_ppm), 0.0)
        self.capacity_adapter_scale = max(float(capacity_adapter_scale), 0.0)
        self.capacity_adapter_trainable_scale_cap = max(
            float(capacity_adapter_trainable_scale_cap),
            0.0,
        )
        self.capacity_adapter_extra_cap_ppm = max(
            float(capacity_adapter_extra_cap_ppm),
            0.0,
        )
        self.capacity_adapter_extra_scale = max(
            float(capacity_adapter_extra_scale),
            0.0,
        )
        self.capacity_adapter_aux_cap_ppm = max(
            float(capacity_adapter_aux_cap_ppm),
            0.0,
        )
        self.capacity_adapter_aux_scale = max(
            float(capacity_adapter_aux_scale),
            0.0,
        )
        self.family_affine_scale_cap = max(float(family_affine_scale_cap), 0.0)
        self.family_affine_shift_cap_ppm = max(
            float(family_affine_shift_cap_ppm),
            0.0,
        )
        extra_family_scale_values = torch.ones(
            int(atom_family_count),
            dtype=torch.float32,
        )
        if capacity_adapter_extra_family_scales:
            family_names = ATOM_FAMILY_NAMES[: int(atom_family_count)]
            for index, family_name in enumerate(family_names):
                if family_name in capacity_adapter_extra_family_scales:
                    extra_family_scale_values[index] = max(
                        float(capacity_adapter_extra_family_scales[family_name]),
                        0.0,
                    )
        self.register_buffer(
            "capacity_adapter_extra_family_scale",
            extra_family_scale_values,
            persistent=False,
        )
        aux_family_scale_values = torch.ones(
            int(atom_family_count),
            dtype=torch.float32,
        )
        if capacity_adapter_aux_family_scales:
            family_names = ATOM_FAMILY_NAMES[: int(atom_family_count)]
            for index, family_name in enumerate(family_names):
                if family_name in capacity_adapter_aux_family_scales:
                    aux_family_scale_values[index] = max(
                        float(capacity_adapter_aux_family_scales[family_name]),
                        0.0,
                    )
        self.register_buffer(
            "capacity_adapter_aux_family_scale",
            aux_family_scale_values,
            persistent=False,
        )
        affine_family_scale_values = torch.ones(
            int(atom_family_count),
            dtype=torch.float32,
        )
        if family_affine_family_scales:
            family_names = ATOM_FAMILY_NAMES[: int(atom_family_count)]
            for index, family_name in enumerate(family_names):
                if family_name in family_affine_family_scales:
                    affine_family_scale_values[index] = max(
                        float(family_affine_family_scales[family_name]),
                        0.0,
                    )
        self.register_buffer(
            "family_affine_family_scale",
            affine_family_scale_values,
            persistent=False,
        )
        self.mechanism_feature_dim = max(int(mechanism_feature_dim), 0)
        self.mechanism_support_basis_count = max(int(mechanism_support_basis_count), 0)
        self.mechanism_support_basis_cap = max(float(mechanism_support_basis_cap), 0.0)
        self.hn_signed_support_basis_count = max(
            int(hn_signed_support_basis_count), 0
        )
        self.hn_signed_support_basis_cap = max(
            float(hn_signed_support_basis_cap), 0.0
        )
        self.hn_direct_support_spread_scale_ppm = float(
            hn_direct_support_spread_scale_ppm
        )
        self.hn_direct_support_spread_trainable_scale_cap_ppm = max(
            float(hn_direct_support_spread_trainable_scale_cap_ppm),
            0.0,
        )
        family_direct_scale_values = torch.zeros(
            int(atom_family_count),
            dtype=torch.float32,
        )
        family_direct_cap_values = torch.zeros(
            int(atom_family_count),
            dtype=torch.float32,
        )
        family_direct_init_values = torch.zeros(
            int(atom_family_count),
            dtype=torch.float32,
        )
        family_names = ATOM_FAMILY_NAMES[: int(atom_family_count)]
        fixed_family_direct = family_direct_support_spread_scale_ppm_by_family or {}
        cap_family_direct = (
            family_direct_support_spread_trainable_scale_cap_ppm_by_family or {}
        )
        init_family_direct = (
            family_direct_support_spread_trainable_scale_init_ppm_by_family or {}
        )
        for index, family_name in enumerate(family_names):
            aliases = {
                family_name,
                family_name.replace("'", "prime"),
                family_name.replace("'", "Prime"),
            }
            for key in aliases:
                if key in fixed_family_direct:
                    family_direct_scale_values[index] = float(
                        fixed_family_direct[key]
                    )
                    break
            for key in aliases:
                if key in cap_family_direct:
                    family_direct_cap_values[index] = max(
                        float(cap_family_direct[key]),
                        0.0,
                    )
                    break
            for key in aliases:
                if key in init_family_direct:
                    family_direct_init_values[index] = float(init_family_direct[key])
                    break
        self.register_buffer(
            "family_direct_support_spread_scale_ppm",
            family_direct_scale_values,
            persistent=False,
        )
        self.register_buffer(
            "family_direct_support_spread_trainable_scale_cap_ppm",
            family_direct_cap_values,
            persistent=False,
        )
        self.cprime_isolated_support_basis_count = max(
            int(cprime_isolated_support_basis_count), 0
        )
        self.cprime_isolated_support_basis_cap = max(
            float(cprime_isolated_support_basis_cap), 0.0
        )
        self.n_ca_cb_support_deviation_enabled = bool(
            n_ca_cb_support_deviation_enabled
        )
        self.emit_support_basis_diagnostics = bool(emit_support_basis_diagnostics)
        self.residue_projection = _make_mlp(
            max(int(residue_feature_dim), 1),
            hidden_dim,
            hidden_dim,
            depth=2,
            dropout=dropout,
        )
        self.mechanism_projection: nn.Module | None = None
        self.family_mechanism_mean_delta: nn.Module | None = None
        self.hn_cprime_specialized_mean_delta: nn.Module | None = None
        self.hn_selected_edge_mean_delta: nn.Module | None = None
        self.hn_support_deviation_mean_delta: nn.Module | None = None
        self.cprime_selected_edge_mean_delta: nn.Module | None = None
        self.cprime_support_deviation_mean_delta: nn.Module | None = None
        self.cprime_plane_support_refinement_mean_delta: nn.Module | None = None
        self.n_ca_cb_support_deviation_mean_delta: nn.Module | None = None
        self.hn_cprime_support_basis_mean_delta: nn.Module | None = None
        self.hn_cprime_support_basis_gate: nn.Module | None = None
        self.hn_signed_support_basis_mean_delta: nn.Module | None = None
        self.hn_signed_support_basis_gate: nn.Module | None = None
        self.hn_direct_support_spread_scale_delta_ppm: nn.Parameter | None = None
        self.family_direct_support_spread_scale_delta_ppm: nn.Parameter | None = None
        self.cprime_isolated_support_basis_mean_delta: nn.Module | None = None
        self.cprime_isolated_support_basis_gate: nn.Module | None = None
        self.capacity_adapter_mean_delta: nn.Module | None = None
        self.capacity_adapter_extra_mean_delta: nn.Module | None = None
        self.capacity_adapter_aux_mean_delta: nn.Module | None = None
        self.capacity_adapter_scale_delta: nn.Parameter | None = None
        self.family_affine_scale_delta: nn.Parameter | None = None
        self.family_affine_shift_delta: nn.Parameter | None = None
        if self.family_affine_scale_cap > 0.0:
            self.family_affine_scale_delta = nn.Parameter(
                torch.zeros(int(atom_family_count), dtype=torch.float32)
            )
        if self.family_affine_shift_cap_ppm > 0.0:
            self.family_affine_shift_delta = nn.Parameter(
                torch.zeros(int(atom_family_count), dtype=torch.float32)
            )
        if self.hn_direct_support_spread_trainable_scale_cap_ppm > 0.0:
            cap = self.hn_direct_support_spread_trainable_scale_cap_ppm
            init = max(
                min(float(hn_direct_support_spread_trainable_scale_init_ppm), cap),
                -cap,
            )
            ratio = max(min(init / max(cap, 1.0e-6), 0.999), -0.999)
            raw_init = cap * torch.atanh(torch.tensor(ratio, dtype=torch.float32))
            self.hn_direct_support_spread_scale_delta_ppm = nn.Parameter(
                torch.tensor(float(raw_init), dtype=torch.float32)
            )
        if bool(torch.any(family_direct_cap_values > 0.0).item()):
            raw_family_init = torch.zeros_like(family_direct_cap_values)
            active = family_direct_cap_values > 0.0
            clipped_init = torch.maximum(
                torch.minimum(
                    family_direct_init_values,
                    family_direct_cap_values,
                ),
                -family_direct_cap_values,
            )
            ratio = clipped_init[active] / family_direct_cap_values[
                active
            ].clamp_min(1.0e-6)
            ratio = ratio.clamp(min=-0.999, max=0.999)
            raw_family_init[active] = family_direct_cap_values[active] * torch.atanh(
                ratio
            )
            self.family_direct_support_spread_scale_delta_ppm = nn.Parameter(
                raw_family_init
            )
        self.mechanism_class_channel_dim = 0
        if int(mechanism_feature_dim) > 0:
            self.mechanism_projection = _make_mlp(
                int(mechanism_feature_dim),
                hidden_dim,
                hidden_dim,
                depth=1,
                zero_init_output=True,
            )
            self.family_mechanism_mean_delta = _make_mlp(
                int(mechanism_feature_dim),
                hidden_dim,
                int(atom_family_count),
                depth=2,
                dropout=dropout,
                zero_init_output=True,
            )
            if int(atom_family_count) >= 5:
                self.hn_cprime_specialized_mean_delta = _make_mlp(
                    int(mechanism_feature_dim) + int(hidden_dim),
                    hidden_dim,
                    2,
                    depth=2,
                    dropout=dropout,
                    zero_init_output=True,
                )
                class_tail_dim = int(mechanism_feature_dim) - int(hidden_dim)
                if class_tail_dim > 0 and class_tail_dim % len(COMPACT_X2D_EDGE_CLASSES) == 0:
                    self.mechanism_class_channel_dim = class_tail_dim // len(COMPACT_X2D_EDGE_CLASSES)
                    selected_input_dim = int(hidden_dim) + int(hidden_dim) + 3 * self.mechanism_class_channel_dim
                    n_ca_cb_input_dim = int(hidden_dim) + int(hidden_dim) + 4 * self.mechanism_class_channel_dim
                    self.hn_selected_edge_mean_delta = _make_mlp(
                        selected_input_dim,
                        hidden_dim,
                        1,
                        depth=2,
                        dropout=dropout,
                        zero_init_output=True,
                    )
                    if self.n_ca_cb_support_deviation_enabled:
                        self.n_ca_cb_support_deviation_mean_delta = _make_mlp(
                            n_ca_cb_input_dim,
                            hidden_dim,
                            3,
                            depth=3,
                            dropout=dropout,
                            zero_init_output=True,
                        )
                    self.hn_support_deviation_mean_delta = _make_mlp(
                        selected_input_dim,
                        hidden_dim,
                        1,
                        depth=3,
                        dropout=dropout,
                        zero_init_output=True,
                    )
                    self.cprime_selected_edge_mean_delta = _make_mlp(
                        selected_input_dim,
                        hidden_dim,
                        1,
                        depth=2,
                        dropout=dropout,
                        zero_init_output=True,
                    )
                    self.cprime_support_deviation_mean_delta = _make_mlp(
                        selected_input_dim,
                        hidden_dim,
                        1,
                        depth=3,
                        dropout=dropout,
                        zero_init_output=True,
                    )
                    cprime_refinement_input_dim = (
                        int(hidden_dim)
                        + int(hidden_dim)
                        + 7 * self.mechanism_class_channel_dim
                    )
                    self.cprime_plane_support_refinement_mean_delta = _make_mlp(
                        cprime_refinement_input_dim,
                        hidden_dim,
                        1,
                        depth=4,
                        dropout=dropout,
                        zero_init_output=True,
                    )
                    if self.mechanism_support_basis_count > 0:
                        support_basis_input_dim = (
                            int(hidden_dim)
                            + int(hidden_dim)
                            + 3
                        )
                        support_basis_output_dim = 2 * self.mechanism_support_basis_count
                        self.hn_cprime_support_basis_mean_delta = _make_mlp(
                            support_basis_input_dim,
                            hidden_dim,
                            support_basis_output_dim,
                            depth=4,
                            dropout=dropout,
                            zero_init_output=True,
                        )
                        self.hn_cprime_support_basis_gate = _make_mlp(
                            support_basis_input_dim,
                            hidden_dim,
                            support_basis_output_dim,
                            depth=2,
                            dropout=dropout,
                            zero_init_output=True,
                        )
                    if self.hn_signed_support_basis_count > 0:
                        hn_signed_basis_input_dim = (
                            int(hidden_dim)
                            + int(hidden_dim)
                            + 4
                        )
                        self.hn_signed_support_basis_mean_delta = _make_mlp(
                            hn_signed_basis_input_dim,
                            hidden_dim,
                            self.hn_signed_support_basis_count,
                            depth=4,
                            dropout=dropout,
                            zero_init_output=True,
                        )
                        self.hn_signed_support_basis_gate = _make_mlp(
                            hn_signed_basis_input_dim,
                            hidden_dim,
                            self.hn_signed_support_basis_count,
                            depth=2,
                            dropout=dropout,
                            zero_init_output=True,
                        )
                    if self.cprime_isolated_support_basis_count > 0:
                        cprime_isolated_basis_input_dim = (
                            int(hidden_dim)
                            + int(hidden_dim)
                            + 3
                        )
                        self.cprime_isolated_support_basis_mean_delta = _make_mlp(
                            cprime_isolated_basis_input_dim,
                            hidden_dim,
                            self.cprime_isolated_support_basis_count,
                            depth=4,
                            dropout=dropout,
                            zero_init_output=True,
                        )
                        self.cprime_isolated_support_basis_gate = _make_mlp(
                            cprime_isolated_basis_input_dim,
                            hidden_dim,
                            self.cprime_isolated_support_basis_count,
                            depth=2,
                            dropout=dropout,
                            zero_init_output=True,
                        )
        adapter_hidden_dim = max(int(capacity_adapter_hidden_dim), 0)
        if adapter_hidden_dim > 0:
            adapter_input_dim = int(hidden_dim) + max(int(mechanism_feature_dim), 0)
            self.capacity_adapter_mean_delta = _make_mlp(
                adapter_input_dim,
                adapter_hidden_dim,
                int(atom_family_count),
                depth=max(int(capacity_adapter_depth), 1),
                dropout=dropout,
                zero_init_output=True,
            )
            if self.capacity_adapter_trainable_scale_cap > 0.0:
                cap = self.capacity_adapter_trainable_scale_cap
                init = max(min(float(capacity_adapter_trainable_scale_init), cap), -cap)
                ratio = max(min(init / max(cap, 1.0e-6), 0.999), -0.999)
                raw_init = cap * torch.atanh(torch.tensor(ratio, dtype=torch.float32))
                self.capacity_adapter_scale_delta = nn.Parameter(
                    torch.full((int(atom_family_count),), float(raw_init))
                )
        extra_adapter_hidden_dim = max(int(capacity_adapter_extra_hidden_dim), 0)
        self.capacity_adapter_extra_requested_hidden_dim = extra_adapter_hidden_dim
        self.capacity_adapter_extra_requested_factorized_rank = max(
            int(capacity_adapter_extra_factorized_rank),
            0,
        )
        self.capacity_adapter_extra_effective_hidden_dim = extra_adapter_hidden_dim
        self.capacity_adapter_extra_effective_factorized_rank = (
            self.capacity_adapter_extra_requested_factorized_rank
        )
        if extra_adapter_hidden_dim > 0:
            adapter_input_dim = int(hidden_dim) + max(int(mechanism_feature_dim), 0)
            extra_expert_count = max(int(capacity_adapter_extra_expert_count), 1)
            extra_output_init_std = max(
                float(capacity_adapter_extra_output_init_std),
                0.0,
            )
            if bool(capacity_adapter_extra_family_specific):
                self.capacity_adapter_extra_mean_delta = FamilySpecificCapacityAdapter(
                    adapter_input_dim,
                    extra_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_extra_depth), 1),
                    expert_count=extra_expert_count,
                    dropout=dropout,
                    output_init_std=extra_output_init_std,
                )
            elif bool(capacity_adapter_extra_family_headed):
                factorized_rank = max(int(capacity_adapter_extra_factorized_rank), 0)
                if factorized_rank > 0:
                    extra_adapter_hidden_dim, factorized_rank = (
                        resolve_factorized_capacity_dims(
                            requested_hidden_dim=extra_adapter_hidden_dim,
                            requested_rank=factorized_rank,
                            depth=max(int(capacity_adapter_extra_depth), 1),
                            effective_hidden_dim=(
                                capacity_adapter_extra_effective_hidden_dim
                            ),
                            effective_rank=(
                                capacity_adapter_extra_effective_factorized_rank
                            ),
                            max_factorized_params=(
                                capacity_adapter_extra_max_factorized_params
                            ),
                        )
                    )
                    self.capacity_adapter_extra_effective_hidden_dim = (
                        extra_adapter_hidden_dim
                    )
                    self.capacity_adapter_extra_effective_factorized_rank = (
                        factorized_rank
                    )
                    if (
                        extra_adapter_hidden_dim >= 4096
                        and self.capacity_adapter_extra_support_head_chunk_size != 1
                    ):
                        # Very wide conformer adapters can exhaust L40S memory even
                        # when the base decoder heads fit.  Keep the requested model
                        # capacity, but process one conformer at a time.
                        self.capacity_adapter_extra_support_head_chunk_size = 1
                    self.capacity_adapter_extra_mean_delta = (
                        FactorizedFamilyHeadedCapacityAdapter(
                            adapter_input_dim,
                            extra_adapter_hidden_dim,
                            int(atom_family_count),
                            factorized_rank=factorized_rank,
                            depth=max(int(capacity_adapter_extra_depth), 1),
                            expert_count=extra_expert_count,
                            dropout=dropout,
                            output_init_std=extra_output_init_std,
                        )
                    )
                else:
                    self.capacity_adapter_extra_mean_delta = (
                        FamilyHeadedCapacityAdapter(
                            adapter_input_dim,
                            extra_adapter_hidden_dim,
                            int(atom_family_count),
                            depth=max(int(capacity_adapter_extra_depth), 1),
                            expert_count=extra_expert_count,
                            dropout=dropout,
                            output_init_std=extra_output_init_std,
                        )
                    )
            elif extra_expert_count > 1:
                self.capacity_adapter_extra_mean_delta = GatedCapacityAdapter(
                    adapter_input_dim,
                    extra_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_extra_depth), 1),
                    expert_count=extra_expert_count,
                    dropout=dropout,
                    family_gated=capacity_adapter_extra_family_gated,
                    output_init_std=extra_output_init_std,
                )
            else:
                self.capacity_adapter_extra_mean_delta = _make_mlp(
                    adapter_input_dim,
                    extra_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_extra_depth), 1),
                    dropout=dropout,
                    zero_init_output=True,
                )
        aux_adapter_hidden_dim = max(int(capacity_adapter_aux_hidden_dim), 0)
        if aux_adapter_hidden_dim > 0:
            adapter_input_dim = int(hidden_dim) + max(int(mechanism_feature_dim), 0)
            aux_expert_count = max(int(capacity_adapter_aux_expert_count), 1)
            aux_output_init_std = max(
                float(capacity_adapter_aux_output_init_std),
                0.0,
            )
            if bool(capacity_adapter_aux_family_specific):
                self.capacity_adapter_aux_mean_delta = FamilySpecificCapacityAdapter(
                    adapter_input_dim,
                    aux_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_aux_depth), 1),
                    expert_count=aux_expert_count,
                    dropout=dropout,
                    output_init_std=aux_output_init_std,
                )
            elif bool(capacity_adapter_aux_family_headed):
                self.capacity_adapter_aux_mean_delta = FamilyHeadedCapacityAdapter(
                    adapter_input_dim,
                    aux_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_aux_depth), 1),
                    expert_count=aux_expert_count,
                    dropout=dropout,
                    output_init_std=aux_output_init_std,
                )
            elif aux_expert_count > 1:
                self.capacity_adapter_aux_mean_delta = GatedCapacityAdapter(
                    adapter_input_dim,
                    aux_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_aux_depth), 1),
                    expert_count=aux_expert_count,
                    dropout=dropout,
                    family_gated=capacity_adapter_aux_family_gated,
                    output_init_std=aux_output_init_std,
                )
            else:
                self.capacity_adapter_aux_mean_delta = _make_mlp(
                    adapter_input_dim,
                    aux_adapter_hidden_dim,
                    int(atom_family_count),
                    depth=max(int(capacity_adapter_aux_depth), 1),
                    dropout=dropout,
                    zero_init_output=aux_output_init_std <= 0.0,
                )
                if aux_output_init_std > 0.0:
                    nn.init.normal_(
                        self.capacity_adapter_aux_mean_delta[-1].weight,
                        mean=0.0,
                        std=aux_output_init_std,
                    )
                    nn.init.zeros_(self.capacity_adapter_aux_mean_delta[-1].bias)
        self.mean_head = nn.Linear(hidden_dim, int(atom_family_count))
        self.log_sigma_head = nn.Linear(hidden_dim, int(atom_family_count))
        centers = torch.zeros(int(atom_family_count), dtype=torch.float32)
        scales = torch.ones(int(atom_family_count), dtype=torch.float32)
        for index, value in enumerate(ATOM_FAMILY_PPM_CENTERS[: int(atom_family_count)]):
            centers[index] = float(value)
        for index, value in enumerate(ATOM_FAMILY_PPM_SCALES[: int(atom_family_count)]):
            scales[index] = max(float(value), 1.0e-3)
        self.register_buffer("family_ppm_centers", centers)
        self.register_buffer("family_ppm_scales", scales)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def _run_head(
        self,
        module: nn.Module,
        features: torch.Tensor,
        *,
        force_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Run a decoder MLP with optional K-axis chunking and checkpointing."""

        chunk_size = self.support_head_chunk_size
        if chunk_size > 0 and features.ndim >= 2 and features.shape[0] > chunk_size:
            chunks = [
                self._run_head_no_chunk(
                    module,
                    chunk,
                    force_checkpoint=force_checkpoint,
                )
                for chunk in torch.split(features, chunk_size, dim=0)
            ]
            return torch.cat(chunks, dim=0)
        return self._run_head_no_chunk(
            module,
            features,
            force_checkpoint=force_checkpoint,
        )

    def _run_head_no_chunk(
        self,
        module: nn.Module,
        features: torch.Tensor,
        *,
        force_checkpoint: bool = False,
    ) -> torch.Tensor:
        use_checkpoint = (
            (self.use_checkpoint or force_checkpoint)
            and torch.is_grad_enabled()
            and _module_or_inputs_require_grad(module, features)
        )
        if use_checkpoint:
            return checkpoint(module, features, use_reentrant=False)
        return module(features)

    def forward(
        self,
        residue_features: torch.Tensor,
        mechanism_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return conformer-level ``phi`` and uncertainty tensors."""

        hidden = self._run_head(self.residue_projection, residue_features)
        mechanism = None
        if mechanism_features is not None:
            mechanism = mechanism_features.to(
                device=residue_features.device,
                dtype=residue_features.dtype,
            )
        if self.mechanism_projection is not None and mechanism is not None:
            hidden = hidden + self._run_head(self.mechanism_projection, mechanism)
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=50.0, neginf=-50.0).clamp(
            min=-50.0,
            max=50.0,
        )
        sigma = torch.nn.functional.softplus(self.log_sigma_head(hidden)).clamp_min(
            self.min_sigma
        )
        sigma = torch.nan_to_num(
            sigma,
            nan=float(self.min_sigma),
            posinf=1.0e3,
            neginf=float(self.min_sigma),
        ).clamp_min(self.min_sigma)
        raw_mean = self.mean_head(hidden)
        raw_mean = torch.nan_to_num(raw_mean, nan=0.0, posinf=50.0, neginf=-50.0)
        if self.family_mechanism_mean_delta is not None and mechanism is not None:
            raw_mean = raw_mean + self._run_head(
                self.family_mechanism_mean_delta,
                mechanism,
            )
        if self.hn_cprime_specialized_mean_delta is not None and mechanism is not None:
            specialized_delta = self._run_head(
                self.hn_cprime_specialized_mean_delta,
                torch.cat([hidden, mechanism], dim=-1)
            )
            raw_mean = raw_mean.clone()
            raw_mean[..., 0] = raw_mean[..., 0] + specialized_delta[..., 0]
            raw_mean[..., 4] = raw_mean[..., 4] + specialized_delta[..., 1]
        mechanism_support_basis_delta = raw_mean.new_zeros(*raw_mean.shape)
        hn_signed_support_basis_delta = raw_mean.new_zeros(*raw_mean.shape)
        hn_direct_support_spread_delta = raw_mean.new_zeros(*raw_mean.shape)
        family_direct_support_spread_delta = raw_mean.new_zeros(*raw_mean.shape)
        cprime_isolated_support_basis_delta = raw_mean.new_zeros(*raw_mean.shape)
        if (
            self.hn_selected_edge_mean_delta is not None
            and self.cprime_selected_edge_mean_delta is not None
            and mechanism is not None
            and self.mechanism_class_channel_dim > 0
        ):
            aggregate = mechanism[..., : hidden.shape[-1]]
            class_channels = mechanism[..., hidden.shape[-1] :].reshape(
                *mechanism.shape[:-1],
                len(COMPACT_X2D_EDGE_CLASSES),
                self.mechanism_class_channel_dim,
            )
            hn_edges = torch.cat(
                [
                    class_channels[..., 3, :],  # hbond
                    class_channels[..., 4, :],  # ring
                    class_channels[..., 5, :],  # electrostatic
                ],
                dim=-1,
            )
            cprime_edges = torch.cat(
                [
                    class_channels[..., 0, :],  # sequence
                    class_channels[..., 1, :],  # peptide_plane
                    class_channels[..., 3, :],  # hbond/carbonyl proxy
                ],
                dim=-1,
            )
            cprime_plane_refinement_edges = torch.cat(
                [
                    class_channels[..., 0, :],  # sequence-local backbone context
                    class_channels[..., 1, :],  # signed peptide-plane direction
                    class_channels[..., 2, :],  # global spatial/contact topology
                    class_channels[..., 3, :],  # H-bond/carbonyl proxy
                    class_channels[..., 5, :],  # electrostatic/contact context
                    class_channels[..., 1, :] * class_channels[..., 3, :],
                    class_channels[..., 1, :] * class_channels[..., 0, :],
                ],
                dim=-1,
            )
            n_ca_cb_edges = torch.cat(
                [
                    class_channels[..., 0, :],  # sequence-local backbone context
                    class_channels[..., 2, :],  # global spatial-contact topology
                    class_channels[..., 3, :],  # H-bond/exchange proxy
                    class_channels[..., 5, :],  # electrostatic context
                ],
                dim=-1,
            )
            hn_direct_support_spread = None

            def normalized_support_unit(signal: torch.Tensor) -> torch.Tensor:
                signal = torch.nan_to_num(
                    signal,
                    nan=0.0,
                    posinf=50.0,
                    neginf=-50.0,
                ).clamp(min=-50.0, max=50.0)
                centered = signal - signal.mean(dim=0, keepdim=True)
                std = torch.sqrt(
                    centered.square().mean(dim=0, keepdim=True).clamp_min(1.0e-12)
                )
                unit = torch.tanh(centered / std)
                unit = unit - unit.mean(dim=0, keepdim=True)
                unit = unit / torch.sqrt(
                    unit.square().mean(dim=0, keepdim=True).clamp_min(1.0e-12)
                )
                unit = unit.clamp(min=-2.5, max=2.5)
                return unit - unit.mean(dim=0, keepdim=True)

            if (
                (
                    self.hn_direct_support_spread_scale_ppm != 0.0
                    or self.hn_direct_support_spread_scale_delta_ppm is not None
                )
                and self.mechanism_class_channel_dim > 0
            ):
                direct_signal = (
                    (
                        class_channels[..., 3, :]
                        - class_channels[..., 4, :]
                    ).mean(dim=-1)
                    + 0.5
                    * (
                        class_channels[..., 5, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                    + 0.25
                    * (
                        class_channels[..., 0, :]
                        * class_channels[..., 3, :]
                    ).mean(dim=-1)
                    - 0.25
                    * (
                        class_channels[..., 4, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                )
                direct_unit = normalized_support_unit(direct_signal)
                scale_ppm = direct_unit.new_tensor(
                    self.hn_direct_support_spread_scale_ppm
                )
                if self.hn_direct_support_spread_scale_delta_ppm is not None:
                    scale_cap = direct_unit.new_tensor(
                        self.hn_direct_support_spread_trainable_scale_cap_ppm
                    )
                    scale_ppm = scale_ppm + scale_cap * torch.tanh(
                        self.hn_direct_support_spread_scale_delta_ppm.to(
                            device=direct_unit.device,
                            dtype=direct_unit.dtype,
                        )
                        / scale_cap.clamp_min(1.0e-6)
                    )
                hn_scale = self.family_ppm_scales.to(
                    device=direct_unit.device,
                    dtype=direct_unit.dtype,
                )[0].clamp_min(1.0e-6)
                hn_direct_support_spread = direct_unit * (scale_ppm / hn_scale)

            family_direct_support_spread = None
            family_direct_base_scale = (
                self.family_direct_support_spread_scale_ppm.to(
                    device=raw_mean.device,
                    dtype=raw_mean.dtype,
                )
            )
            family_direct_cap = (
                self.family_direct_support_spread_trainable_scale_cap_ppm.to(
                    device=raw_mean.device,
                    dtype=raw_mean.dtype,
                )
            )
            if (
                bool(torch.any(family_direct_base_scale != 0.0).item())
                or self.family_direct_support_spread_scale_delta_ppm is not None
            ):
                family_direct_scale_ppm = family_direct_base_scale
                if self.family_direct_support_spread_scale_delta_ppm is not None:
                    family_direct_scale_ppm = family_direct_scale_ppm + torch.where(
                        family_direct_cap > 0.0,
                        family_direct_cap
                        * torch.tanh(
                            self.family_direct_support_spread_scale_delta_ppm.to(
                                device=raw_mean.device,
                                dtype=raw_mean.dtype,
                            )
                            / family_direct_cap.clamp_min(1.0e-6)
                        ),
                        torch.zeros_like(family_direct_cap),
                )
                sequence_signal = class_channels[..., 0, :].mean(dim=-1)
                contact_signal = class_channels[..., 2, :].mean(dim=-1)
                hbond_signal = class_channels[..., 3, :].mean(dim=-1)
                ring_signal = class_channels[..., 4, :].mean(dim=-1)
                electro_signal = class_channels[..., 5, :].mean(dim=-1)
                hn_signal = (
                    hbond_signal
                    - ring_signal
                    + 0.5
                    * (
                        class_channels[..., 5, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                    + 0.25
                    * (
                        class_channels[..., 0, :]
                        * class_channels[..., 3, :]
                    ).mean(dim=-1)
                    - 0.25
                    * (
                        class_channels[..., 4, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                )
                n_signal = (
                    0.65 * hn_signal
                    + 0.25 * sequence_signal
                    + 0.10 * electro_signal
                )
                ca_signal = (
                    sequence_signal
                    + contact_signal
                    + 0.5
                    * (
                        class_channels[..., 1, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                    - 0.25 * hbond_signal
                )
                cb_signal = (
                    contact_signal
                    + electro_signal
                    + 0.25
                    * (
                        class_channels[..., 4, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                )
                cprime_signal = (
                    (
                        class_channels[..., 1, :]
                        * class_channels[..., 3, :]
                    ).mean(dim=-1)
                    + (
                        class_channels[..., 0, :]
                        * class_channels[..., 1, :]
                    ).mean(dim=-1)
                    + 0.5
                    * (
                        class_channels[..., 5, :]
                        * class_channels[..., 1, :]
                    ).mean(dim=-1)
                    - 0.25
                    * (
                        class_channels[..., 4, :]
                        * class_channels[..., 2, :]
                    ).mean(dim=-1)
                )
                family_units = torch.stack(
                    [
                        normalized_support_unit(hn_signal),
                        normalized_support_unit(n_signal),
                        normalized_support_unit(ca_signal),
                        normalized_support_unit(cb_signal),
                        normalized_support_unit(cprime_signal),
                    ],
                    dim=-1,
                )[..., : raw_mean.shape[-1]]
                family_scales = self.family_ppm_scales.to(
                    device=raw_mean.device,
                    dtype=raw_mean.dtype,
                )[: raw_mean.shape[-1]].clamp_min(1.0e-6)
                family_direct_support_spread = family_units * (
                    family_direct_scale_ppm[: raw_mean.shape[-1]] / family_scales
                )

            def run_edge_head(
                module: nn.Module,
                edge_features: torch.Tensor,
                *,
                force_checkpoint: bool = True,
            ) -> torch.Tensor:
                def run_chunk(
                    hidden_chunk: torch.Tensor,
                    aggregate_chunk: torch.Tensor,
                    edge_chunk: torch.Tensor,
                ) -> torch.Tensor:
                    features = torch.cat(
                        [hidden_chunk, aggregate_chunk, edge_chunk],
                        dim=-1,
                    )
                    if torch.is_grad_enabled():
                        return self._run_head(
                            module,
                            features,
                            force_checkpoint=force_checkpoint,
                        )
                    return module(features)

                chunk_size = int(self.support_head_chunk_size)
                if (
                    chunk_size > 0
                    and hidden.ndim >= 2
                    and int(hidden.shape[0]) > chunk_size
                ):
                    outputs = []
                    support_size = int(hidden.shape[0])
                    for start in range(0, support_size, chunk_size):
                        width = min(chunk_size, support_size - start)
                        outputs.append(
                            run_chunk(
                                hidden.narrow(0, start, width),
                                aggregate.narrow(0, start, width),
                                edge_features.narrow(0, start, width),
                            )
                        )
                    return torch.cat(outputs, dim=0)
                return run_chunk(hidden, aggregate, edge_features)

            hn_selected = run_edge_head(
                self.hn_selected_edge_mean_delta,
                hn_edges,
            ).squeeze(-1)
            hn_support_deviation = None
            if self.hn_support_deviation_mean_delta is not None:
                hn_support_deviation = run_edge_head(
                    self.hn_support_deviation_mean_delta,
                    hn_edges,
                ).squeeze(-1)
                # Preserve the residue-wise support mean while allowing HN
                # conformer spread to follow UCBShift teacher variation.
                hn_support_deviation = (
                    hn_support_deviation
                    - hn_support_deviation.mean(dim=0, keepdim=True)
                )
            cprime_selected = run_edge_head(
                self.cprime_selected_edge_mean_delta,
                cprime_edges,
            ).squeeze(-1)
            cprime_support_deviation = None
            if self.cprime_support_deviation_mean_delta is not None:
                cprime_support_deviation = run_edge_head(
                    self.cprime_support_deviation_mean_delta,
                    cprime_edges,
                ).squeeze(-1)
                cprime_support_deviation = (
                    cprime_support_deviation
                    - cprime_support_deviation.mean(dim=0, keepdim=True)
                )
            cprime_plane_refinement = None
            if self.cprime_plane_support_refinement_mean_delta is not None:
                cprime_plane_refinement = run_edge_head(
                    self.cprime_plane_support_refinement_mean_delta,
                    cprime_plane_refinement_edges,
                ).squeeze(-1)
                # Keep C' mean calibration on existing heads; this extra
                # carbonyl/peptide-plane route only reshapes x0 support.
                cprime_plane_refinement = (
                    cprime_plane_refinement
                    - cprime_plane_refinement.mean(dim=0, keepdim=True)
                )
            support_basis_edges = torch.cat(
                [
                    # Keep the support basis compact: local sequence and
                    # global contact topology both matter, but expanding all
                    # x2d channels made the K=512 path too memory-heavy.
                    0.5
                    * (
                        class_channels[..., 0, :] + class_channels[..., 2, :]
                    ).mean(dim=-1, keepdim=True),
                    (
                        class_channels[..., 3, :] * class_channels[..., 4, :]
                    ).mean(dim=-1, keepdim=True),
                    0.5
                    * (
                        class_channels[..., 1, :] * class_channels[..., 3, :]
                        + class_channels[..., 2, :] * class_channels[..., 5, :]
                    ).mean(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            hn_cprime_support_basis = None
            if (
                self.hn_cprime_support_basis_mean_delta is not None
                and self.hn_cprime_support_basis_gate is not None
                and self.mechanism_support_basis_count > 0
            ):
                def run_hn_cprime_support_basis(
                    hidden_chunk: torch.Tensor,
                    aggregate_chunk: torch.Tensor,
                    support_basis_edges_chunk: torch.Tensor,
                ) -> torch.Tensor:
                    basis_input = torch.cat(
                        [hidden_chunk, aggregate_chunk, support_basis_edges_chunk],
                        dim=-1,
                    )
                    basis_shape = (
                        *hidden_chunk.shape[:-1],
                        2,
                        self.mechanism_support_basis_count,
                    )
                    if torch.is_grad_enabled():
                        basis_values_raw = self._run_head(
                            self.hn_cprime_support_basis_mean_delta,
                            basis_input,
                            force_checkpoint=True,
                        )
                        basis_logits_raw = self._run_head(
                            self.hn_cprime_support_basis_gate,
                            basis_input,
                            force_checkpoint=True,
                        )
                    else:
                        basis_values_raw = self.hn_cprime_support_basis_mean_delta(
                            basis_input
                        )
                        basis_logits_raw = self.hn_cprime_support_basis_gate(
                            basis_input
                        )
                    basis_values = basis_values_raw.reshape(basis_shape)
                    basis_logits = basis_logits_raw.reshape(basis_shape)
                    basis_weights = torch.softmax(basis_logits, dim=-1)
                    return torch.sum(
                        basis_values * basis_weights,
                        dim=-1,
                    )

                chunk_size = int(self.support_head_chunk_size)
                if (
                    chunk_size > 0
                    and hidden.ndim >= 2
                    and int(hidden.shape[0]) > chunk_size
                ):
                    basis_chunks = []
                    support_size = int(hidden.shape[0])
                    for start in range(0, support_size, chunk_size):
                        width = min(chunk_size, support_size - start)
                        basis_chunks.append(
                            run_hn_cprime_support_basis(
                                hidden.narrow(0, start, width),
                                aggregate.narrow(0, start, width),
                                support_basis_edges.narrow(0, start, width),
                            )
                        )
                    hn_cprime_support_basis = torch.cat(basis_chunks, dim=0)
                else:
                    hn_cprime_support_basis = run_hn_cprime_support_basis(
                        hidden,
                        aggregate,
                        support_basis_edges,
                    )
                if self.mechanism_support_basis_cap > 0.0:
                    cap = hn_cprime_support_basis.new_tensor(
                        self.mechanism_support_basis_cap
                    )
                    hn_cprime_support_basis = cap * torch.tanh(
                        hn_cprime_support_basis / cap.clamp_min(1.0e-6)
                    )
                hn_cprime_support_basis = (
                    hn_cprime_support_basis
                    - hn_cprime_support_basis.mean(dim=0, keepdim=True)
                )
            hn_signed_support_basis = None
            if (
                self.hn_signed_support_basis_mean_delta is not None
                and self.hn_signed_support_basis_gate is not None
                and self.hn_signed_support_basis_count > 0
            ):
                def run_hn_signed_support_basis(
                    hidden_chunk: torch.Tensor,
                    aggregate_chunk: torch.Tensor,
                    class_channels_chunk: torch.Tensor,
                ) -> torch.Tensor:
                    hn_signed_edges = torch.cat(
                        [
                            (
                                class_channels_chunk[..., 3, :]
                                - class_channels_chunk[..., 4, :]
                            ).mean(dim=-1, keepdim=True),
                            (
                                class_channels_chunk[..., 5, :]
                                * class_channels_chunk[..., 2, :]
                            ).mean(dim=-1, keepdim=True),
                            (
                                class_channels_chunk[..., 0, :]
                                * class_channels_chunk[..., 3, :]
                            ).mean(dim=-1, keepdim=True),
                            (
                                class_channels_chunk[..., 4, :]
                                * class_channels_chunk[..., 2, :]
                            ).mean(dim=-1, keepdim=True),
                        ],
                        dim=-1,
                    )
                    hn_signed_input = torch.cat(
                        [hidden_chunk, aggregate_chunk, hn_signed_edges],
                        dim=-1,
                    )
                    if torch.is_grad_enabled():
                        values = self._run_head(
                            self.hn_signed_support_basis_mean_delta,
                            hn_signed_input,
                            force_checkpoint=True,
                        )
                        logits = self._run_head(
                            self.hn_signed_support_basis_gate,
                            hn_signed_input,
                            force_checkpoint=True,
                        )
                    else:
                        values = self.hn_signed_support_basis_mean_delta(
                            hn_signed_input
                        )
                        logits = self.hn_signed_support_basis_gate(hn_signed_input)
                    weights = torch.softmax(logits, dim=-1)
                    return torch.sum(values * weights, dim=-1)

                chunk_size = int(self.support_head_chunk_size)
                if (
                    chunk_size > 0
                    and hidden.ndim >= 2
                    and int(hidden.shape[0]) > chunk_size
                ):
                    basis_chunks = []
                    support_size = int(hidden.shape[0])
                    for start in range(0, support_size, chunk_size):
                        width = min(chunk_size, support_size - start)
                        hidden_chunk = hidden.narrow(0, start, width)
                        basis_chunks.append(
                            run_hn_signed_support_basis(
                                hidden_chunk,
                                aggregate.narrow(0, start, width),
                                class_channels.narrow(0, start, width),
                            )
                        )
                    hn_signed_support_basis = torch.cat(basis_chunks, dim=0)
                else:
                    hn_signed_support_basis = run_hn_signed_support_basis(
                        hidden,
                        aggregate,
                        class_channels,
                    )
                if self.hn_signed_support_basis_cap > 0.0:
                    cap = hn_signed_support_basis.new_tensor(
                        self.hn_signed_support_basis_cap
                    )
                    hn_signed_support_basis = cap * torch.tanh(
                        hn_signed_support_basis / cap.clamp_min(1.0e-6)
                    )
                hn_signed_support_basis = (
                    hn_signed_support_basis
                    - hn_signed_support_basis.mean(dim=0, keepdim=True)
                )
            cprime_isolated_support_basis = None
            if (
                self.cprime_isolated_support_basis_mean_delta is not None
                and self.cprime_isolated_support_basis_gate is not None
                and self.cprime_isolated_support_basis_count > 0
            ):
                def run_cprime_isolated_support_basis(
                    hidden_chunk: torch.Tensor,
                    aggregate_chunk: torch.Tensor,
                    support_basis_edges_chunk: torch.Tensor,
                ) -> torch.Tensor:
                    cprime_isolated_input = torch.cat(
                        [hidden_chunk, aggregate_chunk, support_basis_edges_chunk],
                        dim=-1,
                    )
                    if torch.is_grad_enabled():
                        cprime_isolated_values = self._run_head(
                            self.cprime_isolated_support_basis_mean_delta,
                            cprime_isolated_input,
                            force_checkpoint=True,
                        )
                        cprime_isolated_logits = self._run_head(
                            self.cprime_isolated_support_basis_gate,
                            cprime_isolated_input,
                            force_checkpoint=True,
                        )
                    else:
                        cprime_isolated_values = (
                            self.cprime_isolated_support_basis_mean_delta(
                                cprime_isolated_input
                            )
                        )
                        cprime_isolated_logits = self.cprime_isolated_support_basis_gate(
                            cprime_isolated_input
                        )
                    cprime_isolated_weights = torch.softmax(
                        cprime_isolated_logits,
                        dim=-1,
                    )
                    return torch.sum(
                        cprime_isolated_values * cprime_isolated_weights,
                        dim=-1,
                    )

                chunk_size = int(self.support_head_chunk_size)
                if (
                    chunk_size > 0
                    and hidden.ndim >= 2
                    and int(hidden.shape[0]) > chunk_size
                ):
                    basis_chunks = []
                    support_size = int(hidden.shape[0])
                    for start in range(0, support_size, chunk_size):
                        width = min(chunk_size, support_size - start)
                        basis_chunks.append(
                            run_cprime_isolated_support_basis(
                                hidden.narrow(0, start, width),
                                aggregate.narrow(0, start, width),
                                support_basis_edges.narrow(0, start, width),
                            )
                        )
                    cprime_isolated_support_basis = torch.cat(basis_chunks, dim=0)
                else:
                    cprime_isolated_support_basis = (
                        run_cprime_isolated_support_basis(
                            hidden,
                            aggregate,
                            support_basis_edges,
                        )
                    )
                if self.cprime_isolated_support_basis_cap > 0.0:
                    cap = cprime_isolated_support_basis.new_tensor(
                        self.cprime_isolated_support_basis_cap
                    )
                    cprime_isolated_support_basis = cap * torch.tanh(
                        cprime_isolated_support_basis / cap.clamp_min(1.0e-6)
                    )
                cprime_isolated_support_basis = (
                    cprime_isolated_support_basis
                    - cprime_isolated_support_basis.mean(dim=0, keepdim=True)
                )
            n_ca_cb_support_deviation = None
            if self.n_ca_cb_support_deviation_mean_delta is not None:
                n_ca_cb_support_deviation = run_edge_head(
                    self.n_ca_cb_support_deviation_mean_delta,
                    n_ca_cb_edges,
                )
                # N/CA/CB are support-limited in the reachability audit; this
                # opens conformer spread without shifting the residue mean.
                n_ca_cb_support_deviation = (
                    n_ca_cb_support_deviation
                    - n_ca_cb_support_deviation.mean(dim=0, keepdim=True)
                )
            raw_mean = raw_mean.clone()
            raw_mean[..., 0] = raw_mean[..., 0] + hn_selected
            if hn_support_deviation is not None:
                raw_mean[..., 0] = raw_mean[..., 0] + hn_support_deviation
            if hn_direct_support_spread is not None:
                raw_mean[..., 0] = raw_mean[..., 0] + hn_direct_support_spread
                hn_direct_support_spread_delta[..., 0] = hn_direct_support_spread
            if family_direct_support_spread is not None:
                raw_mean = raw_mean + family_direct_support_spread
                family_direct_support_spread_delta = family_direct_support_spread
            if hn_cprime_support_basis is not None:
                raw_mean[..., 0] = raw_mean[..., 0] + hn_cprime_support_basis[..., 0]
                mechanism_support_basis_delta[..., 0] = hn_cprime_support_basis[..., 0]
            if hn_signed_support_basis is not None:
                raw_mean[..., 0] = raw_mean[..., 0] + hn_signed_support_basis
                hn_signed_support_basis_delta[..., 0] = hn_signed_support_basis
            if n_ca_cb_support_deviation is not None:
                raw_mean[..., 1:4] = raw_mean[..., 1:4] + n_ca_cb_support_deviation
            raw_mean[..., 4] = raw_mean[..., 4] + cprime_selected
            if cprime_support_deviation is not None:
                raw_mean[..., 4] = raw_mean[..., 4] + cprime_support_deviation
            if cprime_plane_refinement is not None:
                raw_mean[..., 4] = raw_mean[..., 4] + cprime_plane_refinement
            if hn_cprime_support_basis is not None:
                raw_mean[..., 4] = raw_mean[..., 4] + hn_cprime_support_basis[..., 1]
                mechanism_support_basis_delta[..., 4] = hn_cprime_support_basis[..., 1]
            if cprime_isolated_support_basis is not None:
                raw_mean[..., 4] = raw_mean[..., 4] + cprime_isolated_support_basis
                cprime_isolated_support_basis_delta[..., 4] = (
                    cprime_isolated_support_basis
                )
        uses_capacity_adapter = (
            self.capacity_adapter_mean_delta is not None
            or self.capacity_adapter_extra_mean_delta is not None
            or self.capacity_adapter_aux_mean_delta is not None
        )

        def make_adapter_input(
            hidden_chunk: torch.Tensor,
            mechanism_chunk: torch.Tensor | None,
        ) -> torch.Tensor:
            if mechanism_chunk is None and self.mechanism_feature_dim <= 0:
                return hidden_chunk
            if mechanism_chunk is None:
                return torch.cat(
                    [
                        hidden_chunk,
                        hidden_chunk.new_zeros(
                            *hidden_chunk.shape[:-1],
                            self.mechanism_feature_dim,
                        ),
                    ],
                    dim=-1,
                )
            return torch.cat([hidden_chunk, mechanism_chunk], dim=-1)

        adapter_chunk_size = int(self.support_head_chunk_size)
        adapter_chunked = (
            uses_capacity_adapter
            and adapter_chunk_size > 0
            and hidden.ndim >= 2
            and int(hidden.shape[0]) > adapter_chunk_size
        )
        adapter_input = (
            None
            if not uses_capacity_adapter or adapter_chunked
            else make_adapter_input(hidden, mechanism)
        )

        def run_capacity_adapter_head(module: nn.Module) -> torch.Tensor:
            if not adapter_chunked:
                if adapter_input is None:
                    raise RuntimeError("capacity adapter input was not prepared")
                return self._run_head(module, adapter_input)
            chunks = []
            support_size = int(hidden.shape[0])
            for start in range(0, support_size, adapter_chunk_size):
                width = min(adapter_chunk_size, support_size - start)
                hidden_chunk = hidden.narrow(0, start, width)
                mechanism_chunk = (
                    None
                    if mechanism is None
                    else mechanism.narrow(0, start, width)
                )
                chunks.append(
                    self._run_head(
                        module,
                        make_adapter_input(hidden_chunk, mechanism_chunk),
                    )
                )
            return torch.cat(chunks, dim=0)

        def run_capacity_adapter_head_with_weights(
            module: (
                GatedCapacityAdapter
                | FamilySpecificCapacityAdapter
                | FamilyHeadedCapacityAdapter
                | FactorizedFamilyHeadedCapacityAdapter
            ),
            *,
            chunk_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not adapter_chunked:
                if adapter_input is None:
                    raise RuntimeError("capacity adapter input was not prepared")
                return module.forward_with_weights(
                    adapter_input,
                    chunk_size=chunk_size,
                    compact_weights=True,
                )
            delta_chunks = []
            weight_chunks = []
            support_size = int(hidden.shape[0])
            for start in range(0, support_size, adapter_chunk_size):
                width = min(adapter_chunk_size, support_size - start)
                hidden_chunk = hidden.narrow(0, start, width)
                mechanism_chunk = (
                    None
                    if mechanism is None
                    else mechanism.narrow(0, start, width)
                )
                chunk_delta, chunk_weights = module.forward_with_weights(
                    make_adapter_input(hidden_chunk, mechanism_chunk),
                    chunk_size=0,
                    compact_weights=True,
                )
                delta_chunks.append(chunk_delta)
                weight_chunks.append(chunk_weights)
            return torch.cat(delta_chunks, dim=0), torch.cat(weight_chunks, dim=0)
        adapter_delta = raw_mean.new_zeros(*raw_mean.shape)
        aux_delta = raw_mean.new_zeros(*raw_mean.shape)
        extra_expert_gate_entropy = raw_mean.new_zeros(())
        aux_expert_gate_entropy = raw_mean.new_zeros(())
        if self.capacity_adapter_mean_delta is not None and uses_capacity_adapter:
            adapter_delta = run_capacity_adapter_head(self.capacity_adapter_mean_delta)
            if self.capacity_adapter_cap_ppm > 0.0:
                cap_ppm = adapter_delta.new_tensor(self.capacity_adapter_cap_ppm)
                raw_cap = cap_ppm / self.family_ppm_scales.to(
                    device=adapter_delta.device,
                    dtype=adapter_delta.dtype,
                ).clamp_min(1.0e-6)
                adapter_delta = raw_cap * torch.tanh(adapter_delta)
            adapter_scale = adapter_delta.new_tensor(self.capacity_adapter_scale)
            if self.capacity_adapter_scale_delta is not None:
                cap = adapter_delta.new_tensor(self.capacity_adapter_trainable_scale_cap)
                adapter_scale = adapter_scale + cap * torch.tanh(
                    self.capacity_adapter_scale_delta.to(
                        device=adapter_delta.device,
                        dtype=adapter_delta.dtype,
                    )
                    / cap.clamp_min(1.0e-6)
                )
            if self.capacity_adapter_scale != 1.0 or self.capacity_adapter_scale_delta is not None:
                adapter_delta = adapter_delta * adapter_scale
        if self.capacity_adapter_extra_mean_delta is not None and uses_capacity_adapter:
            if isinstance(
                self.capacity_adapter_extra_mean_delta,
                (
                    GatedCapacityAdapter,
                    FamilySpecificCapacityAdapter,
                    FamilyHeadedCapacityAdapter,
                    FactorizedFamilyHeadedCapacityAdapter,
                ),
            ):
                extra_delta, extra_expert_weights = (
                    run_capacity_adapter_head_with_weights(
                        self.capacity_adapter_extra_mean_delta,
                        chunk_size=self.capacity_adapter_extra_support_head_chunk_size,
                    )
                )
                extra_expert_gate_entropy = -torch.sum(
                    extra_expert_weights
                    * torch.log(extra_expert_weights.clamp_min(1.0e-8)),
                    dim=-1,
                ).mean()
            else:
                extra_delta = run_capacity_adapter_head(
                    self.capacity_adapter_extra_mean_delta
                )
            extra_cap_ppm = (
                self.capacity_adapter_extra_cap_ppm
                if self.capacity_adapter_extra_cap_ppm > 0.0
                else self.capacity_adapter_cap_ppm
            )
            if extra_cap_ppm > 0.0:
                cap_ppm = extra_delta.new_tensor(extra_cap_ppm)
                raw_cap = cap_ppm / self.family_ppm_scales.to(
                    device=extra_delta.device,
                    dtype=extra_delta.dtype,
                ).clamp_min(1.0e-6)
                extra_delta = raw_cap * torch.tanh(extra_delta)
            if self.capacity_adapter_extra_scale != 1.0:
                extra_delta = extra_delta * extra_delta.new_tensor(
                    self.capacity_adapter_extra_scale
                )
            extra_family_scale = self.capacity_adapter_extra_family_scale.to(
                device=extra_delta.device,
                dtype=extra_delta.dtype,
            )
            extra_delta = extra_delta * extra_family_scale
            adapter_delta = adapter_delta + extra_delta
        if self.capacity_adapter_aux_mean_delta is not None and uses_capacity_adapter:
            if isinstance(
                self.capacity_adapter_aux_mean_delta,
                (
                    GatedCapacityAdapter,
                    FamilySpecificCapacityAdapter,
                    FamilyHeadedCapacityAdapter,
                    FactorizedFamilyHeadedCapacityAdapter,
                ),
            ):
                aux_delta, aux_expert_weights = (
                    run_capacity_adapter_head_with_weights(
                        self.capacity_adapter_aux_mean_delta,
                        chunk_size=self.capacity_adapter_aux_support_head_chunk_size,
                    )
                )
                aux_expert_gate_entropy = -torch.sum(
                    aux_expert_weights
                    * torch.log(aux_expert_weights.clamp_min(1.0e-8)),
                    dim=-1,
                ).mean()
            else:
                aux_delta = run_capacity_adapter_head(
                    self.capacity_adapter_aux_mean_delta
                )
            aux_cap_ppm = (
                self.capacity_adapter_aux_cap_ppm
                if self.capacity_adapter_aux_cap_ppm > 0.0
                else self.capacity_adapter_extra_cap_ppm
            )
            if aux_cap_ppm > 0.0:
                cap_ppm = aux_delta.new_tensor(aux_cap_ppm)
                raw_cap = cap_ppm / self.family_ppm_scales.to(
                    device=aux_delta.device,
                    dtype=aux_delta.dtype,
                ).clamp_min(1.0e-6)
                aux_delta = raw_cap * torch.tanh(aux_delta)
            if self.capacity_adapter_aux_scale != 1.0:
                aux_delta = aux_delta * aux_delta.new_tensor(
                    self.capacity_adapter_aux_scale
                )
            aux_family_scale = self.capacity_adapter_aux_family_scale.to(
                device=aux_delta.device,
                dtype=aux_delta.dtype,
            )
            aux_delta = aux_delta * aux_family_scale
            adapter_delta = adapter_delta + aux_delta
        raw_mean = raw_mean + adapter_delta
        raw_mean = torch.nan_to_num(raw_mean, nan=0.0, posinf=50.0, neginf=-50.0)
        phi = self.family_ppm_centers.to(
            device=raw_mean.device,
            dtype=raw_mean.dtype,
        ) + raw_mean * self.family_ppm_scales.to(
            device=raw_mean.device,
            dtype=raw_mean.dtype,
        )
        phi = torch.nan_to_num(phi, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        affine_delta_phi = phi.new_zeros(*phi.shape)
        family_affine_scale = phi.new_ones(self.family_ppm_centers.shape)
        family_affine_shift = phi.new_zeros(self.family_ppm_centers.shape)
        if self.family_affine_scale_delta is not None or self.family_affine_shift_delta is not None:
            centers = self.family_ppm_centers.to(device=phi.device, dtype=phi.dtype)
            family_scales = self.family_affine_family_scale.to(
                device=phi.device,
                dtype=phi.dtype,
            )
            if self.family_affine_scale_delta is not None:
                scale_cap = phi.new_tensor(self.family_affine_scale_cap)
                family_affine_scale = 1.0 + family_scales * scale_cap * torch.tanh(
                    self.family_affine_scale_delta.to(
                        device=phi.device,
                        dtype=phi.dtype,
                    )
                    / scale_cap.clamp_min(1.0e-6)
                )
            if self.family_affine_shift_delta is not None:
                shift_cap = phi.new_tensor(self.family_affine_shift_cap_ppm)
                family_affine_shift = family_scales * shift_cap * torch.tanh(
                    self.family_affine_shift_delta.to(
                        device=phi.device,
                        dtype=phi.dtype,
                    )
                    / shift_cap.clamp_min(1.0e-6)
                )
            calibrated_phi = (
                centers
                + (phi - centers) * family_affine_scale
                + family_affine_shift
            )
            affine_delta_phi = calibrated_phi - phi
            phi = calibrated_phi
        family_ppm_scales = self.family_ppm_scales.to(
            device=raw_mean.device,
            dtype=raw_mean.dtype,
        )
        if self.emit_support_basis_diagnostics:
            mechanism_support_basis_delta_phi = (
                mechanism_support_basis_delta * family_ppm_scales
            )
            hn_signed_support_basis_delta_phi = (
                hn_signed_support_basis_delta * family_ppm_scales
            )
            hn_direct_support_spread_delta_phi = (
                hn_direct_support_spread_delta * family_ppm_scales
            )
            family_direct_support_spread_delta_phi = (
                family_direct_support_spread_delta * family_ppm_scales
            )
            cprime_isolated_support_basis_delta_phi = (
                cprime_isolated_support_basis_delta * family_ppm_scales
            )
            support_basis_delta_phi = (
                mechanism_support_basis_delta
                + hn_signed_support_basis_delta
                + hn_direct_support_spread_delta
                + family_direct_support_spread_delta
                + cprime_isolated_support_basis_delta
            ) * family_ppm_scales
        else:
            mechanism_support_basis_delta_phi = None
            hn_signed_support_basis_delta_phi = None
            hn_direct_support_spread_delta_phi = None
            family_direct_support_spread_delta_phi = None
            cprime_isolated_support_basis_delta_phi = None
            support_basis_delta_phi = None
        return {
            "phi": phi,
            "sigma": sigma,
            "hidden": hidden,
            "capacity_adapter_delta_phi": adapter_delta
            * family_ppm_scales,
            "capacity_adapter_aux_delta_phi": aux_delta * family_ppm_scales,
            "mechanism_support_basis_delta_phi": mechanism_support_basis_delta_phi,
            "hn_signed_support_basis_delta_phi": hn_signed_support_basis_delta_phi,
            "hn_direct_support_spread_delta_phi": hn_direct_support_spread_delta_phi,
            "family_direct_support_spread_delta_phi": (
                family_direct_support_spread_delta_phi
            ),
            "cprime_isolated_support_basis_delta_phi": (
                cprime_isolated_support_basis_delta_phi
            ),
            "support_basis_delta_phi": support_basis_delta_phi,
            "capacity_adapter_extra_expert_gate_entropy": extra_expert_gate_entropy,
            "capacity_adapter_aux_expert_gate_entropy": aux_expert_gate_entropy,
            "family_affine_delta_phi": affine_delta_phi,
            "family_affine_scale": family_affine_scale.detach(),
            "family_affine_shift_ppm": family_affine_shift.detach(),
        }


class PosteriorEnergyHead(nn.Module):
    """Learn conformer energies for generalized-Bayes support reweighting."""

    def __init__(
        self,
        conformer_feature_dim: int,
        hidden_dim: int,
        *,
        evidence_feature_dim: int = 0,
        depth: int = 2,
        dropout: float = 0.0,
        init_std: float = 0.0,
        extra_hidden_dim: int = 0,
        extra_depth: int = 2,
        extra_scale: float = 1.0,
        extra_init_std: float = 0.0,
        extra_factorized_rank: int = 0,
        extra_effective_hidden_dim: int = 0,
        extra_effective_factorized_rank: int = 0,
        extra_max_factorized_params: int = 450_000_000,
        extra_shared_block_count: int = 0,
        extra_chunk_size: int = 0,
        extra_use_checkpoint: bool = True,
        support_attention_hidden_dim: int = 0,
        support_attention_head_count: int = 4,
        support_attention_depth: int = 1,
        support_attention_scale: float = 0.0,
        support_attention_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.evidence_feature_dim = max(int(evidence_feature_dim), 0)
        self.extra_scale = max(float(extra_scale), 0.0)
        self.support_attention_scale = max(float(support_attention_scale), 0.0)
        self.requested_extra_hidden_dim = max(int(extra_hidden_dim), 0)
        self.requested_extra_factorized_rank = max(int(extra_factorized_rank), 0)
        self.effective_extra_hidden_dim = self.requested_extra_hidden_dim
        self.effective_extra_factorized_rank = self.requested_extra_factorized_rank
        input_dim = int(conformer_feature_dim) + self.evidence_feature_dim
        self.energy_head = _make_mlp(
            input_dim,
            hidden_dim,
            1,
            depth=depth,
            dropout=dropout,
            zero_init_output=True,
        )
        init_std = max(float(init_std), 0.0)
        if init_std > 0.0:
            final_layer = self.energy_head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.normal_(final_layer.weight, mean=0.0, std=init_std)
                nn.init.zeros_(final_layer.bias)
        self.energy_extra_head: nn.Module | None = None
        extra_hidden_dim = max(int(extra_hidden_dim), 0)
        if extra_hidden_dim > 0:
            factorized_rank = max(int(extra_factorized_rank), 0)
            if factorized_rank > 0:
                extra_hidden_dim, factorized_rank = resolve_factorized_capacity_dims(
                    requested_hidden_dim=extra_hidden_dim,
                    requested_rank=factorized_rank,
                    depth=max(int(extra_depth), 1),
                    effective_hidden_dim=extra_effective_hidden_dim,
                    effective_rank=extra_effective_factorized_rank,
                    max_factorized_params=extra_max_factorized_params,
                )
                self.effective_extra_hidden_dim = extra_hidden_dim
                self.effective_extra_factorized_rank = factorized_rank
                self.energy_extra_head = FactorizedEnergyExtraHead(
                    input_dim,
                    extra_hidden_dim,
                    factorized_rank=factorized_rank,
                    depth=max(int(extra_depth), 1),
                    dropout=dropout,
                    output_init_std=extra_init_std,
                    shared_block_count=extra_shared_block_count,
                    chunk_size=extra_chunk_size,
                    use_checkpoint=extra_use_checkpoint,
                )
            else:
                self.energy_extra_head = _make_mlp(
                    input_dim,
                    extra_hidden_dim,
                    1,
                    depth=max(int(extra_depth), 1),
                    dropout=dropout,
                    zero_init_output=True,
                )
                init_std = max(float(extra_init_std), 0.0)
                if init_std > 0.0:
                    final_layer = self.energy_extra_head[-1]
                    if isinstance(final_layer, nn.Linear):
                        nn.init.normal_(final_layer.weight, mean=0.0, std=init_std)
                        nn.init.zeros_(final_layer.bias)
        self.energy_support_attention_head: nn.Module | None = None
        support_attention_hidden_dim = max(int(support_attention_hidden_dim), 0)
        if support_attention_hidden_dim > 0:
            self.energy_support_attention_head = SupportSetEnergyAttentionHead(
                input_dim,
                support_attention_hidden_dim,
                head_count=support_attention_head_count,
                depth=support_attention_depth,
                dropout=dropout,
                output_init_std=support_attention_init_std,
            )

    def forward(
        self,
        conformer_features: torch.Tensor,
        evidence_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one energy value per conformer support point."""

        return self.forward_parts(conformer_features, evidence_features)["energy"]

    def forward_parts(
        self,
        conformer_features: torch.Tensor,
        evidence_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return total energy plus the base and residual components."""

        features = conformer_features
        if self.evidence_feature_dim > 0:
            if evidence_features is None:
                evidence = conformer_features.new_zeros(
                    conformer_features.shape[0],
                    self.evidence_feature_dim,
                )
            else:
                evidence = evidence_features.to(
                    device=conformer_features.device,
                    dtype=conformer_features.dtype,
                )
                if evidence.ndim == 1:
                    evidence = evidence.reshape(1, -1).expand(conformer_features.shape[0], -1)
            features = torch.cat([features, evidence], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=50.0, neginf=-50.0).clamp(
            min=-50.0,
            max=50.0,
        )
        base_energy = self.energy_head(features).squeeze(-1)
        base_energy = torch.nan_to_num(
            base_energy,
            nan=0.0,
            posinf=80.0,
            neginf=-80.0,
        ).clamp(min=-80.0, max=80.0)
        extra_energy_delta = base_energy.new_zeros(base_energy.shape)
        support_attention_delta = base_energy.new_zeros(base_energy.shape)
        energy = base_energy
        if self.energy_extra_head is not None and self.extra_scale > 0.0:
            extra_energy = self.energy_extra_head(features).squeeze(-1)
            extra_energy = torch.nan_to_num(
                extra_energy,
                nan=0.0,
                posinf=80.0,
                neginf=-80.0,
            ).clamp(min=-80.0, max=80.0)
            extra_energy_delta = extra_energy * extra_energy.new_tensor(self.extra_scale)
            energy = energy + extra_energy_delta
        if (
            self.energy_support_attention_head is not None
            and self.support_attention_scale > 0.0
        ):
            attention_energy = self.energy_support_attention_head(features).squeeze(-1)
            attention_energy = torch.nan_to_num(
                attention_energy,
                nan=0.0,
                posinf=80.0,
                neginf=-80.0,
            ).clamp(min=-80.0, max=80.0)
            support_attention_delta = attention_energy * attention_energy.new_tensor(
                self.support_attention_scale
            )
            energy = energy + support_attention_delta
        energy = torch.nan_to_num(energy, nan=0.0, posinf=80.0, neginf=-80.0).clamp(
            min=-80.0,
            max=80.0,
        )
        return {
            "energy": energy,
            "base_energy": base_energy,
            "energy_extra_delta": extra_energy_delta,
            "energy_support_attention_delta": support_attention_delta,
        }


class ConditionalPriorLogitAdapter(nn.Module):
    """Learn evidence-conditioned support-prior corrections over BioEmu samples.

    When ``set_context_dim`` is positive the adapter becomes support-set aware:
    it scores each conformer using both its local hidden state and pooled
    statistics of the whole generated support.  This is still
    permutation-equivariant over conformers, but it can learn the relative
    mass-shaping needed by CS-reweighted posterior distributions.
    """

    def __init__(
        self,
        conformer_feature_dim: int,
        hidden_dim: int,
        *,
        evidence_feature_dim: int = 0,
        depth: int = 2,
        dropout: float = 0.0,
        scale: float = 1.0,
        max_abs: float = 0.0,
        output_init_std: float = 0.0,
        set_context_dim: int = 0,
        set_context_depth: int = 1,
        include_base_log_prob_features: bool = False,
    ) -> None:
        super().__init__()
        self.evidence_feature_dim = max(int(evidence_feature_dim), 0)
        self.set_context_dim = max(int(set_context_dim), 0)
        self.include_base_log_prob_features = bool(include_base_log_prob_features)
        self.scale = max(float(scale), 0.0)
        self.max_abs = max(float(max_abs), 0.0)
        self.set_projection: nn.Module | None = None
        context_input_dim = int(conformer_feature_dim)
        if self.set_context_dim > 0:
            self.set_projection = _make_mlp(
                int(conformer_feature_dim),
                max(int(hidden_dim), self.set_context_dim),
                self.set_context_dim,
                depth=max(int(set_context_depth), 1),
                dropout=dropout,
                zero_init_output=False,
            )
            context_input_dim += 4 * self.set_context_dim
        if self.include_base_log_prob_features:
            context_input_dim += 4
        input_dim = context_input_dim + self.evidence_feature_dim
        self.logit_head = _make_mlp(
            input_dim,
            max(int(hidden_dim), 1),
            1,
            depth=max(int(depth), 1),
            dropout=dropout,
            zero_init_output=True,
        )
        init_std = max(float(output_init_std), 0.0)
        if init_std > 0.0:
            final_layer = self.logit_head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.normal_(final_layer.weight, mean=0.0, std=init_std)
                nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        conformer_features: torch.Tensor,
        evidence_features: torch.Tensor | None = None,
        base_prior_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = torch.nan_to_num(
            conformer_features,
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        ).clamp(min=-50.0, max=50.0)
        if self.set_projection is not None:
            set_hidden = self.set_projection(features)
            set_hidden = torch.nan_to_num(
                set_hidden,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            ).clamp(min=-50.0, max=50.0)
            set_mean = set_hidden.mean(dim=0, keepdim=True)
            set_centered = set_hidden - set_mean
            set_std = torch.sqrt(
                set_centered.square().mean(dim=0, keepdim=True).clamp_min(1.0e-12)
            )
            set_max = set_hidden.amax(dim=0, keepdim=True)
            set_context = torch.cat([set_mean, set_std, set_max], dim=-1).expand(
                conformer_features.shape[0],
                -1,
            )
            features = torch.cat([features, set_centered, set_context], dim=-1)
        if self.include_base_log_prob_features:
            support_count = int(conformer_features.shape[0])
            if (
                base_prior_log_probs is not None
                and int(base_prior_log_probs.numel()) == support_count
            ):
                prior = base_prior_log_probs.detach().to(
                    device=conformer_features.device,
                    dtype=conformer_features.dtype,
                ).reshape(support_count)
                prior = torch.nan_to_num(prior, nan=0.0, posinf=80.0, neginf=-80.0)
                prior = prior.clamp(min=-80.0, max=80.0)
            else:
                prior = conformer_features.new_zeros(support_count)
            centered_prior = prior - torch.mean(prior)
            prior_scale = torch.std(centered_prior, unbiased=False).clamp_min(1.0e-6)
            prior_z = torch.clamp(centered_prior / prior_scale, -5.0, 5.0)
            prior_q = torch.softmax(prior, dim=0)
            prior_mass_excess = torch.clamp(
                prior_q * float(max(support_count, 1)) - 1.0,
                -5.0,
                5.0,
            )
            prior_rank_order = torch.argsort(torch.argsort(prior))
            if support_count > 1:
                prior_rank = (
                    prior_rank_order.to(dtype=conformer_features.dtype)
                    / float(support_count - 1)
                    - 0.5
                )
            else:
                prior_rank = prior.new_zeros(support_count)
            prior_features = torch.stack(
                (
                    prior_z,
                    torch.square(prior_z),
                    prior_mass_excess,
                    prior_rank,
                ),
                dim=-1,
            )
            features = torch.cat([features, prior_features], dim=-1)
        if self.evidence_feature_dim > 0:
            if evidence_features is None:
                evidence = conformer_features.new_zeros(
                    conformer_features.shape[0],
                    self.evidence_feature_dim,
                )
            else:
                evidence = evidence_features.to(
                    device=conformer_features.device,
                    dtype=conformer_features.dtype,
                )
                if evidence.ndim == 1:
                    evidence = evidence.reshape(1, -1).expand(
                        conformer_features.shape[0],
                        -1,
                    )
            features = torch.cat([features, evidence], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=50.0, neginf=-50.0)
        features = features.clamp(min=-50.0, max=50.0)
        delta = self.logit_head(features).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=80.0, neginf=-80.0)
        if self.max_abs > 0.0:
            max_abs = delta.new_tensor(self.max_abs)
            delta = max_abs * torch.tanh(delta / max_abs.clamp_min(1.0e-6))
        delta = delta - torch.mean(delta, dim=-1, keepdim=True)
        delta = delta * delta.new_tensor(self.scale)
        return torch.nan_to_num(delta, nan=0.0, posinf=80.0, neginf=-80.0).clamp(
            min=-80.0,
            max=80.0,
        )


class PosteriorModeMixture(nn.Module):
    """Reconcile multiple posterior modes into one conformer population.

    Each mode is a generalized-Bayes posterior over the same x0 conformer
    support.  The exported measure remains one ``q(k)`` after convex mixing, so
    downstream chemical shifts are still computed as ``sum_k q_k phi(X_k)``.
    """

    def __init__(
        self,
        conformer_feature_dim: int,
        hidden_dim: int,
        *,
        evidence_feature_dim: int = 0,
        mode_count: int = 1,
        energy_scale: float = 1.0,
        energy_init_std: float = 0.0,
        mode_logit_scale: float = 1.0,
        mode_logit_temperature: float = 1.0,
        mode_logit_prior: list[float] | tuple[float, ...] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode_count = max(int(mode_count), 1)
        self.energy_scale = max(float(energy_scale), 0.0)
        self.mode_logit_scale = max(float(mode_logit_scale), 0.0)
        self.mode_logit_temperature = max(float(mode_logit_temperature), 1.0e-6)
        self.evidence_feature_dim = max(int(evidence_feature_dim), 0)
        prior_values = [0.0 for _ in range(self.mode_count)]
        for index, value in enumerate(list(mode_logit_prior or [])[: self.mode_count]):
            prior_values[index] = float(value)
        self.register_buffer(
            "mode_logit_prior",
            torch.tensor(prior_values, dtype=torch.float32),
            persistent=False,
        )
        input_dim = int(conformer_feature_dim) + self.evidence_feature_dim
        self.energy_delta_head = _make_mlp(
            input_dim,
            int(hidden_dim),
            self.mode_count,
            depth=2,
            dropout=dropout,
            zero_init_output=True,
        )
        init_std = max(float(energy_init_std), 0.0)
        if init_std > 0.0:
            nn.init.normal_(self.energy_delta_head[-1].weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.energy_delta_head[-1].bias)
        if self.evidence_feature_dim > 0:
            self.mode_logit_head = _make_mlp(
                self.evidence_feature_dim,
                int(hidden_dim),
                self.mode_count,
                depth=1,
                dropout=dropout,
                zero_init_output=True,
            )
            self.mode_logits = None
        else:
            self.mode_logit_head = None
            self.mode_logits = nn.Parameter(torch.zeros(self.mode_count))

    def _expanded_evidence(
        self,
        conformer_features: torch.Tensor,
        evidence_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.evidence_feature_dim <= 0:
            return conformer_features.new_zeros(conformer_features.shape[0], 0)
        if evidence_features is None:
            evidence = conformer_features.new_zeros(self.evidence_feature_dim)
        else:
            evidence = evidence_features.to(
                device=conformer_features.device,
                dtype=conformer_features.dtype,
            )
            if evidence.ndim == 2:
                evidence = evidence.mean(dim=0)
        if evidence.ndim != 1 or int(evidence.shape[-1]) != self.evidence_feature_dim:
            raise ValueError(
                "evidence_features must have shape [D] or [K,D] for posterior modes"
            )
        return evidence.reshape(1, -1).expand(conformer_features.shape[0], -1)

    def forward(
        self,
        conformer_features: torch.Tensor,
        base_energy: torch.Tensor,
        *,
        prior_log_probs: torch.Tensor | None = None,
        evidence_features: torch.Tensor | None = None,
        temperature: float = 1.0,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        evidence = self._expanded_evidence(conformer_features, evidence_features)
        mode_features = torch.cat([conformer_features, evidence], dim=-1)
        energy_delta = torch.tanh(self.energy_delta_head(mode_features))
        mode_energy = base_energy.reshape(1, -1) + self.energy_scale * energy_delta.transpose(0, 1)
        component_weights = posterior_weights(
            prior_log_probs,
            mode_energy,
            temperature=temperature,
            support_mask=support_mask,
        )
        if self.mode_logit_head is None:
            assert self.mode_logits is not None
            mode_logits = self.mode_logits.to(
                device=conformer_features.device,
                dtype=conformer_features.dtype,
            )
        else:
            evidence_summary = evidence[:1, :]
            mode_logits = self.mode_logit_head(evidence_summary).squeeze(0)
        mode_logits = mode_logits * self.mode_logit_scale
        mode_prior = self.mode_logit_prior.to(
            device=conformer_features.device,
            dtype=conformer_features.dtype,
        )
        mode_logits = (mode_logits + mode_prior) / self.mode_logit_temperature
        mode_weights = torch.softmax(mode_logits, dim=-1)
        weights = torch.sum(mode_weights.reshape(-1, 1) * component_weights, dim=0)
        return {
            "weights": weights,
            "mode_weights": mode_weights,
            "mode_component_weights": component_weights,
            "mode_energy_delta": energy_delta,
            "mode_logits": mode_logits,
            "mode_logit_prior": mode_prior,
        }


class ResidueLocalModeGate(nn.Module):
    """Residue/family-local mixture over global posterior-mode components."""

    def __init__(
        self,
        hidden_dim: int,
        atom_family_count: int,
        mode_count: int,
        *,
        gate_hidden_dim: int = 0,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.atom_family_count = max(int(atom_family_count), 1)
        self.mode_count = max(int(mode_count), 1)
        self.temperature = max(float(temperature), 1.0e-3)
        hidden = int(gate_hidden_dim) if int(gate_hidden_dim) > 0 else min(int(hidden_dim), 256)
        self.gate = _make_mlp(
            int(hidden_dim),
            max(hidden, 8),
            self.atom_family_count * self.mode_count,
            depth=2,
            dropout=dropout,
            zero_init_output=True,
        )

    def forward(self, measure_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if measure_hidden.ndim != 2:
            raise ValueError("measure_hidden must have shape [L, hidden_dim]")
        logits = self.gate(measure_hidden).view(
            measure_hidden.shape[0],
            self.atom_family_count,
            self.mode_count,
        )
        return logits, torch.softmax(logits / self.temperature, dim=-1)


class FamilyMechanismEvidenceEnergyHead(nn.Module):
    """Learn a family-specific evidence compatibility residual energy.

    The head only consumes conditioning/evidence rows.  Supervised rows stay
    outside this module and remain reserved for the ensemble-mean loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        mechanism_feature_dim: int = 0,
        *,
        family_index: int = 0,
        count_prefix: str = "family",
        dropout: float = 0.0,
        chunk_size: int = 4,
        use_checkpoint: bool = False,
        head_hidden_dim: int | None = None,
        head_depth: int = 2,
    ) -> None:
        super().__init__()
        self.family_index = int(family_index)
        self.count_prefix = str(count_prefix)
        self.hidden_dim = int(hidden_dim)
        self.mechanism_feature_dim = max(int(mechanism_feature_dim), 0)
        self.chunk_size = max(int(chunk_size), 1)
        self.use_checkpoint = bool(use_checkpoint)
        scalar_feature_dim = 6
        input_dim = self.hidden_dim + self.mechanism_feature_dim + scalar_feature_dim
        energy_hidden_dim = max(int(head_hidden_dim or self.hidden_dim), 1)
        self.energy_head = _make_mlp(
            input_dim,
            energy_hidden_dim,
            1,
            depth=max(int(head_depth), 1),
            dropout=dropout,
            zero_init_output=True,
        )

    def _energy_forward_from_parts(
        self,
        hidden_rows: torch.Tensor,
        mechanism_rows: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run the evidence MLP without materializing a wide concat tensor."""

        children = list(self.energy_head.children())
        if children and isinstance(children[0], nn.Linear):
            first = children[0]
            hidden_dim = self.hidden_dim
            mechanism_dim = self.mechanism_feature_dim
            hidden_weight = first.weight[:, :hidden_dim]
            mechanism_weight = first.weight[:, hidden_dim : hidden_dim + mechanism_dim]
            scalar_weight = first.weight[:, hidden_dim + mechanism_dim :]
            x = F.linear(hidden_rows, hidden_weight, first.bias)
            if mechanism_dim > 0:
                x = x + F.linear(mechanism_rows, mechanism_weight, None)
            x = x + F.linear(scalar_features, scalar_weight, None)
            for module in children[1:]:
                x = module(x)
            return torch.tanh(x.squeeze(-1))
        row_features = torch.cat([hidden_rows, mechanism_rows, scalar_features], dim=-1)
        return torch.tanh(self.energy_head(row_features).squeeze(-1))

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        conformer_phi: torch.Tensor,
        evidence_grid: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        mechanism_features: torch.Tensor | None = None,
        energy_scale: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Return a conformer-wise learned evidence residual energy."""

        scale = max(float(energy_scale), 0.0)
        support_count = int(conformer_phi.shape[0])
        count_key = f"{self.count_prefix}_evidence_points"
        if scale <= 0.0:
            return conformer_phi.new_zeros(support_count), {count_key: 0}
        if decoder_hidden.ndim != 3 or conformer_phi.ndim != 3:
            raise ValueError("decoder_hidden and conformer_phi must have shape [K, L, D]")
        if decoder_hidden.shape[:2] != conformer_phi.shape[:2]:
            raise ValueError("decoder_hidden and conformer_phi must share [K, L]")
        target = evidence_grid.to(device=conformer_phi.device, dtype=conformer_phi.dtype)
        mask = observed_mask.to(device=conformer_phi.device, dtype=torch.bool)
        family_index = int(self.family_index)
        if (
            target.ndim != 2
            or mask.ndim != 2
            or target.shape[1] <= family_index
            or family_index < 0
        ):
            return conformer_phi.new_zeros(support_count), {count_key: 0}
        family_mask = mask[:, family_index] & torch.isfinite(target[:, family_index])
        if not bool(torch.any(family_mask)):
            return conformer_phi.new_zeros(support_count), {count_key: 0}

        row_indices = torch.nonzero(family_mask, as_tuple=False).flatten()
        center = conformer_phi.new_tensor(float(ATOM_FAMILY_PPM_CENTERS[family_index]))
        ppm_scale = conformer_phi.new_tensor(
            float(ATOM_FAMILY_PPM_SCALES[family_index])
        ).clamp_min(1.0e-3)
        mechanism_source = None
        if self.mechanism_feature_dim > 0 and mechanism_features is not None:
            mechanism_source = mechanism_features.to(
                device=decoder_hidden.device,
                dtype=decoder_hidden.dtype,
            )
            if mechanism_source.shape[-1] != self.mechanism_feature_dim:
                raise ValueError(
                    "mechanism_features last dimension does not match "
                    f"{self.mechanism_feature_dim}"
                )
        energy_sum = conformer_phi.new_zeros(support_count)
        processed_rows = 0
        chunk_size = self.chunk_size
        use_checkpoint = self.use_checkpoint and torch.is_grad_enabled()

        for _offset, row_start, row_width in _iter_contiguous_index_runs(
            row_indices,
            chunk_size=chunk_size,
        ):
            hidden_rows = decoder_hidden.narrow(1, row_start, row_width)
            pred = conformer_phi.narrow(1, row_start, row_width).select(
                -1,
                family_index,
            )
            evidence_values = target.narrow(0, row_start, row_width).select(
                1,
                family_index,
            )
            evidence = evidence_values.reshape(1, -1)
            evidence = evidence.expand_as(pred)
            normalized_pred = (pred - center) / ppm_scale
            normalized_evidence = (evidence - center) / ppm_scale
            residual = (pred - evidence) / ppm_scale
            scalar_features = torch.stack(
                [
                    normalized_pred.clamp(-6.0, 6.0),
                    normalized_evidence.clamp(-6.0, 6.0),
                    residual.clamp(-6.0, 6.0),
                    torch.abs(residual).clamp(0.0, 6.0),
                    residual.square().clamp(0.0, 36.0),
                    torch.sign(residual),
                ],
                dim=-1,
            )
            if self.mechanism_feature_dim > 0:
                if mechanism_source is None:
                    mechanism_rows = hidden_rows.new_zeros(
                        *hidden_rows.shape[:-1],
                        self.mechanism_feature_dim,
                    )
                else:
                    mechanism_rows = mechanism_source.narrow(1, row_start, row_width)
            else:
                mechanism_rows = hidden_rows.new_zeros(*hidden_rows.shape[:-1], 0)
            if use_checkpoint:
                row_energy = checkpoint(
                    self._energy_forward_from_parts,
                    hidden_rows,
                    mechanism_rows,
                    scalar_features,
                    use_reentrant=False,
                )
            else:
                row_energy = self._energy_forward_from_parts(
                    hidden_rows,
                    mechanism_rows,
                    scalar_features,
                )
            energy_sum = energy_sum + row_energy.sum(dim=1)
            processed_rows += int(row_width)
        energy = scale * energy_sum / max(processed_rows, 1)
        return energy, {count_key: int(torch.count_nonzero(family_mask).detach().cpu())}


class HNMechanismEvidenceEnergyHead(FamilyMechanismEvidenceEnergyHead):
    """HN-specific compatibility energy kept for backwards-compatible imports."""

    def __init__(
        self,
        hidden_dim: int,
        mechanism_feature_dim: int = 0,
        *,
        dropout: float = 0.0,
        chunk_size: int = 4,
        use_checkpoint: bool = False,
        head_hidden_dim: int | None = None,
        head_depth: int = 2,
    ) -> None:
        super().__init__(
            hidden_dim,
            mechanism_feature_dim,
            family_index=0,
            count_prefix="hn",
            dropout=dropout,
            chunk_size=chunk_size,
            use_checkpoint=use_checkpoint,
            head_hidden_dim=head_hidden_dim,
            head_depth=head_depth,
        )


class BoundedEnsembleResidual(nn.Module):
    """Small local calibration residual attached after ensemble averaging."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        atom_family_count: int = len(ATOM_FAMILY_NAMES),
        max_abs: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_abs = max(float(max_abs), 0.0)
        self.residual_head = _make_mlp(
            hidden_dim,
            hidden_dim,
            int(atom_family_count),
            depth=1,
            zero_init_output=True,
        )

    def forward(self, measure_hidden: torch.Tensor) -> torch.Tensor:
        if self.max_abs <= 0.0:
            return measure_hidden.new_zeros(
                measure_hidden.shape[0],
                self.residual_head[-1].out_features,
            )
        return self.max_abs * torch.tanh(self.residual_head(measure_hidden))


class _CSLatentFieldLocalBlock(nn.Module):
    """Cheap local sequence-context block for a residue-indexed CS field."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.local_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("CS latent field hidden must have shape [L, hidden_dim]")
        update = self.norm(hidden).transpose(0, 1).unsqueeze(0)
        update = self.local_conv(update).squeeze(0).transpose(0, 1)
        return hidden + self.output(update)


class MaskAwareCSLatentFieldConditioner(nn.Module):
    """Condition decoder features and posterior energy on a local CS evidence map."""

    def __init__(
        self,
        residue_feature_dim: int,
        decoder_hidden_dim: int,
        *,
        atom_family_count: int = len(ATOM_FAMILY_NAMES),
        hidden_dim: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
        residue_scale: float = 0.0,
        energy_scale: float = 0.0,
        energy_pooling: str = "mean",
        energy_attention_temperature: float = 1.0,
        energy_attention_init_std: float = 0.0,
        energy_salience_attention_weight: float = 0.0,
        energy_std_floor: float = 0.0,
        energy_std_floor_scale_cap: float = 0.0,
        family_token_scale: float = 0.0,
        family_token_temperature: float = 1.0,
        family_token_init_std: float = 0.0,
        film_scale: float = 0.0,
        film_init_std: float = 0.0,
        output_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.atom_family_count = max(int(atom_family_count), 1)
        self.hidden_dim = max(int(hidden_dim), 1)
        self.residue_scale = max(float(residue_scale), 0.0)
        self.energy_scale = max(float(energy_scale), 0.0)
        self.family_token_scale = max(float(family_token_scale), 0.0)
        self.family_token_temperature = max(float(family_token_temperature), 1.0e-4)
        self.film_scale = max(float(film_scale), 0.0)
        self.energy_pooling = (
            "attention"
            if str(energy_pooling or "mean").strip().lower()
            in {"attention", "attn", "residue_attention", "local_attention"}
            else "mean"
        )
        self.energy_attention_temperature = max(
            float(energy_attention_temperature),
            1.0e-4,
        )
        self.energy_salience_attention_weight = max(
            float(energy_salience_attention_weight),
            0.0,
        )
        self.energy_std_floor = max(float(energy_std_floor), 0.0)
        self.energy_std_floor_scale_cap = max(float(energy_std_floor_scale_cap), 0.0)
        input_dim = 2 * self.atom_family_count + 2
        self.input_projection = _make_mlp(
            input_dim,
            self.hidden_dim,
            self.hidden_dim,
            depth=1,
            dropout=dropout,
        )
        self.local_blocks = nn.ModuleList(
            [
                _CSLatentFieldLocalBlock(self.hidden_dim, dropout=dropout)
                for _ in range(max(int(depth), 0))
            ]
        )
        self.residue_adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(residue_feature_dim)),
        )
        self.family_token_embedding = nn.Embedding(
            self.atom_family_count,
            self.hidden_dim,
        )
        self.family_token_projection = _make_mlp(
            2 * self.hidden_dim + 2,
            self.hidden_dim,
            self.hidden_dim,
            depth=1,
            dropout=dropout,
        )
        self.family_token_attention = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.family_token_residue_adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(residue_feature_dim)),
        )
        self.residue_film_adapter = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, 2 * int(residue_feature_dim)),
        )
        self.energy_adapter = _make_mlp(
            int(decoder_hidden_dim) + self.hidden_dim,
            self.hidden_dim,
            1,
            depth=2,
            dropout=dropout,
            zero_init_output=True,
        )
        self.energy_attention_head: nn.Module | None = None
        if self.energy_pooling == "attention":
            self.energy_attention_head = _make_mlp(
                int(decoder_hidden_dim) + self.hidden_dim,
                self.hidden_dim,
                1,
                depth=1,
                dropout=dropout,
                zero_init_output=True,
            )
        init_std = max(float(output_init_std), 0.0)
        if init_std > 0.0:
            residue_layer = self.residue_adapter[-1]
            if isinstance(residue_layer, nn.Linear):
                nn.init.normal_(residue_layer.weight, mean=0.0, std=init_std)
                nn.init.zeros_(residue_layer.bias)
            energy_layer = self.energy_adapter[-1]
            if isinstance(energy_layer, nn.Linear):
                nn.init.normal_(energy_layer.weight, mean=0.0, std=init_std)
                nn.init.zeros_(energy_layer.bias)
        else:
            residue_layer = self.residue_adapter[-1]
            if isinstance(residue_layer, nn.Linear):
                nn.init.zeros_(residue_layer.weight)
                nn.init.zeros_(residue_layer.bias)
        family_token_init_std = max(float(family_token_init_std), 0.0)
        family_token_layer = self.family_token_residue_adapter[-1]
        if isinstance(family_token_layer, nn.Linear):
            if family_token_init_std > 0.0:
                nn.init.normal_(
                    family_token_layer.weight,
                    mean=0.0,
                    std=family_token_init_std,
                )
                nn.init.zeros_(family_token_layer.bias)
            else:
                nn.init.zeros_(family_token_layer.weight)
                nn.init.zeros_(family_token_layer.bias)
        family_attention_layer = self.family_token_attention[-1]
        if family_token_init_std > 0.0 and isinstance(family_attention_layer, nn.Linear):
            nn.init.normal_(
                family_attention_layer.weight,
                mean=0.0,
                std=family_token_init_std,
            )
            nn.init.zeros_(family_attention_layer.bias)
        film_init_std = max(float(film_init_std), 0.0)
        film_layer = self.residue_film_adapter[-1]
        if isinstance(film_layer, nn.Linear):
            if film_init_std > 0.0:
                nn.init.normal_(film_layer.weight, mean=0.0, std=film_init_std)
                nn.init.zeros_(film_layer.bias)
            else:
                nn.init.zeros_(film_layer.weight)
                nn.init.zeros_(film_layer.bias)
        attention_init_std = max(float(energy_attention_init_std), 0.0)
        if attention_init_std > 0.0 and self.energy_attention_head is not None:
            attention_layer = self.energy_attention_head[-1]
            if isinstance(attention_layer, nn.Linear):
                nn.init.normal_(
                    attention_layer.weight,
                    mean=0.0,
                    std=attention_init_std,
                )
                nn.init.zeros_(attention_layer.bias)
        centers = torch.tensor(
            ATOM_FAMILY_PPM_CENTERS[: self.atom_family_count],
            dtype=torch.float32,
        )
        scales = torch.tensor(
            ATOM_FAMILY_PPM_SCALES[: self.atom_family_count],
            dtype=torch.float32,
        ).clamp_min(1.0e-3)
        if centers.numel() < self.atom_family_count:
            centers = F.pad(centers, (0, self.atom_family_count - centers.numel()))
            scales = F.pad(scales, (0, self.atom_family_count - scales.numel()), value=1.0)
        self.register_buffer("ppm_centers", centers.reshape(1, -1), persistent=False)
        self.register_buffer("ppm_scales", scales.reshape(1, -1), persistent=False)

    def _inactive_payload(
        self,
        *,
        reference: torch.Tensor,
        residue_count: int,
        residue_feature_dim: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "residue_delta": reference.new_zeros(residue_count, residue_feature_dim),
            "field_hidden": reference.new_zeros(residue_count, self.hidden_dim),
            "residue_mask": torch.zeros(
                residue_count,
                dtype=torch.bool,
                device=reference.device,
            ),
            "residue_salience": reference.new_zeros(residue_count),
            "active": reference.new_tensor(0.0),
            "observed_cell_count": reference.new_tensor(0.0),
            "observed_residue_count": reference.new_tensor(0.0),
            "observed_residue_fraction": reference.new_tensor(0.0),
            "residue_delta_abs_mean": reference.new_tensor(0.0),
            "residue_delta_abs_max": reference.new_tensor(0.0),
            "residue_film_gamma": reference.new_zeros(
                residue_count,
                residue_feature_dim,
            ),
            "residue_film_beta": reference.new_zeros(
                residue_count,
                residue_feature_dim,
            ),
            "residue_film_active": reference.new_tensor(0.0),
            "residue_film_gamma_abs_mean": reference.new_tensor(0.0),
            "residue_film_gamma_abs_max": reference.new_tensor(0.0),
            "residue_film_beta_abs_mean": reference.new_tensor(0.0),
            "residue_film_beta_abs_max": reference.new_tensor(0.0),
            "family_token_active": reference.new_tensor(0.0),
            "family_token_observed_count": reference.new_tensor(0.0),
            "family_token_attention_entropy": reference.new_tensor(0.0),
            "family_token_attention_top_mass": reference.new_tensor(0.0),
            "family_token_residue_delta_abs_mean": reference.new_tensor(0.0),
            "family_token_residue_delta_abs_max": reference.new_tensor(0.0),
        }

    def _family_token_delta(
        self,
        *,
        field_hidden: torch.Tensor,
        normalized: torch.Tensor,
        observed: torch.Tensor,
        reference: torch.Tensor,
        residue_feature_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        zeros = field_hidden.new_zeros(field_hidden.shape[0], int(residue_feature_dim))
        context_zeros = field_hidden.new_zeros(field_hidden.shape[0], self.hidden_dim)
        scalar_zero = reference.new_tensor(0.0)
        if self.family_token_scale <= 0.0 or field_hidden.ndim != 2:
            return zeros, context_zeros, {
                "family_token_active": scalar_zero,
                "family_token_observed_count": scalar_zero,
                "family_token_attention_entropy": scalar_zero,
                "family_token_attention_top_mass": scalar_zero,
                "family_token_residue_delta_abs_mean": scalar_zero,
                "family_token_residue_delta_abs_max": scalar_zero,
            }
        residue_count = int(field_hidden.shape[0])
        family_count = int(normalized.shape[1])
        if residue_count <= 0 or family_count <= 0:
            return zeros, context_zeros, {
                "family_token_active": scalar_zero,
                "family_token_observed_count": scalar_zero,
                "family_token_attention_entropy": scalar_zero,
                "family_token_attention_top_mass": scalar_zero,
                "family_token_residue_delta_abs_mean": scalar_zero,
                "family_token_residue_delta_abs_max": scalar_zero,
            }
        token_mask = observed[:, :family_count].to(
            device=field_hidden.device,
            dtype=torch.bool,
        )
        if not bool(torch.any(token_mask)):
            return zeros, context_zeros, {
                "family_token_active": scalar_zero,
                "family_token_observed_count": scalar_zero,
                "family_token_attention_entropy": scalar_zero,
                "family_token_attention_top_mass": scalar_zero,
                "family_token_residue_delta_abs_mean": scalar_zero,
                "family_token_residue_delta_abs_max": scalar_zero,
            }
        family_ids = torch.arange(family_count, device=field_hidden.device)
        family_embedding = self.family_token_embedding(family_ids).to(
            dtype=field_hidden.dtype,
        )
        residue_field = field_hidden[:, None, :].expand(-1, family_count, -1)
        family_field = family_embedding[None, :, :].expand(residue_count, -1, -1)
        token_values = normalized[:, :family_count].to(
            device=field_hidden.device,
            dtype=field_hidden.dtype,
        )
        token_inputs = torch.cat(
            [
                residue_field,
                family_field,
                token_values.unsqueeze(-1),
                token_mask.to(dtype=field_hidden.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        token_hidden = self.family_token_projection(token_inputs)
        attention_logits = self.family_token_attention(token_hidden).squeeze(-1)
        attention_logits = attention_logits / attention_logits.new_tensor(
            self.family_token_temperature
        )
        attention_logits = torch.where(
            token_mask,
            attention_logits,
            attention_logits.new_full(attention_logits.shape, -1.0e6),
        )
        token_weights = torch.softmax(attention_logits, dim=-1)
        token_weights = torch.where(
            token_mask,
            token_weights,
            torch.zeros_like(token_weights),
        )
        token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        token_context = torch.sum(token_weights.unsqueeze(-1) * token_hidden, dim=1)
        residue_has_token = torch.any(token_mask, dim=-1)
        token_context = torch.where(
            residue_has_token.unsqueeze(-1),
            token_context,
            torch.zeros_like(token_context),
        )
        token_delta = torch.tanh(self.family_token_residue_adapter(token_context))
        token_delta = token_delta * token_delta.new_tensor(self.family_token_scale)
        token_delta = token_delta * residue_has_token.to(
            dtype=token_delta.dtype
        ).unsqueeze(-1)
        token_delta = torch.nan_to_num(
            token_delta,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        observed_weights = token_weights[residue_has_token]
        if int(observed_weights.numel()) > 0:
            safe_weights = observed_weights.clamp_min(1.0e-12)
            entropy = -torch.sum(safe_weights * torch.log(safe_weights), dim=-1)
            top_mass = torch.max(observed_weights, dim=-1).values
            entropy_mean = torch.mean(entropy.detach())
            top_mass_mean = torch.mean(top_mass.detach())
        else:
            entropy_mean = scalar_zero
            top_mass_mean = scalar_zero
        return token_delta, token_context, {
            "family_token_active": reference.new_tensor(1.0),
            "family_token_observed_count": torch.count_nonzero(token_mask).to(
                dtype=reference.dtype
            ),
            "family_token_attention_entropy": entropy_mean,
            "family_token_attention_top_mass": top_mass_mean,
            "family_token_residue_delta_abs_mean": torch.mean(
                torch.abs(token_delta.detach())
            ),
            "family_token_residue_delta_abs_max": torch.max(
                torch.abs(token_delta.detach())
            ),
        }

    def _residue_film(
        self,
        *,
        field_hidden: torch.Tensor,
        token_context: torch.Tensor,
        residue_mask: torch.Tensor,
        reference: torch.Tensor,
        residue_feature_dim: int,
    ) -> dict[str, torch.Tensor]:
        gamma_zeros = field_hidden.new_zeros(field_hidden.shape[0], residue_feature_dim)
        beta_zeros = field_hidden.new_zeros(field_hidden.shape[0], residue_feature_dim)
        scalar_zero = reference.new_tensor(0.0)
        if self.film_scale <= 0.0:
            return {
                "residue_film_gamma": gamma_zeros,
                "residue_film_beta": beta_zeros,
                "residue_film_active": scalar_zero,
                "residue_film_gamma_abs_mean": scalar_zero,
                "residue_film_gamma_abs_max": scalar_zero,
                "residue_film_beta_abs_mean": scalar_zero,
                "residue_film_beta_abs_max": scalar_zero,
            }
        if field_hidden.ndim != 2 or token_context.shape != field_hidden.shape:
            return {
                "residue_film_gamma": gamma_zeros,
                "residue_film_beta": beta_zeros,
                "residue_film_active": scalar_zero,
                "residue_film_gamma_abs_mean": scalar_zero,
                "residue_film_gamma_abs_max": scalar_zero,
                "residue_film_beta_abs_mean": scalar_zero,
                "residue_film_beta_abs_max": scalar_zero,
            }
        if residue_mask.ndim != 1 or not bool(torch.any(residue_mask)):
            return {
                "residue_film_gamma": gamma_zeros,
                "residue_film_beta": beta_zeros,
                "residue_film_active": scalar_zero,
                "residue_film_gamma_abs_mean": scalar_zero,
                "residue_film_gamma_abs_max": scalar_zero,
                "residue_film_beta_abs_mean": scalar_zero,
                "residue_film_beta_abs_max": scalar_zero,
            }
        film_input = torch.cat([field_hidden, token_context], dim=-1)
        film_raw = torch.tanh(self.residue_film_adapter(film_input))
        film_raw = film_raw * film_raw.new_tensor(self.film_scale)
        gamma, beta = torch.chunk(film_raw, 2, dim=-1)
        mask = residue_mask.to(device=gamma.device, dtype=gamma.dtype).unsqueeze(-1)
        gamma = torch.nan_to_num(
            gamma * mask,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        beta = torch.nan_to_num(
            beta * mask,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        return {
            "residue_film_gamma": gamma,
            "residue_film_beta": beta,
            "residue_film_active": reference.new_tensor(1.0),
            "residue_film_gamma_abs_mean": torch.mean(torch.abs(gamma.detach())),
            "residue_film_gamma_abs_max": torch.max(torch.abs(gamma.detach())),
            "residue_film_beta_abs_mean": torch.mean(torch.abs(beta.detach())),
            "residue_film_beta_abs_max": torch.max(torch.abs(beta.detach())),
        }

    def encode(
        self,
        cs_evidence_grid: torch.Tensor | None,
        cs_evidence_mask: torch.Tensor | None,
        *,
        reference: torch.Tensor,
        residue_count: int,
        residue_feature_dim: int,
    ) -> dict[str, torch.Tensor]:
        if (
            cs_evidence_grid is None
            or cs_evidence_mask is None
            or int(residue_count) <= 0
            or (
                self.residue_scale <= 0.0
                and self.energy_scale <= 0.0
                and self.family_token_scale <= 0.0
                and self.film_scale <= 0.0
            )
        ):
            return self._inactive_payload(
                reference=reference,
                residue_count=int(residue_count),
                residue_feature_dim=int(residue_feature_dim),
            )
        values = cs_evidence_grid.to(device=reference.device, dtype=reference.dtype)
        observed = cs_evidence_mask.to(device=reference.device, dtype=torch.bool)
        if values.ndim != 2 or observed.ndim != 2:
            return self._inactive_payload(
                reference=reference,
                residue_count=int(residue_count),
                residue_feature_dim=int(residue_feature_dim),
            )
        values = values[: int(residue_count), : self.atom_family_count]
        observed = observed[: int(residue_count), : self.atom_family_count]
        residue_pad = int(residue_count) - int(values.shape[0])
        family_pad = self.atom_family_count - int(values.shape[1])
        if residue_pad > 0 or family_pad > 0:
            values = F.pad(values, (0, max(family_pad, 0), 0, max(residue_pad, 0)))
            observed = F.pad(
                observed.to(dtype=reference.dtype),
                (0, max(family_pad, 0), 0, max(residue_pad, 0)),
            ).to(dtype=torch.bool)
        observed = observed & torch.isfinite(values)
        observed_count = torch.count_nonzero(observed)
        if int(observed_count.detach().cpu()) <= 0:
            return self._inactive_payload(
                reference=reference,
                residue_count=int(residue_count),
                residue_feature_dim=int(residue_feature_dim),
            )
        centers = self.ppm_centers.to(device=reference.device, dtype=reference.dtype)
        scales = self.ppm_scales.to(device=reference.device, dtype=reference.dtype)
        normalized = ((values - centers) / scales).clamp(min=-8.0, max=8.0)
        mask_float = observed.to(dtype=reference.dtype)
        normalized = torch.nan_to_num(normalized * mask_float, nan=0.0)
        observed_per_residue = torch.sum(mask_float, dim=-1).clamp_min(1.0)
        residue_salience = torch.sum(torch.abs(normalized), dim=-1) / observed_per_residue
        observed_fraction = mask_float.mean(dim=-1, keepdim=True)
        if int(residue_count) > 1:
            position = torch.linspace(
                -1.0,
                1.0,
                steps=int(residue_count),
                dtype=reference.dtype,
                device=reference.device,
            ).reshape(-1, 1)
        else:
            position = reference.new_zeros(1, 1)
        field_input = torch.cat(
            [normalized, mask_float, observed_fraction, position],
            dim=-1,
        )
        field_hidden = self.input_projection(field_input)
        for block in self.local_blocks:
            field_hidden = block(field_hidden)
        residue_mask = torch.any(observed, dim=-1)
        residue_salience = torch.where(
            residue_mask,
            residue_salience,
            torch.zeros_like(residue_salience),
        )
        residue_delta = torch.tanh(self.residue_adapter(field_hidden))
        residue_delta = residue_delta * residue_delta.new_tensor(self.residue_scale)
        residue_delta = residue_delta * residue_mask.to(dtype=residue_delta.dtype).unsqueeze(-1)
        residue_delta = torch.nan_to_num(residue_delta, nan=0.0, posinf=10.0, neginf=-10.0)
        family_token_delta, family_token_context, family_token_metrics = (
            self._family_token_delta(
            field_hidden=field_hidden,
            normalized=normalized,
            observed=observed,
            reference=reference,
            residue_feature_dim=int(residue_feature_dim),
            )
        )
        residue_delta = torch.nan_to_num(
            residue_delta + family_token_delta,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        film_metrics = self._residue_film(
            field_hidden=field_hidden,
            token_context=family_token_context,
            residue_mask=residue_mask,
            reference=reference,
            residue_feature_dim=int(residue_feature_dim),
        )
        observed_residue_count = torch.count_nonzero(residue_mask).to(dtype=reference.dtype)
        return {
            "residue_delta": residue_delta,
            "field_hidden": field_hidden,
            "residue_mask": residue_mask,
            "residue_salience": residue_salience,
            "active": reference.new_tensor(1.0),
            "observed_cell_count": observed_count.to(dtype=reference.dtype),
            "observed_residue_count": observed_residue_count,
            "observed_residue_fraction": observed_residue_count
            / reference.new_tensor(max(int(residue_count), 1)),
            "residue_delta_abs_mean": torch.mean(torch.abs(residue_delta.detach())),
            "residue_delta_abs_max": torch.max(torch.abs(residue_delta.detach())),
            **film_metrics,
            **family_token_metrics,
        }

    def energy_delta(
        self,
        decoder_hidden: torch.Tensor,
        *,
        field_hidden: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta, _ = self.energy_delta_with_diagnostics(
            decoder_hidden,
            field_hidden=field_hidden,
            residue_mask=residue_mask,
        )
        return delta

    def energy_delta_with_diagnostics(
        self,
        decoder_hidden: torch.Tensor,
        *,
        field_hidden: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_salience: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def inactive() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            zeros = decoder_hidden.new_zeros(decoder_hidden.shape[0])
            scalar_zero = decoder_hidden.new_tensor(0.0)
            return zeros, {
                "cs_latent_field_energy_attention_active": scalar_zero,
                "cs_latent_field_energy_attention_entropy": scalar_zero,
                "cs_latent_field_energy_attention_top_mass": scalar_zero,
                "cs_latent_field_energy_delta_raw_std": scalar_zero,
                "cs_latent_field_energy_delta_floor_scale": scalar_zero,
                "cs_latent_field_energy_attention_salience_bias_active": scalar_zero,
                "cs_latent_field_residue_salience_std": scalar_zero,
            }

        if self.energy_scale <= 0.0 or decoder_hidden.ndim != 3:
            return inactive()
        if field_hidden.ndim != 2 or residue_mask.ndim != 1:
            return inactive()
        residue_count = min(int(decoder_hidden.shape[1]), int(field_hidden.shape[0]))
        if residue_count <= 0:
            return inactive()
        active_mask = residue_mask[:residue_count].to(
            device=decoder_hidden.device,
            dtype=torch.bool,
        )
        if not bool(torch.any(active_mask)):
            return inactive()
        hidden = decoder_hidden[:, :residue_count, :]
        field = field_hidden[:residue_count, :].to(
            device=decoder_hidden.device,
            dtype=decoder_hidden.dtype,
        )
        pair_features = torch.cat(
            [hidden, field.unsqueeze(0).expand(hidden.shape[0], -1, -1)],
            dim=-1,
        )
        residue_scores = self.energy_adapter(pair_features).squeeze(-1)
        mask = active_mask.reshape(1, -1)
        salience_bias_active = decoder_hidden.new_tensor(0.0)
        salience_std = decoder_hidden.new_tensor(0.0)
        if self.energy_pooling == "attention" and self.energy_attention_head is not None:
            attention_logits = self.energy_attention_head(pair_features).squeeze(-1)
            attention_logits = attention_logits / attention_logits.new_tensor(
                self.energy_attention_temperature
            )
            if (
                self.energy_salience_attention_weight > 0.0
                and residue_salience is not None
                and residue_salience.ndim == 1
            ):
                salience = residue_salience[:residue_count].to(
                    device=decoder_hidden.device,
                    dtype=decoder_hidden.dtype,
                )
                salience = torch.where(
                    active_mask,
                    salience,
                    torch.zeros_like(salience),
                )
                active_salience = salience[active_mask]
                salience_mean = active_salience.mean()
                salience_std = torch.std(active_salience, unbiased=False)
                salience_z = (salience - salience_mean) / salience_std.clamp_min(
                    1.0e-6
                )
                attention_logits = attention_logits + attention_logits.new_tensor(
                    self.energy_salience_attention_weight
                ) * salience_z.reshape(1, -1)
                salience_bias_active = decoder_hidden.new_tensor(1.0)
            attention_logits = torch.where(
                mask,
                attention_logits,
                attention_logits.new_full(attention_logits.shape, -1.0e6),
            )
            weights = torch.softmax(attention_logits, dim=-1)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            attention_active = decoder_hidden.new_tensor(1.0)
        else:
            weights = active_mask.to(dtype=decoder_hidden.dtype).reshape(1, -1)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = weights.expand(residue_scores.shape[0], -1)
            attention_active = decoder_hidden.new_tensor(0.0)
        delta = (residue_scores * weights).sum(dim=-1)
        delta = delta * delta.new_tensor(self.energy_scale)
        delta = delta - delta.mean()
        raw_delta_std = torch.std(delta, unbiased=False)
        floor_scale = delta.new_tensor(1.0)
        if self.energy_std_floor > 0.0 and int(delta.numel()) > 1:
            target_std = delta.new_tensor(self.energy_std_floor)
            floor_scale = torch.maximum(
                floor_scale,
                target_std / raw_delta_std.detach().clamp_min(1.0e-8),
            )
            if self.energy_std_floor_scale_cap > 0.0:
                floor_scale = floor_scale.clamp(
                    max=delta.new_tensor(self.energy_std_floor_scale_cap)
                )
            delta = delta * floor_scale
        safe_weights = weights.clamp_min(1.0e-12)
        attention_entropy = -torch.sum(safe_weights * torch.log(safe_weights), dim=-1)
        attention_top_mass = torch.max(weights, dim=-1).values
        delta = torch.nan_to_num(delta, nan=0.0, posinf=80.0, neginf=-80.0).clamp(
            min=-80.0,
            max=80.0,
        )
        return delta, {
            "cs_latent_field_energy_attention_active": attention_active.detach(),
            "cs_latent_field_energy_attention_entropy": torch.mean(
                attention_entropy.detach()
            ),
            "cs_latent_field_energy_attention_top_mass": torch.mean(
                attention_top_mass.detach()
            ),
            "cs_latent_field_energy_delta_raw_std": raw_delta_std.detach(),
            "cs_latent_field_energy_delta_floor_scale": floor_scale.detach(),
            "cs_latent_field_energy_attention_salience_bias_active": (
                salience_bias_active.detach()
            ),
            "cs_latent_field_residue_salience_std": salience_std.detach(),
        }


class X0PosteriorEnsembleModel(nn.Module):
    """Frozen-support posterior model with conformer CS decoder and reweighter."""

    def __init__(
        self,
        residue_feature_dim: int,
        hidden_dim: int,
        *,
        atom_family_count: int = len(ATOM_FAMILY_NAMES),
        edge_feature_dim: int = 0,
        edge_class_count: int = len(COMPACT_X2D_EDGE_CLASSES),
        evidence_feature_dim: int = 0,
        posterior_energy_hidden_dim: int | None = None,
        posterior_energy_init_std: float = 0.0,
        posterior_energy_extra_hidden_dim: int = 0,
        posterior_energy_extra_depth: int = 2,
        posterior_energy_extra_scale: float = 1.0,
        posterior_energy_extra_init_std: float = 0.0,
        posterior_energy_extra_factorized_rank: int = 0,
        posterior_energy_extra_effective_hidden_dim: int = 0,
        posterior_energy_extra_effective_factorized_rank: int = 0,
        posterior_energy_extra_max_factorized_params: int = 450_000_000,
        posterior_energy_extra_shared_block_count: int = 0,
        posterior_energy_extra_chunk_size: int = 0,
        posterior_energy_extra_use_checkpoint: bool = True,
        posterior_energy_support_attention_hidden_dim: int = 0,
        posterior_energy_support_attention_head_count: int = 4,
        posterior_energy_support_attention_depth: int = 1,
        posterior_energy_support_attention_scale: float = 0.0,
        posterior_energy_support_attention_init_std: float = 0.0,
        cs_latent_field_conditioning_hidden_dim: int = 0,
        cs_latent_field_conditioning_depth: int = 1,
        cs_latent_field_conditioning_residue_scale: float = 0.0,
        cs_latent_field_conditioning_energy_scale: float = 0.0,
        cs_latent_field_conditioning_energy_pooling: str = "mean",
        cs_latent_field_conditioning_energy_attention_temperature: float = 1.0,
        cs_latent_field_conditioning_energy_attention_init_std: float = 0.0,
        cs_latent_field_conditioning_energy_salience_attention_weight: float = 0.0,
        cs_latent_field_conditioning_energy_std_floor: float = 0.0,
        cs_latent_field_conditioning_energy_std_floor_scale_cap: float = 0.0,
        cs_latent_field_conditioning_family_token_scale: float = 0.0,
        cs_latent_field_conditioning_family_token_temperature: float = 1.0,
        cs_latent_field_conditioning_family_token_init_std: float = 0.0,
        cs_latent_field_conditioning_film_scale: float = 0.0,
        cs_latent_field_conditioning_film_init_std: float = 0.0,
        cs_latent_field_conditioning_dropout: float = 0.0,
        cs_latent_field_conditioning_init_std: float = 0.0,
        prior_logit_adapter_hidden_dim: int = 0,
        prior_logit_adapter_depth: int = 2,
        prior_logit_adapter_scale: float = 1.0,
        prior_logit_adapter_max_abs: float = 0.0,
        prior_logit_adapter_output_init_std: float = 0.0,
        prior_logit_adapter_set_context_dim: int = 0,
        prior_logit_adapter_set_context_depth: int = 1,
        prior_logit_adapter_include_base_log_prob_features: bool = False,
        residual_max_abs: float = 0.0,
        dropout: float = 0.0,
        evidence_head_chunk_size: int = 4,
        evidence_head_use_checkpoint: bool = False,
        evidence_head_hidden_dim: int = 0,
        evidence_head_depth: int = 2,
        posterior_mode_count: int = 1,
        posterior_mode_energy_scale: float = 0.0,
        posterior_mode_energy_init_std: float = 0.0,
        posterior_mode_logit_scale: float = 1.0,
        posterior_mode_logit_temperature: float = 1.0,
        posterior_mode_logit_prior: list[float] | tuple[float, ...] | None = None,
        residue_local_mode_gate_enabled: bool = False,
        residue_local_mode_gate_hidden_dim: int = 0,
        residue_local_mode_gate_temperature: float = 1.0,
        decoder_mechanism_support_basis_count: int = 0,
        decoder_mechanism_support_basis_cap: float = 0.0,
        decoder_hn_signed_support_basis_count: int = 0,
        decoder_hn_signed_support_basis_cap: float = 0.0,
        decoder_hn_direct_support_spread_scale_ppm: float = 0.0,
        decoder_hn_direct_support_spread_trainable_scale_cap_ppm: float = 0.0,
        decoder_hn_direct_support_spread_trainable_scale_init_ppm: float = 0.0,
        decoder_family_direct_support_spread_scale_ppm_by_family: dict[
            str, float
        ]
        | None = None,
        decoder_family_direct_support_spread_trainable_scale_cap_ppm_by_family: dict[
            str, float
        ]
        | None = None,
        decoder_family_direct_support_spread_trainable_scale_init_ppm_by_family: dict[
            str, float
        ]
        | None = None,
        decoder_cprime_isolated_support_basis_count: int = 0,
        decoder_cprime_isolated_support_basis_cap: float = 0.0,
        decoder_n_ca_cb_support_deviation_enabled: bool = True,
        decoder_emit_support_basis_diagnostics: bool = True,
        decoder_use_checkpoint: bool = False,
        decoder_support_head_chunk_size: int = 0,
        compact_edge_encoder_sample_chunk_size: int = 0,
        decoder_capacity_adapter_hidden_dim: int = 0,
        decoder_capacity_adapter_depth: int = 3,
        decoder_capacity_adapter_cap_ppm: float = 0.0,
        decoder_capacity_adapter_scale: float = 1.0,
        decoder_capacity_adapter_trainable_scale_cap: float = 0.0,
        decoder_capacity_adapter_trainable_scale_init: float = 0.0,
        decoder_capacity_adapter_extra_hidden_dim: int = 0,
        decoder_capacity_adapter_extra_depth: int = 3,
        decoder_capacity_adapter_extra_cap_ppm: float = 0.0,
        decoder_capacity_adapter_extra_scale: float = 1.0,
        decoder_capacity_adapter_extra_expert_count: int = 1,
        decoder_capacity_adapter_extra_factorized_rank: int = 0,
        decoder_capacity_adapter_extra_effective_hidden_dim: int = 0,
        decoder_capacity_adapter_extra_effective_factorized_rank: int = 0,
        decoder_capacity_adapter_extra_max_factorized_params: int = 450_000_000,
        decoder_capacity_adapter_extra_output_init_std: float = 0.0,
        decoder_capacity_adapter_extra_family_scales: dict[str, float] | None = None,
        decoder_capacity_adapter_extra_family_gated: bool = False,
        decoder_capacity_adapter_extra_family_specific: bool = False,
        decoder_capacity_adapter_extra_family_headed: bool = False,
        decoder_capacity_adapter_aux_hidden_dim: int = 0,
        decoder_capacity_adapter_aux_depth: int = 2,
        decoder_capacity_adapter_aux_cap_ppm: float = 0.0,
        decoder_capacity_adapter_aux_scale: float = 1.0,
        decoder_capacity_adapter_aux_expert_count: int = 1,
        decoder_capacity_adapter_aux_output_init_std: float = 0.0,
        decoder_capacity_adapter_aux_family_scales: dict[str, float] | None = None,
        decoder_capacity_adapter_aux_family_gated: bool = False,
        decoder_capacity_adapter_aux_family_specific: bool = False,
        decoder_capacity_adapter_aux_family_headed: bool = False,
        decoder_family_affine_scale_cap: float = 0.0,
        decoder_family_affine_shift_cap_ppm: float = 0.0,
        decoder_family_affine_family_scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.edge_encoder: CompactX2DEdgeEncoder | None = None
        mechanism_dim = 0
        if int(edge_feature_dim) > 0:
            self.edge_encoder = CompactX2DEdgeEncoder(
                edge_feature_dim,
                hidden_dim,
                edge_class_count=edge_class_count,
                dropout=dropout,
                preserve_class_channels=True,
                sample_chunk_size=compact_edge_encoder_sample_chunk_size,
            )
            mechanism_dim = self.edge_encoder.output_dim
        energy_hidden_dim = int(
            posterior_energy_hidden_dim
            if posterior_energy_hidden_dim is not None
            else hidden_dim
        )
        self.decoder = ChemicalShiftConformerDecoder(
            residue_feature_dim,
            hidden_dim,
            atom_family_count=atom_family_count,
            mechanism_feature_dim=mechanism_dim,
            dropout=dropout,
            mechanism_support_basis_count=decoder_mechanism_support_basis_count,
            mechanism_support_basis_cap=decoder_mechanism_support_basis_cap,
            hn_signed_support_basis_count=decoder_hn_signed_support_basis_count,
            hn_signed_support_basis_cap=decoder_hn_signed_support_basis_cap,
            hn_direct_support_spread_scale_ppm=(
                decoder_hn_direct_support_spread_scale_ppm
            ),
            hn_direct_support_spread_trainable_scale_cap_ppm=(
                decoder_hn_direct_support_spread_trainable_scale_cap_ppm
            ),
            hn_direct_support_spread_trainable_scale_init_ppm=(
                decoder_hn_direct_support_spread_trainable_scale_init_ppm
            ),
            family_direct_support_spread_scale_ppm_by_family=(
                decoder_family_direct_support_spread_scale_ppm_by_family
            ),
            family_direct_support_spread_trainable_scale_cap_ppm_by_family=(
                decoder_family_direct_support_spread_trainable_scale_cap_ppm_by_family
            ),
            family_direct_support_spread_trainable_scale_init_ppm_by_family=(
                decoder_family_direct_support_spread_trainable_scale_init_ppm_by_family
            ),
            cprime_isolated_support_basis_count=(
                decoder_cprime_isolated_support_basis_count
            ),
            cprime_isolated_support_basis_cap=(
                decoder_cprime_isolated_support_basis_cap
            ),
            n_ca_cb_support_deviation_enabled=(
                decoder_n_ca_cb_support_deviation_enabled
            ),
            emit_support_basis_diagnostics=(
                decoder_emit_support_basis_diagnostics
            ),
            use_checkpoint=decoder_use_checkpoint,
            support_head_chunk_size=decoder_support_head_chunk_size,
            capacity_adapter_hidden_dim=decoder_capacity_adapter_hidden_dim,
            capacity_adapter_depth=decoder_capacity_adapter_depth,
            capacity_adapter_cap_ppm=decoder_capacity_adapter_cap_ppm,
            capacity_adapter_scale=decoder_capacity_adapter_scale,
            capacity_adapter_trainable_scale_cap=(
                decoder_capacity_adapter_trainable_scale_cap
            ),
            capacity_adapter_trainable_scale_init=(
                decoder_capacity_adapter_trainable_scale_init
            ),
            capacity_adapter_extra_hidden_dim=(
                decoder_capacity_adapter_extra_hidden_dim
            ),
            capacity_adapter_extra_depth=decoder_capacity_adapter_extra_depth,
            capacity_adapter_extra_cap_ppm=decoder_capacity_adapter_extra_cap_ppm,
            capacity_adapter_extra_scale=decoder_capacity_adapter_extra_scale,
            capacity_adapter_extra_expert_count=(
                decoder_capacity_adapter_extra_expert_count
            ),
            capacity_adapter_extra_factorized_rank=(
                decoder_capacity_adapter_extra_factorized_rank
            ),
            capacity_adapter_extra_effective_hidden_dim=(
                decoder_capacity_adapter_extra_effective_hidden_dim
            ),
            capacity_adapter_extra_effective_factorized_rank=(
                decoder_capacity_adapter_extra_effective_factorized_rank
            ),
            capacity_adapter_extra_max_factorized_params=(
                decoder_capacity_adapter_extra_max_factorized_params
            ),
            capacity_adapter_extra_output_init_std=(
                decoder_capacity_adapter_extra_output_init_std
            ),
            capacity_adapter_extra_family_scales=(
                decoder_capacity_adapter_extra_family_scales
            ),
            capacity_adapter_extra_family_gated=(
                decoder_capacity_adapter_extra_family_gated
            ),
            capacity_adapter_extra_family_specific=(
                decoder_capacity_adapter_extra_family_specific
            ),
            capacity_adapter_extra_family_headed=(
                decoder_capacity_adapter_extra_family_headed
            ),
            capacity_adapter_aux_hidden_dim=decoder_capacity_adapter_aux_hidden_dim,
            capacity_adapter_aux_depth=decoder_capacity_adapter_aux_depth,
            capacity_adapter_aux_cap_ppm=decoder_capacity_adapter_aux_cap_ppm,
            capacity_adapter_aux_scale=decoder_capacity_adapter_aux_scale,
            capacity_adapter_aux_expert_count=(
                decoder_capacity_adapter_aux_expert_count
            ),
            capacity_adapter_aux_output_init_std=(
                decoder_capacity_adapter_aux_output_init_std
            ),
            capacity_adapter_aux_family_scales=(
                decoder_capacity_adapter_aux_family_scales
            ),
            capacity_adapter_aux_family_gated=(
                decoder_capacity_adapter_aux_family_gated
            ),
            capacity_adapter_aux_family_specific=(
                decoder_capacity_adapter_aux_family_specific
            ),
            capacity_adapter_aux_family_headed=(
                decoder_capacity_adapter_aux_family_headed
            ),
            family_affine_scale_cap=decoder_family_affine_scale_cap,
            family_affine_shift_cap_ppm=decoder_family_affine_shift_cap_ppm,
            family_affine_family_scales=decoder_family_affine_family_scales,
        )
        self.energy_head = PosteriorEnergyHead(
            hidden_dim,
            energy_hidden_dim,
            evidence_feature_dim=evidence_feature_dim,
            dropout=dropout,
            init_std=posterior_energy_init_std,
            extra_hidden_dim=posterior_energy_extra_hidden_dim,
            extra_depth=posterior_energy_extra_depth,
            extra_scale=posterior_energy_extra_scale,
            extra_init_std=posterior_energy_extra_init_std,
            extra_factorized_rank=posterior_energy_extra_factorized_rank,
            extra_effective_hidden_dim=posterior_energy_extra_effective_hidden_dim,
            extra_effective_factorized_rank=(
                posterior_energy_extra_effective_factorized_rank
            ),
            extra_max_factorized_params=posterior_energy_extra_max_factorized_params,
            extra_shared_block_count=posterior_energy_extra_shared_block_count,
            extra_chunk_size=posterior_energy_extra_chunk_size,
            extra_use_checkpoint=posterior_energy_extra_use_checkpoint,
            support_attention_hidden_dim=(
                posterior_energy_support_attention_hidden_dim
            ),
            support_attention_head_count=(
                posterior_energy_support_attention_head_count
            ),
            support_attention_depth=posterior_energy_support_attention_depth,
            support_attention_scale=posterior_energy_support_attention_scale,
            support_attention_init_std=posterior_energy_support_attention_init_std,
        )
        self.cs_latent_field_conditioner: MaskAwareCSLatentFieldConditioner | None = None
        if (
            int(cs_latent_field_conditioning_hidden_dim) > 0
            and (
                float(cs_latent_field_conditioning_residue_scale) > 0.0
                or float(cs_latent_field_conditioning_energy_scale) > 0.0
                or float(cs_latent_field_conditioning_family_token_scale) > 0.0
                or float(cs_latent_field_conditioning_film_scale) > 0.0
            )
        ):
            self.cs_latent_field_conditioner = MaskAwareCSLatentFieldConditioner(
                residue_feature_dim,
                hidden_dim,
                atom_family_count=atom_family_count,
                hidden_dim=int(cs_latent_field_conditioning_hidden_dim),
                depth=cs_latent_field_conditioning_depth,
                dropout=cs_latent_field_conditioning_dropout,
                residue_scale=cs_latent_field_conditioning_residue_scale,
                energy_scale=cs_latent_field_conditioning_energy_scale,
                energy_pooling=cs_latent_field_conditioning_energy_pooling,
                energy_attention_temperature=(
                    cs_latent_field_conditioning_energy_attention_temperature
                ),
                energy_attention_init_std=(
                    cs_latent_field_conditioning_energy_attention_init_std
                ),
                energy_salience_attention_weight=(
                    cs_latent_field_conditioning_energy_salience_attention_weight
                ),
                energy_std_floor=cs_latent_field_conditioning_energy_std_floor,
                energy_std_floor_scale_cap=(
                    cs_latent_field_conditioning_energy_std_floor_scale_cap
                ),
                family_token_scale=(
                    cs_latent_field_conditioning_family_token_scale
                ),
                family_token_temperature=(
                    cs_latent_field_conditioning_family_token_temperature
                ),
                family_token_init_std=(
                    cs_latent_field_conditioning_family_token_init_std
                ),
                film_scale=cs_latent_field_conditioning_film_scale,
                film_init_std=cs_latent_field_conditioning_film_init_std,
                output_init_std=cs_latent_field_conditioning_init_std,
            )
        self.prior_logit_adapter: ConditionalPriorLogitAdapter | None = None
        if int(prior_logit_adapter_hidden_dim) > 0 and float(prior_logit_adapter_scale) > 0.0:
            self.prior_logit_adapter = ConditionalPriorLogitAdapter(
                hidden_dim,
                int(prior_logit_adapter_hidden_dim),
                evidence_feature_dim=evidence_feature_dim,
                depth=prior_logit_adapter_depth,
                dropout=dropout,
                scale=prior_logit_adapter_scale,
                max_abs=prior_logit_adapter_max_abs,
                output_init_std=prior_logit_adapter_output_init_std,
                set_context_dim=prior_logit_adapter_set_context_dim,
                set_context_depth=prior_logit_adapter_set_context_depth,
                include_base_log_prob_features=(
                    prior_logit_adapter_include_base_log_prob_features
                ),
            )
        mode_count = max(int(posterior_mode_count), 1)
        mode_scale = max(float(posterior_mode_energy_scale), 0.0)
        self.posterior_mode_mixture: PosteriorModeMixture | None = None
        if mode_count > 1 and mode_scale > 0.0:
            self.posterior_mode_mixture = PosteriorModeMixture(
                hidden_dim,
                energy_hidden_dim,
                evidence_feature_dim=evidence_feature_dim,
                mode_count=mode_count,
                energy_scale=mode_scale,
                energy_init_std=posterior_mode_energy_init_std,
                mode_logit_scale=posterior_mode_logit_scale,
                mode_logit_temperature=posterior_mode_logit_temperature,
                mode_logit_prior=posterior_mode_logit_prior,
                dropout=dropout,
            )
        self.residue_local_mode_gate: ResidueLocalModeGate | None = None
        if bool(residue_local_mode_gate_enabled) and mode_count > 1:
            self.residue_local_mode_gate = ResidueLocalModeGate(
                hidden_dim,
                atom_family_count,
                mode_count,
                gate_hidden_dim=residue_local_mode_gate_hidden_dim,
                temperature=residue_local_mode_gate_temperature,
                dropout=dropout,
            )
        evidence_hidden_dim = int(evidence_head_hidden_dim)
        if evidence_hidden_dim <= 0:
            evidence_hidden_dim = hidden_dim
        evidence_depth = max(int(evidence_head_depth), 1)
        self.hn_evidence_head = HNMechanismEvidenceEnergyHead(
            hidden_dim,
            mechanism_dim,
            dropout=dropout,
            chunk_size=evidence_head_chunk_size,
            use_checkpoint=evidence_head_use_checkpoint,
            head_hidden_dim=evidence_hidden_dim,
            head_depth=evidence_depth,
        )
        self.cprime_evidence_head = FamilyMechanismEvidenceEnergyHead(
            hidden_dim,
            mechanism_dim,
            family_index=4,
            count_prefix="cprime",
            dropout=dropout,
            chunk_size=evidence_head_chunk_size,
            use_checkpoint=evidence_head_use_checkpoint,
            head_hidden_dim=evidence_hidden_dim,
            head_depth=evidence_depth,
        )
        family_heads = []
        for family_index, family_name in enumerate(ATOM_FAMILY_NAMES[:atom_family_count]):
            prefix = "cprime" if family_name == "C'" else family_name.lower()
            family_heads.append(
                FamilyMechanismEvidenceEnergyHead(
                    hidden_dim,
                    mechanism_dim,
                    family_index=family_index,
                    count_prefix=prefix,
                    dropout=dropout,
                    chunk_size=evidence_head_chunk_size,
                    use_checkpoint=evidence_head_use_checkpoint,
                    head_hidden_dim=evidence_hidden_dim,
                    head_depth=evidence_depth,
                )
            )
        self.family_evidence_heads = nn.ModuleList(family_heads)
        self.residual = BoundedEnsembleResidual(
            hidden_dim,
            atom_family_count=atom_family_count,
            max_abs=residual_max_abs,
        )

    def posterior_from_energy(
        self,
        conformer_features: torch.Tensor,
        energy: torch.Tensor,
        *,
        prior_log_probs: torch.Tensor | None = None,
        evidence_features: torch.Tensor | None = None,
        temperature: float = 1.0,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the reconciled posterior population for an energy proposal."""

        if self.posterior_mode_mixture is None:
            weights = posterior_weights(
                prior_log_probs,
                energy,
                temperature=temperature,
                support_mask=support_mask,
            )
            return {"weights": weights}
        return self.posterior_mode_mixture(
            conformer_features,
            energy,
            prior_log_probs=prior_log_probs,
            evidence_features=evidence_features,
            temperature=temperature,
            support_mask=support_mask,
        )

    def forward(
        self,
        residue_features: torch.Tensor,
        *,
        prior_log_probs: torch.Tensor | None = None,
        evidence_features: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_target_indices: torch.Tensor | None = None,
        edge_class_indices: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        cs_evidence_grid: torch.Tensor | None = None,
        cs_evidence_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Return posterior weights and ensemble chemical-shift predictions."""

        if residue_features.ndim != 3:
            raise ValueError("residue_features must have shape [K, L, feature_dim]")
        residue_count = int(residue_features.shape[1])
        residue_feature_dim = int(residue_features.shape[-1])
        if self.cs_latent_field_conditioner is None:
            cs_latent_payload: dict[str, torch.Tensor] = {
                "residue_delta": residue_features.new_zeros(
                    residue_count,
                    residue_feature_dim,
                ),
                "field_hidden": residue_features.new_zeros(residue_count, 1),
                "residue_mask": torch.zeros(
                    residue_count,
                    dtype=torch.bool,
                    device=residue_features.device,
                ),
                "active": residue_features.new_tensor(0.0),
                "observed_cell_count": residue_features.new_tensor(0.0),
                "observed_residue_count": residue_features.new_tensor(0.0),
                "observed_residue_fraction": residue_features.new_tensor(0.0),
                "residue_delta_abs_mean": residue_features.new_tensor(0.0),
                "residue_delta_abs_max": residue_features.new_tensor(0.0),
                "residue_film_gamma": residue_features.new_zeros(
                    residue_count,
                    residue_feature_dim,
                ),
                "residue_film_beta": residue_features.new_zeros(
                    residue_count,
                    residue_feature_dim,
                ),
                "residue_film_active": residue_features.new_tensor(0.0),
                "residue_film_gamma_abs_mean": residue_features.new_tensor(0.0),
                "residue_film_gamma_abs_max": residue_features.new_tensor(0.0),
                "residue_film_beta_abs_mean": residue_features.new_tensor(0.0),
                "residue_film_beta_abs_max": residue_features.new_tensor(0.0),
                "family_token_active": residue_features.new_tensor(0.0),
                "family_token_observed_count": residue_features.new_tensor(0.0),
                "family_token_attention_entropy": residue_features.new_tensor(0.0),
                "family_token_attention_top_mass": residue_features.new_tensor(0.0),
                "family_token_residue_delta_abs_mean": (
                    residue_features.new_tensor(0.0)
                ),
                "family_token_residue_delta_abs_max": (
                    residue_features.new_tensor(0.0)
                ),
            }
        else:
            cs_latent_payload = self.cs_latent_field_conditioner.encode(
                cs_evidence_grid,
                cs_evidence_mask,
                reference=residue_features,
                residue_count=residue_count,
                residue_feature_dim=residue_feature_dim,
            )
        residue_film_gamma = cs_latent_payload["residue_film_gamma"].to(
            device=residue_features.device,
            dtype=residue_features.dtype,
        )
        residue_film_beta = cs_latent_payload["residue_film_beta"].to(
            device=residue_features.device,
            dtype=residue_features.dtype,
        )
        conditioned_residue_features = residue_features * (
            1.0 + residue_film_gamma.unsqueeze(0)
        )
        conditioned_residue_features = conditioned_residue_features + (
            residue_film_beta + cs_latent_payload["residue_delta"].to(
                device=residue_features.device,
                dtype=residue_features.dtype,
            )
        ).unsqueeze(0)
        support_inputs_require_grad = bool(residue_features.requires_grad) or bool(
            edge_features is not None and edge_features.requires_grad
        )
        frozen_feature_path = (
            torch.is_grad_enabled()
            and not support_inputs_require_grad
            and not _has_trainable_parameters(self.edge_encoder)
            and not _has_trainable_parameters(self.decoder)
            and not _has_trainable_parameters(self.cs_latent_field_conditioner)
        )

        def frozen_support_features() -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
            mechanism_features = None
            if (
                self.edge_encoder is not None
                and edge_features is not None
                and edge_target_indices is not None
                and edge_class_indices is not None
            ):
                mechanism_features = self.edge_encoder(
                    edge_features,
                    edge_target_indices,
                    edge_class_indices,
                    residue_count=residue_count,
                    edge_mask=edge_mask,
                )
            return self.decoder(
                conditioned_residue_features,
                mechanism_features,
            ), mechanism_features

        if frozen_feature_path:
            # Stage-B/q-only training does not update the conformer decoder or
            # compact x2d edge encoder. Avoiding autograd here keeps K=512
            # posterior-energy experiments on L40S from storing large edge
            # activations while preserving gradients for the energy heads.  If
            # BioEmu/provider features are trainable, however, the frozen
            # decoder must still act as a differentiable readout into those
            # upstream generator parameters.
            with torch.no_grad():
                decoded, mechanism_features = frozen_support_features()
        else:
            decoded, mechanism_features = frozen_support_features()
        conformer_features = decoded["hidden"].mean(dim=1)
        active_prior_log_probs = prior_log_probs
        prior_logit_delta: torch.Tensor | None = None
        if self.prior_logit_adapter is not None:
            prior_logit_delta = self.prior_logit_adapter(
                conformer_features,
                evidence_features,
                base_prior_log_probs=prior_log_probs,
            )
            if prior_log_probs is None:
                base_prior_log_probs = torch.zeros_like(prior_logit_delta)
            else:
                base_prior_log_probs = prior_log_probs.to(
                    device=prior_logit_delta.device,
                    dtype=prior_logit_delta.dtype,
                ).reshape(prior_logit_delta.shape)
                base_prior_log_probs = torch.nan_to_num(
                    base_prior_log_probs,
                    nan=0.0,
                    posinf=80.0,
                    neginf=-80.0,
                ).clamp(min=-80.0, max=80.0)
            active_prior_log_probs = base_prior_log_probs + prior_logit_delta
            active_prior_log_probs = torch.nan_to_num(
                active_prior_log_probs,
                nan=0.0,
                posinf=80.0,
                neginf=-80.0,
            ).clamp(min=-80.0, max=80.0)
        energy_parts = self.energy_head.forward_parts(
            conformer_features,
            evidence_features,
        )
        cs_latent_energy_delta = (
            None
            if self.cs_latent_field_conditioner is not None
            else energy_parts["energy"].new_zeros(energy_parts["energy"].shape)
        )
        cs_latent_energy_metrics: dict[str, torch.Tensor] = {
            "cs_latent_field_energy_attention_active": energy_parts[
                "energy"
            ].new_tensor(0.0),
            "cs_latent_field_energy_attention_entropy": energy_parts[
                "energy"
            ].new_tensor(0.0),
            "cs_latent_field_energy_attention_top_mass": energy_parts[
                "energy"
            ].new_tensor(0.0),
        }
        if self.cs_latent_field_conditioner is not None:
            cs_latent_energy_delta, cs_latent_energy_metrics = (
                self.cs_latent_field_conditioner.energy_delta_with_diagnostics(
                    decoded["hidden"],
                    field_hidden=cs_latent_payload["field_hidden"],
                    residue_mask=cs_latent_payload["residue_mask"],
                    residue_salience=cs_latent_payload["residue_salience"],
                )
            )
        energy = energy_parts["energy"] + cs_latent_energy_delta
        energy = torch.nan_to_num(energy, nan=0.0, posinf=80.0, neginf=-80.0).clamp(
            min=-80.0,
            max=80.0,
        )
        posterior = self.posterior_from_energy(
            conformer_features,
            energy,
            prior_log_probs=active_prior_log_probs,
            evidence_features=evidence_features,
            temperature=temperature,
            support_mask=support_mask,
        )
        weights = posterior["weights"]
        ensemble_phi = weighted_ensemble_observable(decoded["phi"], weights)
        measure_hidden = weighted_ensemble_observable(decoded["hidden"], weights)
        residue_local_mode_logits: torch.Tensor | None = None
        residue_local_mode_weights: torch.Tensor | None = None
        if (
            self.residue_local_mode_gate is not None
            and "mode_component_weights" in posterior
        ):
            residue_local_mode_logits, residue_local_mode_weights = (
                self.residue_local_mode_gate(measure_hidden)
            )
        bounded_delta = self.residual(measure_hidden)
        prediction = ensemble_phi + bounded_delta
        diagnostics = posterior_diagnostics(weights)
        prior_logit_metrics: dict[str, torch.Tensor | None] = {
            "prior_log_probs_corrected": active_prior_log_probs,
            "prior_logit_delta": prior_logit_delta,
        }
        if prior_logit_delta is not None:
            detached_delta = prior_logit_delta.detach()
            corrected_prior_q = torch.softmax(active_prior_log_probs.detach(), dim=-1)
            prior_logit_metrics.update(
                {
                    "prior_logit_adapter_delta_abs_mean": torch.mean(
                        torch.abs(detached_delta)
                    ),
                    "prior_logit_adapter_delta_abs_max": torch.max(
                        torch.abs(detached_delta)
                    ),
                    "prior_logit_adapter_delta_std": torch.std(
                        detached_delta,
                        unbiased=False,
                    ),
                    "prior_logit_adapter_corrected_prior_ess": 1.0
                    / corrected_prior_q.square().sum().clamp_min(1.0e-12),
                    "prior_logit_adapter_corrected_prior_top_mass": torch.max(
                        corrected_prior_q
                    ),
                }
            )
        return {
            "phi": decoded["phi"],
            "sigma": decoded["sigma"],
            "decoder_hidden": decoded["hidden"],
            "capacity_adapter_delta_phi": decoded["capacity_adapter_delta_phi"],
            "mechanism_support_basis_delta_phi": decoded[
                "mechanism_support_basis_delta_phi"
            ],
            "hn_signed_support_basis_delta_phi": decoded[
                "hn_signed_support_basis_delta_phi"
            ],
            "hn_direct_support_spread_delta_phi": decoded[
                "hn_direct_support_spread_delta_phi"
            ],
            "family_direct_support_spread_delta_phi": decoded[
                "family_direct_support_spread_delta_phi"
            ],
            "cprime_isolated_support_basis_delta_phi": decoded[
                "cprime_isolated_support_basis_delta_phi"
            ],
            "support_basis_delta_phi": decoded["support_basis_delta_phi"],
            "capacity_adapter_extra_expert_gate_entropy": decoded[
                "capacity_adapter_extra_expert_gate_entropy"
            ],
            "capacity_adapter_aux_delta_phi": decoded[
                "capacity_adapter_aux_delta_phi"
            ],
            "capacity_adapter_aux_expert_gate_entropy": decoded[
                "capacity_adapter_aux_expert_gate_entropy"
            ],
            "family_affine_delta_phi": decoded["family_affine_delta_phi"],
            "family_affine_scale": decoded["family_affine_scale"],
            "family_affine_shift_ppm": decoded["family_affine_shift_ppm"],
            "mechanism_features": mechanism_features,
            "conformer_features": conformer_features,
            "energy": energy,
            "base_energy": energy_parts["base_energy"],
            "energy_extra_delta": energy_parts["energy_extra_delta"],
            "energy_support_attention_delta": energy_parts[
                "energy_support_attention_delta"
            ],
            "cs_latent_field_energy_delta": cs_latent_energy_delta,
            "cs_latent_field_energy_delta_abs_mean": torch.mean(
                torch.abs(cs_latent_energy_delta.detach())
            ),
            "cs_latent_field_energy_delta_abs_max": torch.max(
                torch.abs(cs_latent_energy_delta.detach())
            ),
            "cs_latent_field_energy_delta_std": torch.std(
                cs_latent_energy_delta.detach(),
                unbiased=False,
            ),
            "cs_latent_field_conditioning_active": cs_latent_payload["active"].detach(),
            "cs_latent_field_observed_cell_count": cs_latent_payload[
                "observed_cell_count"
            ].detach(),
            "cs_latent_field_observed_residue_count": cs_latent_payload[
                "observed_residue_count"
            ].detach(),
            "cs_latent_field_observed_residue_fraction": cs_latent_payload[
                "observed_residue_fraction"
            ].detach(),
            "cs_latent_field_residue_delta_abs_mean": cs_latent_payload[
                "residue_delta_abs_mean"
            ].detach(),
            "cs_latent_field_residue_delta_abs_max": cs_latent_payload[
                "residue_delta_abs_max"
            ].detach(),
            "cs_latent_field_residue_film_active": cs_latent_payload[
                "residue_film_active"
            ].detach(),
            "cs_latent_field_residue_film_gamma_abs_mean": cs_latent_payload[
                "residue_film_gamma_abs_mean"
            ].detach(),
            "cs_latent_field_residue_film_gamma_abs_max": cs_latent_payload[
                "residue_film_gamma_abs_max"
            ].detach(),
            "cs_latent_field_residue_film_beta_abs_mean": cs_latent_payload[
                "residue_film_beta_abs_mean"
            ].detach(),
            "cs_latent_field_residue_film_beta_abs_max": cs_latent_payload[
                "residue_film_beta_abs_max"
            ].detach(),
            "cs_latent_field_family_token_active": cs_latent_payload[
                "family_token_active"
            ].detach(),
            "cs_latent_field_family_token_observed_count": cs_latent_payload[
                "family_token_observed_count"
            ].detach(),
            "cs_latent_field_family_token_attention_entropy": cs_latent_payload[
                "family_token_attention_entropy"
            ].detach(),
            "cs_latent_field_family_token_attention_top_mass": cs_latent_payload[
                "family_token_attention_top_mass"
            ].detach(),
            "cs_latent_field_family_token_residue_delta_abs_mean": cs_latent_payload[
                "family_token_residue_delta_abs_mean"
            ].detach(),
            "cs_latent_field_family_token_residue_delta_abs_max": cs_latent_payload[
                "family_token_residue_delta_abs_max"
            ].detach(),
            **{
                key: value.detach()
                for key, value in cs_latent_energy_metrics.items()
            },
            "weights": weights,
            "ensemble_phi": ensemble_phi,
            "bounded_delta": bounded_delta,
            "prediction": prediction,
            "residue_local_mode_logits": residue_local_mode_logits,
            "residue_local_mode_weights": residue_local_mode_weights,
            **{key: value for key, value in posterior.items() if key != "weights"},
            **prior_logit_metrics,
            **diagnostics,
        }


def regularization_config_from_training_config(
    config: object,
) -> X0PosteriorRegularizationConfig:
    """Build x0 posterior collapse guards from ``StudentTrainingConfig``-like data."""

    return X0PosteriorRegularizationConfig(
        ess_floor=float(getattr(config, "bioemu_x0_posterior_ess_floor", 0.0)),
        entropy_floor=float(
            getattr(config, "bioemu_x0_posterior_entropy_floor", 0.0)
        ),
        top_mass_cap=float(getattr(config, "bioemu_x0_posterior_top_mass_cap", 1.0)),
        prior_kl_weight=float(
            getattr(config, "bioemu_x0_posterior_prior_kl_weight", 0.0)
        ),
        diversity_floor=float(
            getattr(config, "bioemu_x0_posterior_diversity_floor", 0.0)
        ),
        diversity_weight=float(
            getattr(config, "bioemu_x0_posterior_diversity_weight", 0.0)
        ),
    )


def build_x0_posterior_ensemble_model(
    config: object,
    *,
    residue_feature_dim: int,
    edge_feature_dim: int = 0,
    evidence_feature_dim: int = 0,
    atom_family_count: int = len(ATOM_FAMILY_NAMES),
) -> X0PosteriorEnsembleModel:
    """Create the x0 posterior model from a training config object."""

    hidden_dim = int(getattr(config, "bioemu_x0_decoder_hidden_dim", 256))
    return X0PosteriorEnsembleModel(
        residue_feature_dim=residue_feature_dim,
        hidden_dim=hidden_dim,
        atom_family_count=atom_family_count,
        edge_feature_dim=edge_feature_dim,
        edge_class_count=len(
            getattr(config, "bioemu_x0_mechanism_edge_classes", COMPACT_X2D_EDGE_CLASSES)
        ),
        evidence_feature_dim=evidence_feature_dim,
        posterior_energy_hidden_dim=int(
            getattr(config, "bioemu_x0_posterior_energy_hidden_dim", hidden_dim)
        ),
        posterior_energy_extra_hidden_dim=int(
            getattr(config, "bioemu_x0_posterior_energy_extra_hidden_dim", 0)
        ),
        posterior_energy_extra_depth=int(
            getattr(config, "bioemu_x0_posterior_energy_extra_depth", 2)
        ),
        posterior_energy_extra_scale=float(
            getattr(config, "bioemu_x0_posterior_energy_extra_scale", 1.0)
        ),
        posterior_energy_extra_init_std=float(
            getattr(config, "bioemu_x0_posterior_energy_extra_init_std", 0.0)
        ),
        posterior_energy_init_std=float(
            getattr(config, "bioemu_x0_posterior_energy_init_std", 0.0)
        ),
        posterior_energy_extra_factorized_rank=int(
            getattr(config, "bioemu_x0_posterior_energy_extra_factorized_rank", 0)
        ),
        posterior_energy_extra_effective_hidden_dim=int(
            getattr(config, "bioemu_x0_posterior_energy_extra_effective_hidden_dim", 0)
        ),
        posterior_energy_extra_effective_factorized_rank=int(
            getattr(
                config,
                "bioemu_x0_posterior_energy_extra_effective_factorized_rank",
                0,
            )
        ),
        posterior_energy_extra_max_factorized_params=int(
            getattr(
                config,
                "bioemu_x0_posterior_energy_extra_max_factorized_params",
                450_000_000,
            )
        ),
        posterior_energy_extra_shared_block_count=int(
            getattr(config, "bioemu_x0_posterior_energy_extra_shared_block_count", 0)
        ),
        posterior_energy_extra_chunk_size=int(
            getattr(config, "bioemu_x0_posterior_energy_extra_chunk_size", 0)
        ),
        posterior_energy_extra_use_checkpoint=bool(
            getattr(config, "bioemu_x0_posterior_energy_extra_use_checkpoint", True)
        ),
        posterior_energy_support_attention_hidden_dim=int(
            getattr(
                config,
                "bioemu_x0_posterior_energy_support_attention_hidden_dim",
                0,
            )
        ),
        posterior_energy_support_attention_head_count=int(
            getattr(
                config,
                "bioemu_x0_posterior_energy_support_attention_head_count",
                4,
            )
        ),
        posterior_energy_support_attention_depth=int(
            getattr(config, "bioemu_x0_posterior_energy_support_attention_depth", 1)
        ),
        posterior_energy_support_attention_scale=float(
            getattr(config, "bioemu_x0_posterior_energy_support_attention_scale", 0.0)
        ),
        posterior_energy_support_attention_init_std=float(
            getattr(
                config,
                "bioemu_x0_posterior_energy_support_attention_init_std",
                0.0,
            )
        ),
        cs_latent_field_conditioning_hidden_dim=int(
            getattr(config, "bioemu_x0_cs_latent_field_conditioning_hidden_dim", 0)
        ),
        cs_latent_field_conditioning_depth=int(
            getattr(config, "bioemu_x0_cs_latent_field_conditioning_depth", 1)
        ),
        cs_latent_field_conditioning_residue_scale=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_residue_scale",
                0.0,
            )
        ),
        cs_latent_field_conditioning_energy_scale=float(
            getattr(config, "bioemu_x0_cs_latent_field_conditioning_energy_scale", 0.0)
        ),
        cs_latent_field_conditioning_energy_pooling=str(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_energy_pooling",
                "mean",
            )
        ),
        cs_latent_field_conditioning_energy_attention_temperature=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_energy_attention_temperature",
                1.0,
            )
        ),
        cs_latent_field_conditioning_energy_attention_init_std=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_energy_attention_init_std",
                0.0,
            )
        ),
        cs_latent_field_conditioning_energy_salience_attention_weight=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_energy_salience_attention_weight",
                0.0,
            )
        ),
        cs_latent_field_conditioning_energy_std_floor=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_energy_std_floor",
                0.0,
            )
        ),
        cs_latent_field_conditioning_energy_std_floor_scale_cap=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_energy_std_floor_scale_cap",
                0.0,
            )
        ),
        cs_latent_field_conditioning_family_token_scale=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_family_token_scale",
                0.0,
            )
        ),
        cs_latent_field_conditioning_family_token_temperature=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_family_token_temperature",
                1.0,
            )
        ),
        cs_latent_field_conditioning_family_token_init_std=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_family_token_init_std",
                0.0,
            )
        ),
        cs_latent_field_conditioning_film_scale=float(
            getattr(config, "bioemu_x0_cs_latent_field_conditioning_film_scale", 0.0)
        ),
        cs_latent_field_conditioning_film_init_std=float(
            getattr(
                config,
                "bioemu_x0_cs_latent_field_conditioning_film_init_std",
                0.0,
            )
        ),
        cs_latent_field_conditioning_dropout=float(
            getattr(config, "bioemu_x0_cs_latent_field_conditioning_dropout", 0.0)
        ),
        cs_latent_field_conditioning_init_std=float(
            getattr(config, "bioemu_x0_cs_latent_field_conditioning_init_std", 0.0)
        ),
        prior_logit_adapter_hidden_dim=int(
            getattr(config, "bioemu_x0_prior_logit_adapter_hidden_dim", 0)
        ),
        prior_logit_adapter_depth=int(
            getattr(config, "bioemu_x0_prior_logit_adapter_depth", 2)
        ),
        prior_logit_adapter_scale=float(
            getattr(config, "bioemu_x0_prior_logit_adapter_scale", 1.0)
        ),
        prior_logit_adapter_max_abs=float(
            getattr(config, "bioemu_x0_prior_logit_adapter_max_abs", 0.0)
        ),
        prior_logit_adapter_output_init_std=float(
            getattr(config, "bioemu_x0_prior_logit_adapter_output_init_std", 0.0)
        ),
        prior_logit_adapter_set_context_dim=int(
            getattr(config, "bioemu_x0_prior_logit_adapter_set_context_dim", 0)
        ),
        prior_logit_adapter_set_context_depth=int(
            getattr(config, "bioemu_x0_prior_logit_adapter_set_context_depth", 1)
        ),
        prior_logit_adapter_include_base_log_prob_features=bool(
            getattr(
                config,
                "bioemu_x0_prior_logit_adapter_include_base_log_prob_features",
                False,
            )
        ),
        residual_max_abs=float(getattr(config, "bioemu_x0_bounded_delta_max_abs", 0.0)),
        dropout=float(getattr(config, "bioemu_x0_decoder_dropout", 0.0)),
        evidence_head_chunk_size=int(
            getattr(config, "bioemu_x0_evidence_head_chunk_size", 4)
        ),
        evidence_head_use_checkpoint=bool(
            getattr(config, "bioemu_x0_evidence_head_use_checkpoint", False)
        ),
        evidence_head_hidden_dim=int(
            getattr(config, "bioemu_x0_evidence_head_hidden_dim", 0)
        ),
        evidence_head_depth=int(getattr(config, "bioemu_x0_evidence_head_depth", 2)),
        posterior_mode_count=int(getattr(config, "bioemu_x0_posterior_mode_count", 1)),
        posterior_mode_energy_scale=float(
            getattr(config, "bioemu_x0_posterior_mode_energy_scale", 0.0)
        ),
        posterior_mode_energy_init_std=float(
            getattr(config, "bioemu_x0_posterior_mode_energy_init_std", 0.0)
        ),
        posterior_mode_logit_scale=float(
            getattr(config, "bioemu_x0_posterior_mode_logit_scale", 1.0)
        ),
        posterior_mode_logit_temperature=float(
            getattr(config, "bioemu_x0_posterior_mode_logit_temperature", 1.0)
        ),
        posterior_mode_logit_prior=list(
            getattr(config, "bioemu_x0_posterior_mode_logit_prior", []) or []
        ),
        residue_local_mode_gate_enabled=(
            bool(getattr(config, "bioemu_x0_residue_local_mode_observable_enabled", False))
            or float(
                getattr(
                    config,
                    "bioemu_x0_residue_local_mode_gate_teacher_loss_weight",
                    0.0,
                )
            )
            > 0.0
        ),
        residue_local_mode_gate_hidden_dim=int(
            getattr(config, "bioemu_x0_residue_local_mode_gate_hidden_dim", 0)
        ),
        residue_local_mode_gate_temperature=float(
            getattr(config, "bioemu_x0_residue_local_mode_gate_temperature", 1.0)
        ),
        decoder_mechanism_support_basis_count=int(
            getattr(config, "bioemu_x0_decoder_mechanism_support_basis_count", 0)
        ),
        decoder_mechanism_support_basis_cap=float(
            getattr(config, "bioemu_x0_decoder_mechanism_support_basis_cap", 0.0)
        ),
        decoder_hn_signed_support_basis_count=int(
            getattr(config, "bioemu_x0_decoder_hn_signed_support_basis_count", 0)
        ),
        decoder_hn_signed_support_basis_cap=float(
            getattr(config, "bioemu_x0_decoder_hn_signed_support_basis_cap", 0.0)
        ),
        decoder_hn_direct_support_spread_scale_ppm=float(
            getattr(config, "bioemu_x0_decoder_hn_direct_support_spread_scale_ppm", 0.0)
        ),
        decoder_hn_direct_support_spread_trainable_scale_cap_ppm=float(
            getattr(
                config,
                "bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_cap_ppm",
                0.0,
            )
        ),
        decoder_hn_direct_support_spread_trainable_scale_init_ppm=float(
            getattr(
                config,
                "bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_init_ppm",
                0.0,
            )
        ),
        decoder_family_direct_support_spread_scale_ppm_by_family=dict(
            getattr(
                config,
                "bioemu_x0_decoder_family_direct_support_spread_scale_ppm_by_family",
                {},
            )
            or {}
        ),
        decoder_family_direct_support_spread_trainable_scale_cap_ppm_by_family=dict(
            getattr(
                config,
                (
                    "bioemu_x0_decoder_family_direct_support_spread_"
                    "trainable_scale_cap_ppm_by_family"
                ),
                {},
            )
            or {}
        ),
        decoder_family_direct_support_spread_trainable_scale_init_ppm_by_family=dict(
            getattr(
                config,
                (
                    "bioemu_x0_decoder_family_direct_support_spread_"
                    "trainable_scale_init_ppm_by_family"
                ),
                {},
            )
            or {}
        ),
        decoder_cprime_isolated_support_basis_count=int(
            getattr(
                config,
                "bioemu_x0_decoder_cprime_isolated_support_basis_count",
                0,
            )
        ),
        decoder_cprime_isolated_support_basis_cap=float(
            getattr(
                config,
                "bioemu_x0_decoder_cprime_isolated_support_basis_cap",
                0.0,
            )
        ),
        decoder_n_ca_cb_support_deviation_enabled=bool(
            getattr(
                config,
                "bioemu_x0_decoder_n_ca_cb_support_deviation_enabled",
                True,
            )
        ),
        decoder_emit_support_basis_diagnostics=bool(
            getattr(
                config,
                "bioemu_x0_decoder_emit_support_basis_diagnostics",
                True,
            )
        ),
        decoder_use_checkpoint=bool(
            getattr(config, "bioemu_x0_decoder_use_checkpoint", False)
        ),
        decoder_support_head_chunk_size=int(
            getattr(config, "bioemu_x0_decoder_support_head_chunk_size", 0)
        ),
        compact_edge_encoder_sample_chunk_size=int(
            getattr(config, "bioemu_x0_compact_edge_encoder_sample_chunk_size", 0)
        ),
        decoder_capacity_adapter_hidden_dim=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_hidden_dim", 0)
        ),
        decoder_capacity_adapter_depth=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_depth", 3)
        ),
        decoder_capacity_adapter_cap_ppm=float(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_cap_ppm", 0.0)
        ),
        decoder_capacity_adapter_scale=float(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_scale", 1.0)
        ),
        decoder_capacity_adapter_trainable_scale_cap=float(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_trainable_scale_cap",
                0.0,
            )
        ),
        decoder_capacity_adapter_trainable_scale_init=float(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_trainable_scale_init",
                0.0,
            )
        ),
        decoder_capacity_adapter_extra_hidden_dim=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_extra_hidden_dim", 0)
        ),
        decoder_capacity_adapter_extra_depth=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_extra_depth", 3)
        ),
        decoder_capacity_adapter_extra_cap_ppm=float(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_extra_cap_ppm", 0.0)
        ),
        decoder_capacity_adapter_extra_scale=float(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_extra_scale", 1.0)
        ),
        decoder_capacity_adapter_extra_expert_count=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_extra_expert_count", 1)
        ),
        decoder_capacity_adapter_extra_factorized_rank=int(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_factorized_rank",
                0,
            )
        ),
        decoder_capacity_adapter_extra_effective_hidden_dim=int(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_effective_hidden_dim",
                0,
            )
        ),
        decoder_capacity_adapter_extra_effective_factorized_rank=int(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_effective_factorized_rank",
                0,
            )
        ),
        decoder_capacity_adapter_extra_max_factorized_params=int(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_max_factorized_params",
                450_000_000,
            )
        ),
        decoder_capacity_adapter_extra_output_init_std=float(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_output_init_std",
                0.0,
            )
        ),
        decoder_capacity_adapter_extra_family_scales=dict(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_family_scales",
                {},
            )
            or {}
        ),
        decoder_capacity_adapter_extra_family_gated=bool(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_family_gated",
                False,
            )
        ),
        decoder_capacity_adapter_extra_family_specific=bool(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_family_specific",
                False,
            )
        ),
        decoder_capacity_adapter_extra_family_headed=bool(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_extra_family_headed",
                False,
            )
        ),
        decoder_capacity_adapter_aux_hidden_dim=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_aux_hidden_dim", 0)
        ),
        decoder_capacity_adapter_aux_depth=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_aux_depth", 2)
        ),
        decoder_capacity_adapter_aux_cap_ppm=float(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_aux_cap_ppm", 0.0)
        ),
        decoder_capacity_adapter_aux_scale=float(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_aux_scale", 1.0)
        ),
        decoder_capacity_adapter_aux_expert_count=int(
            getattr(config, "bioemu_x0_decoder_capacity_adapter_aux_expert_count", 1)
        ),
        decoder_capacity_adapter_aux_output_init_std=float(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_aux_output_init_std",
                0.0,
            )
        ),
        decoder_capacity_adapter_aux_family_scales=dict(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_aux_family_scales",
                {},
            )
            or {}
        ),
        decoder_capacity_adapter_aux_family_gated=bool(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_aux_family_gated",
                False,
            )
        ),
        decoder_capacity_adapter_aux_family_specific=bool(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_aux_family_specific",
                False,
            )
        ),
        decoder_capacity_adapter_aux_family_headed=bool(
            getattr(
                config,
                "bioemu_x0_decoder_capacity_adapter_aux_family_headed",
                False,
            )
        ),
        decoder_family_affine_scale_cap=float(
            getattr(config, "bioemu_x0_decoder_family_affine_scale_cap", 0.0)
        ),
        decoder_family_affine_shift_cap_ppm=float(
            getattr(config, "bioemu_x0_decoder_family_affine_shift_cap_ppm", 0.0)
        ),
        decoder_family_affine_family_scales=dict(
            getattr(config, "bioemu_x0_decoder_family_affine_family_scales", {})
            or {}
        ),
    )


def x0_posterior_loss_terms(
    *,
    ensemble_phi: torch.Tensor,
    target_shift: torch.Tensor,
    observed_mask: torch.Tensor,
    bounded_delta: torch.Tensor | None = None,
    conformer_phi: torch.Tensor | None = None,
    teacher_phi: torch.Tensor | None = None,
    posterior_q: torch.Tensor | None = None,
    prior_q: torch.Tensor | None = None,
    support_distances: torch.Tensor | None = None,
    teacher_weight: float = 0.0,
    calibration_weight: float = 0.0,
    delta_penalty_weight: float = 1.0,
    regularization: X0PosteriorRegularizationConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return staged losses for decoder pretrain and posterior reweighting.

    ``primary_cs_loss`` is intentionally computed from ``ensemble_phi`` before
    adding ``bounded_delta``.  The optional calibrated prediction can help later
    fine-tuning, but it is kept separate so a direct residual cannot become the
    main chemical-shift predictor.
    """

    mask = observed_mask.to(device=ensemble_phi.device, dtype=torch.bool)
    target = target_shift.to(device=ensemble_phi.device, dtype=ensemble_phi.dtype)
    if mask.shape != ensemble_phi.shape:
        raise ValueError("observed_mask must have the same shape as ensemble_phi")
    safe_mask = mask & torch.isfinite(target)
    if bool(torch.any(safe_mask)):
        primary_cs_loss = torch.mean((ensemble_phi[safe_mask] - target[safe_mask]).square())
    else:
        primary_cs_loss = ensemble_phi.new_tensor(0.0)
    total = primary_cs_loss
    terms: dict[str, torch.Tensor] = {"primary_cs_loss": primary_cs_loss}
    if bounded_delta is not None:
        delta = bounded_delta.to(device=ensemble_phi.device, dtype=ensemble_phi.dtype)
        terms["bounded_delta_penalty"] = torch.mean(delta.square()) * float(delta_penalty_weight)
        total = total + terms["bounded_delta_penalty"]
        if calibration_weight > 0.0 and bool(torch.any(safe_mask)):
            calibrated = ensemble_phi + delta
            terms["calibrated_cs_loss"] = (
                torch.mean((calibrated[safe_mask] - target[safe_mask]).square())
                * float(calibration_weight)
            )
            total = total + terms["calibrated_cs_loss"]
        else:
            terms["calibrated_cs_loss"] = ensemble_phi.new_tensor(0.0)
    if (
        conformer_phi is not None
        and teacher_phi is not None
        and float(teacher_weight) > 0.0
    ):
        teacher = teacher_phi.to(device=conformer_phi.device, dtype=conformer_phi.dtype)
        teacher_mask = torch.isfinite(teacher)
        if bool(torch.any(teacher_mask)):
            terms["decoder_teacher_loss"] = (
                torch.mean((conformer_phi[teacher_mask] - teacher[teacher_mask]).square())
                * float(teacher_weight)
            )
            total = total + terms["decoder_teacher_loss"]
        else:
            terms["decoder_teacher_loss"] = ensemble_phi.new_tensor(0.0)
    if posterior_q is not None and regularization is not None:
        reg_terms = posterior_regularization_terms(
            posterior_q,
            config=regularization,
            prior_weights=prior_q,
            support_distances=support_distances,
        )
        terms.update(reg_terms)
        total = total + reg_terms["collapse_guard_loss"]
    terms["total_loss"] = total
    return terms


def x0_simplex_posterior_loss_terms(
    *,
    conformer_phi: torch.Tensor,
    energy: torch.Tensor,
    target_shift: torch.Tensor,
    observed_mask: torch.Tensor,
    prior_log_probs: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    uncertainty_energy: torch.Tensor | None = None,
    uncertainty_weight: float = 0.0,
    guard_energy: torch.Tensor | None = None,
    guard_weight: float = 0.0,
    prior_q: torch.Tensor | None = None,
    support_distances: torch.Tensor | None = None,
    regularization: X0PosteriorRegularizationConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Build the GPU-friendly shared-q simplex posterior objective.

    This is the online counterpart of the offline shared-q simplex oracle: a
    network predicts one scalar energy per support conformer, optional
    conservative uncertainty/guard energies are added, and ``posterior_weights``
    maps the result to a single support-level q by softmax.  No rowwise or
    family-specific q is created here.
    """

    conservative_energy = energy
    if uncertainty_energy is not None and float(uncertainty_weight) > 0.0:
        conservative_energy = conservative_energy + (
            uncertainty_energy.to(device=energy.device, dtype=energy.dtype)
            * float(uncertainty_weight)
        )
    if guard_energy is not None and float(guard_weight) > 0.0:
        conservative_energy = conservative_energy + (
            guard_energy.to(device=energy.device, dtype=energy.dtype)
            * float(guard_weight)
        )
    posterior_q = posterior_weights(
        prior_log_probs,
        conservative_energy,
        temperature=temperature,
        support_mask=support_mask,
    )
    ensemble_phi = weighted_ensemble_observable(conformer_phi, posterior_q)
    if prior_q is None and prior_log_probs is not None:
        prior_q = torch.softmax(
            prior_log_probs.to(device=posterior_q.device, dtype=posterior_q.dtype),
            dim=-1,
        )
    terms = x0_posterior_loss_terms(
        ensemble_phi=ensemble_phi,
        target_shift=target_shift,
        observed_mask=observed_mask,
        posterior_q=posterior_q,
        prior_q=prior_q,
        support_distances=support_distances,
        regularization=regularization,
    )
    diagnostics = posterior_diagnostics(
        posterior_q,
        prior_weights=prior_q,
        support_distances=support_distances,
    )
    terms.update(
        {
            "simplex_posterior_q": posterior_q,
            "simplex_ensemble_phi": ensemble_phi,
            "simplex_conservative_energy": conservative_energy,
            "simplex_sum_abs_error": torch.abs(
                posterior_q.sum(dim=-1) - posterior_q.new_tensor(1.0)
            ).mean(),
            "simplex_min_weight": torch.amin(posterior_q, dim=-1).mean(),
            "simplex_entropy_mean": diagnostics["entropy"].mean(),
            "simplex_ess_mean": diagnostics["ess"].mean(),
            "simplex_top_mass_mean": diagnostics["top_mass"].mean(),
            "simplex_prior_kl_mean": diagnostics["prior_kl"].mean(),
            "simplex_shared_q_contract": posterior_q.new_tensor(1.0),
        }
    )
    return terms
