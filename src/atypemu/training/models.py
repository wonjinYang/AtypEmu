"""Model building blocks for staged AtypEmu training tasks."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConformerEncoder(nn.Module):
    """Encode per-candidate numeric descriptors into a latent conformer state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """Initialize one numeric conformer encoder."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, numeric_features: torch.Tensor) -> torch.Tensor:
        """Return latent conformer embeddings."""
        return self.network(numeric_features)


class ObservableForwardHead(nn.Module):
    """Predict per-conformer chemical-shift likelihood parameters."""

    def __init__(
        self,
        hidden_dim: int,
        atom_family_count: int = 5,
        min_sigma: float = 1e-3,
    ) -> None:
        """Initialize the chemical-shift forward head."""
        super().__init__()
        self.atom_family_count = int(atom_family_count)
        self.min_sigma = float(min_sigma)
        self.mean_head = nn.Linear(hidden_dim, self.atom_family_count)
        self.log_sigma_head = nn.Linear(hidden_dim, self.atom_family_count)

    def forward(self, conformer_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-family ``mu`` and ``sigma`` tensors."""
        mean = self.mean_head(conformer_embeddings)
        sigma = torch.nn.functional.softplus(self.log_sigma_head(conformer_embeddings))
        sigma = sigma.clamp_min(self.min_sigma)
        return {"mu": mean, "sigma": sigma}


class ForwardResidualHead(nn.Module):
    """Predict atom-family residual corrections for candidate shift matrices."""

    def __init__(
        self,
        hidden_dim: int,
        atom_family_count: int = 5,
        min_sigma: float = 1e-3,
    ) -> None:
        """Initialize one candidate-level residual head."""
        super().__init__()
        self.atom_family_count = int(atom_family_count)
        self.min_sigma = float(min_sigma)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden_dim, self.atom_family_count)
        self.log_sigma_head = nn.Linear(hidden_dim, self.atom_family_count)

    def forward(self, candidate_context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return residual mean and sigma by candidate and atom family."""
        hidden = self.network(candidate_context)
        sigma = torch.nn.functional.softplus(self.log_sigma_head(hidden)).clamp_min(
            self.min_sigma
        )
        return {
            "mu": self.residual_head(hidden),
            "sigma": sigma,
        }


class ResidueAtomResidualHead(nn.Module):
    """Predict residue-atom residual corrections for each target and candidate."""

    def __init__(
        self,
        hidden_dim: int,
        atom_family_count: int = 5,
        amino_acid_count: int = 21,
        max_residue_index: int = 4096,
        min_sigma: float = 1e-3,
    ) -> None:
        """Initialize one target-by-candidate residual head."""
        super().__init__()
        self.max_residue_index = max(int(max_residue_index), 2)
        self.min_sigma = float(min_sigma)
        self.residue_embedding = nn.Embedding(self.max_residue_index, hidden_dim)
        self.amino_acid_embedding = nn.Embedding(amino_acid_count + 1, hidden_dim)
        self.atom_family_embedding = nn.Embedding(atom_family_count, hidden_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_projection = nn.Linear(hidden_dim, hidden_dim)
        self.pair_network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        candidate_context: torch.Tensor,
        target_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        """Return residual mean and sigma with shape ``target x candidate``."""
        if target_features is None or target_features.numel() == 0:
            return None
        residue_indices = (
            target_features[:, 0]
            .long()
            .clamp(
                min=0,
                max=self.max_residue_index - 1,
            )
        )
        amino_acid_indices = (
            target_features[:, 1]
            .long()
            .clamp(
                min=0,
                max=self.amino_acid_embedding.num_embeddings - 1,
            )
        )
        atom_family_indices = (
            target_features[:, 2]
            .long()
            .clamp(
                min=0,
                max=self.atom_family_embedding.num_embeddings - 1,
            )
        )
        positions = target_features[:, 3:4].to(candidate_context.dtype)
        target_context = (
            self.residue_embedding(residue_indices)
            + self.amino_acid_embedding(amino_acid_indices)
            + self.atom_family_embedding(atom_family_indices)
            + self.position_projection(positions)
        )
        target_context = self.target_projection(target_context)
        candidate_context = self.candidate_projection(candidate_context)
        pair_hidden = target_context.unsqueeze(1) + candidate_context.unsqueeze(0)
        pair_hidden = self.pair_network(pair_hidden)
        sigma = torch.nn.functional.softplus(self.log_sigma_head(pair_hidden))
        sigma = sigma.squeeze(-1).clamp_min(self.min_sigma)
        return {
            "mu": self.residual_head(pair_hidden).squeeze(-1),
            "sigma": sigma,
        }


class BayesianPosteriorHead(nn.Module):
    """Combine prior logits with explicit evidence log-likelihood logits."""

    def __init__(self, evidence_temperature: float = 1.0) -> None:
        """Initialize the posterior logit combiner."""
        super().__init__()
        self.evidence_temperature = max(float(evidence_temperature), 1e-6)

    def forward(
        self,
        prior_logits: torch.Tensor,
        evidence_log_likelihood: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return posterior logits from prior and evidence terms."""
        if evidence_log_likelihood is None:
            return prior_logits
        evidence = evidence_log_likelihood.to(prior_logits.dtype)
        evidence = torch.nan_to_num(evidence, nan=-100.0, neginf=-100.0, posinf=10.0)
        return prior_logits + evidence / self.evidence_temperature


class StatePosteriorScorer(nn.Module):
    """Compress candidate weights through soft energy-landscape states."""

    def __init__(self, hidden_dim: int, state_count: int = 8) -> None:
        """Initialize a differentiable state-mixture posterior layer."""
        super().__init__()
        self.state_count = max(int(state_count), 1)
        self.assignment_head = nn.Linear(hidden_dim, self.state_count)

    def forward(
        self,
        logits: torch.Tensor,
        candidate_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return state-compressed candidate weights and state occupancies."""
        if self.state_count <= 1 or logits.numel() <= 1:
            weights = torch.softmax(logits, dim=0)
            return {
                "weights": weights,
                "state_weights": weights.new_ones(1),
                "state_assignments": weights.new_ones(weights.shape[0], 1),
            }
        assignments = torch.softmax(self.assignment_head(candidate_context), dim=1)
        log_assignments = torch.log(assignments.clamp_min(1e-8))
        candidate_state_logits = logits.unsqueeze(1) + log_assignments
        state_logits = torch.logsumexp(candidate_state_logits, dim=0)
        state_weights = torch.softmax(state_logits, dim=0)
        local_weights = torch.softmax(candidate_state_logits, dim=0)
        weights = torch.sum(local_weights * state_weights.unsqueeze(0), dim=1)
        weights = weights / weights.sum().clamp_min(1e-8)
        return {
            "weights": weights,
            "state_weights": state_weights,
            "state_assignments": assignments,
        }


class PosteriorMomentHead(nn.Module):
    """Predict fast residue-atom posterior moments from masked NMR evidence."""

    def __init__(
        self,
        hidden_dim: int,
        example_feature_dim: int,
        atom_family_count: int = 5,
        amino_acid_count: int = 21,
        max_residue_index: int = 4096,
        min_sigma: float = 1e-3,
        max_sigma: float = 12.0,
        enable_family_specific_adapters: bool = False,
        enable_nmr_structural_features: bool = False,
        enable_local_evidence_context: bool = False,
        enable_same_residue_evidence_context: bool = False,
        same_residue_evidence_target_family_indices: tuple[int, ...] = (),
        enable_target_set_evidence_encoder: bool = False,
        enable_target_evidence_token: bool = False,
        target_set_encoder_layers: int = 1,
        target_set_encoder_heads: int = 4,
        target_set_encoder_max_targets: int = 1024,
        target_set_context_gate_init: float = 1.0,
        target_set_encoder_masked_only: bool = False,
        target_set_encoder_target_family_indices: tuple[int, ...] = (),
        enable_residue_grid_evidence_encoder: bool = False,
        residue_grid_encoder_layers: int = 1,
        residue_grid_encoder_heads: int = 4,
        residue_grid_context_gate_init: float = 0.2,
        enable_residue_grid_family_gates: bool = False,
        residue_grid_encoder_masked_only: bool = True,
        residue_grid_encoder_target_family_indices: tuple[int, ...] = (),
        local_evidence_window: int = 3,
        enable_residue_anchor_evidence_context: bool = False,
        residue_anchor_evidence_offsets: tuple[int, ...] = (-1, 0, 1),
        residue_anchor_evidence_target_family_indices: tuple[int, ...] = (),
        enable_backbone_evidence_context: bool = False,
        backbone_evidence_target_family_indices: tuple[int, ...] = (),
        enable_all_family_evidence_affine_calibration: bool = False,
        evidence_affine_family_indices: tuple[int, ...] = (),
        enable_hn_variance_calibration: bool = False,
        enable_cprime_robust_likelihood: bool = False,
        enable_pair_evidence_mean_blend: bool = False,
        pair_evidence_mean_blend_weight: float = 0.0,
        enable_same_family_evidence_interpolation: bool = False,
        same_family_evidence_interpolation_weight: float = 0.0,
        same_family_evidence_interpolation_sigma: float = 6.0,
        same_family_evidence_interpolation_family_indices: tuple[int, ...] = (),
        enable_candidate_observable_context: bool = False,
        candidate_observable_context_gate_init: float = 1.0,
        candidate_observable_context_target_family_indices: tuple[int, ...] = (),
    ) -> None:
        """Initialize the fast moment head."""
        super().__init__()
        self.max_residue_index = max(int(max_residue_index), 2)
        self.atom_family_count = int(atom_family_count)
        self.min_sigma = float(min_sigma)
        self.max_sigma = max(float(max_sigma), self.min_sigma)
        self.enable_family_specific_adapters = bool(enable_family_specific_adapters)
        self.enable_nmr_structural_features = bool(enable_nmr_structural_features)
        self.enable_local_evidence_context = bool(enable_local_evidence_context)
        self.enable_same_residue_evidence_context = bool(
            enable_same_residue_evidence_context
        )
        self.same_residue_evidence_target_family_indices = tuple(
            int(index) for index in same_residue_evidence_target_family_indices
        )
        self.enable_target_set_evidence_encoder = bool(
            enable_target_set_evidence_encoder
        )
        self.enable_target_evidence_token = bool(enable_target_evidence_token)
        self.target_set_encoder_max_targets = max(
            int(target_set_encoder_max_targets), 1
        )
        self.target_set_encoder_masked_only = bool(target_set_encoder_masked_only)
        self.target_set_encoder_target_family_indices = tuple(
            int(index) for index in target_set_encoder_target_family_indices
        )
        self.enable_residue_grid_evidence_encoder = bool(
            enable_residue_grid_evidence_encoder
        )
        self.enable_residue_grid_family_gates = bool(
            enable_residue_grid_family_gates
        )
        self.residue_grid_encoder_masked_only = bool(residue_grid_encoder_masked_only)
        self.residue_grid_encoder_target_family_indices = tuple(
            int(index) for index in residue_grid_encoder_target_family_indices
        )
        self.target_set_context_gate_init = min(
            max(float(target_set_context_gate_init), 1e-4),
            1.0 - 1e-4,
        )
        self.residue_grid_context_gate_init = min(
            max(float(residue_grid_context_gate_init), 1e-4),
            1.0 - 1e-4,
        )
        self.local_evidence_window = max(int(local_evidence_window), 0)
        self.enable_residue_anchor_evidence_context = bool(
            enable_residue_anchor_evidence_context
        )
        self.residue_anchor_evidence_offsets = tuple(
            int(offset) for offset in residue_anchor_evidence_offsets
        ) or (-1, 0, 1)
        self.residue_anchor_evidence_target_family_indices = tuple(
            int(index) for index in residue_anchor_evidence_target_family_indices
        )
        self.enable_backbone_evidence_context = bool(enable_backbone_evidence_context)
        self.backbone_evidence_target_family_indices = tuple(
            int(index) for index in backbone_evidence_target_family_indices
        )
        self.enable_all_family_evidence_affine_calibration = bool(
            enable_all_family_evidence_affine_calibration
        )
        self.evidence_affine_family_indices = tuple(
            int(index) for index in evidence_affine_family_indices
        )
        self.enable_hn_variance_calibration = bool(enable_hn_variance_calibration)
        self.enable_cprime_robust_likelihood = bool(enable_cprime_robust_likelihood)
        self.enable_pair_evidence_mean_blend = bool(enable_pair_evidence_mean_blend)
        self.pair_evidence_mean_blend_weight = min(
            max(float(pair_evidence_mean_blend_weight), 0.0),
            1.0,
        )
        self.enable_same_family_evidence_interpolation = bool(
            enable_same_family_evidence_interpolation
        )
        self.same_family_evidence_interpolation_weight = min(
            max(float(same_family_evidence_interpolation_weight), 0.0),
            1.0,
        )
        self.same_family_evidence_interpolation_sigma = max(
            float(same_family_evidence_interpolation_sigma),
            1e-3,
        )
        self.same_family_evidence_interpolation_family_indices = tuple(
            int(index) for index in same_family_evidence_interpolation_family_indices
        )
        self.enable_candidate_observable_context = bool(
            enable_candidate_observable_context
        )
        self.candidate_observable_context_gate_init = min(
            max(float(candidate_observable_context_gate_init), 1e-4),
            1.0 - 1e-4,
        )
        self.candidate_observable_context_target_family_indices = tuple(
            int(index) for index in candidate_observable_context_target_family_indices
        )
        self.residue_embedding = nn.Embedding(self.max_residue_index, hidden_dim)
        self.amino_acid_embedding = nn.Embedding(amino_acid_count + 1, hidden_dim)
        self.atom_family_embedding = nn.Embedding(atom_family_count, hidden_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.structural_projection = (
            nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_nmr_structural_features
            else None
        )
        self.candidate_observable_projection = (
            nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_candidate_observable_context
            else None
        )
        self.candidate_observable_context_gate_logit = (
            nn.Parameter(
                torch.tensor(
                    math.log(
                        self.candidate_observable_context_gate_init
                        / (1.0 - self.candidate_observable_context_gate_init)
                    ),
                    dtype=torch.float32,
                )
            )
            if self.candidate_observable_projection is not None
            else None
        )
        self.local_evidence_projection = (
            nn.Sequential(
                nn.Linear(atom_family_count * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_local_evidence_context
            else None
        )
        self.same_residue_evidence_projection = (
            nn.Sequential(
                nn.Linear(atom_family_count * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_same_residue_evidence_context
            else None
        )
        self.target_evidence_token_projection = (
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_target_evidence_token
            else None
        )
        self.residue_anchor_evidence_projection = (
            nn.Sequential(
                nn.Linear(
                    atom_family_count * len(self.residue_anchor_evidence_offsets) * 2,
                    hidden_dim,
                ),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_residue_anchor_evidence_context
            else None
        )
        self.backbone_evidence_pairs = (
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (-1, 4),
            (-1, 2),
            (1, 1),
            (1, 0),
            (1, 2),
        )
        self.backbone_evidence_projection = (
            nn.Sequential(
                nn.Linear(len(self.backbone_evidence_pairs) * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_backbone_evidence_context
            else None
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(atom_family_count * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        encoder_heads = max(int(target_set_encoder_heads), 1)
        if hidden_dim % encoder_heads != 0:
            encoder_heads = 1
        encoder_layers = max(int(target_set_encoder_layers), 0)
        self.target_evidence_value_projection = (
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_target_set_evidence_encoder and encoder_layers > 0
            else None
        )
        self.target_set_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=encoder_heads,
                    dim_feedforward=max(hidden_dim * 2, 64),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=encoder_layers,
            )
            if self.target_evidence_value_projection is not None
            else None
        )
        self.target_set_context_gate_logit = (
            nn.Parameter(
                torch.tensor(
                    math.log(
                        self.target_set_context_gate_init
                        / (1.0 - self.target_set_context_gate_init)
                    ),
                    dtype=torch.float32,
                )
            )
            if self.target_set_encoder is not None
            else None
        )
        grid_encoder_heads = max(int(residue_grid_encoder_heads), 1)
        if hidden_dim % grid_encoder_heads != 0:
            grid_encoder_heads = 1
        grid_encoder_layers = max(int(residue_grid_encoder_layers), 0)
        self.residue_grid_evidence_projection = (
            nn.Sequential(
                nn.Linear(atom_family_count * 2 + 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.enable_residue_grid_evidence_encoder
            else None
        )
        self.residue_grid_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=grid_encoder_heads,
                    dim_feedforward=max(hidden_dim * 2, 64),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=grid_encoder_layers,
            )
            if self.residue_grid_evidence_projection is not None
            and grid_encoder_layers > 0
            else None
        )
        self.residue_grid_context_gate_logit = (
            nn.Parameter(
                torch.tensor(
                    math.log(
                        self.residue_grid_context_gate_init
                        / (1.0 - self.residue_grid_context_gate_init)
                    ),
                    dtype=torch.float32,
                )
            )
            if self.residue_grid_evidence_projection is not None
            else None
        )
        self.residue_grid_family_gate_logits = (
            nn.Parameter(
                torch.full(
                    (self.atom_family_count,),
                    math.log(
                        self.residue_grid_context_gate_init
                        / (1.0 - self.residue_grid_context_gate_init)
                    ),
                    dtype=torch.float32,
                )
            )
            if self.residue_grid_evidence_projection is not None
            and self.enable_residue_grid_family_gates
            else None
        )
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3 + example_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.target_network = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)
        self.family_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                for _ in range(self.atom_family_count)
            ]
        )
        self.family_mean_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.atom_family_count)]
        )
        self.family_log_sigma_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.atom_family_count)]
        )
        self.family_sigma_scale_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.atom_family_count)]
        )
        self.family_outlier_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.atom_family_count)]
        )

    def forward(
        self,
        *,
        sequence_context: torch.Tensor,
        example_features: torch.Tensor,
        candidate_context: torch.Tensor,
        target_features: torch.Tensor | None,
        target_values: torch.Tensor | None,
        candidate_observable_values: torch.Tensor | None,
        candidate_observable_mask: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        """Return target-level posterior mean and sigma predictions."""
        if target_features is None or target_features.numel() == 0:
            return None
        target_features = target_features.to(sequence_context.device)
        residue_indices = (
            target_features[:, 0].long().clamp(min=0, max=self.max_residue_index - 1)
        )
        amino_acid_indices = (
            target_features[:, 1]
            .long()
            .clamp(min=0, max=self.amino_acid_embedding.num_embeddings - 1)
        )
        atom_family_indices = (
            target_features[:, 2]
            .long()
            .clamp(min=0, max=self.atom_family_embedding.num_embeddings - 1)
        )
        positions = target_features[:, 3:4].to(sequence_context.dtype)
        target_context = (
            self.residue_embedding(residue_indices)
            + self.amino_acid_embedding(amino_acid_indices)
            + self.atom_family_embedding(atom_family_indices)
            + self.position_projection(positions)
        )
        if self.structural_projection is not None:
            target_context = target_context + self.structural_projection(
                self._nmr_structural_features(
                    residue_indices=residue_indices,
                    amino_acid_indices=amino_acid_indices,
                    atom_family_indices=atom_family_indices,
                    positions=positions,
                    dtype=sequence_context.dtype,
                )
            )
        if self.candidate_observable_projection is not None:
            observable_context = self.candidate_observable_projection(
                self._candidate_observable_features(
                    values=candidate_observable_values,
                    mask=candidate_observable_mask,
                    target_count=target_context.shape[0],
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
            gate = torch.sigmoid(
                self.candidate_observable_context_gate_logit.to(
                    dtype=target_context.dtype,
                    device=target_context.device,
                )
            )
            if self.candidate_observable_context_target_family_indices:
                target_family_mask = torch.zeros(
                    atom_family_indices.shape[0],
                    dtype=torch.bool,
                    device=target_context.device,
                )
                for family_index in (
                    self.candidate_observable_context_target_family_indices
                ):
                    target_family_mask = target_family_mask | (
                        atom_family_indices == int(family_index)
                    )
                observable_context = observable_context * target_family_mask.to(
                    dtype=target_context.dtype
                ).unsqueeze(-1)
            target_context = target_context + gate * observable_context
        if self.local_evidence_projection is not None:
            target_context = target_context + self.local_evidence_projection(
                self._local_evidence_features(
                    residue_indices=residue_indices,
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
        if self.same_residue_evidence_projection is not None:
            same_residue_context = self.same_residue_evidence_projection(
                self._same_residue_evidence_features(
                    residue_indices=residue_indices,
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
            if self.same_residue_evidence_target_family_indices:
                target_family_mask = torch.zeros(
                    atom_family_indices.shape[0],
                    dtype=torch.bool,
                    device=sequence_context.device,
                )
                for family_index in self.same_residue_evidence_target_family_indices:
                    target_family_mask = target_family_mask | (
                        atom_family_indices == int(family_index)
                    )
                same_residue_context = same_residue_context * target_family_mask.to(
                    dtype=same_residue_context.dtype
                ).unsqueeze(-1)
            target_context = target_context + same_residue_context
        if self.target_evidence_token_projection is not None:
            target_context = target_context + self.target_evidence_token_projection(
                self._target_evidence_value_features(
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
        if self.residue_anchor_evidence_projection is not None:
            anchor_context = self.residue_anchor_evidence_projection(
                self._residue_anchor_evidence_features(
                    residue_indices=residue_indices,
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
            if self.residue_anchor_evidence_target_family_indices:
                target_family_mask = torch.zeros(
                    atom_family_indices.shape[0],
                    dtype=torch.bool,
                    device=sequence_context.device,
                )
                for family_index in self.residue_anchor_evidence_target_family_indices:
                    target_family_mask = target_family_mask | (
                        atom_family_indices == int(family_index)
                    )
                anchor_context = anchor_context * target_family_mask.to(
                    dtype=anchor_context.dtype
                ).unsqueeze(-1)
            target_context = target_context + anchor_context
        if self.backbone_evidence_projection is not None:
            backbone_context = self.backbone_evidence_projection(
                self._backbone_evidence_features(
                    residue_indices=residue_indices,
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
            if self.backbone_evidence_target_family_indices:
                target_family_mask = torch.zeros(
                    atom_family_indices.shape[0],
                    dtype=torch.bool,
                    device=sequence_context.device,
                )
                for family_index in self.backbone_evidence_target_family_indices:
                    target_family_mask = target_family_mask | (
                        atom_family_indices == int(family_index)
                    )
                backbone_context = backbone_context * target_family_mask.to(
                    dtype=backbone_context.dtype
                ).unsqueeze(-1)
            target_context = target_context + backbone_context
        if self.residue_grid_evidence_projection is not None:
            grid_context = self._residue_grid_evidence_context(
                residue_indices=residue_indices,
                atom_family_indices=atom_family_indices,
                target_values=target_values,
                evidence_mask=evidence_mask,
                dtype=sequence_context.dtype,
                device=sequence_context.device,
            )
            if self.residue_grid_family_gate_logits is not None:
                family_gates = torch.sigmoid(
                    self.residue_grid_family_gate_logits.to(
                        dtype=target_context.dtype,
                        device=target_context.device,
                    )
                )
                family_indices = atom_family_indices.to(
                    dtype=torch.long,
                    device=target_context.device,
                ).clamp(0, family_gates.numel() - 1)
                grid_context = grid_context * family_gates[family_indices].unsqueeze(
                    -1
                )
            else:
                gate = torch.sigmoid(
                    self.residue_grid_context_gate_logit.to(
                        dtype=target_context.dtype,
                        device=target_context.device,
                    )
                )
                grid_context = gate * grid_context
            if (
                self.residue_grid_encoder_masked_only
                and evidence_mask is not None
                and evidence_mask.numel() == target_context.shape[0]
            ):
                masked_rows = ~evidence_mask.bool().to(target_context.device)
                grid_context = grid_context * masked_rows.to(
                    dtype=target_context.dtype
                ).unsqueeze(-1)
            if self.residue_grid_encoder_target_family_indices:
                target_family_mask = torch.zeros(
                    atom_family_indices.shape[0],
                    dtype=torch.bool,
                    device=target_context.device,
                )
                for family_index in self.residue_grid_encoder_target_family_indices:
                    target_family_mask = target_family_mask | (
                        atom_family_indices == int(family_index)
                    )
                grid_context = grid_context * target_family_mask.to(
                    dtype=target_context.dtype
                ).unsqueeze(-1)
            target_context = target_context + grid_context
        if (
            self.target_evidence_value_projection is not None
            and self.target_set_encoder is not None
            and target_context.shape[0] <= self.target_set_encoder_max_targets
        ):
            target_set_input = target_context + self.target_evidence_value_projection(
                self._target_evidence_value_features(
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    dtype=sequence_context.dtype,
                    device=sequence_context.device,
                )
            )
            encoded_context = self.target_set_encoder(
                target_set_input.unsqueeze(0)
            ).squeeze(0)
            gate = torch.sigmoid(
                self.target_set_context_gate_logit.to(
                    dtype=target_context.dtype,
                    device=target_context.device,
                )
            )
            denoising_delta = gate * (encoded_context - target_context)
            if (
                self.target_set_encoder_masked_only
                and evidence_mask is not None
                and evidence_mask.numel() == target_context.shape[0]
            ):
                masked_rows = ~evidence_mask.bool().to(target_context.device)
                denoising_delta = denoising_delta * masked_rows.to(
                    dtype=target_context.dtype
                ).unsqueeze(-1)
            if self.target_set_encoder_target_family_indices:
                target_family_mask = torch.zeros(
                    atom_family_indices.shape[0],
                    dtype=torch.bool,
                    device=target_context.device,
                )
                for family_index in self.target_set_encoder_target_family_indices:
                    target_family_mask = target_family_mask | (
                        atom_family_indices == int(family_index)
                    )
                denoising_delta = denoising_delta * target_family_mask.to(
                    dtype=target_context.dtype
                ).unsqueeze(-1)
            target_context = target_context + denoising_delta
        evidence_summary = self._evidence_summary(
            atom_family_indices=atom_family_indices,
            target_values=target_values,
            evidence_mask=evidence_mask,
            dtype=sequence_context.dtype,
            device=sequence_context.device,
        )
        evidence_context = self.evidence_projection(evidence_summary)
        candidate_summary = (
            torch.mean(candidate_context, dim=0)
            if candidate_context.numel() > 0
            else torch.zeros_like(sequence_context)
        )
        global_context = self.global_projection(
            torch.cat(
                [
                    sequence_context,
                    candidate_summary,
                    evidence_context,
                    example_features.to(sequence_context.dtype),
                ],
                dim=-1,
            )
        )
        pair_context = torch.cat(
            [
                target_context,
                global_context.unsqueeze(0).expand(target_context.shape[0], -1),
            ],
            dim=-1,
        )
        hidden = self.target_network(pair_context)
        if self.enable_family_specific_adapters:
            payload = self._family_specific_outputs(hidden, atom_family_indices)
            affine_families = self._evidence_affine_family_indices()
            if affine_families:
                payload["mu"] = self._apply_evidence_affine_calibration(
                    mu=payload["mu"],
                    atom_family_indices=atom_family_indices,
                    target_values=target_values,
                    evidence_mask=evidence_mask,
                    family_indices=affine_families,
                )
            payload["mu"] = self._apply_pair_evidence_mean_blend(
                mu=payload["mu"],
                residue_indices=residue_indices,
                atom_family_indices=atom_family_indices,
                target_values=target_values,
                evidence_mask=evidence_mask,
            )
            payload["mu"] = self._apply_same_family_evidence_interpolation(
                mu=payload["mu"],
                residue_indices=residue_indices,
                atom_family_indices=atom_family_indices,
                target_values=target_values,
                evidence_mask=evidence_mask,
            )
            return payload

        sigma = torch.nn.functional.softplus(self.log_sigma_head(hidden))
        mu = self.mean_head(hidden).squeeze(-1)
        affine_families = self._evidence_affine_family_indices()
        if affine_families:
            mu = self._apply_evidence_affine_calibration(
                mu=mu,
                atom_family_indices=atom_family_indices,
                target_values=target_values,
                evidence_mask=evidence_mask,
                family_indices=affine_families,
            )
        mu = self._apply_pair_evidence_mean_blend(
            mu=mu,
            residue_indices=residue_indices,
            atom_family_indices=atom_family_indices,
            target_values=target_values,
            evidence_mask=evidence_mask,
        )
        mu = self._apply_same_family_evidence_interpolation(
            mu=mu,
            residue_indices=residue_indices,
            atom_family_indices=atom_family_indices,
            target_values=target_values,
            evidence_mask=evidence_mask,
        )
        return {
            "mu": mu,
            "sigma": sigma.squeeze(-1).clamp(self.min_sigma, self.max_sigma),
            "sigma_scale": torch.ones(
                hidden.shape[0],
                dtype=hidden.dtype,
                device=hidden.device,
            ),
            "family_adapter_id": atom_family_indices.to(dtype=hidden.dtype),
            "outlier_score": torch.zeros(
                hidden.shape[0],
                dtype=hidden.dtype,
                device=hidden.device,
            ),
        }

    def _family_specific_outputs(
        self,
        hidden: torch.Tensor,
        atom_family_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-family adapter outputs for target-level moments."""
        mu = hidden.new_zeros(hidden.shape[0])
        sigma = hidden.new_ones(hidden.shape[0])
        sigma_scale = hidden.new_ones(hidden.shape[0])
        outlier_score = hidden.new_zeros(hidden.shape[0])
        for family_index in range(self.atom_family_count):
            mask = atom_family_indices == family_index
            if not torch.any(mask):
                continue
            family_hidden = self.family_adapters[family_index](hidden[mask])
            family_sigma = torch.nn.functional.softplus(
                self.family_log_sigma_heads[family_index](family_hidden).squeeze(-1)
            ).clamp(self.min_sigma, self.max_sigma)
            family_scale = (
                torch.nn.functional.softplus(
                    self.family_sigma_scale_heads[family_index](family_hidden).squeeze(
                        -1
                    )
                )
                + 0.05
            )
            if self.enable_hn_variance_calibration and family_index == 0:
                family_sigma = (family_sigma * family_scale).clamp(
                    self.min_sigma,
                    self.max_sigma,
                )
            mu[mask] = self.family_mean_heads[family_index](family_hidden).squeeze(-1)
            sigma[mask] = family_sigma
            sigma_scale[mask] = family_scale
            if self.enable_cprime_robust_likelihood and family_index == 4:
                outlier_score[mask] = torch.sigmoid(
                    self.family_outlier_heads[family_index](family_hidden).squeeze(-1)
                )
        return {
            "mu": mu,
            "sigma": sigma,
            "sigma_scale": sigma_scale,
            "family_adapter_id": atom_family_indices.to(dtype=hidden.dtype),
            "outlier_score": outlier_score,
        }

    def _evidence_affine_family_indices(self) -> tuple[int, ...]:
        """Return atom-family indices that use evidence affine calibration."""
        if self.enable_all_family_evidence_affine_calibration:
            return tuple(range(self.atom_family_count))
        family_indices: list[int] = [
            index
            for index in self.evidence_affine_family_indices
            if 0 <= index < self.atom_family_count
        ]
        if self.enable_hn_variance_calibration:
            family_indices.append(0)
        if self.enable_cprime_robust_likelihood:
            family_indices.append(4)
        return tuple(dict.fromkeys(family_indices))

    def _apply_evidence_affine_calibration(
        self,
        *,
        mu: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        family_indices: tuple[int, ...],
    ) -> torch.Tensor:
        """Calibrate selected family scales from unmasked evidence only."""
        if target_values is None or target_values.numel() != mu.numel():
            return mu
        values = target_values.to(dtype=mu.dtype, device=mu.device)
        calibrated = mu.clone()
        for family_index in family_indices:
            family_mask = atom_family_indices == int(family_index)
            usable = family_mask & torch.isfinite(mu) & torch.isfinite(values)
            if evidence_mask is not None and evidence_mask.numel() == mu.numel():
                usable = usable & evidence_mask.bool().to(mu.device)
            if torch.count_nonzero(usable) < 3:
                continue
            pred = mu[usable]
            target = values[usable]
            pred_mean = torch.mean(pred)
            target_mean = torch.mean(target)
            pred_std = torch.std(pred, unbiased=False).clamp_min(1e-3)
            target_std = torch.std(target, unbiased=False).clamp_min(1e-3)
            scale = (target_std / pred_std).clamp(0.4, 3.0)
            calibrated[family_mask] = (
                target_mean + (mu[family_mask] - pred_mean) * scale
            )
        return calibrated

    def _apply_pair_evidence_mean_blend(
        self,
        *,
        mu: torch.Tensor,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Blend masked HN/C' moments with same-residue evidence imputations."""
        if (
            not self.enable_pair_evidence_mean_blend
            or self.pair_evidence_mean_blend_weight <= 0.0
            or target_values is None
            or target_values.numel() != mu.numel()
            or evidence_mask is None
            or evidence_mask.numel() != mu.numel()
        ):
            return mu
        values = target_values.to(dtype=mu.dtype, device=mu.device)
        evidence = evidence_mask.bool().to(mu.device) & torch.isfinite(values)
        holdout = (~evidence_mask.bool().to(mu.device)) & torch.isfinite(values)
        if not torch.any(holdout):
            return mu
        blended = mu.clone()
        pair_specs = {
            0: (1,),  # HN from same-residue N
            4: (2, 3),  # C' from same-residue CA/CB
        }
        residue_ids = residue_indices.to(device=mu.device)
        family_ids = atom_family_indices.to(device=mu.device)
        for target_family, source_families in pair_specs.items():
            target_mask = holdout & (family_ids == int(target_family))
            if not torch.any(target_mask):
                continue
            imputed_sum = torch.zeros_like(mu)
            imputed_count = torch.zeros_like(mu)
            for source_family in source_families:
                fit_target_indices: list[int] = []
                fit_source_indices: list[int] = []
                apply_target_indices: list[int] = []
                apply_source_indices: list[int] = []
                for row_index in range(int(mu.numel())):
                    if family_ids[row_index] != int(target_family):
                        continue
                    same_residue_source = torch.nonzero(
                        (residue_ids == residue_ids[row_index])
                        & (family_ids == int(source_family)),
                        as_tuple=False,
                    ).flatten()
                    if same_residue_source.numel() == 0:
                        continue
                    source_index = int(same_residue_source[0].item())
                    if bool(evidence[row_index]) and bool(evidence[source_index]):
                        fit_target_indices.append(row_index)
                        fit_source_indices.append(source_index)
                    if bool(target_mask[row_index]) and bool(evidence[source_index]):
                        apply_target_indices.append(row_index)
                        apply_source_indices.append(source_index)
                if len(fit_target_indices) < 4 or len(apply_target_indices) < 1:
                    continue
                fit_targets = torch.tensor(
                    fit_target_indices,
                    dtype=torch.long,
                    device=mu.device,
                )
                fit_sources = torch.tensor(
                    fit_source_indices,
                    dtype=torch.long,
                    device=mu.device,
                )
                apply_targets = torch.tensor(
                    apply_target_indices,
                    dtype=torch.long,
                    device=mu.device,
                )
                apply_sources = torch.tensor(
                    apply_source_indices,
                    dtype=torch.long,
                    device=mu.device,
                )
                x = values[fit_sources].detach()
                y = values[fit_targets].detach()
                x_mean = torch.mean(x)
                y_mean = torch.mean(y)
                x_centered = x - x_mean
                y_centered = y - y_mean
                slope = torch.sum(x_centered * y_centered) / torch.sum(
                    torch.square(x_centered)
                ).clamp_min(1e-6)
                intercept = y_mean - slope * x_mean
                imputed = slope * values[apply_sources].detach() + intercept
                imputed_sum[apply_targets] += imputed
                imputed_count[apply_targets] += 1.0
            valid = imputed_count > 0
            if not torch.any(valid):
                continue
            imputed_mean = imputed_sum[valid] / imputed_count[valid].clamp_min(1.0)
            weight = mu.new_tensor(self.pair_evidence_mean_blend_weight)
            blended[valid] = (1.0 - weight) * blended[valid] + weight * imputed_mean
        return blended

    def _apply_same_family_evidence_interpolation(
        self,
        *,
        mu: torch.Tensor,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Blend masked moments with same-family sequence-local evidence smoothing."""
        if (
            not self.enable_same_family_evidence_interpolation
            or self.same_family_evidence_interpolation_weight <= 0.0
            or target_values is None
            or target_values.numel() != mu.numel()
            or evidence_mask is None
            or evidence_mask.numel() != mu.numel()
        ):
            return mu
        values = target_values.to(dtype=mu.dtype, device=mu.device)
        family_ids = atom_family_indices.to(device=mu.device)
        residue_ids = residue_indices.to(dtype=mu.dtype, device=mu.device)
        evidence = evidence_mask.bool().to(mu.device) & torch.isfinite(values)
        holdout = (~evidence_mask.bool().to(mu.device)) & torch.isfinite(values)
        if not torch.any(holdout):
            return mu
        blended = mu.clone()
        family_indices = self.same_family_evidence_interpolation_family_indices
        if not family_indices:
            family_indices = tuple(range(self.atom_family_count))
        sigma = mu.new_tensor(self.same_family_evidence_interpolation_sigma)
        blend_weight = mu.new_tensor(self.same_family_evidence_interpolation_weight)
        for family_index in family_indices:
            family_mask = family_ids == int(family_index)
            family_evidence = evidence & family_mask
            family_holdout = holdout & family_mask
            if (
                torch.count_nonzero(family_evidence) < 2
                or torch.count_nonzero(family_holdout) == 0
            ):
                continue
            evidence_positions = residue_ids[family_evidence]
            evidence_values = values[family_evidence].detach()
            holdout_positions = residue_ids[family_holdout]
            distances = torch.abs(
                holdout_positions.unsqueeze(1) - evidence_positions.unsqueeze(0)
            )
            weights = torch.exp(-0.5 * torch.square(distances / sigma))
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            imputed = weights @ evidence_values
            blended[family_holdout] = (
                (1.0 - blend_weight) * blended[family_holdout]
                + blend_weight * imputed
            )
        return blended

    def _nmr_structural_features(
        self,
        *,
        residue_indices: torch.Tensor,
        amino_acid_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return lightweight target-level NMR structural proxy features."""
        normalized_position = positions.squeeze(-1).to(dtype=dtype).clamp(0.0, 1.0)
        is_n_terminal = (normalized_position <= 0.05).to(dtype)
        is_c_terminal = (normalized_position >= 0.95).to(dtype)
        is_gly = (amino_acid_indices == 8).to(dtype)
        is_pro = (amino_acid_indices == 15).to(dtype)
        is_hn = (atom_family_indices == 0).to(dtype)
        is_cprime = (atom_family_indices == 4).to(dtype)
        residue_scale = (
            residue_indices.to(dtype=dtype) / float(max(self.max_residue_index - 1, 1))
        ).clamp(0.0, 1.0)
        availability = torch.ones_like(normalized_position)
        return torch.stack(
            [
                normalized_position,
                residue_scale,
                is_n_terminal,
                is_c_terminal,
                is_gly,
                is_pro,
                is_hn + is_cprime,
                availability,
            ],
            dim=-1,
        )

    def _candidate_observable_features(
        self,
        *,
        values: torch.Tensor | None,
        mask: torch.Tensor | None,
        target_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return target-row statistics from the candidate chemical-shift support."""
        if values is None or values.numel() == 0 or int(values.shape[0]) != target_count:
            return torch.zeros(target_count, 5, dtype=dtype, device=device)
        candidate_values = values.to(dtype=dtype, device=device)
        usable = torch.isfinite(candidate_values)
        if mask is not None and mask.shape == values.shape:
            usable = usable & mask.bool().to(device)
        safe_values = torch.where(
            usable,
            candidate_values,
            torch.zeros_like(candidate_values),
        )
        counts = usable.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
        mean = safe_values.sum(dim=1) / counts
        centered = torch.where(
            usable,
            candidate_values - mean.unsqueeze(1),
            torch.zeros_like(candidate_values),
        )
        std = torch.sqrt(torch.square(centered).sum(dim=1) / counts).clamp_min(0.0)
        high_fill = torch.full_like(candidate_values, 1e6)
        low_fill = torch.full_like(candidate_values, -1e6)
        min_value = torch.where(usable, candidate_values, high_fill).min(dim=1).values
        max_value = torch.where(usable, candidate_values, low_fill).max(dim=1).values
        min_value = torch.where(torch.isfinite(min_value), min_value, mean)
        max_value = torch.where(torch.isfinite(max_value), max_value, mean)
        availability = usable.to(dtype=dtype).mean(dim=1)
        return torch.stack(
            [
                (mean / 100.0).clamp(-5.0, 5.0),
                (std / 10.0).clamp(0.0, 5.0),
                (min_value / 100.0).clamp(-5.0, 5.0),
                (max_value / 100.0).clamp(-5.0, 5.0),
                availability.clamp(0.0, 1.0),
            ],
            dim=-1,
        )

    def _evidence_summary(
        self,
        *,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Summarize only unmasked observed targets by atom family."""
        summary = torch.zeros(self.atom_family_count, 3, dtype=dtype, device=device)
        if target_values is None or target_values.numel() == 0:
            return summary.flatten()
        values = target_values.to(dtype=dtype, device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == values.numel():
            usable = usable & evidence_mask.bool().to(device)
        for family_index in range(self.atom_family_count):
            family_mask = usable & (atom_family_indices == family_index)
            if torch.count_nonzero(family_mask) == 0:
                continue
            family_values = values[family_mask]
            summary[family_index, 0] = torch.mean(family_values)
            summary[family_index, 1] = torch.std(family_values, unbiased=False)
            summary[family_index, 2] = family_mask.to(dtype).mean()
        return summary.flatten()

    def _residue_grid_evidence_context(
        self,
        *,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode observed assignment evidence on an ordered residue grid."""
        target_count = int(residue_indices.numel())
        hidden_dim = self.residue_embedding.embedding_dim
        if target_count == 0 or target_values is None:
            return torch.zeros(target_count, hidden_dim, dtype=dtype, device=device)
        values = target_values.to(dtype=dtype, device=device)
        target_residues = residue_indices.to(device=device)
        target_families = atom_family_indices.to(device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == target_count:
            usable = usable & evidence_mask.bool().to(device)
        unique_residues, inverse = torch.unique(
            target_residues,
            sorted=True,
            return_inverse=True,
        )
        residue_count = int(unique_residues.numel())
        if residue_count == 0:
            return torch.zeros(target_count, hidden_dim, dtype=dtype, device=device)
        grid = torch.zeros(
            residue_count,
            self.atom_family_count,
            2,
            dtype=dtype,
            device=device,
        )
        for family_index in range(self.atom_family_count):
            family_usable = usable & (target_families == family_index)
            if torch.count_nonzero(family_usable) == 0:
                continue
            family_values = values[family_usable]
            family_mean = torch.mean(family_values)
            family_std = torch.std(family_values, unbiased=False).clamp_min(1e-3)
            for residue_position in range(residue_count):
                residue_mask = family_usable & (inverse == residue_position)
                if torch.count_nonzero(residue_mask) == 0:
                    continue
                z_value = torch.mean((values[residue_mask] - family_mean) / family_std)
                grid[residue_position, family_index, 0] = z_value.clamp(-8.0, 8.0)
                grid[residue_position, family_index, 1] = 1.0
        residue_scale = (
            unique_residues.to(dtype=dtype) / float(max(self.max_residue_index - 1, 1))
        ).clamp(0.0, 1.0)
        occupancy = grid[:, :, 1].mean(dim=1)
        grid_features = torch.cat(
            [
                grid.reshape(residue_count, -1),
                residue_scale.unsqueeze(-1),
                occupancy.unsqueeze(-1),
            ],
            dim=-1,
        )
        grid_context = self.residue_grid_evidence_projection(grid_features)
        if self.residue_grid_encoder is not None and residue_count > 1:
            grid_context = self.residue_grid_encoder(grid_context.unsqueeze(0)).squeeze(
                0
            )
        return grid_context[inverse]

    def _local_evidence_features(
        self,
        *,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return residue-local observed NMR context without masked-row leakage."""
        target_count = int(residue_indices.numel())
        feature_count = self.atom_family_count * 3
        if target_count == 0:
            return torch.zeros(0, feature_count, dtype=dtype, device=device)
        if target_values is None or target_values.numel() != target_count:
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)
        values = target_values.to(dtype=dtype, device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == target_count:
            usable = usable & evidence_mask.bool().to(device)
        if not torch.any(usable):
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)

        residues = residue_indices.to(device=device)
        family_indices = atom_family_indices.to(device=device)
        residue_distance = torch.abs(residues.unsqueeze(1) - residues.unsqueeze(0))
        if self.local_evidence_window > 0:
            local_residue_mask = residue_distance <= self.local_evidence_window
        else:
            local_residue_mask = residue_distance == 0
        feature_blocks: list[torch.Tensor] = []
        value_vector = values.unsqueeze(1)
        squared_vector = torch.square(values).unsqueeze(1)
        for family_index in range(self.atom_family_count):
            family_usable = usable & (family_indices == family_index)
            local_mask = local_residue_mask & family_usable.unsqueeze(0)
            weights = local_mask.to(dtype=dtype)
            counts = weights.sum(dim=1).clamp_min(0.0)
            safe_counts = counts.clamp_min(1.0)
            means = (weights @ value_vector).squeeze(1) / safe_counts
            second_moments = (weights @ squared_vector).squeeze(1) / safe_counts
            variances = (second_moments - torch.square(means)).clamp_min(0.0)
            stds = torch.sqrt(variances)
            availability = (counts > 0).to(dtype=dtype)
            feature_blocks.extend(
                [
                    means * availability,
                    stds * availability,
                    (counts / float(max(target_count, 1))).clamp(0.0, 1.0),
                ]
            )
        return torch.stack(feature_blocks, dim=1)

    def _target_evidence_value_features(
        self,
        *,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return masked-denoising token features from observed target values."""
        target_count = int(atom_family_indices.numel())
        if target_count == 0:
            return torch.zeros(0, 2, dtype=dtype, device=device)
        if target_values is None or target_values.numel() != target_count:
            return torch.zeros(target_count, 2, dtype=dtype, device=device)
        values = target_values.to(dtype=dtype, device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == target_count:
            usable = usable & evidence_mask.bool().to(device)
        features = torch.zeros(target_count, 2, dtype=dtype, device=device)
        if not torch.any(usable):
            return features
        family_indices = atom_family_indices.to(device=device)
        for family_index in range(self.atom_family_count):
            family_usable = usable & (family_indices == family_index)
            if torch.count_nonzero(family_usable) == 0:
                continue
            family_values = values[family_usable]
            family_mean = torch.mean(family_values)
            family_std = torch.std(family_values, unbiased=False).clamp_min(1e-3)
            z_scores = (values[family_usable] - family_mean) / family_std
            features[family_usable, 0] = z_scores
            features[family_usable, 1] = 1.0
        return features

    def _same_residue_evidence_features(
        self,
        *,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return exact same-residue cross-family evidence without leakage."""
        target_count = int(residue_indices.numel())
        feature_count = self.atom_family_count * 2
        if target_count == 0:
            return torch.zeros(0, feature_count, dtype=dtype, device=device)
        if target_values is None or target_values.numel() != target_count:
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)
        values = target_values.to(dtype=dtype, device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == target_count:
            usable = usable & evidence_mask.bool().to(device)
        if not torch.any(usable):
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)

        residues = residue_indices.to(device=device)
        family_indices = atom_family_indices.to(device=device)
        _, residue_inverse = torch.unique(
            residues,
            sorted=True,
            return_inverse=True,
        )
        residue_count = int(torch.max(residue_inverse).item()) + 1
        family_means = values.new_zeros(self.atom_family_count)
        family_stds = values.new_ones(self.atom_family_count)
        for family_index in range(self.atom_family_count):
            family_usable = usable & (family_indices == family_index)
            if torch.count_nonzero(family_usable) == 0:
                continue
            family_values = values[family_usable]
            family_means[family_index] = torch.mean(family_values)
            family_stds[family_index] = torch.std(
                family_values,
                unbiased=False,
            ).clamp_min(1e-3)

        feature_blocks: list[torch.Tensor] = []
        for family_index in range(self.atom_family_count):
            family_usable = usable & (family_indices == family_index)
            sums_by_residue = values.new_zeros(residue_count)
            counts_by_residue = values.new_zeros(residue_count)
            if torch.any(family_usable):
                family_residue_ids = residue_inverse[family_usable]
                sums_by_residue.scatter_add_(
                    0,
                    family_residue_ids,
                    values[family_usable],
                )
                counts_by_residue.scatter_add_(
                    0,
                    family_residue_ids,
                    torch.ones_like(values[family_usable]),
                )
            counts = counts_by_residue[residue_inverse]
            means = sums_by_residue[residue_inverse] / counts.clamp_min(1.0)
            availability = (counts > 0).to(dtype=dtype)
            z_scores = (
                (means - family_means[family_index]) / family_stds[family_index]
            ) * availability
            feature_blocks.extend([z_scores, availability])
        return torch.stack(feature_blocks, dim=1)

    def _residue_anchor_evidence_features(
        self,
        *,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return offset-specific residue/atom evidence without masked leakage."""
        target_count = int(residue_indices.numel())
        feature_count = (
            len(self.residue_anchor_evidence_offsets) * self.atom_family_count * 2
        )
        if target_count == 0:
            return torch.zeros(0, feature_count, dtype=dtype, device=device)
        if target_values is None or target_values.numel() != target_count:
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)
        values = target_values.to(dtype=dtype, device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == target_count:
            usable = usable & evidence_mask.bool().to(device)
        if not torch.any(usable):
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)

        residues = residue_indices.to(device=device)
        family_indices = atom_family_indices.to(device=device)
        family_means = values.new_zeros(self.atom_family_count)
        family_stds = values.new_ones(self.atom_family_count)
        for family_index in range(self.atom_family_count):
            family_usable = usable & (family_indices == family_index)
            if torch.count_nonzero(family_usable) == 0:
                continue
            family_values = values[family_usable]
            family_means[family_index] = torch.mean(family_values)
            family_stds[family_index] = torch.std(
                family_values,
                unbiased=False,
            ).clamp_min(1e-3)

        feature_blocks: list[torch.Tensor] = []
        evidence_residues = residues.unsqueeze(0)
        query_residues = residues.unsqueeze(1)
        value_vector = values.unsqueeze(1)
        for offset in self.residue_anchor_evidence_offsets:
            offset_mask = evidence_residues == (query_residues + int(offset))
            for family_index in range(self.atom_family_count):
                family_usable = usable & (family_indices == family_index)
                match = offset_mask & family_usable.unsqueeze(0)
                weights = match.to(dtype=dtype)
                counts = weights.sum(dim=1)
                means = (weights @ value_vector).squeeze(1) / counts.clamp_min(1.0)
                availability = (counts > 0).to(dtype=dtype)
                z_scores = (
                    (means - family_means[family_index]) / family_stds[family_index]
                ) * availability
                feature_blocks.extend([z_scores, availability])
        return torch.stack(feature_blocks, dim=1)

    def _backbone_evidence_features(
        self,
        *,
        residue_indices: torch.Tensor,
        atom_family_indices: torch.Tensor,
        target_values: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return hand-picked backbone NMR evidence without masked leakage."""
        target_count = int(residue_indices.numel())
        feature_count = len(self.backbone_evidence_pairs) * 2
        if target_count == 0:
            return torch.zeros(0, feature_count, dtype=dtype, device=device)
        if target_values is None or target_values.numel() != target_count:
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)
        values = target_values.to(dtype=dtype, device=device)
        usable = torch.isfinite(values)
        if evidence_mask is not None and evidence_mask.numel() == target_count:
            usable = usable & evidence_mask.bool().to(device)
        if not torch.any(usable):
            return torch.zeros(target_count, feature_count, dtype=dtype, device=device)

        residues = residue_indices.to(device=device)
        family_indices = atom_family_indices.to(device=device)
        family_means = values.new_zeros(self.atom_family_count)
        family_stds = values.new_ones(self.atom_family_count)
        for family_index in range(self.atom_family_count):
            family_usable = usable & (family_indices == family_index)
            if torch.count_nonzero(family_usable) == 0:
                continue
            family_values = values[family_usable]
            family_means[family_index] = torch.mean(family_values)
            family_stds[family_index] = torch.std(
                family_values,
                unbiased=False,
            ).clamp_min(1e-3)

        evidence_residues = residues.unsqueeze(0)
        query_residues = residues.unsqueeze(1)
        value_vector = values.unsqueeze(1)
        feature_blocks: list[torch.Tensor] = []
        for offset, family_index in self.backbone_evidence_pairs:
            family_usable = usable & (family_indices == int(family_index))
            match = (
                evidence_residues == (query_residues + int(offset))
            ) & family_usable.unsqueeze(0)
            weights = match.to(dtype=dtype)
            counts = weights.sum(dim=1)
            means = (weights @ value_vector).squeeze(1) / counts.clamp_min(1.0)
            availability = (counts > 0).to(dtype=dtype)
            z_scores = (
                (means - family_means[int(family_index)])
                / family_stds[int(family_index)]
            ) * availability
            feature_blocks.extend([z_scores, availability])
        return torch.stack(feature_blocks, dim=1)


class PosteriorStateTokenHead(nn.Module):
    """Return BioEmu-style posterior state tokens and occupancies."""

    def __init__(self, hidden_dim: int, state_count: int = 16) -> None:
        """Initialize learned state queries over candidate latent states."""
        super().__init__()
        self.state_count = max(int(state_count), 1)
        self.state_queries = nn.Parameter(torch.randn(self.state_count, hidden_dim))
        self.logit_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        candidate_context: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return state token embeddings, occupancy, and assignments."""
        if candidate_context.numel() == 0:
            empty = logits.new_zeros(0)
            return {
                "state_tokens": candidate_context.new_zeros(
                    0, candidate_context.shape[-1]
                ),
                "state_weights": empty,
                "state_assignments": candidate_context.new_zeros(
                    candidate_context.shape[0], 0
                ),
            }
        scores = candidate_context @ self.state_queries.t()
        assignments = torch.softmax(scores, dim=1)
        normalizer = assignments.sum(dim=0).clamp_min(1e-8)
        state_tokens = assignments.t() @ candidate_context / normalizer.unsqueeze(1)
        state_logits = self.logit_head(state_tokens).squeeze(-1)
        candidate_mass = torch.softmax(logits, dim=0)
        evidence_mass = assignments.t() @ candidate_mass
        state_weights = torch.softmax(state_logits, dim=0)
        state_weights = 0.5 * state_weights + 0.5 * (
            evidence_mass / evidence_mass.sum().clamp_min(1e-8)
        )
        state_weights = state_weights / state_weights.sum().clamp_min(1e-8)
        return {
            "state_tokens": state_tokens,
            "state_weights": state_weights,
            "state_assignments": assignments,
        }


class LatentPosteriorFlow(nn.Module):
    """Few-step rectified-flow approximation on posterior state tokens."""

    def __init__(self, hidden_dim: int, steps: int = 4) -> None:
        """Initialize the lightweight latent flow."""
        super().__init__()
        self.steps = max(int(steps), 0)
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return latent samples after a few deterministic flow updates."""
        if self.steps <= 0 or state_tokens.numel() == 0:
            return state_tokens
        samples = state_tokens
        context = global_context.unsqueeze(0).expand(samples.shape[0], -1)
        step_size = 1.0 / float(self.steps)
        for _ in range(self.steps):
            samples = samples + step_size * self.update(
                torch.cat([samples, context], dim=-1)
            )
        return samples


class MirrorDescentPosteriorRefiner(nn.Module):
    """Apply a few differentiable CCC-proxy mirror-descent logit updates."""

    def __init__(
        self,
        steps: int = 0,
        step_size: float = 0.0,
        feature_index: int | None = None,
    ) -> None:
        """Initialize the lightweight posterior optimizer layer."""
        super().__init__()
        self.steps = max(int(steps), 0)
        self.step_size = float(step_size)
        self.feature_index = feature_index

    def forward(
        self,
        logits: torch.Tensor,
        candidate_numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits nudged toward the configured CCC proxy direction."""
        if (
            self.steps <= 0
            or self.step_size == 0.0
            or self.feature_index is None
            or candidate_numeric_features.ndim != 2
        ):
            return logits
        index = int(self.feature_index)
        if index < 0:
            index = candidate_numeric_features.shape[1] + index
        if index < 0 or index >= candidate_numeric_features.shape[1]:
            return logits
        proxy = candidate_numeric_features[:, index].to(logits.dtype)
        proxy = torch.nan_to_num(proxy, nan=0.0, neginf=0.0, posinf=0.0)
        proxy = (proxy - proxy.mean()) / proxy.std(unbiased=False).clamp_min(1e-6)
        return logits + float(self.steps) * self.step_size * proxy


class FamilyAffineCalibrationLayer(nn.Module):
    """Stateless marker module for evidence-fit family affine calibration."""

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """Return predictions unchanged; fitting is handled by trainer masks."""
        return predictions


class CandidateSetScorer(nn.Module):
    """Sequence-conditioned scorer over one candidate support set."""

    def __init__(
        self,
        vocab_size: int,
        num_sources: int,
        sequence_embedding_dim: int,
        source_embedding_dim: int,
        example_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        use_candidate_set_encoder: bool = False,
        set_encoder_layers: int = 0,
        set_encoder_heads: int = 4,
        logit_temperature: float = 1.0,
        use_bayesian_posterior: bool = False,
        evidence_feature_index: int | None = None,
        evidence_temperature: float = 1.0,
        enable_forward_residual_head: bool = False,
        enable_residue_atom_residual_head: bool = False,
        residue_atom_max_residue_index: int = 4096,
        enable_state_mixture: bool = False,
        state_mixture_count: int = 8,
        enable_moment_head: bool = False,
        enable_posterior_state_tokens: bool = False,
        posterior_state_count: int = 16,
        enable_latent_posterior_flow: bool = False,
        latent_flow_steps: int = 4,
        enable_family_specific_moment_head: bool = False,
        enable_nmr_structural_features: bool = False,
        enable_local_evidence_context: bool = False,
        enable_same_residue_evidence_context: bool = False,
        same_residue_evidence_target_family_indices: tuple[int, ...] = (),
        enable_target_set_evidence_encoder: bool = False,
        enable_target_evidence_token: bool = False,
        target_set_encoder_layers: int = 1,
        target_set_encoder_heads: int = 4,
        target_set_encoder_max_targets: int = 1024,
        target_set_context_gate_init: float = 1.0,
        target_set_encoder_masked_only: bool = False,
        target_set_encoder_target_family_indices: tuple[int, ...] = (),
        enable_residue_grid_evidence_encoder: bool = False,
        residue_grid_encoder_layers: int = 1,
        residue_grid_encoder_heads: int = 4,
        residue_grid_context_gate_init: float = 0.2,
        enable_residue_grid_family_gates: bool = False,
        residue_grid_encoder_masked_only: bool = True,
        residue_grid_encoder_target_family_indices: tuple[int, ...] = (),
        local_evidence_window: int = 3,
        enable_residue_anchor_evidence_context: bool = False,
        residue_anchor_evidence_offsets: tuple[int, ...] = (-1, 0, 1),
        residue_anchor_evidence_target_family_indices: tuple[int, ...] = (),
        enable_backbone_evidence_context: bool = False,
        backbone_evidence_target_family_indices: tuple[int, ...] = (),
        enable_all_family_evidence_affine_calibration: bool = False,
        evidence_affine_family_indices: tuple[int, ...] = (),
        enable_hn_variance_calibration: bool = False,
        enable_cprime_robust_likelihood: bool = False,
        enable_pair_evidence_mean_blend: bool = False,
        pair_evidence_mean_blend_weight: float = 0.0,
        enable_same_family_evidence_interpolation: bool = False,
        same_family_evidence_interpolation_weight: float = 0.0,
        same_family_evidence_interpolation_sigma: float = 6.0,
        same_family_evidence_interpolation_family_indices: tuple[int, ...] = (),
        enable_candidate_observable_context: bool = False,
        candidate_observable_context_gate_init: float = 1.0,
        candidate_observable_context_target_family_indices: tuple[int, ...] = (),
        nmr_energy_guidance_weight: float = 1.0,
        mirror_descent_steps: int = 0,
        mirror_descent_step_size: float = 0.0,
        ccc_proxy_feature_index: int | None = None,
        robust_likelihood_mode: str = "gaussian",
        student_t_degrees_of_freedom: float = 4.0,
    ) -> None:
        """Initialize a prior/posterior candidate-set scorer."""
        super().__init__()
        encoder_layers = max(int(set_encoder_layers), 0)
        encoder_heads = max(int(set_encoder_heads), 1)
        if hidden_dim % encoder_heads != 0:
            encoder_heads = 1
        self.logit_temperature = max(float(logit_temperature), 1e-6)
        self.use_candidate_set_encoder = bool(use_candidate_set_encoder) and (
            encoder_layers > 0
        )
        self.evidence_feature_index = evidence_feature_index
        self.robust_likelihood_mode = str(robust_likelihood_mode).lower()
        self.nmr_energy_guidance_weight = float(nmr_energy_guidance_weight)
        self.student_t_degrees_of_freedom = max(
            float(student_t_degrees_of_freedom),
            1.0,
        )
        self.posterior_head = (
            BayesianPosteriorHead(evidence_temperature=evidence_temperature)
            if use_bayesian_posterior
            else None
        )
        self.mirror_descent_refiner = MirrorDescentPosteriorRefiner(
            steps=mirror_descent_steps,
            step_size=mirror_descent_step_size,
            feature_index=ccc_proxy_feature_index,
        )
        self.state_posterior = None
        self.moment_head = None
        self.posterior_state_tokens = None
        self.latent_posterior_flow = None
        self.forward_residual_head = (
            ForwardResidualHead(hidden_dim=hidden_dim)
            if enable_forward_residual_head
            else None
        )
        self.residue_atom_residual_head = (
            ResidueAtomResidualHead(
                hidden_dim=hidden_dim,
                max_residue_index=residue_atom_max_residue_index,
            )
            if enable_residue_atom_residual_head
            else None
        )
        self.sequence_embedding = nn.Embedding(
            vocab_size,
            sequence_embedding_dim,
            padding_idx=0,
        )
        self.source_embedding = nn.Embedding(
            max(num_sources, 1),
            source_embedding_dim,
        )
        self.context_projection = nn.Sequential(
            nn.Linear(sequence_embedding_dim + example_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(source_embedding_dim + candidate_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.set_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=encoder_heads,
                    dim_feedforward=max(hidden_dim * 2, 64),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=encoder_layers,
            )
            if self.use_candidate_set_encoder
            else None
        )
        self.state_posterior = (
            StatePosteriorScorer(
                hidden_dim=hidden_dim,
                state_count=state_mixture_count,
            )
            if enable_state_mixture
            else None
        )
        self.moment_head = (
            PosteriorMomentHead(
                hidden_dim=hidden_dim,
                example_feature_dim=example_feature_dim,
                max_residue_index=residue_atom_max_residue_index,
                enable_family_specific_adapters=enable_family_specific_moment_head,
                enable_nmr_structural_features=enable_nmr_structural_features,
                enable_local_evidence_context=enable_local_evidence_context,
                enable_same_residue_evidence_context=(
                    enable_same_residue_evidence_context
                ),
                same_residue_evidence_target_family_indices=(
                    same_residue_evidence_target_family_indices
                ),
                enable_target_set_evidence_encoder=enable_target_set_evidence_encoder,
                enable_target_evidence_token=enable_target_evidence_token,
                target_set_encoder_layers=target_set_encoder_layers,
                target_set_encoder_heads=target_set_encoder_heads,
                target_set_encoder_max_targets=target_set_encoder_max_targets,
                target_set_context_gate_init=target_set_context_gate_init,
                target_set_encoder_masked_only=target_set_encoder_masked_only,
                target_set_encoder_target_family_indices=(
                    target_set_encoder_target_family_indices
                ),
                enable_residue_grid_evidence_encoder=(
                    enable_residue_grid_evidence_encoder
                ),
                residue_grid_encoder_layers=residue_grid_encoder_layers,
                residue_grid_encoder_heads=residue_grid_encoder_heads,
                residue_grid_context_gate_init=residue_grid_context_gate_init,
                enable_residue_grid_family_gates=enable_residue_grid_family_gates,
                residue_grid_encoder_masked_only=residue_grid_encoder_masked_only,
                residue_grid_encoder_target_family_indices=(
                    residue_grid_encoder_target_family_indices
                ),
                local_evidence_window=local_evidence_window,
                enable_residue_anchor_evidence_context=(
                    enable_residue_anchor_evidence_context
                ),
                residue_anchor_evidence_offsets=residue_anchor_evidence_offsets,
                residue_anchor_evidence_target_family_indices=(
                    residue_anchor_evidence_target_family_indices
                ),
                enable_backbone_evidence_context=enable_backbone_evidence_context,
                backbone_evidence_target_family_indices=(
                    backbone_evidence_target_family_indices
                ),
                enable_all_family_evidence_affine_calibration=(
                    enable_all_family_evidence_affine_calibration
                ),
                evidence_affine_family_indices=evidence_affine_family_indices,
                enable_hn_variance_calibration=enable_hn_variance_calibration,
                enable_cprime_robust_likelihood=enable_cprime_robust_likelihood,
                enable_pair_evidence_mean_blend=enable_pair_evidence_mean_blend,
                pair_evidence_mean_blend_weight=pair_evidence_mean_blend_weight,
                enable_same_family_evidence_interpolation=(
                    enable_same_family_evidence_interpolation
                ),
                same_family_evidence_interpolation_weight=(
                    same_family_evidence_interpolation_weight
                ),
                same_family_evidence_interpolation_sigma=(
                    same_family_evidence_interpolation_sigma
                ),
                same_family_evidence_interpolation_family_indices=(
                    same_family_evidence_interpolation_family_indices
                ),
                enable_candidate_observable_context=(
                    enable_candidate_observable_context
                ),
                candidate_observable_context_gate_init=(
                    candidate_observable_context_gate_init
                ),
                candidate_observable_context_target_family_indices=(
                    candidate_observable_context_target_family_indices
                ),
            )
            if enable_moment_head
            else None
        )
        self.posterior_state_tokens = (
            PosteriorStateTokenHead(
                hidden_dim=hidden_dim,
                state_count=posterior_state_count,
            )
            if enable_posterior_state_tokens
            else None
        )
        self.latent_posterior_flow = (
            LatentPosteriorFlow(
                hidden_dim=hidden_dim,
                steps=latent_flow_steps,
            )
            if enable_latent_posterior_flow
            else None
        )
        self.family_affine_calibration = FamilyAffineCalibrationLayer()
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        example_features: torch.Tensor,
        candidate_source_indices: torch.Tensor,
        candidate_numeric_features: torch.Tensor,
        chemical_shift_target_features: torch.Tensor | None = None,
        chemical_shift_values: torch.Tensor | None = None,
        chemical_shift_mask: torch.Tensor | None = None,
        chemical_shift_targets: torch.Tensor | None = None,
        chemical_shift_sigmas: torch.Tensor | None = None,
        chemical_shift_evidence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return prior/posterior logits and normalized support weights."""
        sequence_embeddings = self.sequence_embedding(sequence_tokens)
        token_mask = (sequence_tokens != 0).float().unsqueeze(-1)
        denominator = token_mask.sum(dim=0).clamp_min(1.0)
        sequence_context = (sequence_embeddings * token_mask).sum(dim=0) / denominator

        context_input = torch.cat([sequence_context, example_features], dim=-1)
        context = self.context_projection(context_input)
        source_embeddings = self.source_embedding(candidate_source_indices)
        expanded_context = context.unsqueeze(0).expand(
            candidate_numeric_features.shape[0],
            -1,
        )
        candidate_context = self.candidate_projection(
            torch.cat([source_embeddings, candidate_numeric_features], dim=-1)
        )
        if self.set_encoder is not None:
            candidate_context = self.set_encoder(
                candidate_context.unsqueeze(0)
            ).squeeze(0)
        residual_payload = (
            self.forward_residual_head(candidate_context)
            if self.forward_residual_head is not None
            else None
        )
        residue_atom_payload = (
            self.residue_atom_residual_head(
                candidate_context,
                chemical_shift_target_features,
            )
            if self.residue_atom_residual_head is not None
            else None
        )
        candidate_input = torch.cat([expanded_context, candidate_context], dim=-1)
        prior_logits = self.scorer(candidate_input).squeeze(-1) / self.logit_temperature
        evidence_log_likelihood = self._chemical_shift_evidence_log_likelihood(
            values=chemical_shift_values,
            mask=chemical_shift_mask,
            targets=chemical_shift_targets,
            sigmas=chemical_shift_sigmas,
            evidence_mask=chemical_shift_evidence_mask,
            residual_mu=(
                residue_atom_payload["mu"] if residue_atom_payload is not None else None
            ),
        )
        if evidence_log_likelihood is None:
            evidence_log_likelihood = self._evidence_log_likelihood(
                candidate_numeric_features
            )
        if evidence_log_likelihood is not None:
            evidence_log_likelihood = (
                evidence_log_likelihood * self.nmr_energy_guidance_weight
            )
        effective_residual_mu = (
            residue_atom_payload["mu"]
            if residue_atom_payload is not None
            else (
                residual_payload["mu"]
                if residual_payload is not None
                else torch.zeros(
                    candidate_numeric_features.shape[0],
                    5,
                    dtype=prior_logits.dtype,
                    device=prior_logits.device,
                )
            )
        )
        effective_residual_sigma = (
            residue_atom_payload["sigma"]
            if residue_atom_payload is not None
            else (
                residual_payload["sigma"]
                if residual_payload is not None
                else torch.ones(
                    candidate_numeric_features.shape[0],
                    5,
                    dtype=prior_logits.dtype,
                    device=prior_logits.device,
                )
            )
        )
        logits = (
            self.posterior_head(prior_logits, evidence_log_likelihood)
            if self.posterior_head is not None
            else prior_logits
        )
        logits = self.mirror_descent_refiner(logits, candidate_numeric_features)
        if self.state_posterior is not None:
            state_payload = self.state_posterior(logits, candidate_context)
            weights = state_payload["weights"]
            state_weights = state_payload["state_weights"]
            state_assignments = state_payload["state_assignments"]
        else:
            weights = torch.softmax(logits, dim=0)
            state_weights = logits.new_zeros(0)
            state_assignments = logits.new_zeros(logits.shape[0], 0)
        moment_payload = (
            self.moment_head(
                sequence_context=context,
                example_features=example_features,
                candidate_context=candidate_context,
                target_features=chemical_shift_target_features,
                target_values=chemical_shift_targets,
                candidate_observable_values=chemical_shift_values,
                candidate_observable_mask=chemical_shift_mask,
                evidence_mask=chemical_shift_evidence_mask,
            )
            if self.moment_head is not None
            else None
        )
        token_payload = (
            self.posterior_state_tokens(candidate_context, logits)
            if self.posterior_state_tokens is not None
            else None
        )
        posterior_state_tokens = (
            token_payload["state_tokens"]
            if token_payload is not None
            else logits.new_zeros(0, candidate_context.shape[-1])
        )
        posterior_state_weights = (
            token_payload["state_weights"]
            if token_payload is not None
            else logits.new_zeros(0)
        )
        posterior_state_assignments = (
            token_payload["state_assignments"]
            if token_payload is not None
            else logits.new_zeros(logits.shape[0], 0)
        )
        posterior_latent_samples = (
            self.latent_posterior_flow(posterior_state_tokens, context)
            if self.latent_posterior_flow is not None
            else posterior_state_tokens
        )
        return {
            "logits": logits,
            "weights": weights,
            "prior_logits": prior_logits,
            "prior_weights": torch.softmax(prior_logits, dim=0),
            "state_weights": state_weights,
            "state_assignments": state_assignments,
            "moment_mu": (
                moment_payload["mu"]
                if moment_payload is not None
                else logits.new_zeros(0)
            ),
            "moment_sigma": (
                moment_payload["sigma"]
                if moment_payload is not None
                else logits.new_ones(0)
            ),
            "moment_sigma_scale": (
                moment_payload["sigma_scale"]
                if moment_payload is not None and "sigma_scale" in moment_payload
                else logits.new_ones(0)
            ),
            "moment_family_adapter_id": (
                moment_payload["family_adapter_id"]
                if moment_payload is not None and "family_adapter_id" in moment_payload
                else logits.new_zeros(0)
            ),
            "moment_outlier_score": (
                moment_payload["outlier_score"]
                if moment_payload is not None and "outlier_score" in moment_payload
                else logits.new_zeros(0)
            ),
            "posterior_state_tokens": posterior_state_tokens,
            "posterior_state_weights": posterior_state_weights,
            "posterior_state_assignments": posterior_state_assignments,
            "posterior_latent_samples": posterior_latent_samples,
            "evidence_log_likelihood": (
                evidence_log_likelihood
                if evidence_log_likelihood is not None
                else torch.zeros_like(prior_logits)
            ),
            "forward_residual_mu": effective_residual_mu,
            "forward_residual_sigma": effective_residual_sigma,
            "family_residual_mu": (
                residual_payload["mu"]
                if residual_payload is not None
                else torch.zeros(
                    candidate_numeric_features.shape[0],
                    5,
                    dtype=prior_logits.dtype,
                    device=prior_logits.device,
                )
            ),
            "family_residual_sigma": (
                residual_payload["sigma"]
                if residual_payload is not None
                else torch.ones(
                    candidate_numeric_features.shape[0],
                    5,
                    dtype=prior_logits.dtype,
                    device=prior_logits.device,
                )
            ),
            "residue_atom_residual_mu": (
                residue_atom_payload["mu"]
                if residue_atom_payload is not None
                else torch.zeros(
                    0,
                    candidate_numeric_features.shape[0],
                    dtype=prior_logits.dtype,
                    device=prior_logits.device,
                )
            ),
            "residue_atom_residual_sigma": (
                residue_atom_payload["sigma"]
                if residue_atom_payload is not None
                else torch.ones(
                    0,
                    candidate_numeric_features.shape[0],
                    dtype=prior_logits.dtype,
                    device=prior_logits.device,
                )
            ),
            "sequence_context": sequence_context,
        }

    def _evidence_log_likelihood(
        self,
        candidate_numeric_features: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the configured evidence-likelihood feature when available."""
        if self.evidence_feature_index is None:
            return None
        index = int(self.evidence_feature_index)
        if index < 0:
            index = candidate_numeric_features.shape[1] + index
        if index < 0 or index >= candidate_numeric_features.shape[1]:
            return None
        return candidate_numeric_features[:, index]

    def _chemical_shift_evidence_log_likelihood(
        self,
        *,
        values: torch.Tensor | None,
        mask: torch.Tensor | None,
        targets: torch.Tensor | None,
        sigmas: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        residual_mu: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return corrected per-candidate chemical-shift evidence likelihood."""
        if (
            values is None
            or mask is None
            or targets is None
            or sigmas is None
            or residual_mu is None
            or values.ndim != 2
            or residual_mu.shape != values.shape
        ):
            return None
        usable_mask = mask.bool()
        if evidence_mask is not None:
            usable_mask = usable_mask & evidence_mask.bool().unsqueeze(1)
        corrected = values.to(residual_mu.dtype) + residual_mu
        sigma = sigmas.to(residual_mu.dtype).clamp_min(1e-6)
        target = targets.to(residual_mu.dtype)
        residual = (corrected - target.unsqueeze(1)) / sigma.unsqueeze(1)
        if self.robust_likelihood_mode in {"student_t", "student-t", "t"}:
            nu = corrected.new_tensor(self.student_t_degrees_of_freedom)
            half = corrected.new_tensor(0.5)
            log_likelihood = (
                torch.lgamma((nu + 1.0) * half)
                - torch.lgamma(nu * half)
                - half * torch.log(nu * corrected.new_tensor(torch.pi))
                - torch.log(sigma.unsqueeze(1))
                - ((nu + 1.0) * half) * torch.log1p(torch.square(residual) / nu)
            )
        else:
            log_likelihood = (
                -0.5 * torch.square(residual)
                - torch.log(sigma.unsqueeze(1))
                - 0.5 * torch.log(corrected.new_tensor(2.0 * torch.pi))
            )
        masked = torch.where(
            usable_mask, log_likelihood, torch.zeros_like(log_likelihood)
        )
        counts = usable_mask.sum(dim=0).to(residual_mu.dtype)
        mean_log_likelihood = masked.sum(dim=0) / counts.clamp_min(1.0)
        empty = counts <= 0
        if torch.any(empty):
            mean_log_likelihood = mean_log_likelihood.masked_fill(empty, -100.0)
        return mean_log_likelihood.clamp(min=-100.0, max=10.0)


StudentDensityModel = CandidateSetScorer
