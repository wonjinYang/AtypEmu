"""Torch-free training-data preparation helpers for AtypEmu."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atypemu.datasets import IntegratedDataRegistry
from atypemu.training.config import StudentTrainingConfig
from atypemu.training.materialize import resolve_existing_path
from atypemu.training.tasks import (
    BIOEMU_LATENT_NMR_V1,
    POSTERIOR_NMR_CONDITIONED,
    POSTERIOR_NMR_CONDITIONED_V2,
    POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    POSTERIOR_NMR_PHYSICS_V1,
    POSTERIOR_NMR_RESIDUE_ATOM_V1,
    PRIOR_SEQUENCE_ONLY,
    resolve_training_task,
)
from atypemu.types import (
    CandidatePool,
    NMRTargetBundle,
    ObservableBundle,
    WeightSolution,
)


AA3_VOCAB = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLN": 6,
    "GLU": 7,
    "GLY": 8,
    "HIS": 9,
    "ILE": 10,
    "LEU": 11,
    "LYS": 12,
    "MET": 13,
    "PHE": 14,
    "PRO": 15,
    "SER": 16,
    "THR": 17,
    "TRP": 18,
    "TYR": 19,
    "VAL": 20,
}

CHEMICAL_SHIFT_SIGMA_FLOORS = {
    "HN": 0.03,
    "N": 0.30,
    "CA": 0.20,
    "CB": 0.20,
    "C'": 0.20,
}
CHEMICAL_SHIFT_FAMILIES = ("HN", "N", "CA", "CB", "C'")
CHEMICAL_SHIFT_ATOM_FAMILY_MAP = {
    "H": "HN",
    "HN": "HN",
    "N": "N",
    "CA": "CA",
    "CB": "CB",
    "C": "C'",
}
CHEMICAL_SHIFT_PPM_RANGES = {
    "HN": (4.0, 12.0),
    "N": (80.0, 155.0),
    "CA": (20.0, 85.0),
    "CB": (0.0, 90.0),
    "C'": (150.0, 190.0),
}
TRAINING_INPUT_MODES = {
    PRIOR_SEQUENCE_ONLY,
    POSTERIOR_NMR_CONDITIONED,
    POSTERIOR_NMR_CONDITIONED_V2,
    POSTERIOR_NMR_RESIDUE_ATOM_V1,
    POSTERIOR_NMR_PHYSICS_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
    BIOEMU_LATENT_NMR_V1,
    "legacy_cs_fit",
    "target_conditioned_cs_fit",
    "target_conditioned_ccc_fit",
}
BASE_CANDIDATE_FEATURE_DIM = 3
POSTERIOR_EVIDENCE_FEATURE_COUNT = 3 + len(CHEMICAL_SHIFT_FAMILIES)
POSTERIOR_EVIDENCE_LOG_LIKELIHOOD_OFFSET = 1
POSTERIOR_V2_CCC_FEATURE_COUNT = 4


@dataclass(slots=True)
class PreparedObservableChannel:
    """One prepared observable channel for a student-training example."""

    values: np.ndarray
    mask: np.ndarray
    target_values: np.ndarray
    target_sigmas: np.ndarray
    transform: str
    target_ids: list[str] = field(default_factory=list)
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None


@dataclass(slots=True)
class PreparedTeacherExample:
    """One materialized teacher-training example."""

    entity_uid: str
    split: str
    sequence_tokens: list[int]
    example_features: np.ndarray
    candidate_source_indices: np.ndarray
    candidate_numeric_features: np.ndarray
    teacher_weights: np.ndarray
    channels: dict[str, PreparedObservableChannel] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    candidate_ids: list[str] = field(default_factory=list)


def load_materialized_examples(
    data_root: str | Path,
    integrated_root: str | Path,
    config: StudentTrainingConfig,
) -> tuple[list[PreparedTeacherExample], list[PreparedTeacherExample], dict[str, int]]:
    """Load materialized teacher examples for student training."""
    data_root_path = Path(data_root)
    _ = Path(integrated_root)
    repo_root = data_root_path.parent

    registry = IntegratedDataRegistry.from_data_root(data_root_path)
    teacher_examples = registry.load_teacher_examples()
    eligible = teacher_examples.loc[
        teacher_examples["teacher_ready"]
        & teacher_examples["pool_ready"]
        & teacher_examples["observable_ready"]
        & teacher_examples["target_ready"]
    ].copy()

    train_frame = eligible.loc[eligible["split"].isin(config.train_splits)].copy()
    val_frame = eligible.loc[eligible["split"].isin(config.val_splits)].copy()
    if config.max_train_examples is not None:
        train_frame = train_frame.head(config.max_train_examples)
    if config.max_val_examples is not None:
        val_frame = val_frame.head(config.max_val_examples)

    source_vocab = build_source_vocab(
        list(train_frame.to_dict(orient="records"))
        + list(val_frame.to_dict(orient="records")),
        data_root=data_root_path,
        repo_root=repo_root,
    )
    train_examples = [
        prepare_teacher_example(row, source_vocab, data_root_path, repo_root, config)
        for row in train_frame.to_dict(orient="records")
    ]
    val_examples = [
        prepare_teacher_example(row, source_vocab, data_root_path, repo_root, config)
        for row in val_frame.to_dict(orient="records")
    ]
    return train_examples, val_examples, source_vocab


def build_source_vocab(
    rows: list[dict[str, object]],
    data_root: Path,
    repo_root: Path,
) -> dict[str, int]:
    """Build one stable candidate-source vocabulary from materialized pools."""
    labels: set[str] = set()
    for row in rows:
        pool_path = resolve_existing_path(
            row.get("candidate_pool_path"),
            data_root=data_root,
            repo_root=repo_root,
        )
        if pool_path is None or not pool_path.exists():
            continue
        pool = CandidatePool.from_jsonl(pool_path)
        labels.update(record.source for record in pool.records)
    return {label: index for index, label in enumerate(sorted(labels))}


def prepare_teacher_example(
    row: dict[str, object],
    source_vocab: dict[str, int],
    data_root: Path,
    repo_root: Path,
    config: StudentTrainingConfig | None = None,
) -> PreparedTeacherExample:
    """Prepare one materialized teacher example for student training."""
    config = config or StudentTrainingConfig()
    task_spec = resolve_training_task(config.training_task, config.training_input_mode)
    validate_training_input_mode(task_spec.input_feature_mode)
    pool_path = resolve_existing_path(
        row.get("candidate_pool_path"),
        data_root=data_root,
        repo_root=repo_root,
    )
    observables_path = resolve_existing_path(
        row.get("observable_bundle_path"),
        data_root=data_root,
        repo_root=repo_root,
    )
    target_path = resolve_existing_path(
        row.get("target_bundle_path"),
        data_root=data_root,
        repo_root=repo_root,
    )
    solution_path = resolve_existing_path(
        row.get("teacher_solution_path"),
        data_root=data_root,
        repo_root=repo_root,
    )
    if not all(
        path is not None and path.exists()
        for path in [pool_path, observables_path, target_path, solution_path]
    ):
        raise FileNotFoundError(
            f"Missing materialized artifacts for {row.get('entity_uid')}"
        )

    pool = CandidatePool.from_jsonl(pool_path)
    observables = ObservableBundle.from_npz(observables_path)
    bundle = NMRTargetBundle.from_json(target_path)
    solution = WeightSolution.from_json(solution_path)
    if len(solution.weights) != len(pool.records):
        raise ValueError(
            f"Teacher weight count does not match the pool size for {row['entity_uid']}."
        )

    sequence_tokens = tokenize_sequence(row.get("sequence"))
    sequence_length = max(len(sequence_tokens), 1)
    example_features = np.asarray(
        [
            np.log1p(sequence_length),
            np.log1p(int(row.get("candidate_source_count") or len(pool.records))),
            np.log1p(int(row.get("chemical_shift_count") or 0)),
            np.log1p(int(row.get("j_coupling_count") or 0)),
            np.log1p(int(row.get("noe_count") or 0)),
        ],
        dtype=np.float32,
    )

    candidate_source_indices = np.asarray(
        [source_vocab[record.source] for record in pool.records],
        dtype=np.int64,
    )
    channels = build_prepared_channels(bundle=bundle, observables=observables)
    base_candidate_features = np.asarray(
        [
            [
                float(record.metadata.get("num_residues", len(record.residue_keys)))
                / float(sequence_length),
                float(bool(record.is_protonated)),
                float(bool(record.chemical_shift_path)),
            ]
            for record in pool.records
        ],
        dtype=np.float32,
    )
    feature_parts = [
        base_candidate_features,
        candidate_observable_fit_features(
            channels=channels,
            candidate_count=len(pool.records),
            input_mode=task_spec.input_feature_mode,
        ),
    ]
    if config.enable_residue_geometry_features:
        feature_parts.append(
            candidate_geometry_summary_features(
                pool=pool,
                sequence_length=sequence_length,
            )
        )
    candidate_numeric_features = np.concatenate(feature_parts, axis=1).astype(
        np.float32
    )
    return PreparedTeacherExample(
        entity_uid=str(row["entity_uid"]),
        split=str(row["split"]),
        sequence_tokens=sequence_tokens,
        example_features=example_features,
        candidate_source_indices=candidate_source_indices,
        candidate_numeric_features=candidate_numeric_features,
        teacher_weights=solution.weights.astype(np.float32),
        channels=channels,
        metadata={
            "bmrb_id": str(row["bmrb_id"]),
            "density_support": str(row.get("density_support") or "discrete_support"),
            "density_bridge_mode": str(
                row.get("density_bridge_mode") or "hybrid_bridge"
            ),
        },
        candidate_ids=list(pool.candidate_ids),
    )


def tokenize_sequence(sequence_value: object) -> list[int]:
    """Tokenize one hyphen-delimited amino-acid sequence."""
    if sequence_value is None:
        return [0]
    residues = [token.strip().upper() for token in str(sequence_value).split("-")]
    tokens = [AA3_VOCAB.get(residue, 0) for residue in residues if residue]
    return tokens or [0]


def build_prepared_channels(
    bundle: NMRTargetBundle,
    observables: ObservableBundle,
) -> dict[str, PreparedObservableChannel]:
    """Align observable matrices with bundle-level experimental targets."""
    channels: dict[str, PreparedObservableChannel] = {}
    target_lookup = {
        "chemical_shifts": (
            bundle.chemical_shifts,
            chemical_shift_sigma,
        ),
        "j_couplings": (
            bundle.j_couplings,
            lambda target: float(
                0.5 if target.uncertainty is None else target.uncertainty
            ),
        ),
        "noe_restraints": (
            bundle.noe_restraints,
            lambda target: float(target.uncertainty),
        ),
    }
    for channel_name, matrix in observables.iter_channels():
        targets, sigma_fn = target_lookup[channel_name]
        target_values = np.asarray(
            [
                float(target.value if hasattr(target, "value") else target.target_value)
                for target in targets
            ],
            dtype=np.float32,
        )
        target_sigmas = np.asarray(
            [sigma_fn(target) for target in targets],
            dtype=np.float32,
        )
        target_ids = list(matrix.target_ids)
        values = matrix.values.astype(np.float32)
        mask = matrix.mask.astype(bool)
        if channel_name == "chemical_shifts":
            keep = chemical_shift_training_row_mask(
                target_ids=target_ids,
                target_values=target_values,
            )
            values = values[keep]
            mask = mask[keep]
            target_values = target_values[keep]
            target_sigmas = target_sigmas[keep]
            target_ids = [
                target_id for target_id, keep_row in zip(target_ids, keep) if keep_row
            ]
        channels[channel_name] = PreparedObservableChannel(
            values=values,
            mask=mask,
            target_values=target_values,
            target_sigmas=target_sigmas,
            transform=matrix.transform,
            target_ids=target_ids,
            lower_bounds=(
                np.asarray(
                    [
                        np.nan if target.lower_bound is None else target.lower_bound
                        for target in targets
                    ],
                    dtype=np.float32,
                )
                if channel_name == "noe_restraints"
                else None
            ),
            upper_bounds=(
                np.asarray(
                    [
                        np.nan if target.upper_bound is None else target.upper_bound
                        for target in targets
                    ],
                    dtype=np.float32,
                )
                if channel_name == "noe_restraints"
                else None
            ),
        )
    return channels


def chemical_shift_training_row_mask(
    *,
    target_ids: list[str],
    target_values: np.ndarray,
) -> np.ndarray:
    """Return rows that are canonical protein chemical-shift targets.

    The student posterior should learn protein backbone/CB chemical-shift
    patterns.  BMRB loops can contain nonstandard components or non-backbone
    carbons with atom id ``C``; those look like impossible C' outliers and
    destabilize CCC training, so we filter them at load time.
    """

    rows = [
        chemical_shift_target_is_training_usable(target_id, float(value))
        for target_id, value in zip(target_ids, target_values, strict=False)
    ]
    return np.asarray(rows, dtype=bool)


def chemical_shift_target_is_training_usable(target_id: str, value: float) -> bool:
    """Return whether one chemical-shift row is safe for training."""

    parsed = parse_chemical_shift_target_id(target_id)
    family = parsed["atom_family"]
    residue_name = parsed["residue_name"]
    if family is None or residue_name not in AA3_VOCAB:
        return False
    if family == "HN" and residue_name == "PRO":
        return False
    bounds = CHEMICAL_SHIFT_PPM_RANGES.get(family)
    if bounds is None or not np.isfinite(value):
        return False
    lower, upper = bounds
    return lower <= float(value) <= upper


def parse_chemical_shift_target_id(target_id: str) -> dict[str, str | int | None]:
    """Parse ``cs:{chain}:{seq_id}:{comp_id}:{atom_id}`` target ids."""

    parts = str(target_id).split(":")
    if len(parts) != 5 or parts[0] != "cs":
        return {
            "chain_id": "_",
            "residue_index": 0,
            "residue_name": "",
            "atom_name": "",
            "atom_family": None,
        }
    try:
        residue_index = int(parts[2])
    except ValueError:
        residue_index = 0
    atom_name = parts[4].upper()
    return {
        "chain_id": parts[1],
        "residue_index": max(residue_index, 0),
        "residue_name": parts[3].upper(),
        "atom_name": atom_name,
        "atom_family": CHEMICAL_SHIFT_ATOM_FAMILY_MAP.get(atom_name),
    }


def candidate_geometry_summary_features(
    pool: CandidatePool,
    sequence_length: int,
) -> np.ndarray:
    """Return lightweight candidate geometry summary features.

    These are deterministic fallbacks for the fast physics-aware path.  Rich
    per-residue geometry assets can extend this contract later without changing
    the candidate-level model input shape for missing-coordinate examples.
    """
    denominator = max(float(sequence_length), 1.0)
    rows: list[list[float]] = []
    for record in pool.records:
        metadata = record.metadata or {}
        residue_count = float(metadata.get("num_residues", len(record.residue_keys)))
        residue_coverage = len(record.residue_keys) / denominator
        rows.append(
            [
                residue_count / denominator,
                float(residue_coverage),
                _safe_float(metadata.get("radius_gyration")) / denominator,
                _safe_float(metadata.get("end_to_end_distance")) / denominator,
                _safe_float(metadata.get("mean_contact_degree")) / 20.0,
                float(bool(record.structure_path)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _safe_float(value: object) -> float:
    """Return a finite float or zero for missing geometry metadata."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return parsed


def candidate_observable_fit_features(
    channels: dict[str, PreparedObservableChannel],
    candidate_count: int,
    input_mode: str = "legacy_cs_fit",
) -> np.ndarray:
    """Return candidate-level observable fit summaries for model input."""
    validate_training_input_mode(input_mode)
    if input_mode == PRIOR_SEQUENCE_ONLY:
        return np.zeros((candidate_count, 0), dtype=np.float32)
    if input_mode == POSTERIOR_NMR_CONDITIONED:
        return candidate_evidence_likelihood_features(
            channels=channels,
            candidate_count=candidate_count,
        )
    if input_mode in {
        POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
        POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    }:
        # Moment-diffusion keeps target values out of candidate-level features.
        # Evidence enters through the model's masked evidence encoder so
        # masked-holdout rows cannot leak through precomputed candidate scores.
        return np.zeros((candidate_count, 0), dtype=np.float32)
    if input_mode in {
        POSTERIOR_NMR_CONDITIONED_V2,
        POSTERIOR_NMR_RESIDUE_ATOM_V1,
        POSTERIOR_NMR_PHYSICS_V1,
    }:
        return candidate_posterior_v2_features(
            channels=channels,
            candidate_count=candidate_count,
        )
    base_feature_count = 4
    family_feature_count = len(CHEMICAL_SHIFT_FAMILIES) * 5
    ccc_feature_count = 4
    feature_count = (
        base_feature_count
        if input_mode == "legacy_cs_fit"
        else base_feature_count
        + family_feature_count
        + (ccc_feature_count if input_mode == "target_conditioned_ccc_fit" else 0)
    )
    features = np.zeros((candidate_count, feature_count), dtype=np.float32)
    channel = channels.get("chemical_shifts")
    if channel is None or candidate_count == 0:
        return features
    values = channel.values.astype(np.float64, copy=False)
    mask = channel.mask.astype(bool, copy=False)
    targets = channel.target_values.astype(np.float64, copy=False)
    sigmas = np.clip(channel.target_sigmas.astype(np.float64, copy=False), 1e-6, None)
    family_labels = np.asarray(
        [_atom_family_from_target_id(target_id) for target_id in channel.target_ids],
        dtype=object,
    )
    for candidate_index in range(candidate_count):
        valid = mask[:, candidate_index]
        if not np.any(valid):
            continue
        residual_ppm = np.abs(values[valid, candidate_index] - targets[valid])
        residual_z = residual_ppm / sigmas[valid]
        features[candidate_index, :4] = np.asarray(
            [
                float(np.mean(valid)),
                float(np.log1p(np.mean(residual_ppm))),
                float(np.log1p(np.mean(residual_z))),
                float(np.log1p(np.sqrt(np.mean(np.square(residual_z))))),
            ],
            dtype=np.float32,
        )
        if input_mode in {"target_conditioned_cs_fit", "target_conditioned_ccc_fit"}:
            _fill_family_fit_features(
                features=features,
                feature_offset=base_feature_count,
                candidate_index=candidate_index,
                values=values[:, candidate_index],
                valid=valid,
                targets=targets,
                sigmas=sigmas,
                family_labels=family_labels,
            )
    if input_mode in {"target_conditioned_cs_fit", "target_conditioned_ccc_fit"}:
        _append_within_example_rank_features(features, base_feature_count)
    if input_mode == "target_conditioned_ccc_fit":
        family_ccc_scores = candidate_family_ccc_scores(
            values=values,
            mask=mask,
            targets=targets,
            family_labels=family_labels,
        )
        ccc_offset = base_feature_count + family_feature_count
        _fill_ccc_oracle_features(
            features=features,
            feature_offset=ccc_offset,
            family_ccc_scores=family_ccc_scores,
        )
    return features


def candidate_evidence_likelihood_features(
    channels: dict[str, PreparedObservableChannel],
    candidate_count: int,
) -> np.ndarray:
    """Return explicit NMR-evidence likelihood features for posterior inference.

    The public feature shape stays stable for existing CS-only models.  The
    aggregate likelihood columns now behave as a lightweight product-of-experts:
    dense chemical shifts dominate, while sparse J/NOE channels add small
    reliability-weighted evidence when they are present.
    """
    family_count = len(CHEMICAL_SHIFT_FAMILIES)
    feature_count = 3 + family_count
    features = np.zeros((candidate_count, feature_count), dtype=np.float32)
    channel = channels.get("chemical_shifts")
    if candidate_count == 0:
        return features
    auxiliary_scores = _multi_observable_likelihood_scores(
        channels,
        candidate_count,
        include_chemical_shifts=False,
    )
    if np.any(np.isfinite(auxiliary_scores)):
        features[:, 1] = np.clip(auxiliary_scores, -100.0, 10.0).astype(np.float32)
        features[:, 2] = features[:, 1]
    if channel is None:
        return features
    values = channel.values.astype(np.float64, copy=False)
    mask = channel.mask.astype(bool, copy=False)
    targets = channel.target_values.astype(np.float64, copy=False)
    sigmas = np.clip(channel.target_sigmas.astype(np.float64, copy=False), 1e-6, None)
    family_labels = np.asarray(
        [_atom_family_from_target_id(target_id) for target_id in channel.target_ids],
        dtype=object,
    )
    for candidate_index in range(candidate_count):
        valid = mask[:, candidate_index]
        if not np.any(valid):
            features[candidate_index, 1] = -100.0
            features[candidate_index, 2] = -100.0
            features[candidate_index, 3:] = -100.0
            continue
        log_likelihood = _gaussian_log_likelihood(
            values[valid, candidate_index],
            targets[valid],
            sigmas[valid],
        )
        mean_ll = float(np.mean(log_likelihood))
        combined_ll = mean_ll + float(auxiliary_scores[candidate_index])
        features[candidate_index, 0:3] = np.asarray(
            [
                float(np.mean(valid)),
                float(np.clip(combined_ll, -100.0, 10.0)),
                float(np.clip(combined_ll, -100.0, 10.0)),
            ],
            dtype=np.float32,
        )
        valid_labels = family_labels[valid]
        for family_index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
            family_mask = valid_labels == family_name
            features[candidate_index, 3 + family_index] = (
                float(np.clip(np.mean(log_likelihood[family_mask]), -100.0, 10.0))
                if np.any(family_mask)
                else -100.0
            )
    return features


def _multi_observable_likelihood_scores(
    channels: dict[str, PreparedObservableChannel],
    candidate_count: int,
    include_chemical_shifts: bool = True,
) -> np.ndarray:
    """Return a small product-of-experts score from CS/J/NOE channels."""
    weighted_scores: list[np.ndarray] = []
    weights: list[float] = []
    channel_weights = [
        ("j_couplings", 0.10),
        ("noe_restraints", 0.10),
    ]
    if include_chemical_shifts:
        channel_weights.insert(0, ("chemical_shifts", 1.0))
    for channel_name, base_weight in channel_weights:
        channel = channels.get(channel_name)
        if channel is None:
            continue
        scores = _channel_likelihood_scores(channel_name, channel, candidate_count)
        if scores is None:
            continue
        reliability = base_weight * _channel_reliability(channel)
        if reliability <= 0.0:
            continue
        weighted_scores.append(_standardize_candidate_scores(scores))
        weights.append(reliability)
    if not weighted_scores:
        return np.zeros(candidate_count, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float64)
    stacked = np.vstack(weighted_scores)
    combined = (stacked * weight_array[:, None]).sum(axis=0) / np.clip(
        weight_array.sum(),
        1e-8,
        None,
    )
    return combined.astype(np.float32)


def _channel_likelihood_scores(
    channel_name: str,
    channel: PreparedObservableChannel,
    candidate_count: int,
) -> np.ndarray | None:
    """Return higher-is-better candidate scores for one prepared channel."""
    values = channel.values.astype(np.float64, copy=False)
    mask = channel.mask.astype(bool, copy=False)
    if values.ndim != 2 or values.shape[1] != candidate_count or not np.any(mask):
        return None
    if channel_name == "noe_restraints":
        distances = np.power(np.clip(values, 1e-12, None), -1.0 / 6.0)
        targets = channel.target_values.astype(np.float64, copy=False)[:, None]
        sigmas = np.clip(
            channel.target_sigmas.astype(np.float64, copy=False), 1e-6, None
        )[:, None]
        violation = np.zeros_like(distances)
        if channel.lower_bounds is not None:
            lower = channel.lower_bounds.astype(np.float64, copy=False)[:, None]
            violation += np.where(
                np.isfinite(lower),
                np.maximum(lower - distances, 0.0),
                0.0,
            )
        if channel.upper_bounds is not None:
            upper = channel.upper_bounds.astype(np.float64, copy=False)[:, None]
            violation += np.where(
                np.isfinite(upper),
                np.maximum(distances - upper, 0.0),
                0.0,
            )
        residual = np.where(
            violation > 0.0,
            violation / sigmas,
            np.abs(distances - targets) / sigmas,
        )
        loss = residual
    else:
        targets = channel.target_values.astype(np.float64, copy=False)[:, None]
        sigmas = np.clip(
            channel.target_sigmas.astype(np.float64, copy=False), 1e-6, None
        )[:, None]
        residual = (values - targets) / sigmas
        if channel_name == "j_couplings":
            absolute = np.abs(residual)
            delta = 4.0
            quadratic = np.minimum(absolute, delta)
            loss = 0.5 * np.square(quadratic) + delta * (absolute - quadratic)
        else:
            loss = np.square(residual)
    counts = mask.sum(axis=0).astype(np.float64)
    scores = -np.sum(loss * mask, axis=0) / np.clip(counts, 1.0, None)
    scores[counts <= 0] = -100.0
    return scores


def _channel_reliability(channel: PreparedObservableChannel) -> float:
    """Return a conservative coverage reliability for sparse channels."""
    mask = channel.mask.astype(bool, copy=False)
    if mask.size == 0:
        return 0.0
    target_coverage = float(np.mean(np.any(mask, axis=1)))
    candidate_coverage = float(np.mean(np.any(mask, axis=0)))
    measurement_factor = min(np.log1p(mask.shape[0]) / np.log(32.0), 1.0)
    return max(target_coverage * candidate_coverage * measurement_factor, 0.0)


def _standardize_candidate_scores(scores: np.ndarray) -> np.ndarray:
    """Return z-scored finite candidate scores with safe missing fallbacks."""
    parsed = np.nan_to_num(scores.astype(np.float64), nan=-100.0)
    finite = np.isfinite(parsed)
    if np.count_nonzero(finite) < 2:
        return np.zeros(parsed.shape, dtype=np.float64)
    values = parsed[finite]
    scaled = parsed.copy()
    scaled[finite] = (values - np.mean(values)) / max(np.std(values), 1e-6)
    scaled[~finite] = -5.0
    return np.clip(scaled, -5.0, 5.0)


def candidate_posterior_v2_features(
    channels: dict[str, PreparedObservableChannel],
    candidate_count: int,
) -> np.ndarray:
    """Return explicit evidence and CCC-geometry features for posterior v2."""
    evidence = candidate_evidence_likelihood_features(
        channels=channels,
        candidate_count=candidate_count,
    )
    ccc_features = np.zeros(
        (candidate_count, POSTERIOR_V2_CCC_FEATURE_COUNT),
        dtype=np.float32,
    )
    channel = channels.get("chemical_shifts")
    if channel is not None and candidate_count > 0:
        family_labels = np.asarray(
            [
                _atom_family_from_target_id(target_id)
                for target_id in channel.target_ids
            ],
            dtype=object,
        )
        family_ccc_scores = candidate_family_ccc_scores(
            values=channel.values.astype(np.float64, copy=False),
            mask=channel.mask.astype(bool, copy=False),
            targets=channel.target_values.astype(np.float64, copy=False),
            family_labels=family_labels,
        )
        _fill_ccc_oracle_features(
            features=ccc_features,
            feature_offset=0,
            family_ccc_scores=family_ccc_scores,
        )
    return np.concatenate([evidence, ccc_features], axis=1).astype(np.float32)


def evidence_log_likelihood_feature_index(
    candidate_feature_dim: int,
    input_mode: str,
) -> int | None:
    """Return the full candidate-feature index for explicit evidence likelihood."""
    if input_mode not in {
        POSTERIOR_NMR_CONDITIONED,
        POSTERIOR_NMR_CONDITIONED_V2,
        POSTERIOR_NMR_RESIDUE_ATOM_V1,
        POSTERIOR_NMR_PHYSICS_V1,
        POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
        POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    }:
        return None
    if input_mode in {
        POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
        POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    }:
        return None
    if candidate_feature_dim < BASE_CANDIDATE_FEATURE_DIM + 2:
        return None
    return BASE_CANDIDATE_FEATURE_DIM + POSTERIOR_EVIDENCE_LOG_LIKELIHOOD_OFFSET


def ccc_proxy_feature_index(
    candidate_feature_dim: int,
    input_mode: str,
) -> int | None:
    """Return the full candidate-feature index for CCC proxy scores."""
    if input_mode == "target_conditioned_ccc_fit":
        ccc_offset = 4 + len(CHEMICAL_SHIFT_FAMILIES) * 5
        index = BASE_CANDIDATE_FEATURE_DIM + ccc_offset
    elif input_mode in {
        POSTERIOR_NMR_CONDITIONED_V2,
        POSTERIOR_NMR_RESIDUE_ATOM_V1,
        POSTERIOR_NMR_PHYSICS_V1,
    }:
        index = BASE_CANDIDATE_FEATURE_DIM + POSTERIOR_EVIDENCE_FEATURE_COUNT
    else:
        return None
    return index if 0 <= index < candidate_feature_dim else None


def candidate_family_ccc_scores(
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    family_labels: np.ndarray,
) -> np.ndarray:
    """Return candidate-by-family CCC scores from chemical-shift matrices."""
    candidate_count = int(values.shape[1]) if values.ndim == 2 else 0
    scores = np.full(
        (candidate_count, len(CHEMICAL_SHIFT_FAMILIES)),
        np.nan,
        dtype=np.float32,
    )
    for candidate_index in range(candidate_count):
        valid = mask[:, candidate_index]
        for family_index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
            family_valid = valid & (family_labels == family_name)
            if np.count_nonzero(family_valid) < 2:
                continue
            score = _safe_ccc_np(
                values[family_valid, candidate_index],
                targets[family_valid],
            )
            if np.isfinite(score):
                scores[candidate_index, family_index] = float(score)
    return scores


def validate_training_input_mode(input_mode: str) -> None:
    """Validate one training-input feature mode."""
    if input_mode not in TRAINING_INPUT_MODES:
        allowed = ", ".join(sorted(TRAINING_INPUT_MODES))
        raise ValueError(
            f"Unsupported training_input_mode={input_mode!r}; use {allowed}."
        )


def _fill_family_fit_features(
    *,
    features: np.ndarray,
    feature_offset: int,
    candidate_index: int,
    values: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    sigmas: np.ndarray,
    family_labels: np.ndarray,
) -> None:
    """Fill per-atom-family observable-fit summaries for one candidate."""
    for family_index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
        family_valid = valid & (family_labels == family_name)
        if not np.any(family_valid):
            continue
        signed_residual_ppm = values[family_valid] - targets[family_valid]
        abs_residual_z = np.abs(signed_residual_ppm / sigmas[family_valid])
        offset = feature_offset + family_index * 5
        features[candidate_index, offset : offset + 4] = np.asarray(
            [
                float(np.mean(family_valid)),
                float(np.log1p(np.mean(abs_residual_z))),
                float(np.log1p(np.sqrt(np.mean(np.square(abs_residual_z))))),
                float(np.tanh(np.mean(signed_residual_ppm))),
            ],
            dtype=np.float32,
        )


def _append_within_example_rank_features(
    features: np.ndarray,
    feature_offset: int,
) -> None:
    """Add per-family within-example RMSE rank percentiles in-place."""
    candidate_count = features.shape[0]
    if candidate_count <= 1:
        return
    for family_index, _family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
        family_offset = feature_offset + family_index * 5
        rmse_column = family_offset + 2
        rank_column = family_offset + 4
        values = features[:, rmse_column].astype(np.float64, copy=False)
        missing = values <= 0.0
        if np.all(missing):
            continue
        sortable = values.copy()
        sortable[missing] = np.nanmax(sortable[~missing]) + 1.0
        order = np.argsort(sortable, kind="stable")
        ranks = np.empty(candidate_count, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, candidate_count, dtype=np.float32)
        ranks[missing] = 1.0
        features[:, rank_column] = ranks


def _fill_ccc_oracle_features(
    *,
    features: np.ndarray,
    feature_offset: int,
    family_ccc_scores: np.ndarray,
) -> None:
    """Fill CCC proxy and rank features used by CCC-oracle distillation."""
    if family_ccc_scores.size == 0:
        return
    family_proxy = np.nanmean(family_ccc_scores, axis=1)
    family_proxy = np.where(np.isfinite(family_proxy), family_proxy, -1.0)
    features[:, feature_offset] = family_proxy.astype(np.float32)
    features[:, feature_offset + 1] = _descending_rank_percentiles(family_proxy)

    family_to_index = {
        family_name: index for index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES)
    }
    for output_offset, family_name in [(2, "C'"), (3, "CA")]:
        family_scores = family_ccc_scores[:, family_to_index[family_name]]
        family_scores = np.where(np.isfinite(family_scores), family_scores, -1.0)
        features[:, feature_offset + output_offset] = _descending_rank_percentiles(
            family_scores
        )


def _descending_rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Return rank percentiles where 0 is best and 1 is worst."""
    count = int(values.size)
    if count <= 1:
        return np.zeros(count, dtype=np.float32)
    order = np.argsort(-values.astype(np.float64), kind="stable")
    ranks = np.empty(count, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return ranks


def _safe_ccc_np(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Lin's CCC with a finite fallback."""
    if predictions.size < 2:
        return float("nan")
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_var = float(np.mean(np.square(predictions - pred_mean)))
    target_var = float(np.mean(np.square(targets - target_mean)))
    covariance = float(np.mean((predictions - pred_mean) * (targets - target_mean)))
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    if denominator <= 1e-12:
        return float("nan")
    return float((2.0 * covariance) / denominator)


def _gaussian_log_likelihood(
    predictions: np.ndarray,
    targets: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Return per-target Gaussian log-likelihood values."""
    residual = (predictions - targets) / np.clip(sigmas, 1e-6, None)
    return (
        -0.5 * np.square(residual)
        - np.log(np.clip(sigmas, 1e-6, None))
        - 0.5 * np.log(2.0 * np.pi)
    )


def chemical_shift_sigma(target: object) -> float:
    """Return an atom-family-aware chemical-shift uncertainty floor."""
    raw_uncertainty = getattr(target, "uncertainty", None)
    sigma = 1.0 if raw_uncertainty is None else float(raw_uncertainty)
    family = _atom_family_from_target_id(str(getattr(target, "target_id", "")))
    floor = CHEMICAL_SHIFT_SIGMA_FLOORS.get(family, 0.20)
    return float(max(sigma, floor))


def _atom_family_from_target_id(target_id: str) -> str | None:
    """Return one canonical atom-family label from a chemical-shift target ID."""
    if not target_id.startswith("cs:"):
        return None
    return CHEMICAL_SHIFT_ATOM_FAMILY_MAP.get(target_id.rsplit(":", 1)[-1].upper())
