"""Family-conditioned tabular chemical-shift predictor.

This module provides the first fast-predictor training path for the
UCBShift-parity contract.  It is deliberately lightweight: a per-family ridge
readout over residue, atom-family, local-geometry, and BioEmu-context features.
The exported prediction table is directly consumable by
``cs_prediction_parity`` and includes uncertainty/OOD/reward-safety metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from atypemu.training.cs_prediction_parity import (
    ChemicalShiftParityConfig,
    _boolean_series,
    canonical_atom_family,
    evaluate_chemical_shift_prediction_parity,
    load_table,
    normalize_target_frame,
    write_parity_outputs,
)
from atypemu.training.chemical_shift_shared_q import (
    ChemicalShiftSharedQConfig,
    ChemicalShiftSharedQSweepConfig,
    infer_chemical_shift_shared_q_posterior,
    sweep_chemical_shift_shared_q_posterior,
)


CS_PREDICTOR_BUNDLE_KIND = "atypemu_family_conditioned_tabular_cs_predictor_v1"
CS_PREDICTOR_ARCHITECTURE_ID = (
    "family_conditioned_local_geometry_residue_bioemu_context_"
    "ucbshift2_local_manifold_bounded_interactions_uncertainty_ood_"
    "hierarchical_family_shrinkage_affine_family_calibration_"
    "route_conditioned_support_context_v5"
)
LOCAL_GEOMETRY_TOKENS = (
    "local_environment",
    "geometry",
    "distance",
    "hbond",
    "ring_current",
    "solvent",
    "peptide_plane",
    "contact",
    "neighborhood",
)
CS_LOCAL_MANIFOLD_TOKENS = (
    "ucbshift2",
    "local_manifold",
    "manifold",
    "amide",
    "carbonyl",
    "cprime",
    "c_prime",
    "peptide_plane",
    "backbone_torsion",
    "phi",
    "psi",
    "omega",
    "rama",
    "neighbor_residue",
    "residue_neighbor",
)
HN_N_CPRIME_MANIFOLD_TOKENS = (
    "hn",
    "amide_h",
    "amide_proton",
    "backbone_n",
    "nitrogen",
    "carbonyl",
    "cprime",
    "c_prime",
)
BIOEMU_CONTEXT_TOKENS = (
    "bioemu",
    "latent",
    "score",
    "log_prob",
    "logp",
    "prior",
)
REFERENCE_FEATURE_TOKENS = (
    "random_coil",
    "reference",
    "baseline",
    "offset",
)
ROUTE_CONTEXT_TOKENS = (
    "route",
    "target_coverage",
    "multi_patch",
    "target_direction",
    "support_generation",
    "support_distribution",
    "cs_reweighting",
    "posterior_target",
    "required_generated_family",
)
RESERVED_FEATURE_COLUMNS = {
    "entity_uid",
    "entry_uid",
    "bmrb_uid",
    "target_id",
    "target_uid",
    "measurement_uid",
    "measurement_kind",
    "target_value",
    "observed_value",
    "observed_shift",
    "experimental_shift",
    "chemical_shift",
    "value",
    "prediction",
    "predicted_value",
    "predicted_shift",
    "chemical_shift_prediction",
    "split",
    "dataset_split",
    "target_sigma",
    "uncertainty",
    "sigma",
    "experimental_uncertainty",
    "support_id",
    "conformer_id",
    "candidate_id",
    "sample_id",
    "structure_id",
    "pdb_id",
    "support_valid",
    "conformer_valid",
    "decoder_valid",
    "generated_support_valid",
    "valid",
    "validity_passed",
    "checkpoint_sha256",
    "checkpoint_path",
    "sample_manifest_sha256",
    "sample_manifest_path",
    "manifest_sha256",
    "manifest_path",
    "pdb_path",
    "structure_path",
    "structure_file",
}
PREDICTOR_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "support_id": (
        "support_id",
        "conformer_id",
        "candidate_id",
        "sample_id",
        "sample_index",
        "dense_sample_index",
        "bioemu_generated_sample_index",
        "bioemu_generated_dense_sample_index",
        "structure_id",
        "pdb_id",
    ),
    "sample_index": (
        "sample_index",
        "generated_sample_index",
        "support_sample_index",
        "bioemu_sample_index",
        "bioemu_generated_sample_index",
    ),
    "dense_sample_index": (
        "dense_sample_index",
        "bioemu_generated_dense_sample_index",
    ),
    "candidate_index": (
        "candidate_index",
        "bioemu_generated_candidate_index",
    ),
    "prior_log_prob": (
        "prior_log_prob",
        "log_prior",
        "bioemu_score_log_prob",
        "bioemu_log_prob",
        "log_prob",
    ),
    "support_valid": (
        "support_valid",
        "conformer_valid",
        "decoder_valid",
        "generated_support_valid",
        "valid",
        "validity_passed",
    ),
    "checkpoint_sha256": ("checkpoint_sha256",),
    "checkpoint_path": ("checkpoint_path", "model_checkpoint_path"),
    "sample_manifest_sha256": ("sample_manifest_sha256", "manifest_sha256"),
    "sample_manifest_path": ("sample_manifest_path", "manifest_path"),
    "structure_path": ("structure_path", "pdb_path", "structure_file"),
}
PREDICTOR_OUTPUT_METADATA_COLUMNS: tuple[str, ...] = (
    "support_id",
    "sample_index",
    "dense_sample_index",
    "candidate_index",
    "prior_log_prob",
    "support_valid",
    "checkpoint_sha256",
    "checkpoint_path",
    "sample_manifest_sha256",
    "sample_manifest_path",
    "structure_path",
)


@dataclass(frozen=True, slots=True)
class ChemicalShiftPredictorConfig:
    """Configuration for the tabular CS predictor."""

    ridge_alpha: float = 1.0e-6
    gate_split: str = "val"
    train_splits: tuple[str, ...] = ("train",)
    min_family_rows: int = 2
    uncertainty_floor: float = 1.0e-3
    ood_score_threshold: float = 1.0
    reward_uncertainty_scale: float = 4.0
    enable_pairwise_interactions: bool = True
    max_pairwise_interactions: int = 64
    family_shrinkage_strength: float = 4.0
    enable_family_affine_calibration: bool = True
    family_calibration_strength: float = 4.0
    prediction_label: str = "family_conditioned_tabular_cs_predictor"


@dataclass(frozen=True, slots=True)
class _EncodedFeatureSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_mean: tuple[float, ...]
    numeric_scale: tuple[float, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    interaction_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _RidgeFamilyModel:
    family: str
    coefficients: tuple[float, ...]
    residual_rmse: float
    ood_reference: float
    train_rows: int
    family_blend_weight: float = 1.0
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0
    calibration_blend_weight: float = 0.0
    calibration_rows: int = 0


@dataclass(frozen=True, slots=True)
class FamilyConditionedChemicalShiftPredictor:
    """Serializable family-conditioned CS predictor bundle."""

    feature_spec: _EncodedFeatureSpec
    family_models: dict[str, _RidgeFamilyModel]
    global_model: _RidgeFamilyModel
    config: ChemicalShiftPredictorConfig
    feature_roles: dict[str, bool]
    training_summary: dict[str, Any]

    def predict(self, examples: pd.DataFrame) -> pd.DataFrame:
        """Return parity-compatible predictions with reliability metadata."""

        frame = _normalize_feature_frame(examples)
        encoded = _encode_features(frame, self.feature_spec)
        families = frame["atom_family"].astype(str).to_numpy(dtype=object)
        predictions = np.zeros(len(frame), dtype=np.float64)
        uncertainty = np.zeros(len(frame), dtype=np.float64)
        ood_score = np.zeros(len(frame), dtype=np.float64)
        family_blend_weight = np.zeros(len(frame), dtype=np.float64)
        raw_predictions = np.zeros(len(frame), dtype=np.float64)
        calibration_slope = np.zeros(len(frame), dtype=np.float64)
        calibration_intercept = np.zeros(len(frame), dtype=np.float64)
        calibration_blend_weight = np.zeros(len(frame), dtype=np.float64)

        for row_index, family in enumerate(families.tolist()):
            model = self.family_models.get(str(family), self.global_model)
            beta = np.asarray(model.coefficients, dtype=np.float64)
            raw_prediction = float(encoded[row_index] @ beta)
            raw_predictions[row_index] = raw_prediction
            slope = float(model.calibration_slope)
            intercept = float(model.calibration_intercept)
            predictions[row_index] = slope * raw_prediction + intercept
            family_blend_weight[row_index] = float(model.family_blend_weight)
            calibration_slope[row_index] = slope
            calibration_intercept[row_index] = intercept
            calibration_blend_weight[row_index] = float(model.calibration_blend_weight)
            raw_ood = _ood_raw_score(encoded[row_index])
            reference = max(float(model.ood_reference), 1.0)
            score = raw_ood / max(2.0 * reference, 1.0e-12)
            ood_score[row_index] = score
            uncertainty[row_index] = max(
                float(model.residual_rmse) * (1.0 + 0.1 * score),
                float(self.config.uncertainty_floor),
            )

        uncertainty_threshold = _reward_uncertainty_threshold(
            self.family_models,
            self.global_model,
            self.config,
        )
        uncertainty_gate = uncertainty <= uncertainty_threshold
        ood_gate = ood_score <= float(self.config.ood_score_threshold)
        validity_gate = _support_validity_gate(frame)
        reliability_gate = uncertainty_gate & ood_gate & validity_gate
        reward = np.where(reliability_gate, 1.0 / (1.0 + uncertainty), 0.0)

        out = frame[_prediction_output_columns(frame)].copy()
        out["predicted_value"] = predictions
        out["prediction"] = predictions
        out["cs_predictor_raw_prediction"] = raw_predictions
        out["cs_predictor_calibration_slope"] = calibration_slope
        out["cs_predictor_calibration_intercept"] = calibration_intercept
        out["cs_predictor_calibration_blend_weight"] = calibration_blend_weight
        out["prediction_uncertainty"] = uncertainty
        out["prediction_ood_score"] = ood_score
        out["prediction_ood_flag"] = ~ood_gate
        out["prediction_reward"] = reward
        out["prediction_uncertainty_gate_passed"] = uncertainty_gate
        out["prediction_ood_gate_passed"] = ood_gate
        out["prediction_support_valid_gate_passed"] = validity_gate
        out["prediction_reliability_gate_passed"] = reliability_gate
        out["prediction_reward_safety_gate_passed"] = reliability_gate
        out["cs_predictor_family_blend_weight"] = family_blend_weight
        out["cs_predictor_architecture_id"] = CS_PREDICTOR_ARCHITECTURE_ID
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize the predictor bundle to JSON-compatible data."""

        return {
            "bundle_kind": CS_PREDICTOR_BUNDLE_KIND,
            "architecture_id": CS_PREDICTOR_ARCHITECTURE_ID,
            "config": {
                "ridge_alpha": float(self.config.ridge_alpha),
                "gate_split": str(self.config.gate_split),
                "train_splits": list(self.config.train_splits),
                "min_family_rows": int(self.config.min_family_rows),
                "uncertainty_floor": float(self.config.uncertainty_floor),
                "ood_score_threshold": float(self.config.ood_score_threshold),
                "reward_uncertainty_scale": float(
                    self.config.reward_uncertainty_scale
                ),
                "enable_pairwise_interactions": bool(
                    self.config.enable_pairwise_interactions
                ),
                "max_pairwise_interactions": int(
                    self.config.max_pairwise_interactions
                ),
                "family_shrinkage_strength": float(
                    self.config.family_shrinkage_strength
                ),
                "enable_family_affine_calibration": bool(
                    self.config.enable_family_affine_calibration
                ),
                "family_calibration_strength": float(
                    self.config.family_calibration_strength
                ),
                "prediction_label": str(self.config.prediction_label),
            },
            "feature_spec": {
                "numeric_columns": list(self.feature_spec.numeric_columns),
                "categorical_columns": list(self.feature_spec.categorical_columns),
                "numeric_mean": list(self.feature_spec.numeric_mean),
                "numeric_scale": list(self.feature_spec.numeric_scale),
                "categorical_levels": {
                    key: list(values)
                    for key, values in self.feature_spec.categorical_levels.items()
                },
                "interaction_pairs": [
                    list(pair) for pair in self.feature_spec.interaction_pairs
                ],
            },
            "family_models": {
                family: _family_model_to_dict(model)
                for family, model in self.family_models.items()
            },
            "global_model": _family_model_to_dict(self.global_model),
            "feature_roles": dict(self.feature_roles),
            "training_summary": dict(self.training_summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FamilyConditionedChemicalShiftPredictor":
        """Deserialize a saved predictor bundle."""

        if str(payload.get("bundle_kind")) != CS_PREDICTOR_BUNDLE_KIND:
            raise ValueError(f"expected bundle_kind={CS_PREDICTOR_BUNDLE_KIND}")
        config_payload = dict(payload.get("config", {}))
        config = ChemicalShiftPredictorConfig(
            ridge_alpha=float(config_payload.get("ridge_alpha", 1.0e-6)),
            gate_split=str(config_payload.get("gate_split", "val")),
            train_splits=tuple(config_payload.get("train_splits", ["train"])),
            min_family_rows=int(config_payload.get("min_family_rows", 2)),
            uncertainty_floor=float(config_payload.get("uncertainty_floor", 1.0e-3)),
            ood_score_threshold=float(config_payload.get("ood_score_threshold", 1.0)),
            reward_uncertainty_scale=float(
                config_payload.get("reward_uncertainty_scale", 4.0)
            ),
            enable_pairwise_interactions=bool(
                config_payload.get("enable_pairwise_interactions", True)
            ),
            max_pairwise_interactions=int(
                config_payload.get("max_pairwise_interactions", 64)
            ),
            family_shrinkage_strength=float(
                config_payload.get("family_shrinkage_strength", 0.0)
            ),
            enable_family_affine_calibration=bool(
                config_payload.get("enable_family_affine_calibration", True)
            ),
            family_calibration_strength=float(
                config_payload.get("family_calibration_strength", 4.0)
            ),
            prediction_label=str(
                config_payload.get(
                    "prediction_label",
                    "family_conditioned_tabular_cs_predictor",
                )
            ),
        )
        spec_payload = dict(payload["feature_spec"])
        spec = _EncodedFeatureSpec(
            numeric_columns=tuple(map(str, spec_payload.get("numeric_columns", []))),
            categorical_columns=tuple(
                map(str, spec_payload.get("categorical_columns", []))
            ),
            numeric_mean=tuple(map(float, spec_payload.get("numeric_mean", []))),
            numeric_scale=tuple(map(float, spec_payload.get("numeric_scale", []))),
            categorical_levels={
                str(key): tuple(map(str, values))
                for key, values in dict(
                    spec_payload.get("categorical_levels", {})
                ).items()
            },
            interaction_pairs=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in spec_payload.get("interaction_pairs", [])
                if len(pair) == 2
            ),
        )
        return cls(
            feature_spec=spec,
            family_models={
                str(family): _family_model_from_dict(model)
                for family, model in dict(payload.get("family_models", {})).items()
            },
            global_model=_family_model_from_dict(dict(payload["global_model"])),
            config=config,
            feature_roles={str(k): bool(v) for k, v in dict(payload.get("feature_roles", {})).items()},
            training_summary=dict(payload.get("training_summary", {})),
        )


def train_family_conditioned_chemical_shift_predictor(
    *,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    config: ChemicalShiftPredictorConfig | None = None,
) -> FamilyConditionedChemicalShiftPredictor:
    """Fit a family-conditioned CS predictor on the configured train split."""

    cfg = config or ChemicalShiftPredictorConfig()
    examples = build_chemical_shift_predictor_examples(
        targets=targets,
        splits=splits,
        features=features,
    )
    train_mask = _training_mask(examples, cfg)
    train = examples.loc[train_mask].reset_index(drop=True)
    if train.empty:
        raise ValueError("No rows are available for CS predictor training")

    feature_spec = _fit_feature_spec(train, cfg)
    encoded = _encode_features(train, feature_spec)
    target = train["target_value"].to_numpy(dtype=np.float64)
    global_model = _fit_one_family_model(
        family="all",
        encoded=encoded,
        target=target,
        alpha=float(cfg.ridge_alpha),
        uncertainty_floor=float(cfg.uncertainty_floor),
    )
    family_models: dict[str, _RidgeFamilyModel] = {}
    for family in sorted(train["atom_family"].dropna().astype(str).unique()):
        mask = train["atom_family"].astype(str).to_numpy() == family
        if int(np.sum(mask.astype(np.int64))) < int(cfg.min_family_rows):
            continue
        family_models[family] = _fit_one_family_model(
            family=family,
            encoded=encoded[mask],
            target=target[mask],
            alpha=float(cfg.ridge_alpha),
            uncertainty_floor=float(cfg.uncertainty_floor),
        )
    family_models = _apply_family_shrinkage(
        family_models=family_models,
        global_model=global_model,
        encoded=encoded,
        target=target,
        train_frame=train,
        config=cfg,
    )
    global_model, family_models = _apply_affine_family_calibration(
        family_models=family_models,
        global_model=global_model,
        encoded=encoded,
        target=target,
        train_frame=train,
        config=cfg,
    )

    roles = _feature_roles(
        numeric_columns=feature_spec.numeric_columns,
        categorical_columns=feature_spec.categorical_columns,
        interaction_pairs=feature_spec.interaction_pairs,
    )
    roles["hierarchical_family_shrinkage"] = _family_shrinkage_enabled(cfg)
    roles["affine_family_calibration"] = _family_calibration_enabled(cfg)
    local_manifold_feature_columns = _columns_with_tokens(
        feature_spec.numeric_columns,
        CS_LOCAL_MANIFOLD_TOKENS,
    )
    local_manifold_interaction_pairs = _interaction_pairs_with_role(
        feature_spec.interaction_pairs,
        role="manifold",
    )
    return FamilyConditionedChemicalShiftPredictor(
        feature_spec=feature_spec,
        family_models=family_models,
        global_model=global_model,
        config=cfg,
        feature_roles=roles,
        training_summary={
            "bundle_kind": CS_PREDICTOR_BUNDLE_KIND,
            "architecture_id": CS_PREDICTOR_ARCHITECTURE_ID,
            "train_rows": int(len(train)),
            "example_rows": int(len(examples)),
            "trained_family_count": int(len(family_models)),
            "trained_families": sorted(family_models),
            "numeric_feature_columns": list(feature_spec.numeric_columns),
            "categorical_feature_columns": list(feature_spec.categorical_columns),
            "interaction_feature_pairs": [
                list(pair) for pair in feature_spec.interaction_pairs
            ],
            "interaction_feature_count": int(len(feature_spec.interaction_pairs)),
            "family_shrinkage_strength": float(cfg.family_shrinkage_strength),
            "family_shrinkage_enabled": _family_shrinkage_enabled(cfg),
            "family_model_blend_weights": {
                family: float(model.family_blend_weight)
                for family, model in sorted(family_models.items())
            },
            "family_affine_calibration_enabled": _family_calibration_enabled(cfg),
            "family_calibration_strength": float(cfg.family_calibration_strength),
            "family_calibration_parameters": {
                family: {
                    "slope": float(model.calibration_slope),
                    "intercept": float(model.calibration_intercept),
                    "blend_weight": float(model.calibration_blend_weight),
                    "rows": int(model.calibration_rows),
                }
                for family, model in sorted(family_models.items())
            },
            "global_calibration_parameters": {
                "slope": float(global_model.calibration_slope),
                "intercept": float(global_model.calibration_intercept),
                "blend_weight": float(global_model.calibration_blend_weight),
                "rows": int(global_model.calibration_rows),
            },
            "local_manifold_feature_columns": list(local_manifold_feature_columns),
            "local_manifold_interaction_feature_pairs": [
                list(pair) for pair in local_manifold_interaction_pairs
            ],
            "local_manifold_interaction_feature_count": int(
                len(local_manifold_interaction_pairs)
            ),
            "feature_roles": roles,
        },
    )


def build_chemical_shift_predictor_examples(
    *,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return normalized target rows merged with optional feature sidecars."""

    target_frame = normalize_target_frame(targets)
    target_frame = _attach_splits(target_frame, splits)
    if features is None:
        return target_frame.reset_index(drop=True)
    feature_frame = _normalize_feature_frame(features)
    join_keys = _feature_join_keys(target_frame, feature_frame)
    extra_columns = [
        column
        for column in feature_frame.columns
        if column not in set(join_keys)
        and column
        not in {
            "entity_uid",
            "target_id",
            "seq_id",
            "comp_id",
            "atom_id",
            "atom_family",
            "split",
        }
    ]
    merged = target_frame.merge(
        feature_frame[list(join_keys) + extra_columns],
        on=list(join_keys),
        how="left",
    )
    return merged.reset_index(drop=True)


def train_and_evaluate_chemical_shift_predictor(
    *,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    predictor_config: ChemicalShiftPredictorConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
) -> dict[str, Any]:
    """Fit, predict, and run the UCBShift-parity benchmark."""

    cfg = predictor_config or ChemicalShiftPredictorConfig()
    predictor = train_family_conditioned_chemical_shift_predictor(
        targets=targets,
        splits=splits,
        features=features,
        config=cfg,
    )
    prediction_source = features if features is not None else targets
    predictions = predictor.predict(prediction_source)
    parity = evaluate_chemical_shift_prediction_parity(
        predictions=predictions,
        targets=targets,
        splits=splits,
        config=parity_config
        or ChemicalShiftParityConfig(
            gate_split=cfg.gate_split,
            prediction_label=cfg.prediction_label,
            require_uncertainty=True,
            require_ood=True,
            uncertainty_threshold=_reward_uncertainty_threshold(
                predictor.family_models,
                predictor.global_model,
                cfg,
            ),
            ood_score_threshold=cfg.ood_score_threshold,
            max_high_uncertainty_positive_reward_fraction=0.0,
        ),
    )
    return {"predictor": predictor, "predictions": predictions, "parity": parity}


def train_and_evaluate_chemical_shift_predictor_shared_q(
    *,
    targets: pd.DataFrame,
    support_features: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    predictor_config: ChemicalShiftPredictorConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
    shared_q_config: ChemicalShiftSharedQConfig | None = None,
    shared_q_sweep_config: ChemicalShiftSharedQSweepConfig | None = None,
) -> dict[str, Any]:
    """Fit a CS predictor, then evaluate generated support through one shared q."""

    base = train_and_evaluate_chemical_shift_predictor(
        targets=targets,
        splits=splits,
        features=features,
        predictor_config=predictor_config,
        parity_config=parity_config,
    )
    raw_support_predictions = base["predictor"].predict(support_features)
    resolved_shared_q_config = shared_q_config or ChemicalShiftSharedQConfig()
    if shared_q_sweep_config is not None:
        shared_q = sweep_chemical_shift_shared_q_posterior(
            predictions=raw_support_predictions,
            targets=targets,
            splits=splits,
            posterior_config=resolved_shared_q_config,
            parity_config=_shared_q_parity_config(
                parity_config,
                resolved_shared_q_config,
            ),
            sweep_config=shared_q_sweep_config,
        )
    else:
        shared_q = infer_chemical_shift_shared_q_posterior(
            predictions=raw_support_predictions,
            targets=targets,
            splits=splits,
            posterior_config=resolved_shared_q_config,
            parity_config=_shared_q_parity_config(
                parity_config,
                resolved_shared_q_config,
            ),
        )
    support_predictions = _attach_source_feature_columns(
        predictions=raw_support_predictions,
        source_features=support_features,
    )
    support_predictions = _attach_target_values_to_predictions(
        predictions=support_predictions,
        targets=targets,
    )
    support_predictions = attach_shared_q_reward_signals_to_support_predictions(
        support_predictions=support_predictions,
        support_weights=shared_q["support_weights"],
    )
    support_predictions = attach_shared_q_family_pressure_to_support_predictions(
        support_predictions=support_predictions,
        shared_q_summary=shared_q["summary"],
    )
    return {
        **base,
        "support_predictions": support_predictions,
        "shared_q": shared_q,
    }


def save_predictor_bundle(
    predictor: FamilyConditionedChemicalShiftPredictor,
    path: str | Path,
) -> None:
    """Write a predictor bundle JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictor.to_dict(), indent=2, sort_keys=True) + "\n")


def load_predictor_bundle(path: str | Path) -> FamilyConditionedChemicalShiftPredictor:
    """Load a predictor bundle JSON."""

    return FamilyConditionedChemicalShiftPredictor.from_dict(
        json.loads(Path(path).read_text())
    )


def train_predictor_from_paths(
    *,
    targets_path: str | Path,
    splits_path: str | Path | None,
    features_path: str | Path | None,
    predictions_output: str | Path,
    bundle_output: str | Path,
    summary_output: str | Path,
    metrics_output: str | Path | None = None,
    pairs_output: str | Path | None = None,
    support_features_path: str | Path | None = None,
    support_predictions_output: str | Path | None = None,
    shared_q_posterior_predictions_output: str | Path | None = None,
    shared_q_support_weights_output: str | Path | None = None,
    shared_q_summary_output: str | Path | None = None,
    shared_q_metrics_output: str | Path | None = None,
    shared_q_pairs_output: str | Path | None = None,
    shared_q_sweep_report_output: str | Path | None = None,
    predictor_config: ChemicalShiftPredictorConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
    shared_q_config: ChemicalShiftSharedQConfig | None = None,
    shared_q_sweep_config: ChemicalShiftSharedQSweepConfig | None = None,
) -> dict[str, Any]:
    """CLI-oriented helper that trains and writes all predictor artifacts."""

    targets = load_table(targets_path)
    splits = load_table(splits_path) if splits_path and Path(splits_path).exists() else None
    features = (
        load_table(features_path)
        if features_path and Path(features_path).exists()
        else None
    )
    result = train_and_evaluate_chemical_shift_predictor(
        targets=targets,
        splits=splits,
        features=features,
        predictor_config=predictor_config,
        parity_config=parity_config,
    )
    _write_table(result["predictions"], Path(predictions_output))
    save_predictor_bundle(result["predictor"], bundle_output)
    write_parity_outputs(
        result=result["parity"],
        summary_path=summary_output,
        metrics_path=metrics_output,
        pairs_path=pairs_output,
    )
    summary = {
        "bundle_output": str(bundle_output),
        "predictions_output": str(predictions_output),
        "summary_output": str(summary_output),
        "parity_decision": result["parity"]["summary"]["decision"],
        "joined_pair_count": int(result["parity"]["summary"]["joined_pair_count"]),
        "predictor_training_summary": result["predictor"].training_summary,
    }
    if support_features_path and Path(support_features_path).exists():
        support_features = load_table(support_features_path)
        raw_support_predictions = result["predictor"].predict(support_features)
        resolved_shared_q_config = shared_q_config or ChemicalShiftSharedQConfig()
        if shared_q_sweep_config is not None:
            shared_q = sweep_chemical_shift_shared_q_posterior(
                predictions=raw_support_predictions,
                targets=targets,
                splits=splits,
                posterior_config=resolved_shared_q_config,
                parity_config=_shared_q_parity_config(
                    parity_config,
                    resolved_shared_q_config,
                ),
                sweep_config=shared_q_sweep_config,
            )
        else:
            shared_q = infer_chemical_shift_shared_q_posterior(
                predictions=raw_support_predictions,
                targets=targets,
                splits=splits,
                posterior_config=resolved_shared_q_config,
                parity_config=_shared_q_parity_config(
                    parity_config,
                    resolved_shared_q_config,
                ),
            )
        support_predictions = _attach_source_feature_columns(
            predictions=raw_support_predictions,
            source_features=support_features,
        )
        support_predictions = _attach_target_values_to_predictions(
            predictions=support_predictions,
            targets=targets,
        )
        support_predictions = attach_shared_q_reward_signals_to_support_predictions(
            support_predictions=support_predictions,
            support_weights=shared_q["support_weights"],
        )
        support_predictions = attach_shared_q_family_pressure_to_support_predictions(
            support_predictions=support_predictions,
            shared_q_summary=shared_q["summary"],
        )
        if support_predictions_output is not None:
            _write_table(support_predictions, Path(support_predictions_output))
        if shared_q_posterior_predictions_output is not None:
            _write_table(
                shared_q["posterior_predictions"],
                Path(shared_q_posterior_predictions_output),
            )
        if shared_q_support_weights_output is not None:
            _write_table(
                shared_q["support_weights"],
                Path(shared_q_support_weights_output),
            )
        if shared_q_summary_output is not None:
            output = Path(shared_q_summary_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(shared_q["summary"], indent=2, sort_keys=True) + "\n"
            )
        if (
            shared_q_sweep_report_output is not None
            and "sweep_results" in shared_q
        ):
            _write_table(shared_q["sweep_results"], Path(shared_q_sweep_report_output))
        if shared_q_metrics_output is not None or shared_q_pairs_output is not None:
            if shared_q_summary_output is None:
                raise ValueError(
                    "shared_q_summary_output is required when writing shared-q "
                    "metrics or pairs"
                )
            write_parity_outputs(
                result=shared_q["parity"],
                summary_path=Path(shared_q_summary_output).with_name(
                    f"{Path(shared_q_summary_output).stem}.parity.json"
                ),
                metrics_path=shared_q_metrics_output,
                pairs_path=shared_q_pairs_output,
            )
        summary.update(
            {
                "support_predictions_output": str(support_predictions_output or ""),
                "shared_q_summary_output": str(shared_q_summary_output or ""),
                "shared_q_sweep_report_output": str(
                    shared_q_sweep_report_output or ""
                ),
                "shared_q_decision": shared_q["summary"]["decision"],
                "shared_q_support_count": int(shared_q["summary"]["support_count"]),
                "shared_q_solver_sweep_enabled": "sweep" in shared_q["summary"],
                "support_prediction_source_feature_columns_preserved": True,
                "support_prediction_target_values_attached": bool(
                    "target_value" in support_predictions.columns
                    and support_predictions["target_value"].notna().any()
                ),
                "support_prediction_shared_q_reward_columns_appended": True,
                "support_prediction_shared_q_family_pressure_columns_appended": True,
                "support_prediction_positive_shared_q_gain_rows": int(
                    (
                        pd.to_numeric(
                            support_predictions.get(
                                "shared_q_gain",
                                pd.Series(dtype=float),
                            ),
                            errors="coerce",
                        ).fillna(0.0)
                        > 0.0
                    ).sum()
                ),
                "support_prediction_positive_cosatisfaction_gain_rows": int(
                    (
                        pd.to_numeric(
                            support_predictions.get(
                                "cosatisfaction_gain",
                                pd.Series(dtype=float),
                            ),
                            errors="coerce",
                        ).fillna(0.0)
                        > 0.0
                    ).sum()
                ),
            }
        )
    return summary


def _attach_source_feature_columns(
    *,
    predictions: pd.DataFrame,
    source_features: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve generated-support feature/manifest columns in prediction output."""

    if predictions.empty or source_features.empty:
        return predictions.copy()
    if len(predictions) != len(source_features):
        return predictions.copy()
    out = predictions.reset_index(drop=True).copy()
    source = source_features.reset_index(drop=True)
    for column in source.columns:
        name = str(column)
        if name in out.columns:
            continue
        out[name] = source[column]
    return out


def _attach_target_values_to_predictions(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Attach target shifts after shared-q inference for teacher-feature reuse."""

    if predictions.empty or targets.empty:
        return predictions.copy()
    out = predictions.copy()
    existing = (
        pd.to_numeric(out["target_value"], errors="coerce")
        if "target_value" in out.columns
        else pd.Series(math.nan, index=out.index)
    )
    out = out.drop(columns=["target_value"], errors="ignore")
    target_frame = normalize_target_frame(targets)
    try:
        join_keys = _feature_join_keys(target_frame, out)
    except ValueError:
        out["target_value"] = existing
        return out
    target_values = (
        target_frame[list(join_keys) + ["target_value"]]
        .dropna(subset=["target_value"])
        .drop_duplicates(list(join_keys))
    )
    merged = out.merge(target_values, on=list(join_keys), how="left")
    merged["target_value"] = existing.reset_index(drop=True).combine_first(
        pd.to_numeric(merged["target_value"], errors="coerce")
    )
    return merged


def attach_shared_q_reward_signals_to_support_predictions(
    *,
    support_predictions: pd.DataFrame,
    support_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one-shared-q posterior preference as router teacher signals."""

    out = support_predictions.copy()
    if out.empty or support_weights.empty:
        return out
    if "entity_uid" not in out.columns or "support_id" not in out.columns:
        return out
    weights = support_weights.copy()
    if "entity_uid" not in weights.columns or "support_id" not in weights.columns:
        return out
    weights["entity_uid"] = weights["entity_uid"].astype(str)
    weights["support_id"] = weights["support_id"].astype(str)
    out["entity_uid"] = out["entity_uid"].astype(str)
    out["support_id"] = out["support_id"].astype(str)
    weights = _shared_q_support_signal_frame(weights)
    merge_columns = [
        "entity_uid",
        "support_id",
        "shared_q_posterior_weight",
        "shared_q_posterior_weight_initial",
        "shared_q_uniform_valid_weight",
        "shared_q_gain",
        "cosatisfaction_gain",
        "shared_q_energy",
        "shared_q_energy_advantage",
        "shared_q_support_rank",
        "shared_q_support_selected",
        "support_replay_priority",
    ]
    enriched = out.merge(
        weights[merge_columns],
        on=["entity_uid", "support_id"],
        how="left",
    )
    for column in (
        "shared_q_posterior_weight",
        "shared_q_posterior_weight_initial",
        "shared_q_uniform_valid_weight",
        "shared_q_gain",
        "cosatisfaction_gain",
        "shared_q_energy",
        "shared_q_energy_advantage",
        "support_replay_priority",
    ):
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    enriched["shared_q_support_rank"] = (
        pd.to_numeric(enriched["shared_q_support_rank"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    enriched["shared_q_support_selected"] = _boolean_series(
        enriched["shared_q_support_selected"]
    ).fillna(False)
    return enriched


def attach_shared_q_family_pressure_to_support_predictions(
    *,
    support_predictions: pd.DataFrame,
    shared_q_summary: dict[str, Any],
) -> pd.DataFrame:
    """Attach family CCC gap pressure from one-shared-q posterior parity."""

    out = support_predictions.copy()
    if out.empty or "atom_family" not in out.columns:
        return out
    pressure = _shared_q_family_pressure(shared_q_summary)
    if not pressure:
        return out
    family_gap = pressure["family_gap"]
    family_ccc = pressure["family_ccc"]
    family_values = out["atom_family"].map(canonical_atom_family).astype(str)
    out["shared_q_family_ccc"] = family_values.map(family_ccc).fillna(math.nan)
    out["family_ccc_gap_to_target"] = family_values.map(family_gap).fillna(0.0)
    out["target_family_ccc_gap"] = out["family_ccc_gap_to_target"]
    out["macro_family_ccc_gap_to_target"] = float(pressure["macro_gap"])
    out["worst_family_ccc_gap_to_target"] = float(pressure["worst_gap"])
    out["worst_family"] = str(pressure["worst_family"])
    out["shared_q_worst_family"] = str(pressure["worst_family"])
    out["shared_q_family_min_ccc"] = float(pressure["family_min_ccc"])
    out["shared_q_family_macro_ccc"] = float(pressure["family_macro_ccc"])
    out["shared_q_failing_required_family"] = (
        out["family_ccc_gap_to_target"].to_numpy(dtype=float) > 0.0
    )
    priority = pd.to_numeric(
        out.get("support_replay_priority", pd.Series(1.0, index=out.index)),
        errors="coerce",
    ).fillna(1.0)
    pressure_boost = (
        1.0
        + 2.0 * out["family_ccc_gap_to_target"].to_numpy(dtype=float)
        + float(pressure["macro_gap"])
        + float(pressure["worst_gap"])
    )
    out["support_replay_priority"] = np.clip(
        priority.to_numpy(dtype=float) * pressure_boost,
        0.1,
        8.0,
    )
    return out


def _shared_q_family_pressure(shared_q_summary: dict[str, Any]) -> dict[str, Any]:
    parity_summary = dict(shared_q_summary.get("parity_summary") or {})
    target_ccc = _finite_summary_float(
        shared_q_summary.get("target_ccc", parity_summary.get("target_ccc", 0.95)),
        default=0.95,
    )
    gate_split = str(
        shared_q_summary.get("gate_split")
        or parity_summary.get("gate_split")
        or "all"
    )
    split_summaries = dict(parity_summary.get("split_summaries") or {})
    split_summary = dict(
        split_summaries.get(gate_split)
        or split_summaries.get("all")
        or next(iter(split_summaries.values()), {})
    )
    per_family = dict(split_summary.get("per_family") or {})
    if not per_family:
        return {}
    family_ccc: dict[str, float] = {}
    family_gap: dict[str, float] = {}
    for family, row in per_family.items():
        canonical = canonical_atom_family(str(family))
        if canonical is None:
            canonical = str(family)
        ccc = _finite_summary_float(dict(row).get("ccc"), default=math.nan)
        family_ccc[str(canonical)] = ccc
        family_gap[str(canonical)] = (
            max(0.0, target_ccc - ccc) if math.isfinite(ccc) else target_ccc
        )
    if not family_gap:
        return {}
    worst_family = max(family_gap, key=lambda family: family_gap[family])
    worst_gap = float(family_gap[worst_family])
    macro_ccc = _finite_summary_float(
        split_summary.get("family_macro_ccc"),
        default=math.nan,
    )
    family_min_ccc = _finite_summary_float(
        split_summary.get("family_min_ccc"),
        default=math.nan,
    )
    macro_gap = (
        max(0.0, target_ccc - macro_ccc)
        if math.isfinite(macro_ccc)
        else float(np.mean(list(family_gap.values())))
    )
    return {
        "family_ccc": family_ccc,
        "family_gap": family_gap,
        "target_ccc": target_ccc,
        "family_macro_ccc": macro_ccc,
        "family_min_ccc": family_min_ccc,
        "macro_gap": float(macro_gap),
        "worst_family": str(worst_family),
        "worst_gap": worst_gap,
    }


def _finite_summary_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _shared_q_support_signal_frame(support_weights: pd.DataFrame) -> pd.DataFrame:
    frame = support_weights.copy()
    frame["posterior_weight"] = pd.to_numeric(
        frame.get("posterior_weight", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    frame["posterior_weight_initial"] = pd.to_numeric(
        frame.get(
            "posterior_weight_initial",
            frame["posterior_weight"],
        ),
        errors="coerce",
    ).fillna(0.0)
    frame["shared_q_energy"] = pd.to_numeric(
        frame.get("shared_q_energy", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    )
    frame["family_balanced_residual_energy"] = pd.to_numeric(
        frame.get("family_balanced_residual_energy", frame["shared_q_energy"]),
        errors="coerce",
    )
    if "support_valid" in frame.columns:
        valid = _boolean_series(frame["support_valid"]).fillna(True)
    else:
        valid = pd.Series(True, index=frame.index)
    frame["support_valid"] = valid.astype(bool)

    uniform_weights: dict[tuple[str, str], float] = {}
    energy_means: dict[str, float] = {}
    for entity_uid, group in frame.groupby("entity_uid", sort=False):
        group_valid = group["support_valid"].to_numpy(dtype=bool)
        valid_count = int(group_valid.sum())
        uniform = 1.0 / float(valid_count) if valid_count > 0 else 0.0
        for _, row in group.iterrows():
            uniform_weights[(str(entity_uid), str(row["support_id"]))] = (
                uniform if bool(row["support_valid"]) else 0.0
            )
        energies = group.loc[group["support_valid"], "family_balanced_residual_energy"]
        finite = pd.to_numeric(energies, errors="coerce")
        finite = finite[np.isfinite(finite.to_numpy(dtype=float))]
        energy_means[str(entity_uid)] = (
            float(finite.mean()) if not finite.empty else math.nan
        )

    frame["shared_q_uniform_valid_weight"] = [
        float(uniform_weights.get((str(row["entity_uid"]), str(row["support_id"])), 0.0))
        for _, row in frame.iterrows()
    ]
    posterior_advantage = (
        frame["posterior_weight"] - frame["shared_q_uniform_valid_weight"]
    ).clip(lower=0.0)
    refinement_advantage = (
        frame["posterior_weight"] - frame["posterior_weight_initial"]
    ).clip(lower=0.0)
    frame["shared_q_gain"] = (posterior_advantage + refinement_advantage).clip(
        lower=0.0,
        upper=5.0,
    )
    mean_energy = frame["entity_uid"].astype(str).map(energy_means)
    energy_advantage = mean_energy - frame["family_balanced_residual_energy"]
    denominator = np.maximum(np.abs(mean_energy.to_numpy(dtype=float)), 1.0)
    normalized_advantage = energy_advantage.to_numpy(dtype=float) / denominator
    normalized_advantage = np.where(
        np.isfinite(normalized_advantage),
        np.maximum(normalized_advantage, 0.0),
        0.0,
    )
    frame["shared_q_energy_advantage"] = normalized_advantage
    frame["cosatisfaction_gain"] = np.clip(
        normalized_advantage * (1.0 + frame["posterior_weight"].to_numpy(dtype=float)),
        0.0,
        5.0,
    )
    rank = frame.get("posterior_weight_rank")
    if rank is None:
        frame["shared_q_support_rank"] = (
            frame.groupby("entity_uid")["posterior_weight"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
    else:
        frame["shared_q_support_rank"] = (
            pd.to_numeric(rank, errors="coerce").fillna(0).astype(int)
        )
    frame["shared_q_support_selected"] = (
        frame["support_valid"]
        & (
            (frame["shared_q_gain"] > 0.0)
            | (frame["shared_q_support_rank"].astype(int) == 1)
        )
    )
    frame["support_replay_priority"] = np.clip(
        1.0 + 4.0 * frame["shared_q_gain"] + 4.0 * frame["cosatisfaction_gain"],
        0.1,
        8.0,
    )
    frame = frame.rename(
        columns={
            "posterior_weight": "shared_q_posterior_weight",
            "posterior_weight_initial": "shared_q_posterior_weight_initial",
        }
    )
    return frame


def _shared_q_parity_config(
    parity_config: ChemicalShiftParityConfig | None,
    shared_q_config: ChemicalShiftSharedQConfig,
) -> ChemicalShiftParityConfig | None:
    if parity_config is None:
        return None
    return replace(parity_config, prediction_label=shared_q_config.posterior_label)


def _normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    normalized["entity_uid"] = _entity_uid_series(frame)
    normalized["target_id"] = _string_column(frame, ("target_id", "target_uid", "measurement_uid"))
    normalized["seq_id"] = pd.to_numeric(
        _column_or_na(frame, ("seq_id", "residue_index", "residue_id", "residue_number")),
        errors="coerce",
    )
    normalized["comp_id"] = _string_column(
        frame,
        ("comp_id", "residue_name", "resname", "aa"),
    ).str.upper()
    normalized["atom_id"] = _string_column(frame, ("atom_id", "atom_name", "atom")).str.upper()
    family = _string_column(frame, ("atom_family", "family"))
    normalized["atom_family"] = family.map(canonical_atom_family)
    normalized["atom_family"] = normalized["atom_family"].fillna(
        normalized["atom_id"].map(canonical_atom_family)
    )
    _attach_predictor_metadata(normalized, frame)
    for column in frame.columns:
        if column in normalized.columns:
            continue
        if str(column).lower() in RESERVED_FEATURE_COLUMNS:
            continue
        normalized[str(column)] = frame[column]
    normalized = normalized.dropna(subset=["entity_uid", "atom_family"]).reset_index(
        drop=True
    )
    return normalized


def _attach_predictor_metadata(
    normalized: pd.DataFrame,
    frame: pd.DataFrame,
) -> None:
    for output_column, aliases in PREDICTOR_METADATA_ALIASES.items():
        values = _first_available_column(frame, aliases)
        if values is None:
            continue
        if output_column == "support_valid":
            normalized[output_column] = _boolean_series(values).fillna(True)
        elif output_column == "prior_log_prob":
            normalized[output_column] = pd.to_numeric(values, errors="coerce")
        else:
            normalized[output_column] = values.astype("string").str.strip()


def _first_available_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> pd.Series | None:
    for alias in aliases:
        if alias in frame.columns:
            return frame[alias]
    return None


def _prediction_output_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["entity_uid", "target_id", "seq_id", "comp_id", "atom_id", "atom_family"]
    columns.extend(
        column
        for column in PREDICTOR_OUTPUT_METADATA_COLUMNS
        if column in frame.columns and column not in columns
    )
    return columns


def _support_validity_gate(frame: pd.DataFrame) -> np.ndarray:
    if "support_valid" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    validity = _boolean_series(frame["support_valid"]).fillna(True)
    return validity.to_numpy(dtype=bool)


def _attach_splits(frame: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    if splits is None:
        out["split"] = out.get("split", pd.Series("all", index=out.index)).fillna("all")
        return out
    if "split" in out.columns and out["split"].notna().any():
        out["split"] = out["split"].fillna("all")
        return out
    if "split" not in splits.columns:
        out["split"] = "all"
        return out
    out = out.drop(columns=["split"], errors="ignore")
    split_frame = pd.DataFrame(
        {
            "entity_uid": _entity_uid_series(splits),
            "split": splits["split"].astype("string").str.strip(),
        }
    ).drop_duplicates("entity_uid")
    out = out.merge(split_frame, on="entity_uid", how="left")
    out["split"] = out["split"].fillna("all")
    return out


def _training_mask(
    examples: pd.DataFrame,
    config: ChemicalShiftPredictorConfig,
) -> np.ndarray:
    splits = examples.get("split", pd.Series("all", index=examples.index)).astype(str)
    requested = {str(split) for split in config.train_splits}
    mask = splits.isin(requested).to_numpy(dtype=bool)
    if bool(mask.any()):
        return mask
    not_gate = (splits != str(config.gate_split)).to_numpy(dtype=bool)
    return not_gate if bool(not_gate.any()) else np.ones(len(examples), dtype=bool)


def _fit_feature_spec(
    frame: pd.DataFrame,
    config: ChemicalShiftPredictorConfig,
) -> _EncodedFeatureSpec:
    categorical_columns = tuple(
        column
        for column in ("atom_family", "atom_id", "comp_id")
        if column in frame.columns and frame[column].notna().any()
    )
    numeric_candidates = tuple(
        column
        for column in frame.columns
        if column not in RESERVED_FEATURE_COLUMNS
        and column not in categorical_columns
        and column not in {"target_id", "atom_id", "atom_family", "comp_id", "split"}
        and pd.api.types.is_numeric_dtype(frame[column])
    )
    numeric_candidate_frame = frame.loc[:, list(numeric_candidates)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    numeric_columns = tuple(
        column
        for column in numeric_candidates
        if np.isfinite(numeric_candidate_frame[column].to_numpy(dtype=float)).any()
    )
    numeric = numeric_candidate_frame.loc[:, list(numeric_columns)]
    numeric_mean = tuple(float(numeric[column].mean()) for column in numeric_columns)
    numeric_scale = tuple(
        max(float(numeric[column].std(ddof=0)), 1.0e-6)
        for column in numeric_columns
    )
    levels = {
        column: tuple(sorted(frame[column].dropna().astype(str).unique()))
        for column in categorical_columns
    }
    interaction_pairs = _select_interaction_pairs(
        numeric_columns=numeric_columns,
        config=config,
    )
    return _EncodedFeatureSpec(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_mean=numeric_mean,
        numeric_scale=numeric_scale,
        categorical_levels=levels,
        interaction_pairs=interaction_pairs,
    )


def _encode_features(frame: pd.DataFrame, spec: _EncodedFeatureSpec) -> np.ndarray:
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=np.float64)]
    standardized_numeric: dict[str, np.ndarray] = {}
    for index, column in enumerate(spec.numeric_columns):
        standardized = _standardize_numeric_feature(
            frame,
            column=column,
            mean=float(spec.numeric_mean[index]),
            scale=float(spec.numeric_scale[index]),
        )
        standardized_numeric[column] = standardized
        columns.append(standardized)
    for left, right in spec.interaction_pairs:
        left_values = standardized_numeric.get(left)
        right_values = standardized_numeric.get(right)
        if left_values is None or right_values is None:
            columns.append(np.zeros(len(frame), dtype=np.float64))
            continue
        columns.append(left_values * right_values)
    for column in spec.categorical_columns:
        values = frame.get(column, pd.Series(pd.NA, index=frame.index)).astype(str)
        for level in spec.categorical_levels.get(column, ()):
            columns.append((values == str(level)).to_numpy(dtype=np.float64))
    return np.vstack(columns).T.astype(np.float64)


def _standardize_numeric_feature(
    frame: pd.DataFrame,
    *,
    column: str,
    mean: float,
    scale: float,
) -> np.ndarray:
    if column in frame.columns:
        raw = pd.to_numeric(frame[column], errors="coerce")
    else:
        raw = pd.Series(math.nan, index=frame.index)
    centered = raw.fillna(float(mean)).to_numpy(dtype=np.float64)
    return (centered - float(mean)) / max(float(scale), 1.0e-6)


def _select_interaction_pairs(
    *,
    numeric_columns: Sequence[str],
    config: ChemicalShiftPredictorConfig,
) -> tuple[tuple[str, str], ...]:
    if not config.enable_pairwise_interactions:
        return ()
    max_pairs = max(int(config.max_pairwise_interactions), 0)
    if max_pairs <= 0:
        return ()
    candidates: list[tuple[tuple[int, str, str], tuple[str, str]]] = []
    columns = tuple(str(column) for column in numeric_columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            priority = _interaction_priority(left, right)
            if priority is None:
                continue
            candidates.append(((priority, left, right), (left, right)))
    candidates.sort(key=lambda item: item[0])
    return tuple(pair for _, pair in candidates[:max_pairs])


def _interaction_priority(left: str, right: str) -> int | None:
    left_roles = _numeric_feature_roles(left)
    right_roles = _numeric_feature_roles(right)
    pair_roles = {
        (left_role, right_role)
        for left_role in left_roles
        for right_role in right_roles
            if left_role != right_role
    }
    if ("manifold", "bioemu") in pair_roles or ("bioemu", "manifold") in pair_roles:
        return 0
    if (
        ("manifold", "reference") in pair_roles
        or ("reference", "manifold") in pair_roles
    ):
        return 1
    if (
        ("manifold", "geometry") in pair_roles
        or ("geometry", "manifold") in pair_roles
    ):
        return 2
    if ("geometry", "bioemu") in pair_roles or ("bioemu", "geometry") in pair_roles:
        return 3
    if ("route", "bioemu") in pair_roles or ("bioemu", "route") in pair_roles:
        return 4
    if ("route", "manifold") in pair_roles or ("manifold", "route") in pair_roles:
        return 5
    if ("route", "geometry") in pair_roles or ("geometry", "route") in pair_roles:
        return 6
    if (
        ("geometry", "reference") in pair_roles
        or ("reference", "geometry") in pair_roles
    ):
        return 7
    if ("bioemu", "reference") in pair_roles or ("reference", "bioemu") in pair_roles:
        return 8
    if ("route", "reference") in pair_roles or ("reference", "route") in pair_roles:
        return 9
    if "manifold" in left_roles or "manifold" in right_roles:
        return 10
    if "geometry" in left_roles or "geometry" in right_roles:
        return 11
    if "bioemu" in left_roles or "bioemu" in right_roles:
        return 12
    if "route" in left_roles or "route" in right_roles:
        return 13
    return None


def _numeric_feature_roles(column: str) -> set[str]:
    lowered = str(column).lower()
    roles: set[str] = set()
    if _contains_token(lowered, LOCAL_GEOMETRY_TOKENS):
        roles.add("geometry")
    if _contains_token(lowered, CS_LOCAL_MANIFOLD_TOKENS):
        roles.add("manifold")
    if _contains_token(lowered, BIOEMU_CONTEXT_TOKENS):
        roles.add("bioemu")
    if _contains_token(lowered, REFERENCE_FEATURE_TOKENS):
        roles.add("reference")
    if _contains_token(lowered, ROUTE_CONTEXT_TOKENS):
        roles.add("route")
    return roles


def _fit_one_family_model(
    *,
    family: str,
    encoded: np.ndarray,
    target: np.ndarray,
    alpha: float,
    uncertainty_floor: float,
) -> _RidgeFamilyModel:
    beta = _ridge_solve(encoded, target, alpha=alpha)
    residual = target - encoded @ beta
    rmse = max(_rmse(residual), float(uncertainty_floor))
    ood_raw = np.asarray([_ood_raw_score(row) for row in encoded], dtype=np.float64)
    finite = ood_raw[np.isfinite(ood_raw)]
    reference = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    return _RidgeFamilyModel(
        family=str(family),
        coefficients=tuple(float(value) for value in beta.tolist()),
        residual_rmse=float(rmse),
        ood_reference=max(reference, 1.0),
        train_rows=int(target.size),
    )


def _apply_family_shrinkage(
    *,
    family_models: dict[str, _RidgeFamilyModel],
    global_model: _RidgeFamilyModel,
    encoded: np.ndarray,
    target: np.ndarray,
    train_frame: pd.DataFrame,
    config: ChemicalShiftPredictorConfig,
) -> dict[str, _RidgeFamilyModel]:
    if not _family_shrinkage_enabled(config) or not family_models:
        return {
            family: replace(model, family_blend_weight=1.0)
            for family, model in family_models.items()
        }
    global_beta = np.asarray(global_model.coefficients, dtype=np.float64)
    family_values = train_frame["atom_family"].astype(str).to_numpy(dtype=object)
    out: dict[str, _RidgeFamilyModel] = {}
    for family, model in family_models.items():
        blend_weight = _family_shrinkage_blend_weight(
            train_rows=int(model.train_rows),
            config=config,
        )
        family_beta = np.asarray(model.coefficients, dtype=np.float64)
        if family_beta.shape != global_beta.shape:
            out[family] = replace(model, family_blend_weight=1.0)
            continue
        beta = blend_weight * family_beta + (1.0 - blend_weight) * global_beta
        mask = family_values == str(family)
        residual = target[mask] - encoded[mask] @ beta if bool(mask.any()) else target
        rmse = max(_rmse(residual), float(config.uncertainty_floor))
        out[family] = replace(
            model,
            coefficients=tuple(float(value) for value in beta.tolist()),
            residual_rmse=float(rmse),
            family_blend_weight=float(blend_weight),
        )
    return out


def _family_shrinkage_enabled(config: ChemicalShiftPredictorConfig) -> bool:
    try:
        strength = float(config.family_shrinkage_strength)
    except (TypeError, ValueError):
        return False
    return math.isfinite(strength) and strength > 0.0


def _family_shrinkage_blend_weight(
    *,
    train_rows: int,
    config: ChemicalShiftPredictorConfig,
) -> float:
    if not _family_shrinkage_enabled(config):
        return 1.0
    rows = max(float(train_rows), 0.0)
    strength = max(float(config.family_shrinkage_strength), 0.0)
    if rows <= 0.0:
        return 0.0
    return float(np.clip(rows / (rows + strength), 0.0, 1.0))


def _apply_affine_family_calibration(
    *,
    family_models: dict[str, _RidgeFamilyModel],
    global_model: _RidgeFamilyModel,
    encoded: np.ndarray,
    target: np.ndarray,
    train_frame: pd.DataFrame,
    config: ChemicalShiftPredictorConfig,
) -> tuple[_RidgeFamilyModel, dict[str, _RidgeFamilyModel]]:
    if not _family_calibration_enabled(config):
        return (
            _with_identity_calibration(global_model),
            {
                family: _with_identity_calibration(model)
                for family, model in family_models.items()
            },
        )
    calibrated_global = _calibrate_one_family_model(
        global_model,
        encoded=encoded,
        target=target,
        config=config,
    )
    family_values = train_frame["atom_family"].astype(str).to_numpy(dtype=object)
    calibrated: dict[str, _RidgeFamilyModel] = {}
    for family, model in family_models.items():
        mask = family_values == str(family)
        if not bool(mask.any()):
            calibrated[family] = _with_identity_calibration(model)
            continue
        calibrated[family] = _calibrate_one_family_model(
            model,
            encoded=encoded[mask],
            target=target[mask],
            config=config,
        )
    return calibrated_global, calibrated


def _family_calibration_enabled(config: ChemicalShiftPredictorConfig) -> bool:
    if not bool(config.enable_family_affine_calibration):
        return False
    try:
        strength = float(config.family_calibration_strength)
    except (TypeError, ValueError):
        return False
    return math.isfinite(strength) and strength >= 0.0


def _with_identity_calibration(model: _RidgeFamilyModel) -> _RidgeFamilyModel:
    return replace(
        model,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        calibration_blend_weight=0.0,
        calibration_rows=0,
    )


def _calibrate_one_family_model(
    model: _RidgeFamilyModel,
    *,
    encoded: np.ndarray,
    target: np.ndarray,
    config: ChemicalShiftPredictorConfig,
) -> _RidgeFamilyModel:
    beta = np.asarray(model.coefficients, dtype=np.float64)
    raw_prediction = encoded @ beta if encoded.size else np.asarray([], dtype=np.float64)
    slope, intercept, blend_weight, rows = _fit_affine_calibration(
        raw_prediction=raw_prediction,
        target=target,
        config=config,
    )
    calibrated = slope * raw_prediction + intercept
    residual = target - calibrated
    rmse = max(_rmse(residual), float(config.uncertainty_floor))
    return replace(
        model,
        residual_rmse=float(rmse),
        calibration_slope=float(slope),
        calibration_intercept=float(intercept),
        calibration_blend_weight=float(blend_weight),
        calibration_rows=int(rows),
    )


def _fit_affine_calibration(
    *,
    raw_prediction: np.ndarray,
    target: np.ndarray,
    config: ChemicalShiftPredictorConfig,
) -> tuple[float, float, float, int]:
    raw = np.asarray(raw_prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(raw) & np.isfinite(observed)
    raw = raw[finite]
    observed = observed[finite]
    rows = int(raw.size)
    if rows <= 0:
        return 1.0, 0.0, 0.0, 0
    strength = max(float(config.family_calibration_strength), 0.0)
    blend_weight = float(np.clip(rows / (rows + strength), 0.0, 1.0))
    raw_mean = float(np.mean(raw))
    target_mean = float(np.mean(observed))
    raw_centered = raw - raw_mean
    target_centered = observed - target_mean
    raw_var = float(np.mean(np.square(raw_centered)))
    if raw_var <= 1.0e-12:
        fitted_slope = 1.0
    else:
        fitted_slope = float(np.mean(raw_centered * target_centered) / raw_var)
        fitted_slope = float(np.clip(fitted_slope, 0.05, 20.0))
    fitted_intercept = target_mean - fitted_slope * raw_mean
    slope = 1.0 + blend_weight * (fitted_slope - 1.0)
    intercept = blend_weight * fitted_intercept
    return float(slope), float(intercept), blend_weight, rows


def _ridge_solve(encoded: np.ndarray, target: np.ndarray, *, alpha: float) -> np.ndarray:
    regularizer = max(float(alpha), 0.0) * np.eye(encoded.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    lhs = encoded.T @ encoded + regularizer
    rhs = encoded.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _ood_raw_score(encoded_row: np.ndarray) -> float:
    if encoded_row.size <= 1:
        return 0.0
    values = encoded_row[1:]
    return float(np.sqrt(np.mean(np.square(values))))


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def _reward_uncertainty_threshold(
    family_models: dict[str, _RidgeFamilyModel],
    global_model: _RidgeFamilyModel,
    config: ChemicalShiftPredictorConfig,
) -> float:
    rmse_values = [global_model.residual_rmse]
    rmse_values.extend(model.residual_rmse for model in family_models.values())
    finite = [float(value) for value in rmse_values if math.isfinite(float(value))]
    base = max(finite) if finite else float(config.uncertainty_floor)
    return max(base * float(config.reward_uncertainty_scale), config.uncertainty_floor)


def _feature_roles(
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    interaction_pairs: Sequence[tuple[str, str]],
) -> dict[str, bool]:
    lowered = [str(column).lower() for column in [*numeric_columns, *categorical_columns]]
    manifold_feature_columns = [
        column
        for column in lowered
        if _contains_token(column, CS_LOCAL_MANIFOLD_TOKENS)
    ]
    interaction_roles = [
        (_numeric_feature_roles(left), _numeric_feature_roles(right))
        for left, right in interaction_pairs
    ]
    return {
        "target_family": "atom_family" in set(categorical_columns),
        "residue_identity": any(column in {"comp_id", "atom_id"} for column in categorical_columns),
        "local_geometry": any(
            _contains_token(column, LOCAL_GEOMETRY_TOKENS)
            for column in lowered
        ),
        "ucbshift2_local_manifold": bool(manifold_feature_columns),
        "hn_n_cprime_backbone_manifold": any(
            _contains_token(column, HN_N_CPRIME_MANIFOLD_TOKENS)
            for column in manifold_feature_columns
        ),
        "bioemu_context": any(
            _contains_token(column, BIOEMU_CONTEXT_TOKENS)
            for column in lowered
        ),
        "residual_route_context": any(
            _contains_token(column, ROUTE_CONTEXT_TOKENS)
            for column in lowered
        ),
        "bounded_pairwise_interactions": bool(interaction_pairs),
        "geometry_bioemu_interactions": any(
            ("geometry" in left_roles and "bioemu" in right_roles)
            or ("bioemu" in left_roles and "geometry" in right_roles)
            for left_roles, right_roles in interaction_roles
        ),
        "manifold_bioemu_interactions": any(
            ("manifold" in left_roles and "bioemu" in right_roles)
            or ("bioemu" in left_roles and "manifold" in right_roles)
            for left_roles, right_roles in interaction_roles
        ),
        "manifold_reference_interactions": any(
            ("manifold" in left_roles and "reference" in right_roles)
            or ("reference" in left_roles and "manifold" in right_roles)
            for left_roles, right_roles in interaction_roles
        ),
        "route_bioemu_interactions": any(
            ("route" in left_roles and "bioemu" in right_roles)
            or ("bioemu" in left_roles and "route" in right_roles)
            for left_roles, right_roles in interaction_roles
        ),
        "route_manifold_interactions": any(
            ("route" in left_roles and "manifold" in right_roles)
            or ("manifold" in left_roles and "route" in right_roles)
            for left_roles, right_roles in interaction_roles
        ),
        "route_geometry_interactions": any(
            ("route" in left_roles and "geometry" in right_roles)
            or ("geometry" in left_roles and "route" in right_roles)
            for left_roles, right_roles in interaction_roles
        ),
        "uncertainty_head": True,
        "ood_head": True,
        "reward_safety_gate": True,
    }


def _contains_token(value: str, tokens: Sequence[str]) -> bool:
    lowered = str(value).lower()
    return any(str(token).lower() in lowered for token in tokens)


def _columns_with_tokens(
    columns: Sequence[str],
    tokens: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in columns
        if _contains_token(str(column), tokens)
    )


def _interaction_pairs_with_role(
    pairs: Sequence[tuple[str, str]],
    *,
    role: str,
) -> tuple[tuple[str, str], ...]:
    selected: list[tuple[str, str]] = []
    for left, right in pairs:
        if role in _numeric_feature_roles(left) or role in _numeric_feature_roles(right):
            selected.append((str(left), str(right)))
    return tuple(selected)


def _feature_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, ...]:
    support_scoped = (
        "support_id" in left.columns
        and "support_id" in right.columns
        and left["support_id"].notna().any()
        and right["support_id"].notna().any()
    )
    if left["target_id"].notna().any() and right["target_id"].notna().any():
        if support_scoped:
            return ("entity_uid", "target_id", "support_id")
        return ("entity_uid", "target_id")
    if left["seq_id"].notna().any() and right["seq_id"].notna().any():
        if support_scoped:
            return ("entity_uid", "seq_id", "atom_family", "support_id")
        return ("entity_uid", "seq_id", "atom_family")
    raise ValueError("features must share target_id or seq_id/atom_family with targets")


def _entity_uid_series(frame: pd.DataFrame) -> pd.Series:
    values = _column_or_na(frame, ("entity_uid", "entry_uid", "bmrb_uid"))
    cleaned = values.astype("string").str.strip()
    if cleaned.notna().any():
        return cleaned
    for alias in ("bmrb_id", "native_id", "entry_id"):
        if alias in frame.columns:
            fallback = frame[alias].astype("string").str.strip()
            return fallback.map(
                lambda value: (
                    value
                    if pd.isna(value) or str(value).startswith("bmrb:")
                    else f"bmrb:{value}"
                )
            )
    raise ValueError("table is missing entity_uid/entry_uid/bmrb_uid")


def _string_column(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    return _column_or_na(frame, aliases).astype("string").str.strip()


def _column_or_na(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return frame[alias]
    return pd.Series(pd.NA, index=frame.index)


def _family_model_to_dict(model: _RidgeFamilyModel) -> dict[str, Any]:
    return {
        "family": str(model.family),
        "coefficients": list(model.coefficients),
        "residual_rmse": float(model.residual_rmse),
        "ood_reference": float(model.ood_reference),
        "train_rows": int(model.train_rows),
        "family_blend_weight": float(model.family_blend_weight),
        "calibration_slope": float(model.calibration_slope),
        "calibration_intercept": float(model.calibration_intercept),
        "calibration_blend_weight": float(model.calibration_blend_weight),
        "calibration_rows": int(model.calibration_rows),
    }


def _family_model_from_dict(payload: dict[str, Any]) -> _RidgeFamilyModel:
    return _RidgeFamilyModel(
        family=str(payload["family"]),
        coefficients=tuple(map(float, payload.get("coefficients", []))),
        residual_rmse=float(payload.get("residual_rmse", 1.0)),
        ood_reference=float(payload.get("ood_reference", 1.0)),
        train_rows=int(payload.get("train_rows", 0)),
        family_blend_weight=float(payload.get("family_blend_weight", 1.0)),
        calibration_slope=float(payload.get("calibration_slope", 1.0)),
        calibration_intercept=float(payload.get("calibration_intercept", 0.0)),
        calibration_blend_weight=float(payload.get("calibration_blend_weight", 0.0)),
        calibration_rows=int(payload.get("calibration_rows", 0)),
    )


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix in {".tsv", ".tab"}:
        frame.to_csv(path, sep="\t", index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    elif suffix == ".json":
        frame.to_json(path, orient="records", indent=2)
    else:
        frame.to_csv(path, index=False)
