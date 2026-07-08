"""Prior-tethered low-rank score adapter training for BioEmu denoising.

The adapter learned here is deliberately small: it distills offline teacher
signals into a low-rank residual score field that can be loaded by a BioEmu
sampler wrapper.  It is not itself final evidence of success.  Final acceptance
still requires generated conformers, offline UCBShift, and one shared-q CCC
audits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ArrayLike = np.ndarray | list[float] | list[list[float]]
RUNTIME_POSITION_DELTA_SEMANTICS = "position_delta_nm"
RUNTIME_POSITION_DELTA_MODE = "score_model_constant_position_delta_nm"
RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS = "residue_position_delta_nm"
RUNTIME_RESIDUE_POSITION_DELTA_MODE = "score_model_residue_position_delta_nm"
RUNTIME_RESIDUE_ONEHOT_POSITION_DELTA_MODE = (
    "score_model_residue_onehot_position_delta_nm"
)
RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_MODE = (
    "score_model_residue_local_patch_position_delta_nm"
)
RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_MODE = (
    "score_model_residue_response_geometry_position_delta_nm"
)
RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_MODE = (
    "score_model_residue_multifamily_patch_position_delta_nm"
)
RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_MODE = (
    "score_model_residue_pair_conditioned_local_patch_position_delta_nm"
)
RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_PROTOTYPE_POSITION_DELTA_MODE = (
    "score_model_residue_pair_conditioned_local_patch_prototype_position_delta_nm"
)
RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_ATTENTION_POSITION_DELTA_MODE = (
    "score_model_residue_pair_conditioned_local_patch_attention_position_delta_nm"
)
RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_MODE = (
    "score_model_residue_pair_conditioned_multifamily_patch_position_delta_nm"
)
RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_ATTENTION_POSITION_DELTA_MODE = (
    "score_model_residue_pair_conditioned_multifamily_patch_attention_position_delta_nm"
)
RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_MODE = (
    "score_model_residue_pair_conditioned_multifamily_cross_position_delta_nm"
)
RUNTIME_RESIDUE_POSITION_DELTA_FEATURE_DIM = 3
RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM = 66
RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_FEATURE_DIM = 85
RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM = 117
RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM = 96
RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM = 147
RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_FEATURE_DIM = 1154


@dataclass(frozen=True, slots=True)
class BioEmuScoreAdapterLoRAConfig:
    """Configuration for a conservative low-rank score residual adapter."""

    rank: int = 4
    ridge_lambda: float = 1.0e-3
    adapter_norm_tether: float = 1.0e-4
    center_features: bool = True
    fit_bias: bool = True
    target_scale: float = 1.0
    min_samples: int = 4
    base_bioemu_checkpoint_path: str = ""
    base_bioemu_model_config_path: str = ""
    teacher_bundle_path: str = ""
    runtime_conditioning_path: str = ""
    target_shared_q_ccc: float = 0.95
    target_semantics: str = "posterior_energy_residual"
    runtime_application_mode: str = "offline_distillation_only"

    def validated(self) -> "BioEmuScoreAdapterLoRAConfig":
        """Return a validated copy with numerically safe values."""

        if int(self.rank) <= 0:
            raise ValueError("rank must be positive")
        if float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge_lambda must be non-negative")
        if float(self.adapter_norm_tether) < 0.0:
            raise ValueError("adapter_norm_tether must be non-negative")
        if float(self.target_scale) <= 0.0:
            raise ValueError("target_scale must be positive")
        if int(self.min_samples) <= 0:
            raise ValueError("min_samples must be positive")
        if not str(self.target_semantics).strip():
            raise ValueError("target_semantics must be non-empty")
        if not str(self.runtime_application_mode).strip():
            raise ValueError("runtime_application_mode must be non-empty")
        return self


@dataclass(frozen=True, slots=True)
class BioEmuScoreAdapterLoRACheckpoint:
    """Serializable low-rank score residual adapter."""

    adapter_down: list[list[float]]
    adapter_up: list[list[float]]
    bias: list[float]
    feature_mean: list[float]
    rank: int
    input_dim: int
    output_dim: int
    train_mse: float
    baseline_mse: float
    explained_mse_fraction: float
    adapter_frobenius_norm: float
    sample_count: int
    config: dict[str, Any]

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Apply the score residual adapter to feature rows."""

        x = _as_2d_float_array(features, name="features")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} feature columns, got {x.shape[1]}"
            )
        centered = x - np.asarray(self.feature_mean, dtype=np.float64)[None, :]
        down = np.asarray(self.adapter_down, dtype=np.float64)
        up = np.asarray(self.adapter_up, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        return centered @ down @ up + bias[None, :]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""

        return {
            "checkpoint_kind": "bioemu_score_adapter_lora_v1",
            "model_class": "BioEmuScoreAdapterLoRACheckpoint",
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BioEmuScoreAdapterLoRACheckpoint":
        """Deserialize from a dictionary written by :meth:`to_dict`."""

        return cls(
            adapter_down=[list(map(float, row)) for row in payload["adapter_down"]],
            adapter_up=[list(map(float, row)) for row in payload["adapter_up"]],
            bias=list(map(float, payload["bias"])),
            feature_mean=list(map(float, payload["feature_mean"])),
            rank=int(payload["rank"]),
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            train_mse=float(payload["train_mse"]),
            baseline_mse=float(payload["baseline_mse"]),
            explained_mse_fraction=float(payload["explained_mse_fraction"]),
            adapter_frobenius_norm=float(payload["adapter_frobenius_norm"]),
            sample_count=int(payload["sample_count"]),
            config=dict(payload.get("config", {})),
        )


class BioEmuScoreAdapterLoRATrainer:
    """Fit a low-rank residual score adapter from offline teacher targets."""

    def __init__(self, config: BioEmuScoreAdapterLoRAConfig | None = None) -> None:
        self.config = (config or BioEmuScoreAdapterLoRAConfig()).validated()

    def fit(
        self,
        features: ArrayLike,
        target_delta: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> BioEmuScoreAdapterLoRACheckpoint:
        """Fit a ridge-tethered low-rank adapter.

        ``target_delta`` should be a teacher residual score or coordinate-score
        delta derived from offline UCBShift/shared-q promotion evidence.  The
        adapter uses one shared support-level signal; atom-specific q shortcuts
        should be resolved before creating these targets.
        """

        x = _as_2d_float_array(features, name="features")
        y = _as_2d_float_array(target_delta, name="target_delta")
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and target_delta must have the same row count")
        if x.shape[0] < int(self.config.min_samples):
            raise ValueError(
                f"need at least {self.config.min_samples} samples, got {x.shape[0]}"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("features and target_delta must be finite")

        y = y / float(self.config.target_scale)
        weights = _sample_weights(sample_weight, x.shape[0])
        feature_mean = (
            np.average(x, axis=0, weights=weights)
            if bool(self.config.center_features)
            else np.zeros(x.shape[1], dtype=np.float64)
        )
        x_centered = x - feature_mean[None, :]
        bias = (
            np.average(y, axis=0, weights=weights)
            if bool(self.config.fit_bias)
            else np.zeros(y.shape[1], dtype=np.float64)
        )
        y_centered = y - bias[None, :]

        weighted_x = x_centered * np.sqrt(weights)[:, None]
        weighted_y = y_centered * np.sqrt(weights)[:, None]
        penalty = float(self.config.ridge_lambda) + float(
            self.config.adapter_norm_tether
        )
        xtx = weighted_x.T @ weighted_x
        if penalty > 0.0:
            xtx = xtx + np.eye(xtx.shape[0], dtype=np.float64) * penalty
        full_w = np.linalg.pinv(xtx) @ weighted_x.T @ weighted_y

        rank = min(int(self.config.rank), full_w.shape[0], full_w.shape[1])
        u, singular_values, vt = np.linalg.svd(full_w, full_matrices=False)
        root_s = np.sqrt(singular_values[:rank])
        adapter_down = u[:, :rank] * root_s[None, :]
        adapter_up = root_s[:, None] * vt[:rank, :]

        prediction = x_centered @ adapter_down @ adapter_up + bias[None, :]
        residual = prediction - y
        baseline = bias[None, :] - y
        train_mse = float(np.average(np.mean(residual * residual, axis=1), weights=weights))
        baseline_mse = float(
            np.average(np.mean(baseline * baseline, axis=1), weights=weights)
        )
        explained = 0.0
        if baseline_mse > 0.0:
            explained = max(0.0, min(1.0, 1.0 - train_mse / baseline_mse))
        adapter_norm = float(
            np.linalg.norm(adapter_down @ adapter_up, ord="fro")
        )

        return BioEmuScoreAdapterLoRACheckpoint(
            adapter_down=adapter_down.astype(float).tolist(),
            adapter_up=adapter_up.astype(float).tolist(),
            bias=bias.astype(float).tolist(),
            feature_mean=feature_mean.astype(float).tolist(),
            rank=rank,
            input_dim=int(x.shape[1]),
            output_dim=int(y.shape[1]),
            train_mse=train_mse,
            baseline_mse=baseline_mse,
            explained_mse_fraction=float(explained),
            adapter_frobenius_norm=adapter_norm,
            sample_count=int(x.shape[0]),
            config=asdict(self.config),
        )


def save_checkpoint(
    checkpoint: BioEmuScoreAdapterLoRACheckpoint,
    path: str | Path,
) -> None:
    """Write a JSON checkpoint atomically enough for async handoff."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> BioEmuScoreAdapterLoRACheckpoint:
    """Load a JSON checkpoint written by :func:`save_checkpoint`."""

    payload = json.loads(Path(path).read_text())
    if payload.get("checkpoint_kind") != "bioemu_score_adapter_lora_v1":
        raise ValueError("not a bioemu_score_adapter_lora_v1 checkpoint")
    return BioEmuScoreAdapterLoRACheckpoint.from_dict(payload)


def train_from_npz(
    npz_path: str | Path,
    *,
    output_checkpoint_path: str | Path,
    config: BioEmuScoreAdapterLoRAConfig | None = None,
) -> BioEmuScoreAdapterLoRACheckpoint:
    """Train from an ``.npz`` file with features and target_delta arrays."""

    data = np.load(npz_path)
    if "features" not in data or "target_delta" not in data:
        raise ValueError("npz must contain 'features' and 'target_delta' arrays")
    sample_weight = data["sample_weight"] if "sample_weight" in data else None
    checkpoint = BioEmuScoreAdapterLoRATrainer(config).fit(
        data["features"],
        data["target_delta"],
        sample_weight=sample_weight,
    )
    save_checkpoint(checkpoint, output_checkpoint_path)
    return checkpoint


def structural_generator_metadata(
    *,
    checkpoint_path: str | Path,
    checkpoint: BioEmuScoreAdapterLoRACheckpoint,
    mark_online_generator: bool = False,
) -> dict[str, Any]:
    """Return READY-compatible metadata for downstream BioEmu samplers.

    ``mark_online_generator`` should only be true after the BioEmu denoising
    wrapper actually loads this adapter and emits conformers from it.  Keeping
    this explicit avoids repeating the x0-as-generator mistake.
    """

    config = dict(checkpoint.config)
    base_checkpoint = str(config.get("base_bioemu_checkpoint_path") or "")
    if mark_online_generator and not base_checkpoint:
        raise ValueError(
            "mark_online_generator=True requires base_bioemu_checkpoint_path"
        )
    runtime_contract = score_adapter_runtime_contract(checkpoint)
    if mark_online_generator and not bool(runtime_contract["runtime_applicable"]):
        raise ValueError(
            "mark_online_generator=True requires a runtime-applicable score "
            f"adapter: {runtime_contract['reason']}"
        )
    return {
        "structural_generator_checkpoint_path": str(checkpoint_path),
        "structural_generator_model_config_path": str(
            config.get("base_bioemu_model_config_path") or ""
        ),
        "structural_generator_checkpoint_is_online_generator": bool(
            mark_online_generator
        ),
        "bioemu_score_checkpoint_path": str(checkpoint_path),
        "bioemu_score_model_config_path": str(
            config.get("base_bioemu_model_config_path") or ""
        ),
        "bioemu_score_checkpoint_is_online_generator": bool(mark_online_generator),
        "bioemu_score_adapter_checkpoint_path": str(checkpoint_path),
        "bioemu_score_adapter_checkpoint_kind": "bioemu_score_adapter_lora_v1",
        "bioemu_score_adapter_target_semantics": str(
            config.get("target_semantics") or ""
        ),
        "bioemu_score_adapter_runtime_application_mode": str(
            config.get("runtime_application_mode") or ""
        ),
        "bioemu_score_adapter_runtime_applicable": bool(
            runtime_contract["runtime_applicable"]
        ),
        "bioemu_score_adapter_runtime_contract_reason": str(
            runtime_contract["reason"]
        ),
        "bioemu_score_adapter_rank": int(checkpoint.rank),
        "bioemu_score_adapter_train_mse": float(checkpoint.train_mse),
        "bioemu_score_adapter_explained_mse_fraction": float(
            checkpoint.explained_mse_fraction
        ),
        "bioemu_score_adapter_prior_tethered": True,
        "base_bioemu_checkpoint_path": base_checkpoint,
        "teacher_bundle_path": str(config.get("teacher_bundle_path") or ""),
        "bioemu_score_adapter_runtime_conditioning_path": str(
            config.get("runtime_conditioning_path") or ""
        ),
    }


def save_structural_generator_metadata(
    *,
    checkpoint_path: str | Path,
    checkpoint: BioEmuScoreAdapterLoRACheckpoint,
    output_path: str | Path,
    mark_online_generator: bool = False,
) -> dict[str, Any]:
    """Write READY-compatible score adapter metadata."""

    payload = structural_generator_metadata(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        mark_online_generator=mark_online_generator,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(output)
    return payload


def score_adapter_runtime_contract(
    checkpoint: BioEmuScoreAdapterLoRACheckpoint,
) -> dict[str, Any]:
    """Return whether a checkpoint is safe to apply inside BioEmu denoising.

    Energy-residual adapters are valid distillation artifacts, but they are not
    coordinate score fields.  Runtime application is therefore deliberately
    narrow until richer feature providers are implemented.
    """

    config = dict(checkpoint.config)
    semantics = str(config.get("target_semantics") or "").strip()
    mode = str(config.get("runtime_application_mode") or "").strip()
    constant_applicable = (
        semantics == RUNTIME_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_POSITION_DELTA_MODE
        and int(checkpoint.input_dim) == 1
        and int(checkpoint.output_dim) == 3
    )
    residue_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_POSITION_DELTA_MODE
        and int(checkpoint.input_dim) == RUNTIME_RESIDUE_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    residue_onehot_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_ONEHOT_POSITION_DELTA_MODE
        and int(checkpoint.input_dim) > 0
        and int(checkpoint.output_dim) == 3
    )
    residue_local_patch_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        == RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    residue_response_geometry_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        == RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    residue_multifamily_patch_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        == RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    residue_pair_conditioned_multifamily_patch_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        in {
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_ATTENTION_POSITION_DELTA_MODE,
        }
        and int(checkpoint.input_dim)
        == RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    residue_pair_conditioned_local_patch_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        in {
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_PROTOTYPE_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_ATTENTION_POSITION_DELTA_MODE,
        }
        and int(checkpoint.input_dim)
        == RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    residue_pair_conditioned_multifamily_cross_applicable = (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        == RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        == RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_FEATURE_DIM
        and int(checkpoint.output_dim) == 3
    )
    applicable = (
        constant_applicable
        or residue_applicable
        or residue_onehot_applicable
        or residue_local_patch_applicable
        or residue_response_geometry_applicable
        or residue_multifamily_patch_applicable
        or residue_pair_conditioned_local_patch_applicable
        or residue_pair_conditioned_multifamily_patch_applicable
        or residue_pair_conditioned_multifamily_cross_applicable
    )
    if constant_applicable:
        reason = "runtime_position_delta_adapter_supported"
    elif residue_applicable:
        reason = "runtime_residue_position_delta_adapter_supported"
    elif residue_onehot_applicable:
        reason = "runtime_residue_onehot_position_delta_adapter_supported"
    elif residue_local_patch_applicable:
        reason = "runtime_residue_local_patch_position_delta_adapter_supported"
    elif residue_response_geometry_applicable:
        reason = "runtime_residue_response_geometry_position_delta_adapter_supported"
    elif residue_multifamily_patch_applicable:
        reason = "runtime_residue_multifamily_patch_position_delta_adapter_supported"
    elif residue_pair_conditioned_local_patch_applicable:
        if mode == RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_PROTOTYPE_POSITION_DELTA_MODE:
            reason = (
                "runtime_residue_pair_conditioned_local_patch_prototype_"
                "position_delta_adapter_supported"
            )
        elif mode == RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_ATTENTION_POSITION_DELTA_MODE:
            reason = (
                "runtime_residue_pair_conditioned_local_patch_attention_"
                "position_delta_adapter_supported"
            )
        else:
            reason = (
                "runtime_residue_pair_conditioned_local_patch_position_delta_"
                "adapter_supported"
            )
    elif residue_pair_conditioned_multifamily_patch_applicable:
        if (
            mode
            == RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_ATTENTION_POSITION_DELTA_MODE
        ):
            reason = (
                "runtime_residue_pair_conditioned_multifamily_patch_attention_"
                "position_delta_adapter_supported"
            )
        else:
            reason = (
                "runtime_residue_pair_conditioned_multifamily_patch_position_delta_"
                "adapter_supported"
            )
    elif residue_pair_conditioned_multifamily_cross_applicable:
        reason = (
            "runtime_residue_pair_conditioned_multifamily_cross_position_delta_"
            "adapter_supported"
        )
    elif semantics not in {
        RUNTIME_POSITION_DELTA_SEMANTICS,
        RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS,
    }:
        reason = (
            f"target_semantics={semantics or '<missing>'}"
            "_is_not_position_delta_nm_or_residue_position_delta_nm"
        )
    elif semantics == RUNTIME_POSITION_DELTA_SEMANTICS and mode != RUNTIME_POSITION_DELTA_MODE:
        reason = f"runtime_application_mode={mode or '<missing>'}_is_not_supported"
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        not in {
            RUNTIME_RESIDUE_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_ONEHOT_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_PROTOTYPE_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_ATTENTION_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_ATTENTION_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_MODE,
        }
    ):
        reason = f"runtime_application_mode={mode or '<missing>'}_is_not_supported"
    elif semantics == RUNTIME_POSITION_DELTA_SEMANTICS and int(checkpoint.input_dim) != 1:
        reason = f"input_dim={checkpoint.input_dim}_requires_feature_provider"
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_POSITION_DELTA_MODE
        and int(checkpoint.input_dim) != RUNTIME_RESIDUE_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_feature_dim_"
            f"{RUNTIME_RESIDUE_POSITION_DELTA_FEATURE_DIM}"
        )
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_ONEHOT_POSITION_DELTA_MODE
        and int(checkpoint.input_dim) <= 0
    ):
        reason = f"input_dim={checkpoint.input_dim}_must_be_positive_for_onehot"
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        != RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_local_patch_"
            f"feature_dim_{RUNTIME_RESIDUE_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM}"
        )
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        != RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_response_"
            f"geometry_feature_dim_{RUNTIME_RESIDUE_RESPONSE_GEOMETRY_POSITION_DELTA_FEATURE_DIM}"
        )
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode == RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        != RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_multifamily_"
            f"patch_feature_dim_{RUNTIME_RESIDUE_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM}"
        )
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        in {
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_PROTOTYPE_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_ATTENTION_POSITION_DELTA_MODE,
        }
        and int(checkpoint.input_dim)
        != RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_pair_"
            "conditioned_local_patch_feature_dim_"
            f"{RUNTIME_RESIDUE_PAIR_CONDITIONED_LOCAL_PATCH_POSITION_DELTA_FEATURE_DIM}"
        )
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        in {
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_MODE,
            RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_ATTENTION_POSITION_DELTA_MODE,
        }
        and int(checkpoint.input_dim)
        != RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_pair_"
            "conditioned_multifamily_patch_feature_dim_"
            f"{RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_PATCH_POSITION_DELTA_FEATURE_DIM}"
        )
    elif (
        semantics == RUNTIME_RESIDUE_POSITION_DELTA_SEMANTICS
        and mode
        == RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_MODE
        and int(checkpoint.input_dim)
        != RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_FEATURE_DIM
    ):
        reason = (
            f"input_dim={checkpoint.input_dim}_does_not_match_residue_pair_"
            "conditioned_multifamily_cross_feature_dim_"
            f"{RUNTIME_RESIDUE_PAIR_CONDITIONED_MULTIFAMILY_CROSS_POSITION_DELTA_FEATURE_DIM}"
        )
    elif int(checkpoint.output_dim) != 3:
        reason = f"output_dim={checkpoint.output_dim}_is_not_xyz_delta"
    else:
        reason = "unsupported_runtime_contract"
    return {
        "runtime_applicable": bool(applicable),
        "reason": reason,
        "target_semantics": semantics,
        "runtime_application_mode": mode,
        "input_dim": int(checkpoint.input_dim),
        "output_dim": int(checkpoint.output_dim),
    }


def _as_2d_float_array(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    return array


def _sample_weights(value: ArrayLike | None, row_count: int) -> np.ndarray:
    if value is None:
        return np.full(row_count, 1.0 / float(row_count), dtype=np.float64)
    weights = np.asarray(value, dtype=np.float64).reshape(-1)
    if weights.shape[0] != row_count:
        raise ValueError("sample_weight must have one value per row")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("sample_weight must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("sample_weight must have positive total mass")
    return weights / total
