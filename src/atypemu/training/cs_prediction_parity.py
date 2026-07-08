"""UCBShift-parity benchmarks for chemical-shift predictors.

This module evaluates a model-produced chemical-shift table against the same
BMRB target rows, split assignments, atom-family mapping, and Lin CCC metric
used by the AtypEmu UCBShift benchmark path.  It is intentionally independent
of UCBShift execution: callers provide predictions that are already materialized
as a table.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from atypemu.training.metrics import ATOM_FAMILY_MAP, _safe_ccc, _safe_pearson


CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES: tuple[str, ...] = (
    "HN",
    "N",
    "CA",
    "CB",
    "C'",
)

PREDICTION_VALUE_ALIASES: tuple[str, ...] = (
    "predicted_value",
    "prediction",
    "predicted_shift",
    "chemical_shift_prediction",
    "posterior_mean",
    "mean",
    "y_pred",
    "shift",
    "value",
)
TARGET_VALUE_ALIASES: tuple[str, ...] = (
    "target_value",
    "observed_value",
    "observed_shift",
    "experimental_shift",
    "chemical_shift",
    "value",
)
TARGET_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "support_id": (
        "support_id",
        "conformer_id",
        "candidate_id",
        "sample_id",
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
    "structure_path": ("structure_path", "pdb_path", "structure_file"),
}
ENTITY_UID_ALIASES: tuple[str, ...] = ("entity_uid", "entry_uid", "bmrb_uid")
TARGET_ID_ALIASES: tuple[str, ...] = ("target_id", "target_uid", "measurement_uid")
SEQ_ID_ALIASES: tuple[str, ...] = (
    "seq_id",
    "residue_index",
    "residue_id",
    "residue_number",
    "residue",
)
ATOM_ID_ALIASES: tuple[str, ...] = ("atom_id", "atom_name", "atom")
COMP_ID_ALIASES: tuple[str, ...] = ("comp_id", "residue_name", "resname", "aa")
SIGMA_ALIASES: tuple[str, ...] = (
    "target_sigma",
    "uncertainty",
    "sigma",
    "experimental_uncertainty",
)
PREDICTION_UNCERTAINTY_ALIASES: tuple[str, ...] = (
    "prediction_uncertainty",
    "predicted_uncertainty",
    "uncertainty",
    "sigma",
    "prediction_sigma",
    "cs_predictor_uncertainty",
)
PREDICTION_OOD_SCORE_ALIASES: tuple[str, ...] = (
    "prediction_ood_score",
    "ood_score",
    "generated_support_ood_score",
    "cs_predictor_ood_score",
)
PREDICTION_OOD_FLAG_ALIASES: tuple[str, ...] = (
    "prediction_ood_flag",
    "ood_flag",
    "is_ood",
    "generated_support_ood",
    "cs_predictor_ood",
)
PREDICTION_REWARD_ALIASES: tuple[str, ...] = (
    "prediction_reward",
    "training_reward",
    "reward",
    "posterior_weight",
    "q",
)


@dataclass(frozen=True, slots=True)
class ChemicalShiftParityConfig:
    """Configuration for one UCBShift-parity chemical-shift benchmark."""

    target_ccc: float = 0.95
    min_family_count: int = 2
    gate_split: str = "val"
    required_atom_families: tuple[str, ...] = CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES
    join_keys: tuple[str, ...] | None = None
    aggregate_duplicates: bool = True
    prediction_label: str = "chemical_shift_predictor"
    target_label: str = "bmrb_targets"
    require_uncertainty: bool = False
    require_ood: bool = False
    uncertainty_threshold: float | None = None
    ood_score_threshold: float | None = None
    max_ood_fraction: float = 0.0
    max_high_uncertainty_fraction: float = 0.0
    positive_reward_threshold: float = 0.0
    max_high_uncertainty_positive_reward_fraction: float = 0.0


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from parquet, CSV, TSV, JSON, or JSONL."""

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(table_path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(table_path, lines=True)
    if suffix == ".json":
        return pd.read_json(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(
        f"Unsupported table suffix for {table_path}. "
        "Expected parquet, csv, tsv, json, or jsonl."
    )


def evaluate_chemical_shift_prediction_parity(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    config: ChemicalShiftParityConfig | None = None,
    prediction_value_column: str | None = None,
    target_value_column: str | None = None,
) -> dict[str, Any]:
    """Evaluate predictions against targets using AtypEmu CS benchmark metrics."""

    resolved_config = config or ChemicalShiftParityConfig()
    pairs = build_prediction_target_pairs(
        predictions=predictions,
        targets=targets,
        splits=splits,
        config=resolved_config,
        prediction_value_column=prediction_value_column,
        target_value_column=target_value_column,
    )
    metrics = chemical_shift_metric_frame(pairs, config=resolved_config)
    summary = summarize_chemical_shift_parity(
        pairs=pairs,
        metrics=metrics,
        config=resolved_config,
    )
    return {"summary": summary, "metrics": metrics, "pairs": pairs}


def build_prediction_target_pairs(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    config: ChemicalShiftParityConfig | None = None,
    prediction_value_column: str | None = None,
    target_value_column: str | None = None,
) -> pd.DataFrame:
    """Join normalized prediction and target rows on the strictest shared key."""

    resolved_config = config or ChemicalShiftParityConfig()
    prediction_frame = normalize_prediction_frame(
        predictions,
        value_column=prediction_value_column,
    )
    target_frame = normalize_target_frame(targets, value_column=target_value_column)
    join_keys = resolve_join_keys(
        prediction_frame=prediction_frame,
        target_frame=target_frame,
        requested_join_keys=resolved_config.join_keys,
    )
    prediction_frame = _prepare_join_frame(
        prediction_frame,
        value_column="predicted_value",
        join_keys=join_keys,
        aggregate_duplicates=resolved_config.aggregate_duplicates,
    )
    target_frame = _prepare_join_frame(
        target_frame,
        value_column="target_value",
        join_keys=join_keys,
        aggregate_duplicates=resolved_config.aggregate_duplicates,
    )
    joined = prediction_frame.merge(
        target_frame,
        on=list(join_keys),
        how="inner",
        suffixes=("_prediction", "_target"),
    )
    if joined.empty:
        raise ValueError(
            "No prediction-target pairs joined. Check entity_uid, target_id, "
            "seq_id, and atom_id/atom_family alignment."
        )
    joined = _finalize_joined_pairs(joined, join_keys=join_keys)
    if splits is not None:
        joined = _attach_split_frame(joined, splits)
    joined["split"] = joined["split"].fillna("all").astype(str)
    joined = joined[
        np.isfinite(joined["predicted_value"].to_numpy(dtype=float))
        & np.isfinite(joined["target_value"].to_numpy(dtype=float))
    ].copy()
    joined = joined.sort_values(
        ["split", "entity_uid", "seq_id", "target_id", "atom_family"],
        na_position="last",
    ).reset_index(drop=True)
    return joined


def normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    value_column: str | None = None,
) -> pd.DataFrame:
    """Normalize a chemical-shift prediction table to the parity schema."""

    normalized = _normalize_common_columns(frame)
    value_series, _ = _column_from_aliases(
        frame,
        (value_column,) if value_column else PREDICTION_VALUE_ALIASES,
        required=True,
        label="prediction value",
    )
    normalized["predicted_value"] = pd.to_numeric(value_series, errors="coerce")
    uncertainty_series, _ = _column_from_aliases(
        frame,
        PREDICTION_UNCERTAINTY_ALIASES,
        required=False,
        label="prediction uncertainty",
    )
    normalized["prediction_uncertainty"] = pd.to_numeric(
        uncertainty_series,
        errors="coerce",
    )
    ood_score_series, _ = _column_from_aliases(
        frame,
        PREDICTION_OOD_SCORE_ALIASES,
        required=False,
        label="prediction OOD score",
    )
    normalized["prediction_ood_score"] = pd.to_numeric(
        ood_score_series,
        errors="coerce",
    )
    ood_flag_series, _ = _column_from_aliases(
        frame,
        PREDICTION_OOD_FLAG_ALIASES,
        required=False,
        label="prediction OOD flag",
    )
    normalized["prediction_ood_flag"] = _boolean_series(ood_flag_series)
    reward_series, _ = _column_from_aliases(
        frame,
        PREDICTION_REWARD_ALIASES,
        required=False,
        label="prediction reward",
    )
    normalized["prediction_reward"] = pd.to_numeric(reward_series, errors="coerce")
    normalized = normalized.dropna(
        subset=["entity_uid", "atom_family", "predicted_value"]
    )
    return normalized.reset_index(drop=True)


def normalize_target_frame(
    frame: pd.DataFrame,
    *,
    value_column: str | None = None,
) -> pd.DataFrame:
    """Normalize a BMRB or benchmark target table to the parity schema."""

    target_source = frame
    if "measurement_kind" in target_source.columns:
        target_source = target_source[
            target_source["measurement_kind"].astype(str) == "chemical_shift"
        ].copy()
    normalized = _normalize_common_columns(target_source)
    value_series, _ = _column_from_aliases(
        target_source,
        (value_column,) if value_column else TARGET_VALUE_ALIASES,
        required=True,
        label="target value",
    )
    normalized["target_value"] = pd.to_numeric(value_series, errors="coerce")
    sigma_series, _ = _column_from_aliases(
        target_source,
        SIGMA_ALIASES,
        required=False,
        label="target sigma",
    )
    normalized["target_sigma"] = pd.to_numeric(sigma_series, errors="coerce")
    for output_column, aliases in TARGET_METADATA_ALIASES.items():
        values, source_column = _column_from_aliases(
            target_source,
            aliases,
            required=False,
            label=output_column,
        )
        if source_column is None:
            continue
        if output_column in {"sample_index", "dense_sample_index", "candidate_index"}:
            normalized[output_column] = pd.to_numeric(values, errors="coerce")
        else:
            normalized[output_column] = _clean_string_series(values)
    normalized = normalized.dropna(subset=["entity_uid", "atom_family", "target_value"])
    return normalized.reset_index(drop=True)


def resolve_join_keys(
    *,
    prediction_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    requested_join_keys: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve the join key used for prediction-target pairing."""

    if requested_join_keys:
        join_keys = tuple(str(key) for key in requested_join_keys)
        missing = [
            key
            for key in join_keys
            if key not in prediction_frame.columns or key not in target_frame.columns
        ]
        if missing:
            raise ValueError(f"Requested join keys are unavailable: {missing}")
        return join_keys

    if _has_non_null(prediction_frame, "target_id") and _has_non_null(
        target_frame,
        "target_id",
    ):
        return ("entity_uid", "target_id")
    if _has_non_null(prediction_frame, "seq_id") and _has_non_null(
        target_frame,
        "seq_id",
    ):
        return ("entity_uid", "seq_id", "atom_family")
    raise ValueError(
        "Could not resolve join keys. Provide target_id on both tables or "
        "seq_id plus atom_id/atom_family on both tables."
    )


def chemical_shift_metric_frame(
    pairs: pd.DataFrame,
    *,
    config: ChemicalShiftParityConfig | None = None,
) -> pd.DataFrame:
    """Compute overall and atom-family chemical-shift benchmark metrics."""

    resolved_config = config or ChemicalShiftParityConfig()
    split_values = ["all"]
    split_values.extend(
        split
        for split in sorted(pairs["split"].dropna().astype(str).unique())
        if split != "all"
    )
    rows: list[dict[str, Any]] = []
    for split in split_values:
        split_frame = pairs if split == "all" else pairs[pairs["split"] == split]
        if split_frame.empty:
            continue
        rows.append(_metric_row(split_frame, split=split, atom_family="all"))
        for family in resolved_config.required_atom_families:
            family_frame = split_frame[split_frame["atom_family"] == family]
            if family_frame.empty:
                continue
            rows.append(_metric_row(family_frame, split=split, atom_family=family))
    return pd.DataFrame(rows)


def summarize_chemical_shift_parity(
    *,
    pairs: pd.DataFrame,
    metrics: pd.DataFrame,
    config: ChemicalShiftParityConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable parity gate summary."""

    resolved_config = config or ChemicalShiftParityConfig()
    available_splits = set(metrics["split"].astype(str)) if not metrics.empty else set()
    gate_split = (
        resolved_config.gate_split
        if resolved_config.gate_split in available_splits
        else "all"
    )
    split_summaries = {
        split: _split_summary(
            metrics=metrics,
            split=split,
            config=resolved_config,
        )
        for split in sorted(available_splits)
    }
    gate_summary = split_summaries.get(gate_split, {})
    passes_gate = bool(gate_summary.get("passes_family_ccc_gate", False))
    reliability = _predictor_reliability_summary(
        pairs=pairs,
        gate_split=gate_split,
        config=resolved_config,
    )
    reliability_passed = bool(reliability.get("passes_predictor_reliability_gate"))
    summary = {
        "decision": "pass" if passes_gate and reliability_passed else "fail",
        "passes_family_ccc_gate": passes_gate,
        "predictor_reliability_gate_passed": reliability_passed,
        "gate_split": gate_split,
        "target_ccc": float(resolved_config.target_ccc),
        "min_family_count": int(resolved_config.min_family_count),
        "required_atom_families": list(resolved_config.required_atom_families),
        "prediction_label": resolved_config.prediction_label,
        "target_label": resolved_config.target_label,
        "joined_pair_count": int(len(pairs)),
        "entity_count": int(pairs["entity_uid"].nunique()),
        "split_summaries": split_summaries,
        "predictor_reliability": reliability,
        "data_signature": _frame_signature(
            pairs,
            [
                "entity_uid",
                "split",
                "target_id",
                "seq_id",
                "atom_family",
                "predicted_value",
                "target_value",
                "prediction_uncertainty",
                "prediction_ood_score",
                "prediction_ood_flag",
                "prediction_reward",
            ],
        ),
    }
    return _json_safe(summary)


def write_parity_outputs(
    *,
    result: dict[str, Any],
    summary_path: str | Path,
    metrics_path: str | Path | None = None,
    pairs_path: str | Path | None = None,
) -> None:
    """Write benchmark summary, metric rows, and optional joined pairs."""

    import json

    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(_json_safe(result["summary"]), indent=2, sort_keys=True) + "\n"
    )
    if metrics_path is not None:
        _write_table(result["metrics"], Path(metrics_path))
    if pairs_path is not None:
        _write_table(result["pairs"], Path(pairs_path))


def _normalize_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    normalized["entity_uid"] = _entity_uid_series(frame)
    target_id, _ = _column_from_aliases(
        frame,
        TARGET_ID_ALIASES,
        required=False,
        label="target id",
    )
    normalized["target_id"] = _clean_string_series(target_id)
    seq_id, _ = _column_from_aliases(
        frame,
        SEQ_ID_ALIASES,
        required=False,
        label="seq id",
    )
    normalized["seq_id"] = pd.to_numeric(seq_id, errors="coerce")
    comp_id, _ = _column_from_aliases(
        frame,
        COMP_ID_ALIASES,
        required=False,
        label="residue name",
    )
    normalized["comp_id"] = _clean_string_series(comp_id).str.upper()
    atom_id, _ = _column_from_aliases(
        frame,
        ATOM_ID_ALIASES,
        required=False,
        label="atom id",
    )
    normalized["atom_id"] = _clean_string_series(atom_id).str.upper()
    family, _ = _column_from_aliases(
        frame,
        ("atom_family", "family"),
        required=False,
        label="atom family",
    )
    normalized["atom_family"] = _canonical_family_series(family)
    normalized["atom_family"] = normalized["atom_family"].fillna(
        _clean_string_series(target_id).map(_family_from_target_id)
    )
    normalized["atom_family"] = normalized["atom_family"].fillna(
        normalized["atom_id"].map(canonical_atom_family)
    )
    split, _ = _column_from_aliases(
        frame,
        ("split", "dataset_split"),
        required=False,
        label="split",
    )
    normalized["split"] = _clean_string_series(split)
    return normalized


def _entity_uid_series(frame: pd.DataFrame) -> pd.Series:
    entity, _ = _column_from_aliases(
        frame,
        ENTITY_UID_ALIASES,
        required=False,
        label="entity uid",
    )
    entity = _clean_string_series(entity)
    if entity.notna().any():
        return entity
    for alias in ("bmrb_id", "native_id", "entry_id"):
        if alias in frame.columns:
            values = _clean_string_series(frame[alias])
            values = values.map(
                lambda value: (
                    value
                    if value is None or str(value).startswith("bmrb:")
                    else f"bmrb:{value}"
                )
            )
            if values.notna().any():
                return values
    raise ValueError(
        "Table is missing entity_uid/entry_uid, and no bmrb_id/native_id fallback "
        "was available."
    )


def _prepare_join_frame(
    frame: pd.DataFrame,
    *,
    value_column: str,
    join_keys: Sequence[str],
    aggregate_duplicates: bool,
) -> pd.DataFrame:
    join_frame = frame.dropna(subset=list(join_keys)).copy()
    if join_frame.empty:
        raise ValueError(f"No rows have complete join keys: {join_keys}")
    if not aggregate_duplicates and join_frame.duplicated(list(join_keys)).any():
        duplicate_count = int(join_frame.duplicated(list(join_keys)).sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate prediction-target join keys. "
            "Enable duplicate aggregation or make rows unique."
        )
    aggregation: dict[str, Any] = {value_column: "mean"}
    for column in [
        "target_sigma",
        "target_id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_family",
        "split",
        "prediction_uncertainty",
        "prediction_ood_score",
        "prediction_ood_flag",
        "prediction_reward",
    ]:
        if column in join_frame.columns and column not in join_keys:
            if column == "prediction_ood_flag":
                aggregation[column] = _any_truthy
            elif column == "prediction_ood_score":
                aggregation[column] = "max"
            elif column in {"prediction_uncertainty", "prediction_reward"}:
                aggregation[column] = "mean"
            else:
                aggregation[column] = _first_non_null
    return (
        join_frame.groupby(list(join_keys), dropna=False, as_index=False)
        .agg(aggregation)
        .reset_index(drop=True)
    )


def _finalize_joined_pairs(
    joined: pd.DataFrame,
    *,
    join_keys: Sequence[str],
) -> pd.DataFrame:
    result = pd.DataFrame(index=joined.index)
    result["entity_uid"] = joined["entity_uid"].astype(str)
    for column in ["target_id", "seq_id", "comp_id", "atom_id", "atom_family"]:
        result[column] = _coalesce_joined_column(joined, column, join_keys=join_keys)
    result["atom_family"] = _canonical_family_series(result["atom_family"])
    result["predicted_value"] = pd.to_numeric(joined["predicted_value"], errors="coerce")
    result["target_value"] = pd.to_numeric(joined["target_value"], errors="coerce")
    if "target_sigma" in joined.columns:
        result["target_sigma"] = pd.to_numeric(joined["target_sigma"], errors="coerce")
    else:
        result["target_sigma"] = math.nan
    for column in [
        "prediction_uncertainty",
        "prediction_ood_score",
        "prediction_reward",
    ]:
        result[column] = pd.to_numeric(
            _coalesce_joined_column(joined, column, join_keys=join_keys),
            errors="coerce",
        )
    result["prediction_ood_flag"] = _boolean_series(
        _coalesce_joined_column(joined, "prediction_ood_flag", join_keys=join_keys)
    )
    split = _coalesce_joined_column(joined, "split", join_keys=join_keys)
    result["split"] = _clean_string_series(split)
    return result


def _attach_split_frame(joined: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    if "split" not in splits.columns:
        return joined
    split_frame = splits.copy()
    split_frame["entity_uid"] = _entity_uid_series(split_frame)
    split_frame["split_from_file"] = _clean_string_series(split_frame["split"])
    split_frame = split_frame[["entity_uid", "split_from_file"]].drop_duplicates(
        "entity_uid"
    )
    merged = joined.merge(split_frame, on="entity_uid", how="left")
    merged["split"] = merged["split"].where(
        merged["split"].notna(),
        merged["split_from_file"],
    )
    return merged.drop(columns=["split_from_file"])


def _metric_row(frame: pd.DataFrame, *, split: str, atom_family: str) -> dict[str, Any]:
    predictions = frame["predicted_value"].to_numpy(dtype=np.float64)
    targets = frame["target_value"].to_numpy(dtype=np.float64)
    residual = predictions - targets
    return {
        "split": split,
        "atom_family": atom_family,
        "count": int(predictions.size),
        "entity_count": int(frame["entity_uid"].nunique()),
        "mae": _finite_mean(np.abs(residual)),
        "rmse": _finite_rmse(residual),
        "bias": _finite_mean(residual),
        "pearson": _safe_pearson(predictions, targets),
        "ccc": _safe_ccc(predictions, targets),
        "target_mean": _finite_mean(targets),
        "prediction_mean": _finite_mean(predictions),
    }


def _split_summary(
    *,
    metrics: pd.DataFrame,
    split: str,
    config: ChemicalShiftParityConfig,
) -> dict[str, Any]:
    split_metrics = metrics[metrics["split"].astype(str) == split]
    family_rows = {
        str(row["atom_family"]): row.to_dict()
        for _, row in split_metrics.iterrows()
        if str(row["atom_family"]) != "all"
    }
    per_family = {
        family: _json_safe(family_rows.get(family, {"atom_family": family, "count": 0}))
        for family in config.required_atom_families
    }
    missing_families = [
        family
        for family in config.required_atom_families
        if int(per_family[family].get("count") or 0) < config.min_family_count
    ]
    failing_families = []
    ccc_values: list[float] = []
    for family, row in per_family.items():
        count = int(row.get("count") or 0)
        if count < config.min_family_count:
            continue
        ccc = _safe_float(row.get("ccc"))
        if math.isfinite(ccc):
            ccc_values.append(ccc)
        if (not math.isfinite(ccc)) or ccc < config.target_ccc:
            failing_families.append(family)
    overall_rows = split_metrics[split_metrics["atom_family"].astype(str) == "all"]
    overall = overall_rows.iloc[0].to_dict() if not overall_rows.empty else {}
    return _json_safe(
        {
            "passes_family_ccc_gate": not missing_families and not failing_families,
            "missing_required_families": missing_families,
            "failing_required_families": failing_families,
            "family_macro_ccc": float(np.mean(ccc_values)) if ccc_values else math.nan,
            "family_min_ccc": float(np.min(ccc_values)) if ccc_values else math.nan,
            "families_passing": [
                family
                for family in config.required_atom_families
                if family not in missing_families and family not in failing_families
            ],
            "overall": overall,
            "per_family": per_family,
        }
    )


def _predictor_reliability_summary(
    *,
    pairs: pd.DataFrame,
    gate_split: str,
    config: ChemicalShiftParityConfig,
) -> dict[str, Any]:
    gate_pairs = pairs if gate_split == "all" else pairs[pairs["split"] == gate_split]
    gate_pairs = gate_pairs.copy()
    row_count = int(len(gate_pairs))
    reasons: list[str] = []
    if row_count <= 0:
        return {
            "passes_predictor_reliability_gate": False,
            "failed_reasons": ["no_gate_split_pairs_for_reliability_audit"],
            "gate_split": gate_split,
            "row_count": 0,
        }

    uncertainty = pd.to_numeric(
        gate_pairs.get("prediction_uncertainty", pd.Series(math.nan, index=gate_pairs.index)),
        errors="coerce",
    )
    uncertainty_finite = np.isfinite(uncertainty.to_numpy(dtype=float))
    uncertainty_coverage = float(np.mean(uncertainty_finite.astype(float)))
    uncertainty_threshold = _optional_finite_float(config.uncertainty_threshold)
    high_uncertainty = (
        uncertainty_finite & (uncertainty.to_numpy(dtype=float) > uncertainty_threshold)
        if uncertainty_threshold is not None
        else np.zeros(row_count, dtype=bool)
    )
    if config.require_uncertainty and uncertainty_coverage < 1.0:
        reasons.append("prediction_uncertainty_missing_or_incomplete")
    high_uncertainty_fraction = float(np.mean(high_uncertainty.astype(float)))
    if high_uncertainty_fraction > float(config.max_high_uncertainty_fraction):
        reasons.append("high_uncertainty_fraction_exceeds_gate")

    ood_score = pd.to_numeric(
        gate_pairs.get("prediction_ood_score", pd.Series(math.nan, index=gate_pairs.index)),
        errors="coerce",
    )
    ood_score_finite = np.isfinite(ood_score.to_numpy(dtype=float))
    ood_score_threshold = _optional_finite_float(config.ood_score_threshold)
    score_ood = (
        ood_score_finite & (ood_score.to_numpy(dtype=float) > ood_score_threshold)
        if ood_score_threshold is not None
        else np.zeros(row_count, dtype=bool)
    )
    ood_flag = _boolean_series(
        gate_pairs.get("prediction_ood_flag", pd.Series(pd.NA, index=gate_pairs.index))
    )
    ood_flag_available = bool(ood_flag.notna().any())
    ood_flag_bool = ood_flag.fillna(False).to_numpy(dtype=bool)
    ood_info_available = bool(ood_score_finite.any() or ood_flag_available)
    if config.require_ood and not ood_info_available:
        reasons.append("prediction_ood_signal_missing")
    ood_mask = score_ood | ood_flag_bool
    ood_fraction = float(np.mean(ood_mask.astype(float)))
    if ood_fraction > float(config.max_ood_fraction):
        reasons.append("ood_fraction_exceeds_gate")

    reward = pd.to_numeric(
        gate_pairs.get("prediction_reward", pd.Series(math.nan, index=gate_pairs.index)),
        errors="coerce",
    )
    reward_finite = np.isfinite(reward.to_numpy(dtype=float))
    positive_reward = (
        reward_finite
        & (reward.to_numpy(dtype=float) > float(config.positive_reward_threshold))
    )
    positive_reward_count = int(np.sum(positive_reward.astype(np.int64)))
    high_uncertainty_positive_reward = high_uncertainty & positive_reward
    high_uncertainty_positive_reward_fraction = (
        float(np.sum(high_uncertainty_positive_reward.astype(np.int64)) / positive_reward_count)
        if positive_reward_count > 0
        else 0.0
    )
    if (
        high_uncertainty_positive_reward_fraction
        > float(config.max_high_uncertainty_positive_reward_fraction)
    ):
        reasons.append("high_uncertainty_positive_reward_fraction_exceeds_gate")

    residual = (
        gate_pairs["predicted_value"].to_numpy(dtype=float)
        - gate_pairs["target_value"].to_numpy(dtype=float)
    )
    uncertainty_residual_pearson = (
        _safe_pearson(
            uncertainty.to_numpy(dtype=float)[uncertainty_finite],
            np.abs(residual)[uncertainty_finite],
        )
        if int(np.sum(uncertainty_finite.astype(np.int64))) >= 2
        else math.nan
    )

    return _json_safe(
        {
            "passes_predictor_reliability_gate": not reasons,
            "failed_reasons": reasons,
            "gate_split": gate_split,
            "row_count": row_count,
            "require_uncertainty": bool(config.require_uncertainty),
            "require_ood": bool(config.require_ood),
            "uncertainty_coverage": uncertainty_coverage,
            "uncertainty_threshold": uncertainty_threshold,
            "high_uncertainty_count": int(np.sum(high_uncertainty.astype(np.int64))),
            "high_uncertainty_fraction": high_uncertainty_fraction,
            "max_high_uncertainty_fraction": float(
                config.max_high_uncertainty_fraction
            ),
            "uncertainty_abs_residual_pearson": uncertainty_residual_pearson,
            "ood_info_available": ood_info_available,
            "ood_score_threshold": ood_score_threshold,
            "ood_count": int(np.sum(ood_mask.astype(np.int64))),
            "ood_fraction": ood_fraction,
            "max_ood_fraction": float(config.max_ood_fraction),
            "positive_reward_threshold": float(config.positive_reward_threshold),
            "positive_reward_count": positive_reward_count,
            "high_uncertainty_positive_reward_count": int(
                np.sum(high_uncertainty_positive_reward.astype(np.int64))
            ),
            "high_uncertainty_positive_reward_fraction": (
                high_uncertainty_positive_reward_fraction
            ),
            "max_high_uncertainty_positive_reward_fraction": float(
                config.max_high_uncertainty_positive_reward_fraction
            ),
        }
    )


def _column_from_aliases(
    frame: pd.DataFrame,
    aliases: Iterable[str | None],
    *,
    required: bool,
    label: str,
) -> tuple[pd.Series, str | None]:
    for alias in aliases:
        if alias and alias in frame.columns:
            return frame[alias], str(alias)
    if required:
        checked = [str(alias) for alias in aliases if alias]
        raise ValueError(f"Missing {label} column. Checked aliases: {checked}")
    return pd.Series(pd.NA, index=frame.index), None


def _coalesce_joined_column(
    joined: pd.DataFrame,
    column: str,
    *,
    join_keys: Sequence[str],
) -> pd.Series:
    if column in join_keys and column in joined.columns:
        return joined[column]
    candidates = [
        name
        for name in (f"{column}_target", f"{column}_prediction", column)
        if name in joined.columns
    ]
    if not candidates:
        return pd.Series(pd.NA, index=joined.index)
    result = _plain_object_series(joined[candidates[0]])
    for name in candidates[1:]:
        fallback = _plain_object_series(joined[name])
        result = result.where(result.notna(), fallback)
    return result


def _plain_object_series(series: pd.Series) -> pd.Series:
    """Return a pandas object series safe for null coalescing.

    Pandas groupby aggregation can produce ``null[pyarrow]`` columns when an
    optional string column, such as ``split``, is entirely missing.  Some Arrow
    builds abort when those columns are used in ``Series.where``.  Coalescing
    through object dtype keeps the parity path independent of Arrow null arrays.
    """

    if str(series.dtype) == "null[pyarrow]":
        return pd.Series(pd.NA, index=series.index, dtype=object)
    try:
        return series.astype(object)
    except Exception:
        return pd.Series(list(series), index=series.index, dtype=object)


def canonical_atom_family(atom: Any) -> str | None:
    """Return AtypEmu's canonical atom-family label for one atom token."""

    if atom is None or pd.isna(atom):
        return None
    token = str(atom).strip().upper()
    if not token or token in {"NAN", "NONE", "NULL"}:
        return None
    if token in {"C'", "CO", "C_PRIME", "CPRIME"}:
        return "C'"
    return ATOM_FAMILY_MAP.get(token)


def _family_from_target_id(target_id: Any) -> str | None:
    if target_id is None or pd.isna(target_id):
        return None
    value = str(target_id)
    if not value.startswith("cs:"):
        return None
    return canonical_atom_family(value.rsplit(":", 1)[-1])


def _canonical_family_series(series: pd.Series) -> pd.Series:
    return series.map(canonical_atom_family)


def _clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.isin(["", "nan", "NaN", "None", "<NA>"]))


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _boolean_series(series: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    if series.empty:
        return result
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype(float) != 0.0

    text = series.astype("string").str.strip().str.lower()
    result.loc[text.isin({"1", "1.0", "true", "t", "yes", "y", "on"})] = True
    result.loc[text.isin({"0", "0.0", "false", "f", "no", "n", "off"})] = False
    return result


def _any_truthy(series: pd.Series) -> Any:
    values = _boolean_series(series)
    if not bool(values.notna().any()):
        return pd.NA
    return bool(values.fillna(False).any())


def _optional_finite_float(value: Any) -> float | None:
    resolved = _safe_float(value)
    return resolved if math.isfinite(resolved) else None


def _has_non_null(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and bool(frame[column].notna().any())


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite))


def _finite_rmse(residual: np.ndarray) -> float:
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def _safe_float(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return math.nan
    return resolved


def _frame_signature(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    available_columns = [column for column in columns if column in frame.columns]
    payload = frame[available_columns].astype("string").fillna("").to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix in {".tsv", ".tab"}:
        frame.to_csv(path, sep="\t", index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    else:
        frame.to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


__all__ = [
    "CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES",
    "ChemicalShiftParityConfig",
    "PREDICTION_OOD_FLAG_ALIASES",
    "PREDICTION_OOD_SCORE_ALIASES",
    "PREDICTION_REWARD_ALIASES",
    "PREDICTION_UNCERTAINTY_ALIASES",
    "build_prediction_target_pairs",
    "canonical_atom_family",
    "chemical_shift_metric_frame",
    "evaluate_chemical_shift_prediction_parity",
    "load_table",
    "normalize_prediction_frame",
    "normalize_target_frame",
    "summarize_chemical_shift_parity",
    "write_parity_outputs",
]
