"""Readiness-audit helpers for production student-training campaigns."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.datasets import IntegratedDataRegistry
from atypemu.training.config import TrainingReadinessConfig
from atypemu.training.data import build_source_vocab


def audit_training_readiness(
    data_root: str | Path,
    integrated_root: str | Path,
    config: TrainingReadinessConfig,
) -> dict[str, Any]:
    """Audit whether an integrated workspace is ready for production training.

    Args:
        data_root: Repository data root.
        integrated_root: Integrated workspace root such as ``data/integrated``.
        config: Readiness thresholds and split policy.

    Returns:
        Serializable readiness report with threshold checks and status counts.
    """
    data_root_path = Path(data_root)
    integrated_root_path = Path(integrated_root)
    summary_path = integrated_root_path / "teachers" / "materialization_summary.json"
    ucbshift_summary_path = (
        integrated_root_path / "ucbshift" / "sidecar_generation_summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            "Missing materialization summary. Run teacher materialization first: "
            f"{summary_path}"
        )

    summary_payload = json.loads(summary_path.read_text())
    results = list(summary_payload.get("results", []))
    registry = IntegratedDataRegistry.from_data_root(data_root_path)
    teacher_examples = registry.load_teacher_examples()
    split_frame = teacher_examples.loc[
        teacher_examples["split"].isin(config.selected_splits)
    ].copy()
    split_lookup = (
        split_frame.set_index("entity_uid")["split"].astype(str).to_dict()
        if not split_frame.empty
        else {}
    )

    effective_results = [
        _hydrate_existing_result(
            result=result,
            integrated_root=integrated_root_path,
        )
        for result in results
    ]
    written_results = [
        result for result in effective_results if _is_valid_teacher_result(result)
    ]
    validation_written = [
        result
        for result in written_results
        if split_lookup.get(str(result.get("entity_uid"))) in config.validation_splits
    ]
    cs_valid_fractions = [
        float(result.get("chemical_shift_valid_fraction", math.nan))
        for result in written_results
        if math.isfinite(float(result.get("chemical_shift_valid_fraction", math.nan)))
    ]

    requested_examples = int(summary_payload.get("requested_examples", len(results)))
    status_counts = (
        pd.Series([str(result.get("status", "")) for result in effective_results])
        .value_counts()
        .to_dict()
        if effective_results
        else {}
    )
    missing_shift_fraction = _status_fraction(
        status_counts=status_counts,
        requested_examples=requested_examples,
        status="skipped_missing_shifts",
    )
    no_cs_coverage_fraction = _status_fraction(
        status_counts=status_counts,
        requested_examples=requested_examples,
        status="skipped_no_cs_coverage",
    )
    ucbshift_payload = (
        json.loads(ucbshift_summary_path.read_text())
        if ucbshift_summary_path.exists()
        else {}
    )
    ucbshift_observed = _extract_ucbshift_observed(ucbshift_payload)

    eligible = split_frame.loc[
        split_frame["target_ready"]
        & split_frame["pool_ready"]
        & split_frame["observable_ready"]
        & split_frame["teacher_ready"]
    ].copy()
    repo_root = data_root_path.parent
    source_vocab = build_source_vocab(
        eligible.to_dict(orient="records"),
        data_root=data_root_path,
        repo_root=repo_root,
    )

    checks = {
        "zero_failed_examples": (
            int(summary_payload.get("failed_examples", 0)) == 0
            if config.require_zero_failed_examples
            else True
        ),
        "written_examples": len(written_results) >= config.required_written_examples,
        "validation_examples": len(validation_written)
        >= config.required_validation_examples,
        "chemical_shift_valid_fraction": (
            bool(cs_valid_fractions)
            and float(pd.Series(cs_valid_fractions).median())
            >= config.minimum_median_chemical_shift_valid_fraction
        ),
        "source_count": len(source_vocab) >= config.minimum_source_count,
        "ucbshift_summary": (
            (not config.require_ucbshift_summary) or ucbshift_summary_path.exists()
        ),
        "requested_candidate_fraction": (
            ucbshift_observed["requested_candidate_fraction"]
            >= config.minimum_requested_candidate_fraction
        ),
        "ucbshift_written_example_fraction": (
            ucbshift_observed["written_example_fraction"]
            >= config.minimum_ucbshift_written_example_fraction
        ),
        "ucbshift_baseline_example_fraction": (
            ucbshift_observed["baseline_example_fraction"]
            >= config.minimum_ucbshift_baseline_example_fraction
        ),
        "missing_shift_fraction": (
            missing_shift_fraction <= config.max_missing_shift_fraction
        ),
        "no_cs_coverage_fraction": (
            no_cs_coverage_fraction <= config.max_no_cs_coverage_fraction
        ),
    }
    ready = all(checks.values())

    return {
        "ready": ready,
        "summary_path": str(summary_path),
        "thresholds": config.as_dict(),
        "observed": {
            "requested_examples": requested_examples,
            "written_examples": len(written_results),
            "validation_written_examples": len(validation_written),
            "failed_examples": int(summary_payload.get("failed_examples", 0)),
            "median_chemical_shift_valid_fraction": (
                float(pd.Series(cs_valid_fractions).median())
                if cs_valid_fractions
                else None
            ),
            "source_vocab": source_vocab,
            "source_count": len(source_vocab),
            "missing_shift_fraction": missing_shift_fraction,
            "no_cs_coverage_fraction": no_cs_coverage_fraction,
            "ucbshift_summary_path": (
                str(ucbshift_summary_path) if ucbshift_summary_path.exists() else None
            ),
            "ucbshift_status": ucbshift_observed["status"],
            "requested_candidate_examples": ucbshift_observed[
                "requested_candidate_examples"
            ],
            "written_candidate_examples": ucbshift_observed[
                "written_candidate_examples"
            ],
            "baseline_examples": ucbshift_observed["baseline_examples"],
            "requested_candidate_fraction": ucbshift_observed[
                "requested_candidate_fraction"
            ],
            "ucbshift_written_example_fraction": ucbshift_observed[
                "written_example_fraction"
            ],
            "ucbshift_baseline_example_fraction": ucbshift_observed[
                "baseline_example_fraction"
            ],
        },
        "status_counts": status_counts,
        "checks": checks,
        "failing_checks": [
            check_name for check_name, passed in checks.items() if not passed
        ],
    }


def _hydrate_existing_result(
    result: dict[str, Any],
    integrated_root: Path,
) -> dict[str, Any]:
    """Load cached coverage values for one ``skipped_existing`` teacher row.

    Args:
        result: One row from ``materialization_summary.json``.
        integrated_root: Integrated workspace root such as ``data/integrated``.

    Returns:
        Result dictionary enriched with cached coverage values when available.
    """
    if str(result.get("status", "")).strip() != "skipped_existing":
        return result

    bmrb_id = str(result.get("bmrb_id", "")).strip()
    if not bmrb_id:
        return result

    summary_path = integrated_root / "teachers" / bmrb_id / "materialization.json"
    if not summary_path.exists():
        return result

    payload = json.loads(summary_path.read_text())
    hydrated = dict(result)
    for key in [
        "candidate_count",
        "chemical_shift_valid_fraction",
        "j_coupling_valid_fraction",
        "noe_valid_fraction",
    ]:
        if key in payload:
            hydrated[key] = payload[key]
    return hydrated


def _is_valid_teacher_result(result: dict[str, Any]) -> bool:
    """Return whether one materialized teacher result is usable for training."""
    if str(result.get("status", "")).strip() not in {"written", "skipped_existing"}:
        return False
    candidate_count = int(result.get("candidate_count", 0) or 0)
    chemical_shift_valid_fraction = float(
        result.get("chemical_shift_valid_fraction", 0.0) or 0.0
    )
    return not (candidate_count == 0 and chemical_shift_valid_fraction == 0.0)


def _extract_ucbshift_observed(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized UCBShift readiness counters from one summary payload."""
    requested_examples = int(payload.get("requested_examples", 0) or 0)
    requested_candidate_examples = int(
        payload.get("requested_candidate_examples", 0) or 0
    )
    written_candidate_examples = int(payload.get("written_candidate_examples", 0) or 0)
    baseline_examples = int(payload.get("baseline_examples", 0) or 0)

    def fraction(value: int) -> float:
        if requested_examples <= 0:
            return 0.0
        return float(value) / float(requested_examples)

    return {
        "status": str(payload.get("status", "missing")),
        "requested_examples": requested_examples,
        "requested_candidate_examples": requested_candidate_examples,
        "written_candidate_examples": written_candidate_examples,
        "baseline_examples": baseline_examples,
        "requested_candidate_fraction": fraction(requested_candidate_examples),
        "written_example_fraction": fraction(written_candidate_examples),
        "baseline_example_fraction": fraction(baseline_examples),
    }


def _status_fraction(
    status_counts: dict[str, int],
    requested_examples: int,
    status: str,
) -> float:
    """Return the fraction of examples in one materialization status bucket."""
    if requested_examples <= 0:
        return 0.0
    return float(status_counts.get(status, 0)) / float(requested_examples)
