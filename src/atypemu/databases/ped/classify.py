"""Classify PED entries by ensemble-generation strategy."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atypemu.databases.ped.layout import (
    ensure_ped_workspace,
    normalize_ped_workspace,
    resolve_ped_table,
)


PED_GENERATION_COLUMNS = [
    "ped_id",
    "protein_name",
    "generation_bucket",
    "generation_family",
    "validation_group",
    "is_modern_ml",
    "is_generator_based",
    "experimental_procedures",
    "structural_ensemble_calculation_tags",
    "classification_rationale",
]


@dataclass(frozen=True)
class _PedGenerationRow:
    """One PED generation-classification record."""

    ped_id: str
    protein_name: str | None
    generation_bucket: str
    generation_family: str
    validation_group: str
    is_modern_ml: bool
    is_generator_based: bool
    experimental_procedures: str | None
    structural_ensemble_calculation_tags: str | None
    classification_rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "ped_id": self.ped_id,
            "protein_name": self.protein_name,
            "generation_bucket": self.generation_bucket,
            "generation_family": self.generation_family,
            "validation_group": self.validation_group,
            "is_modern_ml": self.is_modern_ml,
            "is_generator_based": self.is_generator_based,
            "experimental_procedures": self.experimental_procedures,
            "structural_ensemble_calculation_tags": (
                self.structural_ensemble_calculation_tags
            ),
            "classification_rationale": self.classification_rationale,
        }


def classify_ped_entries(
    ped_root: str | Path,
    workspace: str = "benchmark",
) -> dict[str, Any]:
    """Classify PED entries by ensemble-generation strategy.

    Args:
        ped_root: PED source root such as ``data/ped``.

    Returns:
        Summary dictionary describing the generated classification tables.
    """
    ped_root_path = Path(ped_root)
    workspace_name = normalize_ped_workspace(workspace)
    public_root, debug_root = ensure_ped_workspace(ped_root_path, workspace_name)
    entry_path = resolve_ped_table(ped_root_path, workspace_name, "ped_entries.tsv")
    rows = _load_tsv(entry_path)
    classified_rows = [_classify_row(row) for row in rows]

    _write_tsv(
        public_root / "ped_generation_classes.tsv",
        PED_GENERATION_COLUMNS,
        [row.to_dict() for row in classified_rows],
    )
    _write_jsonl(
        debug_root / "ped_generation_classes.jsonl",
        [row.to_dict() for row in classified_rows],
    )

    bucket_counts = Counter(row.generation_bucket for row in classified_rows)
    family_counts = Counter(row.generation_family for row in classified_rows)
    validation_counts = Counter(row.validation_group for row in classified_rows)
    summary = {
        "ped_entries": len(classified_rows),
        "generation_bucket_counts": dict(bucket_counts),
        "generation_family_counts": dict(family_counts),
        "validation_group_counts": dict(validation_counts),
        "ped_root": str(ped_root_path),
        "workspace": workspace_name,
        "output_root": str(public_root),
    }
    (debug_root / "ped_generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _classify_row(row: dict[str, str]) -> _PedGenerationRow:
    """Classify one PED entry."""
    protein_name = row.get("protein_name") or ""
    experimental_text = row.get("experimental_procedures") or ""
    tags_text = row.get("structural_ensemble_calculation_tags") or ""
    raw_text = row.get("structural_ensemble_calculation_raw") or ""
    combined = f"{protein_name} {experimental_text} {tags_text} {raw_text}".lower()
    rationale: list[str] = []

    if _has_any(
        combined,
        [
            "idpgan",
            "alphaflex",
            "deep generative model",
            "machine learning",
            "cg2all network",
        ],
    ):
        bucket = "modern_ml_generator"
        family = "deep_ml_generator"
        validation_group = "holdout_ml_generative"
        rationale.append("modern_ml_keywords")
    elif "idpconformergenerator" in combined:
        bucket = "generator_based"
        family = "algorithmic_conformer_generator"
        validation_group = "holdout_algorithmic_generator"
        rationale.append("idpconformergenerator")
    elif "fastfloppytail" in combined:
        bucket = "generator_based"
        family = "algorithmic_conformer_generator"
        validation_group = "holdout_algorithmic_generator"
        rationale.append("fastfloppytail")
    elif _has_any(
        combined,
        [
            "mmmx",
            "rigiflex",
            "flex module",
            "coral",
            "mmm2018",
            "in-silico growth",
            "force-field free algorithm",
            "generated stochastically",
        ],
    ):
        bucket = "generator_based"
        family = "algorithmic_conformer_generator"
        validation_group = "holdout_algorithmic_generator"
        rationale.append("algorithmic_generator_keywords")
    elif _has_any(
        combined,
        [
            "flexible-meccano",
            "trades",
            "ensemble optimization",
            "eom",
            "gajoe",
            "ranch",
            "asteroids",
            "ensemble generated",
        ],
    ):
        bucket = "generator_based"
        family = "classical_pool_generator_selection"
        validation_group = "classical_ensemble_reference"
        rationale.append("classical_pool_generator")
    elif _has_any(combined, ["campari", "profasi", "monte carlo"]):
        bucket = "classical_sampling"
        family = "monte_carlo_sampling"
        validation_group = "classical_ensemble_reference"
        rationale.append("monte_carlo")
    elif _has_any(combined, ["molecular dynamics", "charmm", "amber", "gromacs"]):
        bucket = "classical_sampling"
        family = "molecular_dynamics"
        validation_group = "classical_ensemble_reference"
        rationale.append("molecular_dynamics")
    elif _has_any(
        combined,
        [
            "dyana",
            "cyana",
            "cs-rosetta",
            "rosetta",
            "x-plor",
            "xplor-nih",
            "xplor",
            "cns",
            "aria",
            "candid",
            "structure calculation",
        ],
    ):
        bucket = "restraint_structure_models"
        family = "nmr_structure_calculation"
        validation_group = "structure_model_reference"
        rationale.append("restraint_structure_calculation")
    elif _has_any(combined, ["structural ensemble", "solution structure"]) and _has_any(
        combined,
        [
            "nmr",
            "noesy",
            "cosy",
            "tocsy",
            "roesy",
            "chemical shift",
            "relaxation",
        ],
    ):
        bucket = "restraint_structure_models"
        family = "nmr_structure_calculation"
        validation_group = "structure_model_reference"
        rationale.append("nmr_only_structure_ensemble")
    else:
        bucket = "unknown"
        family = "unknown"
        validation_group = "manual_review"
        rationale.append("no_rule_match")

    return _PedGenerationRow(
        ped_id=row["ped_id"],
        protein_name=row.get("protein_name") or None,
        generation_bucket=bucket,
        generation_family=family,
        validation_group=validation_group,
        is_modern_ml=bucket == "modern_ml_generator",
        is_generator_based=bucket in {"modern_ml_generator", "generator_based"},
        experimental_procedures=row.get("experimental_procedures") or None,
        structural_ensemble_calculation_tags=(
            row.get("structural_ensemble_calculation_tags") or None
        ),
        classification_rationale="|".join(rationale),
    )


def _has_any(text: str, patterns: list[str]) -> bool:
    """Return whether any plain-text pattern exists in the target text."""
    return any(pattern in text for pattern in patterns)


def _resolve_ped_table(ped_root: Path, filename: str) -> Path:
    """Resolve a PED table with backward-compatible fallback paths."""
    direct_path = ped_root / filename
    if direct_path.exists():
        return direct_path
    return ped_root / "manifests" / filename


def _load_tsv(path: Path) -> list[dict[str, str]]:
    """Load a TSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a TSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
