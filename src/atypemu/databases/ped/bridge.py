"""Bridge PED entry tables to BMRB and AtypEmu training splits."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atypemu.databases.ped.layout import (
    ensure_ped_workspace,
    normalize_ped_workspace,
    ped_dataset_root,
    resolve_ped_table,
)


PED_BRIDGE_COLUMNS = [
    "ped_id",
    "protein_name",
    "uniprot_accessions",
    "ped_bmrb_ids",
    "overlapping_bmrb_ids",
    "overlap_count",
    "overlap_sources",
    "split_assignment",
    "split_bmrb_ids",
    "experimental_procedures",
    "structural_ensemble_calculation_tags",
    "generation_bucket",
    "generation_family",
    "validation_group",
    "recommended_role",
]

PED_VALIDATION_COLUMNS = [
    "ped_id",
    "protein_name",
    "split_assignment",
    "recommended_role",
    "validation_group",
    "generation_bucket",
    "generation_family",
    "overlapping_bmrb_ids",
    "experimental_procedures",
    "structural_ensemble_calculation_tags",
]


@dataclass(frozen=True)
class _PedBridgeRow:
    """Bridge row linking one PED entry to BMRB and split metadata."""

    ped_id: str
    protein_name: str | None
    uniprot_accessions: str | None
    ped_bmrb_ids: str | None
    overlapping_bmrb_ids: str | None
    overlap_count: int
    overlap_sources: str | None
    split_assignment: str
    split_bmrb_ids: str | None
    experimental_procedures: str | None
    structural_ensemble_calculation_tags: str | None
    generation_bucket: str
    generation_family: str
    validation_group: str
    recommended_role: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for the bridge row."""
        return {
            "ped_id": self.ped_id,
            "protein_name": self.protein_name,
            "uniprot_accessions": self.uniprot_accessions,
            "ped_bmrb_ids": self.ped_bmrb_ids,
            "overlapping_bmrb_ids": self.overlapping_bmrb_ids,
            "overlap_count": self.overlap_count,
            "overlap_sources": self.overlap_sources,
            "split_assignment": self.split_assignment,
            "split_bmrb_ids": self.split_bmrb_ids,
            "experimental_procedures": self.experimental_procedures,
            "structural_ensemble_calculation_tags": (
                self.structural_ensemble_calculation_tags
            ),
            "generation_bucket": self.generation_bucket,
            "generation_family": self.generation_family,
            "validation_group": self.validation_group,
            "recommended_role": self.recommended_role,
        }


def prepare_ped_bridge(
    ped_root: str | Path,
    training_root: str | Path,
    workspace: str = "catalog",
) -> dict[str, Any]:
    """Build PED-to-BMRB bridge manifests and training-aware PED tables.

    Args:
        ped_root: PED source root such as ``data/ped``.
        training_root: Integrated cross-source workspace root.

    Returns:
        Summary dictionary describing the generated bridge outputs.
    """
    ped_root_path = Path(ped_root)
    training_root_path = Path(training_root)
    workspace_name = normalize_ped_workspace(workspace)
    manifests_dir, debug_root = ensure_ped_workspace(ped_root_path, workspace_name)

    entry_rows = _load_tsv(
        resolve_ped_table(ped_root_path, workspace_name, "ped_entries.tsv")
    )
    generation_map = _load_generation_map(ped_root_path, workspace_name)
    accessions = json.loads(
        (training_root_path / "inventories" / "accessions.json").read_text()
    )
    split_payload = json.loads(
        (training_root_path / "splits" / "default_split.json").read_text()
    )

    accession_by_id = {row["bmrb_id"]: row for row in accessions}
    accession_by_hash = {
        row["sequence_hash"]: row["bmrb_id"]
        for row in accessions
        if row.get("sequence_hash")
    }
    split_by_bmrb = {
        bmrb_id: split_name
        for split_name, values in split_payload.items()
        for bmrb_id in values
    }

    bridge_rows: list[_PedBridgeRow] = []
    validation_rows: list[dict[str, Any]] = []
    experimental_tag_rows: list[dict[str, str]] = []
    structural_tag_rows: list[dict[str, str]] = []

    for entry in entry_rows:
        generation_payload = generation_map.get(
            entry["ped_id"],
            {
                "generation_bucket": "unknown",
                "generation_family": "unknown",
                "validation_group": "manual_review",
            },
        )
        overlap_map: dict[str, set[str]] = {}

        for bmrb_id in _normalize_ped_bmrb_ids(entry.get("bmrb_ids")):
            if bmrb_id in accession_by_id:
                overlap_map.setdefault(bmrb_id, set()).add("cross_ref")

        sequence = entry.get("sequence") or ""
        if sequence:
            sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            matched_id = accession_by_hash.get(sequence_hash)
            if matched_id:
                overlap_map.setdefault(matched_id, set()).add("exact_sequence_hash")
            for accession in accessions:
                candidate_sequence = accession.get("sequence") or ""
                if not candidate_sequence or candidate_sequence == sequence:
                    continue
                if sequence in candidate_sequence or candidate_sequence in sequence:
                    overlap_map.setdefault(accession["bmrb_id"], set()).add(
                        "sequence_substring"
                    )

        overlapping_bmrb_ids = sorted(overlap_map)
        split_hits = sorted(
            {split_by_bmrb[bmrb_id] for bmrb_id in overlapping_bmrb_ids}
        )
        if not split_hits:
            split_assignment = "unassigned"
        elif len(split_hits) == 1:
            split_assignment = split_hits[0]
        else:
            split_assignment = "mixed"

        overlap_sources = _pipe_join(
            sorted({item for sources in overlap_map.values() for item in sources})
        )
        recommended_role = _recommended_role(
            split_assignment=split_assignment,
            overlap_sources=overlap_sources,
            validation_group=generation_payload["validation_group"],
        )
        bridge_row = _PedBridgeRow(
            ped_id=entry["ped_id"],
            protein_name=entry.get("protein_name") or None,
            uniprot_accessions=entry.get("uniprot_accessions") or None,
            ped_bmrb_ids=_pipe_join(_normalize_ped_bmrb_ids(entry.get("bmrb_ids"))),
            overlapping_bmrb_ids=_pipe_join(overlapping_bmrb_ids),
            overlap_count=len(overlapping_bmrb_ids),
            overlap_sources=overlap_sources,
            split_assignment=split_assignment,
            split_bmrb_ids=(
                _pipe_join(
                    [
                        bmrb_id
                        for bmrb_id in overlapping_bmrb_ids
                        if split_by_bmrb.get(bmrb_id) == split_assignment
                    ]
                )
                if split_assignment in {"train", "val", "test"}
                else _pipe_join(overlapping_bmrb_ids)
            ),
            experimental_procedures=entry.get("experimental_procedures") or None,
            structural_ensemble_calculation_tags=(
                entry.get("structural_ensemble_calculation_tags") or None
            ),
            generation_bucket=generation_payload["generation_bucket"],
            generation_family=generation_payload["generation_family"],
            validation_group=generation_payload["validation_group"],
            recommended_role=recommended_role,
        )
        bridge_rows.append(bridge_row)
        validation_rows.append(
            {
                "ped_id": bridge_row.ped_id,
                "protein_name": bridge_row.protein_name,
                "split_assignment": bridge_row.split_assignment,
                "recommended_role": bridge_row.recommended_role,
                "validation_group": bridge_row.validation_group,
                "generation_bucket": bridge_row.generation_bucket,
                "generation_family": bridge_row.generation_family,
                "overlapping_bmrb_ids": bridge_row.overlapping_bmrb_ids,
                "experimental_procedures": bridge_row.experimental_procedures,
                "structural_ensemble_calculation_tags": (
                    bridge_row.structural_ensemble_calculation_tags
                ),
            }
        )
        for tag in _split_pipe(entry.get("experimental_procedures")):
            experimental_tag_rows.append({"ped_id": entry["ped_id"], "tag": tag})
        for tag in _split_pipe(entry.get("structural_ensemble_calculation_tags")):
            structural_tag_rows.append({"ped_id": entry["ped_id"], "tag": tag})

    _write_tsv(
        manifests_dir / "ped_bmrb_bridge.tsv",
        PED_BRIDGE_COLUMNS,
        [row.to_dict() for row in bridge_rows],
    )
    _write_jsonl(
        debug_root / "ped_bmrb_bridge.jsonl",
        [row.to_dict() for row in bridge_rows],
    )
    _write_tsv(
        manifests_dir / "ped_validation_manifest.tsv",
        PED_VALIDATION_COLUMNS,
        validation_rows,
    )
    _write_jsonl(
        debug_root / "ped_validation_manifest.jsonl",
        validation_rows,
    )
    _write_tsv(
        manifests_dir / "ped_experimental_tags.tsv",
        ["ped_id", "tag"],
        experimental_tag_rows,
    )
    _write_tsv(
        manifests_dir / "ped_structural_tags.tsv",
        ["ped_id", "tag"],
        structural_tag_rows,
    )

    summary = {
        "ped_entries": len(bridge_rows),
        "overlap_count": sum(row.overlap_count > 0 for row in bridge_rows),
        "train_overlap_count": sum(
            row.split_assignment == "train" for row in bridge_rows
        ),
        "val_overlap_count": sum(row.split_assignment == "val" for row in bridge_rows),
        "test_overlap_count": sum(
            row.split_assignment == "test" for row in bridge_rows
        ),
        "validation_group_counts": _counter_dict(
            row.validation_group for row in bridge_rows
        ),
        "recommended_role_counts": _counter_dict(
            row.recommended_role for row in bridge_rows
        ),
        "manual_review_count": sum(
            row.validation_group == "manual_review" for row in bridge_rows
        ),
        "ped_root": str(ped_root_path),
        "workspace": workspace_name,
        "output_root": str(manifests_dir),
    }
    (debug_root / "ped_bmrb_bridge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    _write_source_validation_policy(
        training_root=training_root_path,
        ped_root=ped_root_path,
        workspace=workspace_name,
        summary=summary,
    )
    return summary


def _recommended_role(
    split_assignment: str,
    overlap_sources: str | None,
    validation_group: str,
) -> str:
    """Infer a PED usage recommendation from overlap and split information."""
    if validation_group in {
        "holdout_ml_generative",
        "holdout_algorithmic_generator",
        "structure_model_reference",
    }:
        return "validation_only"
    if validation_group == "manual_review":
        return "manual_review"
    if split_assignment == "test":
        return "validation_only"
    if split_assignment == "val":
        return "validation_only"
    if split_assignment == "train":
        return (
            "manual_review"
            if overlap_sources and "sequence_substring" in overlap_sources
            else "training_augmentation_candidate"
        )
    if split_assignment == "mixed":
        return "manual_review"
    if overlap_sources and "sequence_substring" in overlap_sources:
        return "manual_review"
    return "external_benchmark_only"


def _load_tsv(path: Path) -> list[dict[str, str]]:
    """Load a TSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a TSV file with deterministic row order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _normalize_ped_bmrb_ids(value: str | None) -> list[str]:
    """Normalize PED BMRB identifier strings to ``bmrNNNN`` form."""
    if not value:
        return []
    normalized = []
    for item in value.split("|"):
        digits = "".join(char for char in item if char.isdigit())
        if digits:
            normalized.append(f"bmr{digits}")
    return sorted(set(normalized))


def _pipe_join(values: list[str]) -> str | None:
    """Join values as a pipe-separated string when non-empty."""
    items = [item for item in values if item]
    return "|".join(items) if items else None


def _split_pipe(value: str | None) -> list[str]:
    """Split a pipe-separated string into deterministic tokens."""
    if not value:
        return []
    return [item for item in value.split("|") if item]


def _load_generation_map(
    ped_root: Path,
    workspace: str,
) -> dict[str, dict[str, str]]:
    """Load PED generation-classification rows keyed by PED ID."""
    path = resolve_ped_table(ped_root, workspace, "ped_generation_classes.tsv")
    if not path.exists():
        return {}
    rows = _load_tsv(path)
    return {
        row["ped_id"]: {
            "generation_bucket": row["generation_bucket"],
            "generation_family": row["generation_family"],
            "validation_group": row["validation_group"],
        }
        for row in rows
    }


def _counter_dict(values: Any) -> dict[str, int]:
    """Count values and return a deterministically sorted mapping."""
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _write_source_validation_policy(
    training_root: Path,
    ped_root: Path,
    workspace: str,
    summary: dict[str, Any],
) -> None:
    """Write PED validation policy into the source root and integrated config."""
    ped_dataset_dir = ped_dataset_root(ped_root)
    ped_dataset_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = training_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    validation_policy = {
        "ped": {
            "ped_root": str(ped_root.resolve()),
            "workspace": workspace,
            "bridge_tsv": str(
                (
                    resolve_ped_table(ped_root, workspace, "ped_bmrb_bridge.tsv")
                ).resolve()
            ),
            "validation_manifest_tsv": str(
                (
                    resolve_ped_table(
                        ped_root, workspace, "ped_validation_manifest.tsv"
                    )
                ).resolve()
            ),
            "track_priority": [
                "holdout_ml_generative",
                "holdout_algorithmic_generator",
                "classical_ensemble_reference",
                "structure_model_reference",
                "manual_review",
            ],
            "training_allowed_validation_groups": ["classical_ensemble_reference"],
            "training_excluded_validation_groups": [
                "holdout_ml_generative",
                "holdout_algorithmic_generator",
                "structure_model_reference",
                "manual_review",
            ],
            "summary": summary,
        }
    }
    policy_path = ped_dataset_dir / "ped_validation_policy.json"
    policy_path.write_text(
        json.dumps(validation_policy, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    workspace_config_path = configs_dir / "training_workspace.json"
    if workspace_config_path.exists():
        payload = json.loads(workspace_config_path.read_text(encoding="utf-8"))
        payload.setdefault("source_validation", {})
        payload["source_validation"]["ped"] = {
            **validation_policy["ped"],
            "policy_path": str(policy_path.resolve()),
        }
        workspace_config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
