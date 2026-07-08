"""Archive-aware staging and provenance helpers for AtypEmu data."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from atypemu.errors import ValidationError


STAGE_MANIFEST_FILENAME = ".atypemu_stage_manifest.json"

PHYSICAL_TO_LOGICAL_SOURCE = {
    "AF": "AF3",
    "CALVADOS": "CALVADOS2",
    "BioEmu": "BioEmu",
}

PREPROCESSING_PIPELINES = {
    "AF": "af3_faspr_pdbfixer_openmm100_checkpdb",
    "CALVADOS": "calvados2_cg2all_openmm100_checkpdb",
    "BioEmu": "bioemu_faspr_pdbfixer_openmm100_checkpdb",
}

_CANDIDATE_NAME_PATTERN = re.compile(
    r"^(?P<bmrb_id>bmr\d+)_(?P<physical_source>[A-Za-z0-9]+)_(?P<sample_index>\d+)\.pdb$"
)


def parse_candidate_filename(path_or_name: str | Path) -> dict[str, Any]:
    """Parse canonical candidate information from a PDB filename.

    Args:
        path_or_name: Candidate filename or full path.

    Returns:
        A dictionary containing parsed identifier fields. Returns an empty
        dictionary when the filename does not follow the canonical v1 contract.
    """
    name = Path(path_or_name).name
    match = _CANDIDATE_NAME_PATTERN.match(name)
    if match is None:
        return {}

    payload = match.groupdict()
    physical_source = payload["physical_source"]
    return {
        "bmrb_id": payload["bmrb_id"],
        "physical_source_dir": physical_source,
        "generator_source": PHYSICAL_TO_LOGICAL_SOURCE.get(
            physical_source, physical_source
        ),
        "sample_index": int(payload["sample_index"]),
        "preprocessing_pipeline": PREPROCESSING_PIPELINES.get(physical_source),
        "is_canonical_v1": True,
        "requires_bond_topology_caution": True,
    }


def infer_candidate_metadata(path: str | Path) -> dict[str, Any]:
    """Infer provenance metadata from a candidate path.

    Args:
        path: Candidate path on disk.

    Returns:
        Metadata inferred from the canonical filename contract. If the path does
        not match the v1 naming convention, an empty dictionary is returned.
    """
    metadata = parse_candidate_filename(path)
    if not metadata:
        return {}

    metadata["storage_mode"] = "directory_file"
    metadata["archive_path"] = None
    metadata["archive_member"] = None
    return metadata


def load_stage_manifest(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Load a per-directory staging manifest when available.

    Args:
        directory: Structure directory that may contain a stage manifest.

    Returns:
        Mapping from relative file path to metadata payload.
    """
    manifest_path = Path(directory) / STAGE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text())
    entries = payload.get("entries", {})
    return {str(key): dict(value) for key, value in entries.items()}


def stage_bmrb_accession(
    data_root: str | Path,
    bmrb_id: str,
    output_dir: str | Path,
    bioemu_mode: str = "symlink",
    overwrite: bool = False,
) -> Path:
    """Stage one BMRB accession from the canonical AtypEmu data layout.

    Args:
        data_root: Root directory containing ``AF``, ``CALVADOS``, and
            ``BioEmu`` subdirectories.
        bmrb_id: Canonical BMRB accession such as ``bmr10077``.
        output_dir: Root directory where the staged accession will be created.
        bioemu_mode: How to materialize BioEmu directories. Supported values are
            ``"symlink"`` and ``"copy"``.
        overwrite: Whether to replace an existing staged accession.

    Returns:
        The created staging directory for the accession.

    Raises:
        ValidationError: If inputs are invalid or staging cannot proceed safely.
    """
    normalized_bmrb_id = bmrb_id.strip().lower()
    if not normalized_bmrb_id.startswith("bmr"):
        raise ValidationError("BMRB accession must look like 'bmrXXXX'.")
    if bioemu_mode not in {"symlink", "copy"}:
        raise ValidationError("bioemu_mode must be 'symlink' or 'copy'.")

    data_root_path = Path(data_root)
    stage_root = Path(output_dir) / normalized_bmrb_id

    if stage_root.exists():
        if not overwrite:
            raise ValidationError(f"Stage directory already exists: {stage_root}")
        shutil.rmtree(stage_root)

    stage_root.mkdir(parents=True, exist_ok=True)

    manifests = {
        "AF": {},
        "CALVADOS": {},
        "BioEmu": {},
    }

    _stage_archived_source(
        source_root=data_root_path / "AF",
        physical_source="AF",
        bmrb_id=normalized_bmrb_id,
        stage_root=stage_root,
        manifest=manifests["AF"],
    )
    _stage_archived_source(
        source_root=data_root_path / "CALVADOS",
        physical_source="CALVADOS",
        bmrb_id=normalized_bmrb_id,
        stage_root=stage_root,
        manifest=manifests["CALVADOS"],
    )
    _stage_bioemu_source(
        source_root=data_root_path / "BioEmu",
        bmrb_id=normalized_bmrb_id,
        stage_root=stage_root,
        manifest=manifests["BioEmu"],
        mode=bioemu_mode,
    )

    if not any(manifests.values()):
        shutil.rmtree(stage_root)
        raise ValidationError(
            f"No candidates found for accession: {normalized_bmrb_id}"
        )

    for physical_source, entries in manifests.items():
        source_root = stage_root / physical_source
        source_root.mkdir(parents=True, exist_ok=True)
        _write_stage_manifest(
            manifest_path=source_root / STAGE_MANIFEST_FILENAME,
            bmrb_id=normalized_bmrb_id,
            physical_source=physical_source,
            entries=entries,
        )

    return stage_root


def _stage_archived_source(
    source_root: Path,
    physical_source: str,
    bmrb_id: str,
    stage_root: Path,
    manifest: dict[str, dict[str, Any]],
) -> None:
    """Extract one accession from chunked zip archives."""
    if not source_root.exists():
        return

    prefix = f"{bmrb_id}/"
    destination_root = stage_root / physical_source

    for archive_path in sorted(source_root.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if not member.startswith(prefix) or not member.endswith(".pdb"):
                    continue

                destination_path = destination_root / member
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination_path.write_bytes(archive.read(member))

                metadata = infer_candidate_metadata(destination_path)
                metadata.update(
                    {
                        "bmrb_id": bmrb_id,
                        "physical_source_dir": physical_source,
                        "generator_source": PHYSICAL_TO_LOGICAL_SOURCE[physical_source],
                        "preprocessing_pipeline": PREPROCESSING_PIPELINES[
                            physical_source
                        ],
                        "storage_mode": "zip_member",
                        "archive_path": str(archive_path.resolve()),
                        "archive_member": member,
                        "is_canonical_v1": True,
                        "requires_bond_topology_caution": True,
                    }
                )
                manifest[member] = metadata


def _stage_bioemu_source(
    source_root: Path,
    bmrb_id: str,
    stage_root: Path,
    manifest: dict[str, dict[str, Any]],
    mode: str,
) -> None:
    """Stage one BioEmu accession by linking or copying the existing directory."""
    source_dir = source_root / bmrb_id
    destination_root = stage_root / "BioEmu"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root / bmrb_id

    if not source_dir.exists():
        return

    if destination_dir.exists():
        if destination_dir.is_symlink() or destination_dir.is_file():
            destination_dir.unlink()
        else:
            shutil.rmtree(destination_dir)

    if mode == "copy":
        shutil.copytree(source_dir, destination_dir)
    else:
        try:
            destination_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(source_dir, destination_dir)

    for path in sorted(source_dir.glob("*.pdb")):
        relative_path = f"{bmrb_id}/{path.name}"
        metadata = infer_candidate_metadata(path)
        metadata.update(
            {
                "bmrb_id": bmrb_id,
                "physical_source_dir": "BioEmu",
                "generator_source": PHYSICAL_TO_LOGICAL_SOURCE["BioEmu"],
                "preprocessing_pipeline": PREPROCESSING_PIPELINES["BioEmu"],
                "storage_mode": "directory_file",
                "archive_path": None,
                "archive_member": None,
                "is_canonical_v1": True,
                "requires_bond_topology_caution": True,
            }
        )
        manifest[relative_path] = metadata


def _write_stage_manifest(
    manifest_path: Path,
    bmrb_id: str,
    physical_source: str,
    entries: dict[str, dict[str, Any]],
) -> None:
    """Write a stage manifest for one staged source subtree."""
    payload = {
        "bmrb_id": bmrb_id,
        "physical_source": physical_source,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
