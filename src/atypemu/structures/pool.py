"""Candidate-pool discovery and structure validation helpers."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from atypemu.adapters.cnnls_adapter import build_prediction_map, normalize_candidate_key
from atypemu.errors import ValidationError
from atypemu.structures.staging import infer_candidate_metadata, load_stage_manifest
from atypemu.types import CandidatePool, CandidateRecord


class StructureLoader:
    """Load atomic structures from local PDB files."""

    def __init__(self) -> None:
        """Initialize the structure loader."""
        self._parser = PDBParser(QUIET=True)

    def load_structure(self, path: str | Path):
        """Load one structure from disk.

        Args:
            path: Input PDB file path.

        Returns:
            Parsed Biopython structure object.
        """
        return self._parser.get_structure(Path(path).stem, str(path))


class ResidueMapper:
    """Extract residue keys from a parsed structure."""

    @staticmethod
    def extract_residue_keys(structure) -> list[str]:
        """Return residue identifiers in chain and sequence order."""
        residue_keys: list[str] = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    residue_keys.append(
                        f"{chain.id}:{residue.id[1]}:{residue.resname.strip()}"
                    )
            break
        return residue_keys


class ProtonationValidator:
    """Validate whether a structure contains explicit hydrogens."""

    @staticmethod
    def has_hydrogens(structure) -> bool:
        """Return whether the structure contains any hydrogen atom."""
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.id.upper().startswith("H"):
                            return True
            break
        return False

    @classmethod
    def validate(cls, structure, require_hydrogens: bool = False) -> bool:
        """Validate hydrogen presence when requested.

        Args:
            structure: Biopython structure object.
            require_hydrogens: Whether hydrogen atoms are mandatory.

        Returns:
            Whether hydrogens are present.

        Raises:
            ValidationError: If hydrogens are required and missing.
        """
        has_hydrogens = cls.has_hydrogens(structure)
        if require_hydrogens and not has_hydrogens:
            raise ValidationError("Structure is not protonated.")
        return has_hydrogens


def build_candidate_pool(
    structure_dir: str | Path,
    source: str,
    chemical_shift_dir: str | Path | None = None,
    chemical_shift_format: str | None = None,
    glob_pattern: str = "*.pdb",
) -> CandidatePool:
    """Build a candidate pool from a local structure directory.

    Args:
        structure_dir: Directory containing candidate structures.
        source: Source label written to the pool manifest.
        chemical_shift_dir: Optional directory with precomputed shift sidecars.
        chemical_shift_format: Optional sidecar format.
        glob_pattern: PDB glob pattern.

    Returns:
        Candidate pool manifest.
    """
    structure_root = Path(structure_dir)
    loader = StructureLoader()
    prediction_map = (
        build_prediction_map(chemical_shift_dir)
        if chemical_shift_dir is not None
        else {}
    )
    records: list[CandidateRecord] = []
    stage_manifest = load_stage_manifest(structure_root)

    for path in _iter_structure_paths(structure_root, glob_pattern):
        if not path.is_file():
            continue
        candidate_id = normalize_candidate_key(path)
        structure = loader.load_structure(path)
        residue_keys = ResidueMapper.extract_residue_keys(structure)
        is_protonated = ProtonationValidator.has_hydrogens(structure)
        prediction_path = prediction_map.get(candidate_id)
        relative_path = path.relative_to(structure_root).as_posix()

        metadata = infer_candidate_metadata(path)
        metadata.update(stage_manifest.get(relative_path, {}))
        metadata["num_residues"] = len(residue_keys)

        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                structure_path=str(path),
                source=source,
                chemical_shift_path=prediction_path,
                chemical_shift_format=chemical_shift_format,
                is_protonated=is_protonated,
                residue_keys=residue_keys,
                metadata=metadata,
            )
        )

    return CandidatePool(records=records)


def _iter_structure_paths(structure_root: Path, glob_pattern: str) -> list[Path]:
    """Return matching PDB paths while following staged symlink directories."""
    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(structure_root, followlinks=True):
        for filename in filenames:
            path = Path(dirpath) / filename
            relative_path = path.relative_to(structure_root).as_posix()
            if fnmatch.fnmatch(filename, glob_pattern) or fnmatch.fnmatch(
                relative_path, glob_pattern
            ):
                paths.append(path)
    return sorted(paths)
