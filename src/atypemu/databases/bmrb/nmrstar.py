"""NMR-STAR target parsing with a lightweight fallback implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from atypemu.databases.bmrb._star_utils import (
    LoopTable,
    as_float,
    as_int,
    as_none,
    parse_fallback_loops,
)
from atypemu.databases.bmrb.restraints import NoeRestraintParser
from atypemu.errors import ParseError
from atypemu.types import (
    ChemicalShiftTarget,
    JCouplingTarget,
    NMRTargetBundle,
    canonical_atom_name,
)


def _infer_three_j_hnha(atom_1: str, atom_2: str) -> bool:
    """Return whether a scalar coupling target matches the v1 JC scope."""
    atom_set = {canonical_atom_name(atom_1), canonical_atom_name(atom_2)}
    return (
        atom_set == {"H", "HA"}
        or atom_set == {"H", "HA2"}
        or atom_set
        == {
            "H",
            "HA3",
        }
    )


class PyNMRStarTargetParser:
    """Parse chemical shifts and scalar couplings from local NMR-STAR files."""

    def __init__(
        self,
        allowed_chemical_shift_atoms: set[str] | None = None,
        default_j_uncertainty: float = 0.5,
    ) -> None:
        """Initialize the parser.

        Args:
            allowed_chemical_shift_atoms: Optional atom filter for chemical shifts.
            default_j_uncertainty: Default JC uncertainty when missing from the file.
        """
        default_atoms = {"H", "N", "CA", "CB", "C"}
        self.allowed_chemical_shift_atoms = (
            {canonical_atom_name(atom) for atom in allowed_chemical_shift_atoms}
            if allowed_chemical_shift_atoms is not None
            else default_atoms
        )
        self.default_j_uncertainty = default_j_uncertainty

    def parse(
        self,
        input_star: str | Path,
        jc_star: str | Path | None = None,
        noe_star: str | Path | None = None,
        noe_error_mode: Literal["raise", "skip"] = "raise",
    ) -> NMRTargetBundle:
        """Parse the requested targets into one bundle.

        Args:
            input_star: Primary BMRB-style STAR file.
            jc_star: Optional separate STAR file for scalar couplings.
            noe_star: Optional separate STAR file for NOE restraints.
            noe_error_mode: Whether NOE parsing failures should raise or be
                recorded and skipped.

        Returns:
            Parsed target bundle.
        """
        input_path = Path(input_star)
        cs_loops = self._load_loops(input_path)
        jc_loops = self._load_loops(Path(jc_star)) if jc_star else cs_loops
        noe_targets = []
        noe_status = "not_requested"
        noe_parse_error: str | None = None
        if noe_star is not None:
            try:
                noe_targets = NoeRestraintParser().parse(noe_star)
                noe_status = "ok" if noe_targets else "empty"
            except ParseError as exc:
                if noe_error_mode != "skip":
                    raise
                noe_status = "skipped_parse_error"
                noe_parse_error = str(exc)

        return NMRTargetBundle(
            chemical_shifts=self._parse_chemical_shifts(cs_loops),
            j_couplings=self._parse_j_couplings(jc_loops),
            noe_restraints=noe_targets,
            metadata={
                "input_star": str(input_path),
                "jc_star": str(jc_star) if jc_star else str(input_path),
                "noe_star": str(noe_star) if noe_star else None,
                "noe_status": noe_status,
                "noe_parse_error": noe_parse_error,
            },
        )

    def _load_loops(self, path: Path) -> list[LoopTable]:
        """Load loop tables from a STAR file."""
        text = path.read_text()
        try:
            import pynmrstar  # type: ignore

            entry = pynmrstar.Entry.from_file(str(path))
            loops: list[LoopTable] = []
            for saveframe in entry:
                for loop in saveframe:
                    category = str(getattr(loop, "category", "")).lstrip("_")
                    tags = []
                    for tag in loop.tags:
                        tag_name = str(tag)
                        if "." in tag_name:
                            tags.append(
                                tag_name if tag_name.startswith("_") else f"_{tag_name}"
                            )
                        elif category:
                            tags.append(f"_{category}.{tag_name.lstrip('_')}")
                        else:
                            tags.append(tag_name)
                    rows = [[str(value) for value in row] for row in loop.data]
                    loops.append(LoopTable(tags=tags, rows=rows))
            return loops
        except Exception:
            return parse_fallback_loops(text)

    def _parse_chemical_shifts(
        self, loops: list[LoopTable]
    ) -> list[ChemicalShiftTarget]:
        """Parse chemical shifts from loop tables."""
        targets: list[ChemicalShiftTarget] = []
        for loop in loops:
            if loop.category() != "Atom_chem_shift":
                continue
            for row in loop.iter_dicts():
                seq_id = as_int(row.get("Seq_ID")) or as_int(row.get("Comp_index_ID"))
                comp_id = as_none(row.get("Comp_ID")) or as_none(
                    row.get("Auth_comp_ID")
                )
                atom_id = canonical_atom_name(
                    as_none(row.get("Atom_ID"))
                    or as_none(row.get("Auth_atom_ID"))
                    or ""
                )
                value = as_float(row.get("Val"))
                uncertainty = as_float(row.get("Val_err"))
                if seq_id is None or comp_id is None or value is None or not atom_id:
                    continue
                if atom_id not in self.allowed_chemical_shift_atoms:
                    continue
                targets.append(
                    ChemicalShiftTarget(
                        seq_id=seq_id,
                        comp_id=comp_id,
                        atom_id=atom_id,
                        value=value,
                        uncertainty=uncertainty,
                        chain_id=as_none(row.get("Auth_asym_ID")),
                    )
                )
        return targets

    def _parse_j_couplings(self, loops: list[LoopTable]) -> list[JCouplingTarget]:
        """Parse scalar couplings from loop tables."""
        targets: list[JCouplingTarget] = []
        category_names = {"Coupling_constant", "Scalar_coupling", "J_coupling"}
        for loop in loops:
            if loop.category() not in category_names:
                continue
            for row in loop.iter_dicts():
                seq_id = as_int(row.get("Seq_ID")) or as_int(row.get("Seq_ID_1"))
                comp_id = as_none(row.get("Comp_ID")) or as_none(row.get("Comp_ID_1"))
                atom_1 = canonical_atom_name(
                    as_none(row.get("Atom_ID_1")) or as_none(row.get("Atom_ID")) or ""
                )
                atom_2 = canonical_atom_name(as_none(row.get("Atom_ID_2")) or "")
                if seq_id is None or comp_id is None or not atom_1 or not atom_2:
                    continue
                if not _infer_three_j_hnha(atom_1, atom_2):
                    continue
                value = (
                    as_float(row.get("Val"))
                    or as_float(row.get("Coupling_val"))
                    or as_float(row.get("Coupling_constant_val"))
                )
                if value is None:
                    continue
                uncertainty = (
                    as_float(row.get("Val_err"))
                    or as_float(row.get("Val_error"))
                    or self.default_j_uncertainty
                )
                targets.append(
                    JCouplingTarget(
                        seq_id=seq_id,
                        comp_id=comp_id,
                        atom_id_1="H",
                        atom_id_2="HA",
                        value=value,
                        uncertainty=uncertainty,
                        coupling_type="3J_HNHA",
                        chain_id=as_none(row.get("Auth_asym_ID")),
                    )
                )
        return targets
