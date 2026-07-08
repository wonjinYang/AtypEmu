"""Simple NOE restraint parsing for local STAR-like files."""

from __future__ import annotations

from pathlib import Path

from atypemu.databases.bmrb._star_utils import (
    LoopTable,
    as_float,
    as_int,
    as_none,
    parse_fallback_loops,
)
from atypemu.errors import ParseError
from atypemu.types import NOERestraint, canonical_atom_name


def _is_ambiguous_atom(atom_id: str) -> bool:
    """Return whether an atom identifier looks ambiguous."""
    return any(token in atom_id for token in [",", ";", "{", "}", "*"])


class NoeRestraintParser:
    """Parse simple unique NOE restraints from a local STAR file."""

    def parse(self, path: str | Path) -> list[NOERestraint]:
        """Parse NOE restraints.

        Args:
            path: Input STAR-like file path.

        Returns:
            Parsed simple NOE restraints.
        """
        loops = self._load_loops(Path(path))
        targets: list[NOERestraint] = []
        valid_categories = {"Gen_dist_constraint", "Distance_constraint"}

        for loop in loops:
            if loop.category() not in valid_categories:
                continue
            for row in loop.iter_dicts():
                atom_1 = canonical_atom_name(
                    as_none(row.get("Atom_ID_1"))
                    or as_none(row.get("Atom_name_1"))
                    or ""
                )
                atom_2 = canonical_atom_name(
                    as_none(row.get("Atom_ID_2"))
                    or as_none(row.get("Atom_name_2"))
                    or ""
                )
                if not atom_1 or not atom_2:
                    continue
                if _is_ambiguous_atom(atom_1) or _is_ambiguous_atom(atom_2):
                    continue

                seq_id_1 = as_int(row.get("Seq_ID_1"))
                seq_id_2 = as_int(row.get("Seq_ID_2"))
                comp_id_1 = as_none(row.get("Comp_ID_1"))
                comp_id_2 = as_none(row.get("Comp_ID_2"))
                if any(
                    value is None
                    for value in (seq_id_1, seq_id_2, comp_id_1, comp_id_2)
                ):
                    continue

                lower_bound = as_float(
                    row.get("Distance_lower_bound_val")
                    or row.get("Lower_limit")
                    or row.get("Distance_lower_bound")
                )
                upper_bound = as_float(
                    row.get("Distance_upper_bound_val")
                    or row.get("Upper_limit")
                    or row.get("Distance_upper_bound")
                )
                if upper_bound is None and lower_bound is None:
                    continue

                if upper_bound is not None and lower_bound is None:
                    target_value = upper_bound / 2.0
                    uncertainty = upper_bound / 2.0
                elif upper_bound is not None and lower_bound is not None:
                    target_value = (lower_bound + upper_bound) / 2.0
                    uncertainty = (upper_bound - lower_bound) / 2.0
                else:
                    raise ParseError(
                        "NOE restraints with only a lower bound are unsupported."
                    )

                targets.append(
                    NOERestraint(
                        seq_id_1=seq_id_1,
                        comp_id_1=comp_id_1,
                        atom_id_1=atom_1,
                        seq_id_2=seq_id_2,
                        comp_id_2=comp_id_2,
                        atom_id_2=atom_2,
                        target_value=target_value,
                        uncertainty=uncertainty,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        chain_id_1=as_none(row.get("Auth_asym_ID_1")),
                        chain_id_2=as_none(row.get("Auth_asym_ID_2")),
                    )
                )

        return targets

    def _load_loops(self, path: Path) -> list[LoopTable]:
        """Load loop tables from a STAR-like NOE file."""
        return parse_fallback_loops(path.read_text())
