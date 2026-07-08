"""Shared geometry helpers for structure-derived interface annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

try:
    import freesasa
except ImportError:  # pragma: no cover - guarded at runtime in callers
    freesasa = None


@dataclass(slots=True)
class ResidueNode:
    """One residue node used in a residue-level contact graph."""

    chain_id: str
    residue_id: str
    resname: str
    atom_coords: np.ndarray


@dataclass(slots=True)
class ResidueContactGraph:
    """Residue-level contact graph between two polymer chains."""

    chain_id_a: str
    chain_id_b: str
    residue_ids_a: list[str]
    residue_ids_b: list[str]
    residue_names_a: list[str]
    residue_names_b: list[str]
    contact_index_pairs: list[tuple[int, int]]
    interface_index_a: list[int]
    interface_index_b: list[int]
    model_selection: str = "first_model"


@dataclass(slots=True)
class PairSurfaceArea:
    """Surface-area summary for one chain pair."""

    chain_id_a: str
    chain_id_b: str
    sasa_chain_a: float
    sasa_chain_b: float
    sasa_complex: float
    buried_surface_area: float
    interface_area: float
    backend: str = "freesasa"


def load_first_model(cif_path: str | Path) -> dict[str, list[str]]:
    """Load one CIF payload and keep only atom rows from the first model."""
    payload = MMCIF2Dict(str(cif_path))
    atom_count = len(
        _ensure_list(
            payload.get("_atom_site.auth_asym_id")
            or payload.get("_atom_site.label_asym_id")
        )
    )
    if atom_count == 0:
        raise ValueError(f"No atom rows found in CIF file: {cif_path}")

    model_numbers = _expand_to_length(
        payload.get("_atom_site.pdbx_PDB_model_num"),
        atom_count,
        default="1",
    )
    first_model = next((value for value in model_numbers if value.strip()), "1")
    keep_indices = [
        index
        for index, value in enumerate(model_numbers)
        if (value or "1") == first_model
    ]
    if not keep_indices:
        raise ValueError(f"No first-model atom rows found in CIF file: {cif_path}")

    filtered: dict[str, list[str]] = {}
    for key, value in payload.items():
        values = _ensure_list(value)
        if len(values) == atom_count:
            filtered[key] = [values[index] for index in keep_indices]
        else:
            filtered[key] = values
    return filtered


def extract_polymer_residue_nodes(
    model_payload: dict[str, list[str]],
) -> dict[str, list[ResidueNode]]:
    """Extract polymer residue nodes keyed by chain ID from one CIF payload."""
    atom_chain_ids = _ensure_list(
        model_payload.get("_atom_site.auth_asym_id")
        or model_payload.get("_atom_site.label_asym_id")
    )
    atom_groups = _expand_to_length(
        model_payload.get("_atom_site.group_PDB"),
        len(atom_chain_ids),
        default="ATOM",
    )
    atom_seq_ids = _expand_to_length(
        model_payload.get("_atom_site.auth_seq_id")
        or model_payload.get("_atom_site.label_seq_id"),
        len(atom_chain_ids),
    )
    atom_comp_ids = _expand_to_length(
        model_payload.get("_atom_site.label_comp_id"),
        len(atom_chain_ids),
    )
    atom_ins_codes = _expand_to_length(
        model_payload.get("_atom_site.pdbx_PDB_ins_code"),
        len(atom_chain_ids),
    )
    atom_x = _expand_to_length(
        model_payload.get("_atom_site.Cartn_x"),
        len(atom_chain_ids),
    )
    atom_y = _expand_to_length(
        model_payload.get("_atom_site.Cartn_y"),
        len(atom_chain_ids),
    )
    atom_z = _expand_to_length(
        model_payload.get("_atom_site.Cartn_z"),
        len(atom_chain_ids),
    )

    polymer_chains = _polymer_chain_ids(model_payload)
    residue_atoms: dict[tuple[str, str, str], list[np.ndarray]] = {}

    for chain_id, group, seq_id, comp_id, ins_code, x, y, z in zip(
        atom_chain_ids,
        atom_groups,
        atom_seq_ids,
        atom_comp_ids,
        atom_ins_codes,
        atom_x,
        atom_y,
        atom_z,
    ):
        normalized_chain = _clean_text(chain_id)
        normalized_group = _clean_text(group)
        normalized_seq = _clean_text(seq_id)
        normalized_comp = _clean_text(comp_id)
        if normalized_chain is None or normalized_chain not in polymer_chains:
            continue
        if normalized_group != "ATOM":
            continue
        if normalized_seq is None or normalized_comp is None:
            continue
        try:
            coords = np.asarray(
                [float(x), float(y), float(z)],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            continue
        residue_id = _residue_identifier(normalized_seq, ins_code)
        residue_key = (normalized_chain, residue_id, normalized_comp)
        residue_atoms.setdefault(residue_key, []).append(coords)

    chain_nodes: dict[str, list[ResidueNode]] = {}
    for (chain_id, residue_id, resname), coords in sorted(
        residue_atoms.items(),
        key=lambda item: (item[0][0], _sort_residue_id(item[0][1])),
    ):
        chain_nodes.setdefault(chain_id, []).append(
            ResidueNode(
                chain_id=chain_id,
                residue_id=residue_id,
                resname=resname,
                atom_coords=np.stack(coords),
            )
        )
    return chain_nodes


def compute_residue_contact_graph(
    residues_a: list[ResidueNode],
    residues_b: list[ResidueNode],
    distance_cutoff: float = 5.0,
) -> ResidueContactGraph:
    """Compute a residue-level contact graph between two polymer chains."""
    if not residues_a or not residues_b:
        raise ValueError("Residue contact graphs require two non-empty chains.")

    cutoff_sq = float(distance_cutoff) ** 2
    contact_pairs: list[tuple[int, int]] = []
    interface_a: set[int] = set()
    interface_b: set[int] = set()

    for index_a, residue_a in enumerate(residues_a):
        for index_b, residue_b in enumerate(residues_b):
            deltas = (
                residue_a.atom_coords[:, np.newaxis, :]
                - residue_b.atom_coords[np.newaxis, :, :]
            )
            if np.any(np.sum(deltas * deltas, axis=2) <= cutoff_sq):
                contact_pairs.append((index_a, index_b))
                interface_a.add(index_a)
                interface_b.add(index_b)

    return ResidueContactGraph(
        chain_id_a=residues_a[0].chain_id,
        chain_id_b=residues_b[0].chain_id,
        residue_ids_a=[node.residue_id for node in residues_a],
        residue_ids_b=[node.residue_id for node in residues_b],
        residue_names_a=[node.resname for node in residues_a],
        residue_names_b=[node.resname for node in residues_b],
        contact_index_pairs=contact_pairs,
        interface_index_a=sorted(interface_a),
        interface_index_b=sorted(interface_b),
    )


def write_contact_graph_npz(path: str | Path, graph: ResidueContactGraph) -> None:
    """Serialize one residue-level contact graph to a compressed NPZ file."""
    pair_index_a = np.asarray(
        [index_a for index_a, _ in graph.contact_index_pairs],
        dtype=np.int32,
    )
    pair_index_b = np.asarray(
        [index_b for _, index_b in graph.contact_index_pairs],
        dtype=np.int32,
    )
    np.savez_compressed(
        path,
        chain_id_a=np.asarray(graph.chain_id_a),
        chain_id_b=np.asarray(graph.chain_id_b),
        residue_ids_a=np.asarray(graph.residue_ids_a, dtype=str),
        residue_ids_b=np.asarray(graph.residue_ids_b, dtype=str),
        residue_names_a=np.asarray(graph.residue_names_a, dtype=str),
        residue_names_b=np.asarray(graph.residue_names_b, dtype=str),
        contact_index_a=pair_index_a,
        contact_index_b=pair_index_b,
        interface_index_a=np.asarray(graph.interface_index_a, dtype=np.int32),
        interface_index_b=np.asarray(graph.interface_index_b, dtype=np.int32),
        model_selection=np.asarray(graph.model_selection),
    )


def compute_chain_surface_area(
    model_payload: dict[str, list[str]],
    chain_id: str,
) -> float:
    """Compute solvent-accessible surface area for one polymer chain."""
    atom_rows = list(_iter_polymer_atom_rows(model_payload, [chain_id]))
    if not atom_rows:
        raise ValueError(f"No polymer atom rows found for chain {chain_id}.")
    return _compute_sasa_from_atom_rows(atom_rows, chain_id_map={chain_id: "A"})


def compute_pair_surface_area(
    model_payload: dict[str, list[str]],
    chain_id_a: str,
    chain_id_b: str,
    sasa_chain_a: float | None = None,
    sasa_chain_b: float | None = None,
) -> PairSurfaceArea:
    """Compute buried and interface surface areas for one chain pair."""
    if freesasa is None:  # pragma: no cover - exercised only in missing-backend envs
        raise RuntimeError(
            "FreeSASA is required for interface-area calculation. "
            "Install the 'freesasa' Python package first."
        )

    atom_rows = list(_iter_polymer_atom_rows(model_payload, [chain_id_a, chain_id_b]))
    if not atom_rows:
        raise ValueError(
            f"No polymer atom rows found for pair {chain_id_a}/{chain_id_b}."
        )

    chain_area_a = (
        float(sasa_chain_a)
        if sasa_chain_a is not None
        else compute_chain_surface_area(model_payload, chain_id_a)
    )
    chain_area_b = (
        float(sasa_chain_b)
        if sasa_chain_b is not None
        else compute_chain_surface_area(model_payload, chain_id_b)
    )
    complex_area = _compute_sasa_from_atom_rows(
        atom_rows,
        chain_id_map={chain_id_a: "A", chain_id_b: "B"},
    )
    buried_surface_area = max(0.0, chain_area_a + chain_area_b - complex_area)
    interface_area = buried_surface_area / 2.0
    return PairSurfaceArea(
        chain_id_a=chain_id_a,
        chain_id_b=chain_id_b,
        sasa_chain_a=chain_area_a,
        sasa_chain_b=chain_area_b,
        sasa_complex=complex_area,
        buried_surface_area=buried_surface_area,
        interface_area=interface_area,
    )


def _polymer_chain_ids(payload: dict[str, list[str]]) -> set[str]:
    """Return polymer chain identifiers described by the CIF entity section."""
    strand_entries = _ensure_list(payload.get("_entity_poly.pdbx_strand_id"))
    chain_ids: set[str] = set()
    for value in strand_entries:
        for chain_id in str(value).split(","):
            normalized = _clean_text(chain_id)
            if normalized:
                chain_ids.add(normalized)
    return chain_ids


def _iter_polymer_atom_rows(
    payload: dict[str, list[str]],
    selected_chain_ids: list[str],
) -> list[dict[str, Any]]:
    """Yield atom rows for selected polymer chains in CIF order."""
    atom_chain_ids = _ensure_list(
        payload.get("_atom_site.auth_asym_id")
        or payload.get("_atom_site.label_asym_id")
    )
    atom_groups = _expand_to_length(
        payload.get("_atom_site.group_PDB"),
        len(atom_chain_ids),
        default="ATOM",
    )
    atom_seq_ids = _expand_to_length(
        payload.get("_atom_site.auth_seq_id") or payload.get("_atom_site.label_seq_id"),
        len(atom_chain_ids),
    )
    atom_comp_ids = _expand_to_length(
        payload.get("_atom_site.label_comp_id"),
        len(atom_chain_ids),
    )
    atom_ins_codes = _expand_to_length(
        payload.get("_atom_site.pdbx_PDB_ins_code"),
        len(atom_chain_ids),
    )
    atom_names = _expand_to_length(
        payload.get("_atom_site.auth_atom_id")
        or payload.get("_atom_site.label_atom_id"),
        len(atom_chain_ids),
    )
    atom_elements = _expand_to_length(
        payload.get("_atom_site.type_symbol"),
        len(atom_chain_ids),
    )
    atom_x = _expand_to_length(
        payload.get("_atom_site.Cartn_x"),
        len(atom_chain_ids),
    )
    atom_y = _expand_to_length(
        payload.get("_atom_site.Cartn_y"),
        len(atom_chain_ids),
    )
    atom_z = _expand_to_length(
        payload.get("_atom_site.Cartn_z"),
        len(atom_chain_ids),
    )
    atom_occupancy = _expand_to_length(
        payload.get("_atom_site.occupancy"),
        len(atom_chain_ids),
        default="1.00",
    )
    atom_b_factor = _expand_to_length(
        payload.get("_atom_site.B_iso_or_equiv"),
        len(atom_chain_ids),
        default="0.00",
    )

    polymer_chains = _polymer_chain_ids(payload)
    selected = set(selected_chain_ids)
    rows: list[dict[str, Any]] = []
    for (
        chain_id,
        group,
        seq_id,
        comp_id,
        ins_code,
        atom_name,
        element,
        x,
        y,
        z,
        occupancy,
        b_factor,
    ) in zip(
        atom_chain_ids,
        atom_groups,
        atom_seq_ids,
        atom_comp_ids,
        atom_ins_codes,
        atom_names,
        atom_elements,
        atom_x,
        atom_y,
        atom_z,
        atom_occupancy,
        atom_b_factor,
    ):
        normalized_chain = _clean_text(chain_id)
        normalized_group = _clean_text(group)
        normalized_seq = _clean_text(seq_id)
        normalized_comp = _clean_text(comp_id)
        if normalized_chain is None or normalized_chain not in selected:
            continue
        if normalized_chain not in polymer_chains:
            continue
        if normalized_group != "ATOM":
            continue
        if normalized_seq is None or normalized_comp is None:
            continue
        try:
            rows.append(
                {
                    "chain_id": normalized_chain,
                    "residue_id": _residue_identifier(normalized_seq, ins_code),
                    "resname": normalized_comp,
                    "atom_name": _clean_text(atom_name) or "X",
                    "element": (_clean_text(element) or "C").upper(),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "occupancy": float(occupancy),
                    "b_factor": float(b_factor),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def _compute_sasa_from_atom_rows(
    atom_rows: list[dict[str, Any]],
    chain_id_map: dict[str, str],
) -> float:
    """Compute SASA for one atom-row selection by writing a temporary PDB file."""
    pdb_lines = _atom_rows_to_pdb_lines(atom_rows, chain_id_map)
    with NamedTemporaryFile(
        "w",
        suffix=".pdb",
        encoding="utf-8",
        delete=True,
    ) as handle:
        handle.write("".join(pdb_lines))
        handle.flush()
        structure = freesasa.Structure(
            handle.name,
            None,
            {
                "hetatm": False,
                "hydrogen": False,
                "join-models": False,
                "skip-unknown": False,
                "halt-at-unknown": False,
            },
        )
        result = freesasa.calc(structure)
    return float(result.totalArea())


def _atom_rows_to_pdb_lines(
    atom_rows: list[dict[str, Any]],
    chain_id_map: dict[str, str],
) -> list[str]:
    """Convert one atom-row selection into PDB-formatted text lines."""
    residue_index_map: dict[tuple[str, str], int] = {}
    next_residue_index_by_chain: dict[str, int] = {}
    lines: list[str] = []

    for serial, row in enumerate(atom_rows, start=1):
        original_chain_id = str(row["chain_id"])
        chain_id = chain_id_map[original_chain_id]
        residue_key = (original_chain_id, str(row["residue_id"]))
        if residue_key not in residue_index_map:
            next_index = next_residue_index_by_chain.get(original_chain_id, 0) + 1
            next_residue_index_by_chain[original_chain_id] = next_index
            residue_index_map[residue_key] = next_index
        residue_index = residue_index_map[residue_key]
        atom_name = str(row["atom_name"])[:4]
        resname = str(row["resname"])[:3]
        element = str(row["element"])[:2]
        lines.append(
            (
                f"ATOM  {serial:>5} {atom_name:>4} {resname:>3} {chain_id:1}"
                f"{residue_index:>4}    {float(row['x']):>8.3f}{float(row['y']):>8.3f}"
                f"{float(row['z']):>8.3f}{float(row['occupancy']):>6.2f}"
                f"{float(row['b_factor']):>6.2f}          {element:>2}\n"
            )
        )
    lines.append("END\n")
    return lines


def _ensure_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Return one CIF scalar or sequence as a plain string list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _expand_to_length(
    value: str | list[str] | tuple[str, ...] | None,
    length: int,
    default: str = "",
) -> list[str]:
    """Expand one scalar or short CIF column to a target row count."""
    values = _ensure_list(value)
    if not values:
        return [default] * length
    if len(values) == 1 and length > 1:
        return values * length
    if len(values) < length:
        return values + [default] * (length - len(values))
    return values[:length]


def _clean_text(value: object) -> str | None:
    """Normalize one CIF scalar into stripped text or null."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"?", ".", "None"}:
        return None
    return text


def _residue_identifier(seq_id: str, insertion_code: str | None) -> str:
    """Build one stable residue identifier from sequence and insertion code."""
    normalized_insertion = _clean_text(insertion_code)
    if normalized_insertion is not None:
        return f"{seq_id}{normalized_insertion}"
    return seq_id


def _sort_residue_id(residue_id: str) -> tuple[int, str]:
    """Sort residue identifiers numerically when possible."""
    digits = "".join(character for character in residue_id if character.isdigit())
    if digits:
        return int(digits), residue_id
    return 0, residue_id
