"""Derived heavy-geometry features for MFIB CIF assets."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.mfib.config import SOURCE_ID
from atypemu.databases.mfib.layout import (
    ensure_mfib_workspace,
    mfib_contact_map_root,
    mfib_debug_root,
    mfib_interface_residue_set_root,
    mfib_tables_root,
)
from atypemu.databases.mfib.records import (
    MfibGeometryEntryFeatureRecord,
    MfibInterfacePairRecord,
    MfibInterfaceResidueRecord,
)
from atypemu.databases.mfib.tables import write_tsv
from atypemu.structures.geometry import (
    compute_chain_surface_area,
    compute_pair_surface_area,
    compute_residue_contact_graph,
    extract_polymer_residue_nodes,
    load_first_model,
    write_contact_graph_npz,
)


INTERFACE_PAIR_COLUMNS = list(MfibInterfacePairRecord.__dataclass_fields__)
INTERFACE_RESIDUE_COLUMNS = list(MfibInterfaceResidueRecord.__dataclass_fields__)
GEOMETRY_ENTRY_COLUMNS = list(MfibGeometryEntryFeatureRecord.__dataclass_fields__)


def compute_mfib_geometry_features(
    mfib_root: str | Path,
    refresh: bool = False,
    contact_level: str = "residue",
    distance_cutoff: float = 5.0,
    compute_interface_area: bool = False,
    compute_contact_map: bool = True,
) -> dict[str, Any]:
    """Compute derived MFIB chain-pair geometry features from CIF assets."""
    if contact_level != "residue":
        raise ValueError("MFIB geometry v1 only supports residue-level contacts.")

    source_root = mfib_root if isinstance(mfib_root, Path) else Path(mfib_root)
    if source_root.name != "mfib":
        source_root = source_root / "mfib"
    ensure_mfib_workspace(source_root)

    tables_root = mfib_tables_root(source_root)
    entries = _read_tsv(tables_root / "entries.tsv")
    assets = _read_tsv(tables_root / "assets.tsv")
    cif_chain_features = _read_tsv(tables_root / "cif_chain_features.tsv")

    polymer_chains = cif_chain_features.loc[
        cif_chain_features["is_polypeptide"].apply(_truthy)
    ].copy()
    chain_groups = {
        entry_uid: sorted(
            {
                str(value)
                for value in frame["chain_id"].dropna().tolist()
                if str(value).strip()
            }
        )
        for entry_uid, frame in polymer_chains.groupby("entry_uid", dropna=False)
    }
    asset_map = {
        str(row["entry_uid"]): row
        for row in assets.to_dict(orient="records")
        if row.get("asset_kind") == "cif" and _truthy(row.get("available"))
    }

    contact_map_root = mfib_contact_map_root(source_root)
    interface_set_root = mfib_interface_residue_set_root(source_root)
    contact_map_root.mkdir(parents=True, exist_ok=True)
    interface_set_root.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []

    for entry_row in entries.to_dict(orient="records"):
        entry_uid = str(entry_row["entry_uid"])
        entry_asset = asset_map.get(entry_uid)
        polymer_chain_ids = chain_groups.get(entry_uid, [])
        entry_pair_rows: list[dict[str, Any]] = []

        if entry_asset is not None and len(polymer_chain_ids) >= 2:
            cif_path = source_root.parent / str(entry_asset["asset_path"])
            model = load_first_model(cif_path)
            residue_nodes = extract_polymer_residue_nodes(model)
            chain_surface_area_cache: dict[str, float] = {}

            for chain_id_a, chain_id_b in combinations(polymer_chain_ids, 2):
                if chain_id_a not in residue_nodes or chain_id_b not in residue_nodes:
                    continue
                pair_uid = (
                    f"{SOURCE_ID}:{entry_row['accession']}:{chain_id_a}:{chain_id_b}"
                )
                graph = compute_residue_contact_graph(
                    residues_a=residue_nodes[chain_id_a],
                    residues_b=residue_nodes[chain_id_b],
                    distance_cutoff=distance_cutoff,
                )

                contact_map_path = None
                if compute_contact_map:
                    contact_map_file = (
                        contact_map_root / f"{_pair_filename(pair_uid)}.npz"
                    )
                    if refresh or not contact_map_file.exists():
                        write_contact_graph_npz(contact_map_file, graph)
                    contact_map_path = str(
                        contact_map_file.relative_to(source_root.parent)
                    )

                residue_set_file = (
                    interface_set_root / f"{_pair_filename(pair_uid)}.json"
                )
                residue_payload = {
                    "pair_uid": pair_uid,
                    "chain_id_a": chain_id_a,
                    "chain_id_b": chain_id_b,
                    "model_selection": "first_model",
                    "contact_level": contact_level,
                    "distance_cutoff": distance_cutoff,
                    "residue_ids_a": [
                        graph.residue_ids_a[index] for index in graph.interface_index_a
                    ],
                    "residue_ids_b": [
                        graph.residue_ids_b[index] for index in graph.interface_index_b
                    ],
                }
                if refresh or not residue_set_file.exists():
                    residue_set_file.write_text(
                        json.dumps(residue_payload, indent=2, sort_keys=True)
                    )
                residue_set_path = str(residue_set_file.relative_to(source_root.parent))

                interface_area = None
                buried_surface_area = None
                if compute_interface_area:
                    chain_surface_area_cache.setdefault(
                        chain_id_a,
                        compute_chain_surface_area(model, chain_id_a),
                    )
                    chain_surface_area_cache.setdefault(
                        chain_id_b,
                        compute_chain_surface_area(model, chain_id_b),
                    )
                    pair_surface_area = compute_pair_surface_area(
                        model_payload=model,
                        chain_id_a=chain_id_a,
                        chain_id_b=chain_id_b,
                        sasa_chain_a=chain_surface_area_cache[chain_id_a],
                        sasa_chain_b=chain_surface_area_cache[chain_id_b],
                    )
                    interface_area = pair_surface_area.interface_area
                    buried_surface_area = pair_surface_area.buried_surface_area

                pair_row = MfibInterfacePairRecord(
                    accession=str(entry_row["accession"]),
                    entry_uid=entry_uid,
                    pair_uid=pair_uid,
                    source_id=SOURCE_ID,
                    native_id=str(entry_row["native_id"]),
                    parent_uid=entry_uid,
                    pdb_id=str(entry_row["pdb_id"]),
                    chain_id_a=chain_id_a,
                    chain_id_b=chain_id_b,
                    model_selection="first_model",
                    contact_level=contact_level,
                    distance_cutoff=float(distance_cutoff),
                    interface_area=interface_area,
                    buried_surface_area=buried_surface_area,
                    contact_count=len(graph.contact_index_pairs),
                    interface_residue_count_a=len(graph.interface_index_a),
                    interface_residue_count_b=len(graph.interface_index_b),
                    contact_map_path=contact_map_path,
                    interface_residue_set_path=residue_set_path,
                ).as_dict()
                pair_rows.append(pair_row)
                entry_pair_rows.append(pair_row)

                for index in graph.interface_index_a:
                    residue_rows.append(
                        MfibInterfaceResidueRecord(
                            accession=str(entry_row["accession"]),
                            entry_uid=entry_uid,
                            pair_uid=pair_uid,
                            chain_uid=f"{entry_uid}:{chain_id_a}",
                            source_id=SOURCE_ID,
                            native_id=str(entry_row["native_id"]),
                            parent_uid=entry_uid,
                            chain_id=chain_id_a,
                            residue_id=graph.residue_ids_a[index],
                            resname=graph.residue_names_a[index],
                            is_interface=True,
                        ).as_dict()
                    )
                for index in graph.interface_index_b:
                    residue_rows.append(
                        MfibInterfaceResidueRecord(
                            accession=str(entry_row["accession"]),
                            entry_uid=entry_uid,
                            pair_uid=pair_uid,
                            chain_uid=f"{entry_uid}:{chain_id_b}",
                            source_id=SOURCE_ID,
                            native_id=str(entry_row["native_id"]),
                            parent_uid=entry_uid,
                            chain_id=chain_id_b,
                            residue_id=graph.residue_ids_b[index],
                            resname=graph.residue_names_b[index],
                            is_interface=True,
                        ).as_dict()
                    )

        geometry_rows.append(
            _build_geometry_entry_row(
                entry_row=entry_row,
                pair_rows=entry_pair_rows,
                contact_level=contact_level,
                distance_cutoff=distance_cutoff,
            )
        )

    write_tsv(
        tables_root / "interface_pairs.tsv",
        pair_rows,
        INTERFACE_PAIR_COLUMNS,
    )
    write_tsv(
        tables_root / "interface_residues.tsv",
        residue_rows,
        INTERFACE_RESIDUE_COLUMNS,
    )
    write_tsv(
        tables_root / "geometry_entry_features.tsv",
        geometry_rows,
        GEOMETRY_ENTRY_COLUMNS,
    )

    summary = {
        "source_id": SOURCE_ID,
        "source_root": str(source_root),
        "entry_rows": len(geometry_rows),
        "pair_rows": len(pair_rows),
        "interface_residue_rows": len(residue_rows),
        "contact_level": contact_level,
        "distance_cutoff": float(distance_cutoff),
        "model_selection": "first_model",
        "compute_contact_map": compute_contact_map,
        "compute_interface_area": compute_interface_area,
        "interface_area_backend": "freesasa" if compute_interface_area else None,
        "interface_area_pair_rows": sum(
            1 for row in pair_rows if row.get("interface_area") is not None
        ),
    }
    debug_root = mfib_debug_root(source_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    (debug_root / "mfib_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read one MFIB curated TSV table into a nullable dataframe."""
    dataframe = pd.read_csv(path, sep="\t").replace({r"^\s*$": pd.NA}, regex=True)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _truthy(value: Any) -> bool:
    """Return whether one dataframe-like value should be treated as true."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _pair_filename(pair_uid: str) -> str:
    """Return a filesystem-safe filename stem for one pair UID."""
    return pair_uid.replace(":", "__")


def _build_geometry_entry_row(
    entry_row: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    contact_level: str,
    distance_cutoff: float,
) -> dict[str, Any]:
    """Build one accession-level geometry summary row."""
    contact_counts = [int(row["contact_count"]) for row in pair_rows]
    interface_areas = [
        row["interface_area"]
        for row in pair_rows
        if row.get("interface_area") is not None
    ]
    return MfibGeometryEntryFeatureRecord(
        accession=str(entry_row["accession"]),
        entry_uid=str(entry_row["entry_uid"]),
        source_id=SOURCE_ID,
        native_id=str(entry_row["native_id"]),
        geometry_available=bool(pair_rows),
        pair_count=len(pair_rows),
        max_interface_area=max(interface_areas) if interface_areas else None,
        total_interface_area=sum(interface_areas) if interface_areas else None,
        max_contact_count=max(contact_counts) if contact_counts else None,
        model_selection="first_model",
        contact_level=contact_level,
        distance_cutoff=float(distance_cutoff),
    ).as_dict()
