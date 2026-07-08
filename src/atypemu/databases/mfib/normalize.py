"""Deterministic normalization helpers for MFIB bundle downloads."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from atypemu.databases.mfib.config import SOURCE_ID
from atypemu.databases.mfib.records import (
    MfibAssetRecord,
    MfibChainRecord,
    MfibCifChainFeatureRecord,
    MfibCifEntryFeatureRecord,
    MfibCrossrefRecord,
    MfibEntryRecord,
    MfibEvidenceRecord,
    MfibGoRecord,
    MfibRegionRecord,
    MfibRelatedRecord,
    MfibSearchRecord,
)


ENTRY_COLUMNS = list(MfibEntryRecord.__dataclass_fields__)
CHAIN_COLUMNS = list(MfibChainRecord.__dataclass_fields__)
REGION_COLUMNS = list(MfibRegionRecord.__dataclass_fields__)
EVIDENCE_COLUMNS = list(MfibEvidenceRecord.__dataclass_fields__)
GO_COLUMNS = list(MfibGoRecord.__dataclass_fields__)
RELATED_COLUMNS = list(MfibRelatedRecord.__dataclass_fields__)
CROSSREF_COLUMNS = list(MfibCrossrefRecord.__dataclass_fields__)
SEARCH_COLUMNS = list(MfibSearchRecord.__dataclass_fields__)
ASSET_COLUMNS = list(MfibAssetRecord.__dataclass_fields__)
CIF_ENTRY_COLUMNS = list(MfibCifEntryFeatureRecord.__dataclass_fields__)
CIF_CHAIN_COLUMNS = list(MfibCifChainFeatureRecord.__dataclass_fields__)

_FIELD_RE = re.compile(r"^\[(?P<key>.+?)\]=(.*)$")
_BOUNDARY_RE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)$")
_WHITESPACE_RE = re.compile(r"\s+")


def parse_mfib_json_archive(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse the full MFIB JSON archive keyed by accession."""
    archive_path = Path(path)
    entries: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".json"):
                continue
            payload = json.loads(archive.read(member))
            entry = payload.get("entry") or {}
            accession = clean_text(entry.get("accession"))
            if accession is None:
                continue
            entries[accession] = entry
    return entries


def parse_mfib_txt_entries(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse the flat MFIB TXT export keyed by accession."""
    txt_path = Path(path)
    entries: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for raw_line in txt_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if current.get("Accession"):
                entries[current["Accession"]] = dict(current)
                current.clear()
            continue
        if line == "[Entry]":
            if current.get("Accession"):
                entries[current["Accession"]] = dict(current)
                current.clear()
            continue
        match = _FIELD_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        value = raw_line.split("=", maxsplit=1)[1].strip()
        current[key] = value
    if current.get("Accession"):
        entries[current["Accession"]] = dict(current)
    return entries


def normalize_entry_bundle(
    accession: str,
    json_entry: dict[str, Any] | None,
    txt_entry: dict[str, str] | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Normalize one accession into curated MFIB table rows."""
    entry_uid = f"{SOURCE_ID}:{accession}"
    entry_row = build_entry_row(accession, entry_uid, json_entry, txt_entry)
    chain_rows = build_chain_rows(accession, entry_uid, json_entry, txt_entry)
    region_rows = build_region_rows(accession, entry_uid, json_entry, chain_rows)
    evidence_rows = build_evidence_rows(
        accession=accession,
        entry_uid=entry_uid,
        json_entry=json_entry,
        txt_entry=txt_entry,
        chain_rows=chain_rows,
    )
    go_rows = build_go_rows(accession, entry_uid, json_entry)
    related_rows = build_related_rows(accession, entry_uid, json_entry, txt_entry)
    crossref_rows = build_crossref_rows(entry_row, chain_rows)
    search_rows = build_search_rows(
        entry_row=entry_row,
        chain_rows=chain_rows,
        go_rows=go_rows,
        related_rows=related_rows,
    )
    return (
        entry_row,
        chain_rows,
        region_rows,
        evidence_rows,
        go_rows,
        related_rows,
        crossref_rows,
        search_rows,
    )


def build_entry_row(
    accession: str,
    entry_uid: str,
    json_entry: dict[str, Any] | None,
    txt_entry: dict[str, str] | None,
) -> dict[str, Any]:
    """Build the accession-level MFIB row."""
    general = (json_entry or {}).get("general") or {}
    publication = general.get("publication") or {}
    macromolecules = (json_entry or {}).get("macromolecules") or {}
    macro_general = macromolecules.get("general") or {}
    evidence = (json_entry or {}).get("evidence") or {}
    txt_entry = txt_entry or {}

    return MfibEntryRecord(
        accession=accession,
        entry_uid=entry_uid,
        source_id=SOURCE_ID,
        native_id=accession,
        name=first_text(general.get("name"), txt_entry.get("Entry name")),
        pdb_id=uppercase_text(
            first_text(general.get("pdb_id"), txt_entry.get("PDB ID"))
        ),
        exp_method=first_text(general.get("exp_method"), txt_entry.get("ExpTech")),
        resolution=first_float(general.get("resolution"), txt_entry.get("Resolution")),
        assembly=first_text(general.get("assembly"), txt_entry.get("Assembly")),
        total_number_of_chains=first_int(
            macro_general.get("nr_of_chains"),
            txt_entry.get("Total number of chains"),
        ),
        number_of_unique_proteins=first_int(
            macro_general.get("nr_of_unique_protein_segments"),
            txt_entry.get("Number of unique proteins"),
        ),
        pdb_note=first_text(macro_general.get("note"), txt_entry.get("PDB note")),
        class_name=first_text(macro_general.get("class"), txt_entry.get("Class")),
        subclass_name=first_text(
            macro_general.get("subclass"),
            txt_entry.get("Subclass"),
        ),
        sequence_domain=first_text(
            evidence.get("sequence_domain"),
            txt_entry.get("Sequence domain"),
        ),
        evidence_level=first_text(
            evidence.get("evidence_level"),
            txt_entry.get("Evidence level"),
        ),
        evidence_coverage=first_text(
            evidence.get("evidence_coverage"),
            txt_entry.get("Evidence coverage"),
        ),
        evidence_text=first_text(
            evidence.get("complex_evidence"),
            txt_entry.get("Evidence text"),
        ),
        source_organism=first_text(
            general.get("source_organism"),
            txt_entry.get("Source organism"),
        ),
        publication_pmid=first_text(publication.get("pmid")),
        publication_authors=first_text(publication.get("authors")),
        publication_title=first_text(publication.get("title")),
        publication_journal=first_text(publication.get("journal")),
        publication_year=first_text(publication.get("year")),
        publication_volume=first_text(publication.get("volume")),
        publication_issue=first_text(publication.get("issue")),
        publication_pages=first_text(publication.get("pages")),
        publication_abstract=first_text(publication.get("abstract")),
    ).as_dict()


def build_chain_rows(
    accession: str,
    entry_uid: str,
    json_entry: dict[str, Any] | None,
    txt_entry: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Build chain-level MFIB rows from JSON with TXT fallback."""
    txt_entry = txt_entry or {}
    json_chain_map = {
        chain_id: payload
        for chain_id, payload in _json_chain_payloads(json_entry).items()
    }
    chain_ids = sorted(set(json_chain_map) | set(_txt_chain_ids(txt_entry)))
    rows: list[dict[str, Any]] = []
    for chain_id in chain_ids:
        payload = json_chain_map.get(chain_id) or {}
        uniprot = payload.get("uniprot") or {}
        txt_boundaries = _parse_boundaries(
            txt_entry.get(f"UniProt boundaries chain {chain_id}")
        )
        sequence = first_text(
            uniprot.get("sequence"),
            txt_entry.get(f"UniProt sequence chain {chain_id}"),
        )
        rows.append(
            MfibChainRecord(
                accession=accession,
                entry_uid=entry_uid,
                chain_uid=f"{entry_uid}:{chain_id}",
                source_id=SOURCE_ID,
                native_id=accession,
                parent_uid=entry_uid,
                chain_id=chain_id,
                chain_name=first_text(
                    payload.get("name"),
                    txt_entry.get(f"Name chain {chain_id}"),
                ),
                source_organism=first_text(
                    payload.get("source_organism"),
                    txt_entry.get(f"Source organism chain {chain_id}"),
                ),
                uniprot_id=uppercase_text(
                    first_text(
                        uniprot.get("id"),
                        txt_entry.get(f"UniProt ID chain {chain_id}"),
                    )
                ),
                uniprot_start=first_int(
                    uniprot.get("start"),
                    txt_boundaries[0],
                ),
                uniprot_end=first_int(
                    uniprot.get("end"),
                    txt_boundaries[1],
                ),
                uniprot_coverage=first_text(
                    uniprot.get("coverage"),
                    txt_entry.get(f"UniProt coverage chain {chain_id}"),
                ),
                uniprot_sequence=sequence,
                uniprot_length=first_int(uniprot.get("length"), len(sequence or "")),
                sequence_hash=sequence_hash(sequence),
            ).as_dict()
        )
    return rows


def build_region_rows(
    accession: str,
    entry_uid: str,
    json_entry: dict[str, Any] | None,
    chain_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build chain-region rows from the JSON archive."""
    chain_uid_by_id = {row["chain_id"]: row["chain_uid"] for row in chain_rows}
    rows: list[dict[str, Any]] = []
    for chain_id, payload in _json_chain_payloads(json_entry).items():
        regions = payload.get("regions") or {}
        for region in ensure_list(regions.get("region")):
            if not isinstance(region, dict):
                continue
            rows.append(
                MfibRegionRecord(
                    accession=accession,
                    entry_uid=entry_uid,
                    chain_uid=chain_uid_by_id.get(chain_id, f"{entry_uid}:{chain_id}"),
                    source_id=SOURCE_ID,
                    native_id=accession,
                    parent_uid=entry_uid,
                    chain_id=chain_id,
                    region_type=clean_text(region.get("region_type")),
                    region_name=clean_text(region.get("region_name")),
                    region_id=clean_text(region.get("region_id")),
                    region_start=to_int(region.get("region_start")),
                    region_end=to_int(region.get("region_end")),
                ).as_dict()
            )
    return rows


def build_evidence_rows(
    accession: str,
    entry_uid: str,
    json_entry: dict[str, Any] | None,
    txt_entry: dict[str, str] | None,
    chain_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build complex and chain-level evidence rows."""
    evidence = (json_entry or {}).get("evidence") or {}
    txt_entry = txt_entry or {}
    rows: list[dict[str, Any]] = [
        MfibEvidenceRecord(
            accession=accession,
            entry_uid=entry_uid,
            source_id=SOURCE_ID,
            native_id=accession,
            scope="complex",
            chain_id=None,
            evidence_level=first_text(
                evidence.get("evidence_level"),
                txt_entry.get("Evidence level"),
            ),
            evidence_coverage=first_text(
                evidence.get("evidence_coverage"),
                txt_entry.get("Evidence coverage"),
            ),
            sequence_domain=first_text(
                evidence.get("sequence_domain"),
                txt_entry.get("Sequence domain"),
            ),
            complex_evidence=first_text(
                evidence.get("complex_evidence"),
                txt_entry.get("Evidence text"),
            ),
            support=None,
        ).as_dict()
    ]

    chain_evidence_map = {
        clean_text(payload.get("chain_id")): payload
        for payload in ensure_list(evidence.get("chain_evidence"))
        if isinstance(payload, dict) and clean_text(payload.get("chain_id"))
    }
    for chain_row in chain_rows:
        chain_id = chain_row["chain_id"]
        payload = chain_evidence_map.get(chain_id) or {}
        support = first_text(
            payload.get("support"),
            txt_entry.get(f"Evidence chain {chain_id}"),
        )
        rows.append(
            MfibEvidenceRecord(
                accession=accession,
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=accession,
                scope="chain",
                chain_id=chain_id,
                evidence_level=None,
                evidence_coverage=None,
                sequence_domain=None,
                complex_evidence=None,
                support=support,
            ).as_dict()
        )
    return rows


def build_go_rows(
    accession: str,
    entry_uid: str,
    json_entry: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build GO rows from the JSON archive."""
    function_payload = (json_entry or {}).get("function") or {}
    rows: list[dict[str, Any]] = []
    for namespace in [
        "molecular_function",
        "biological_process",
        "cellular_component",
    ]:
        payload = function_payload.get(namespace) or {}
        for go_payload in ensure_list(payload.get("go")):
            if not isinstance(go_payload, dict):
                continue
            rows.append(
                MfibGoRecord(
                    accession=accession,
                    entry_uid=entry_uid,
                    source_id=SOURCE_ID,
                    native_id=accession,
                    namespace=namespace,
                    go_accession=clean_text(go_payload.get("accession")),
                    go_name=clean_text(go_payload.get("name")),
                ).as_dict()
            )
    return rows


def build_related_rows(
    accession: str,
    entry_uid: str,
    json_entry: dict[str, Any] | None,
    txt_entry: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Build related-structure rows."""
    txt_entry = txt_entry or {}
    related_structures = (json_entry or {}).get("related_structures") or {}
    related_ids = [
        clean_text(value)
        for value in ensure_list(related_structures.get("id"))
        if clean_text(value) is not None
    ]
    if not related_ids:
        raw_related = clean_text(txt_entry.get("Similar structures"))
        if raw_related and raw_related.lower() != "none":
            related_ids = [clean_text(value) for value in raw_related.split(",")]
            related_ids = [value for value in related_ids if value is not None]
    rows: list[dict[str, Any]] = []
    for related_accession in sorted(set(related_ids)):
        rows.append(
            MfibRelatedRecord(
                accession=accession,
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=accession,
                related_accession=related_accession,
                related_entry_uid=f"{SOURCE_ID}:{related_accession}",
            ).as_dict()
        )
    return rows


def build_crossref_rows(
    entry_row: dict[str, Any],
    chain_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build accession and chain-level cross-reference rows."""
    rows: list[dict[str, Any]] = []
    if entry_row.get("pdb_id"):
        rows.append(
            MfibCrossrefRecord(
                accession=entry_row["accession"],
                entry_uid=entry_row["entry_uid"],
                source_id=SOURCE_ID,
                native_id=entry_row["native_id"],
                chain_id=None,
                namespace="PDB",
                xref_value=str(entry_row["pdb_id"]),
                source_field="pdb_id",
            ).as_dict()
        )
    if entry_row.get("publication_pmid"):
        rows.append(
            MfibCrossrefRecord(
                accession=entry_row["accession"],
                entry_uid=entry_row["entry_uid"],
                source_id=SOURCE_ID,
                native_id=entry_row["native_id"],
                chain_id=None,
                namespace="PMID",
                xref_value=str(entry_row["publication_pmid"]),
                source_field="publication_pmid",
            ).as_dict()
        )
    for chain_row in chain_rows:
        if not chain_row.get("uniprot_id"):
            continue
        rows.append(
            MfibCrossrefRecord(
                accession=chain_row["accession"],
                entry_uid=chain_row["entry_uid"],
                source_id=SOURCE_ID,
                native_id=chain_row["native_id"],
                chain_id=chain_row["chain_id"],
                namespace="UniProt",
                xref_value=str(chain_row["uniprot_id"]),
                source_field="uniprot_id",
            ).as_dict()
        )
    return rows


def build_search_rows(
    entry_row: dict[str, Any],
    chain_rows: list[dict[str, Any]],
    go_rows: list[dict[str, Any]],
    related_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the long-form MFIB search index."""
    rows: list[dict[str, Any]] = []
    entry_values = [
        ("accession", entry_row.get("accession")),
        ("name", entry_row.get("name")),
        ("pdb_id", entry_row.get("pdb_id")),
        ("assembly", entry_row.get("assembly")),
        ("source_organism", entry_row.get("source_organism")),
        ("exp_method", entry_row.get("exp_method")),
        ("class", entry_row.get("class_name")),
        ("subclass", entry_row.get("subclass_name")),
        ("sequence_domain", entry_row.get("sequence_domain")),
        ("evidence_level", entry_row.get("evidence_level")),
        ("evidence_coverage", entry_row.get("evidence_coverage")),
    ]
    for namespace, value in entry_values:
        search_row = build_search_row(
            accession=entry_row["accession"],
            entry_uid=entry_row["entry_uid"],
            native_id=entry_row["native_id"],
            scope="entry",
            chain_id=None,
            namespace=namespace,
            value=value,
        )
        if search_row is not None:
            rows.append(search_row)

    publication_values = [
        ("pmid", entry_row.get("publication_pmid")),
        ("publication_title", entry_row.get("publication_title")),
        ("journal", entry_row.get("publication_journal")),
        ("year", entry_row.get("publication_year")),
    ]
    for namespace, value in publication_values:
        search_row = build_search_row(
            accession=entry_row["accession"],
            entry_uid=entry_row["entry_uid"],
            native_id=entry_row["native_id"],
            scope="publication",
            chain_id=None,
            namespace=namespace,
            value=value,
        )
        if search_row is not None:
            rows.append(search_row)

    for chain_row in chain_rows:
        for namespace, value in [
            ("uniprot_id", chain_row.get("uniprot_id")),
            ("chain_name", chain_row.get("chain_name")),
            ("chain_organism", chain_row.get("source_organism")),
        ]:
            search_row = build_search_row(
                accession=chain_row["accession"],
                entry_uid=chain_row["entry_uid"],
                native_id=chain_row["native_id"],
                scope="chain",
                chain_id=chain_row["chain_id"],
                namespace=namespace,
                value=value,
            )
            if search_row is not None:
                rows.append(search_row)

    for go_row in go_rows:
        for namespace, value in [
            ("go_accession", go_row.get("go_accession")),
            ("go_name", go_row.get("go_name")),
        ]:
            search_row = build_search_row(
                accession=go_row["accession"],
                entry_uid=go_row["entry_uid"],
                native_id=go_row["native_id"],
                scope="function",
                chain_id=None,
                namespace=namespace,
                value=value,
            )
            if search_row is not None:
                rows.append(search_row)

    for related_row in related_rows:
        search_row = build_search_row(
            accession=related_row["accession"],
            entry_uid=related_row["entry_uid"],
            native_id=related_row["native_id"],
            scope="related",
            chain_id=None,
            namespace="related_accession",
            value=related_row.get("related_accession"),
        )
        if search_row is not None:
            rows.append(search_row)
    return rows


def build_search_row(
    accession: str,
    entry_uid: str,
    native_id: str,
    scope: str,
    chain_id: str | None,
    namespace: str,
    value: Any,
) -> dict[str, Any] | None:
    """Build one search-index row when a value is present."""
    raw_value = clean_text(value)
    if raw_value is None:
        return None
    return MfibSearchRecord(
        accession=accession,
        entry_uid=entry_uid,
        source_id=SOURCE_ID,
        native_id=native_id,
        scope=scope,
        chain_id=chain_id,
        search_namespace=namespace,
        search_value_raw=raw_value,
        search_value_normalized=normalize_search_value(raw_value),
    ).as_dict()


def build_cif_asset_record(
    accession: str,
    entry_uid: str,
    pdb_id: str | None,
    asset_path: str | None,
    available: bool,
    shared_across_entries: bool,
) -> dict[str, Any]:
    """Build one curated CIF-asset inventory row."""
    normalized_pdb_id = uppercase_text(pdb_id)
    return MfibAssetRecord(
        accession=accession,
        entry_uid=entry_uid,
        source_id=SOURCE_ID,
        native_id=accession,
        parent_uid=entry_uid,
        asset_uid=f"mfib_cif:{accession}:{normalized_pdb_id or 'missing'}",
        pdb_id=normalized_pdb_id,
        asset_kind="cif",
        asset_path=asset_path,
        available=available,
        shared_across_entries=shared_across_entries,
    ).as_dict()


def parse_mfib_cif_features(
    cif_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse basic deterministic structure features from one CIF asset."""
    payload = MMCIF2Dict(str(cif_path))
    atom_chain_ids = ensure_list(
        payload.get("_atom_site.auth_asym_id")
        or payload.get("_atom_site.label_asym_id")
    )
    atom_groups = _expand_to_length(
        payload.get("_atom_site.group_PDB"), len(atom_chain_ids)
    )
    atom_seq_ids = _expand_to_length(
        payload.get("_atom_site.auth_seq_id") or payload.get("_atom_site.label_seq_id"),
        len(atom_chain_ids),
    )
    atom_comp_ids = _expand_to_length(
        payload.get("_atom_site.label_comp_id"), len(atom_chain_ids)
    )
    model_numbers = ensure_list(payload.get("_atom_site.pdbx_PDB_model_num"))

    polymer_chains = _polymer_chain_ids(payload)
    unique_chain_ids = sorted(
        {clean_text(value) for value in atom_chain_ids if clean_text(value)}
    )
    unique_models = sorted(
        {clean_text(value) for value in model_numbers if clean_text(value)}
    )
    atom_count = len(atom_chain_ids)
    polymer_atom_count = sum(
        1 for chain_id in atom_chain_ids if clean_text(chain_id) in polymer_chains
    )

    residue_keys = set()
    ligand_keys = set()
    chain_stats: dict[str, Counter[str]] = {}
    chain_polymer_residues: dict[str, set[tuple[str, str, str]]] = {}
    chain_all_residues: dict[str, set[tuple[str, str, str]]] = {}
    chain_hetero_residues: dict[str, set[tuple[str, str, str]]] = {}

    for chain_id, group, seq_id, comp_id in zip(
        atom_chain_ids,
        atom_groups,
        atom_seq_ids,
        atom_comp_ids,
    ):
        normalized_chain = clean_text(chain_id)
        normalized_seq = clean_text(seq_id)
        normalized_comp = clean_text(comp_id)
        if normalized_chain is None:
            continue
        chain_stats.setdefault(normalized_chain, Counter())
        chain_polymer_residues.setdefault(normalized_chain, set())
        chain_all_residues.setdefault(normalized_chain, set())
        chain_hetero_residues.setdefault(normalized_chain, set())
        chain_stats[normalized_chain]["atom_count"] += 1

        if normalized_seq and normalized_comp:
            residue_key = (normalized_chain, normalized_seq, normalized_comp)
            residue_keys.add(residue_key)
            chain_all_residues[normalized_chain].add(residue_key)
        else:
            residue_key = None

        if normalized_chain in polymer_chains:
            chain_stats[normalized_chain]["polymer_atom_count"] += 1
            if residue_key is not None:
                chain_polymer_residues[normalized_chain].add(residue_key)
        elif residue_key is not None:
            ligand_keys.add(residue_key)
            chain_hetero_residues[normalized_chain].add(residue_key)

        if clean_text(group) == "HETATM" and residue_key is not None:
            ligand_keys.add(residue_key)
            chain_hetero_residues[normalized_chain].add(residue_key)

    entry_features = {
        "model_count": max(len(unique_models), 1 if atom_count else 0),
        "chain_count_in_cif": len(unique_chain_ids),
        "polymer_chain_count": len(polymer_chains),
        "atom_count": atom_count,
        "polymer_atom_count": polymer_atom_count,
        "residue_count": len(residue_keys),
        "ligand_residue_count": len(ligand_keys),
        "has_multiple_models": max(len(unique_models), 1 if atom_count else 0) > 1,
        "experimental_method_cif": _first_scalar(payload.get("_exptl.method")),
        "resolution_cif": first_float(
            _first_scalar(payload.get("_refine.ls_d_res_high")),
            _first_scalar(payload.get("_em_3d_reconstruction.resolution")),
        ),
    }

    chain_features: list[dict[str, Any]] = []
    for chain_id in unique_chain_ids:
        chain_features.append(
            {
                "chain_id": chain_id,
                "atom_count": chain_stats[chain_id]["atom_count"],
                "polymer_atom_count": chain_stats[chain_id]["polymer_atom_count"],
                "residue_count": len(chain_all_residues.get(chain_id, set())),
                "hetero_residue_count": len(chain_hetero_residues.get(chain_id, set())),
                "is_polypeptide": chain_id in polymer_chains,
            }
        )
    return entry_features, chain_features


def json_txt_mismatch_counts(
    json_entries: dict[str, dict[str, Any]],
    txt_entries: dict[str, dict[str, str]],
) -> dict[str, int]:
    """Return mismatch counts between JSON and TXT flat fields."""
    fields = {
        "name": (
            lambda payload: clean_text((payload.get("general") or {}).get("name")),
            lambda payload: clean_text(payload.get("Entry name")),
        ),
        "pdb_id": (
            lambda payload: uppercase_text(
                (payload.get("general") or {}).get("pdb_id")
            ),
            lambda payload: uppercase_text(payload.get("PDB ID")),
        ),
        "assembly": (
            lambda payload: clean_text((payload.get("general") or {}).get("assembly")),
            lambda payload: clean_text(payload.get("Assembly")),
        ),
        "exp_method": (
            lambda payload: clean_text(
                (payload.get("general") or {}).get("exp_method")
            ),
            lambda payload: clean_text(payload.get("ExpTech")),
        ),
        "source_organism": (
            lambda payload: clean_text(
                (payload.get("general") or {}).get("source_organism")
            ),
            lambda payload: clean_text(payload.get("Source organism")),
        ),
    }
    mismatches = {field: 0 for field in fields}
    for accession in sorted(set(json_entries) & set(txt_entries)):
        json_payload = json_entries[accession]
        txt_payload = txt_entries[accession]
        for field_name, (json_getter, txt_getter) in fields.items():
            json_value = json_getter(json_payload)
            txt_value = txt_getter(txt_payload)
            if json_value is None or txt_value is None:
                continue
            if str(json_value) != str(txt_value):
                mismatches[field_name] += 1
    return mismatches


def ensure_list(value: Any) -> list[Any]:
    """Return one payload as a list while preserving dict values."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clean_text(value: Any) -> str | None:
    """Normalize a text-like field and drop empty placeholders."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "N/A", "None", "null"}:
        return None
    return text


def uppercase_text(value: Any) -> str | None:
    """Return one cleaned text field as uppercase."""
    text = clean_text(value)
    return text.upper() if text is not None else None


def to_int(value: Any) -> int | None:
    """Convert one scalar-like value to ``int`` when possible."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = clean_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    """Convert one scalar-like value to ``float`` when possible."""
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = clean_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def first_text(*values: Any) -> str | None:
    """Return the first non-null cleaned text value."""
    for value in values:
        text = clean_text(value)
        if text is not None:
            return text
    return None


def first_int(*values: Any) -> int | None:
    """Return the first successfully parsed integer value."""
    for value in values:
        integer = to_int(value)
        if integer is not None:
            return integer
    return None


def first_float(*values: Any) -> float | None:
    """Return the first successfully parsed float value."""
    for value in values:
        number = to_float(value)
        if number is not None:
            return number
    return None


def normalize_search_value(value: str) -> str:
    """Normalize one search value for stable case-insensitive lookups."""
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def sequence_hash(sequence: Any) -> str | None:
    """Return a deterministic SHA-256 sequence hash when a sequence exists."""
    text = clean_text(sequence)
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_chain_payloads(
    json_entry: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Return JSON chain payloads keyed by chain ID."""
    macromolecules = (json_entry or {}).get("macromolecules") or {}
    chains = macromolecules.get("chain")
    payloads: dict[str, dict[str, Any]] = {}
    for chain in ensure_list(chains):
        if not isinstance(chain, dict):
            continue
        chain_id = clean_text(chain.get("id"))
        if chain_id is None:
            continue
        payloads[chain_id] = chain
    return payloads


def _txt_chain_ids(txt_entry: dict[str, str]) -> list[str]:
    """Return chain IDs inferred from the flat TXT payload."""
    chains = clean_text(txt_entry.get("Chains"))
    if chains:
        values = [clean_text(value) for value in chains.split(",")]
        return [value for value in values if value is not None]
    chain_ids: set[str] = set()
    for key in txt_entry:
        if " chain " not in key:
            continue
        chain_ids.add(key.rsplit(" ", maxsplit=1)[-1])
    return sorted(chain_ids)


def _parse_boundaries(value: Any) -> tuple[int | None, int | None]:
    """Parse one ``start-end`` UniProt boundary string."""
    text = clean_text(value)
    if text is None:
        return None, None
    match = _BOUNDARY_RE.match(text)
    if match is None:
        return None, None
    return int(match.group("start")), int(match.group("end"))


def _expand_to_length(value: Any, length: int) -> list[Any]:
    """Expand one MMCIF scalar or sequence to a fixed row count."""
    values = ensure_list(value)
    if len(values) == length:
        return values
    if not values:
        return [None] * length
    if len(values) == 1 and length > 1:
        return values * length
    return values[:length] + [None] * max(length - len(values), 0)


def _polymer_chain_ids(payload: dict[str, Any]) -> set[str]:
    """Infer polymer chain IDs from one MMCIF dictionary payload."""
    polymer_chains: set[str] = set()
    strand_values = ensure_list(payload.get("_entity_poly.pdbx_strand_id"))
    for strand_value in strand_values:
        if strand_value is None:
            continue
        for token in str(strand_value).replace(";", ",").split(","):
            normalized = clean_text(token)
            if normalized is not None:
                polymer_chains.add(normalized)
    if polymer_chains:
        return polymer_chains

    polymer_entities = {
        clean_text(value)
        for value in ensure_list(payload.get("_entity_poly.entity_id"))
        if clean_text(value) is not None
    }
    if not polymer_entities:
        return set()
    struct_asym_ids = ensure_list(payload.get("_struct_asym.id"))
    struct_asym_entities = _expand_to_length(
        payload.get("_struct_asym.entity_id"),
        len(struct_asym_ids),
    )
    for chain_id, entity_id in zip(struct_asym_ids, struct_asym_entities):
        normalized_chain = clean_text(chain_id)
        normalized_entity = clean_text(entity_id)
        if normalized_chain is None or normalized_entity is None:
            continue
        if normalized_entity in polymer_entities:
            polymer_chains.add(normalized_chain)
    return polymer_chains


def _first_scalar(value: Any) -> Any:
    """Return the first scalar from one MMCIF payload value."""
    values = ensure_list(value)
    return values[0] if values else None
