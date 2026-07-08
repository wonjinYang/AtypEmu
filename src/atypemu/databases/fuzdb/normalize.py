"""Deterministic normalization helpers for FuzDB API payloads."""

from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any

from atypemu.databases.fuzdb.config import ENTRY_URL_TEMPLATE, SOURCE_ID
from atypemu.databases.fuzdb.records import (
    FuzdbCondensateRecord,
    FuzdbCrossrefRecord,
    FuzdbEntryRecord,
    FuzdbFunctionalSiteRecord,
    FuzdbIsoformRecord,
    FuzdbPtmRecord,
    FuzdbReferenceRecord,
    FuzdbRegionRecord,
    FuzdbSearchRecord,
    FuzdbStructureLinkRecord,
)


ENTRY_COLUMNS = list(FuzdbEntryRecord.__dataclass_fields__)
REGION_COLUMNS = list(FuzdbRegionRecord.__dataclass_fields__)
STRUCTURE_LINK_COLUMNS = list(FuzdbStructureLinkRecord.__dataclass_fields__)
FUNCTIONAL_SITE_COLUMNS = list(FuzdbFunctionalSiteRecord.__dataclass_fields__)
PTM_COLUMNS = list(FuzdbPtmRecord.__dataclass_fields__)
ISOFORM_COLUMNS = list(FuzdbIsoformRecord.__dataclass_fields__)
CONDENSATE_COLUMNS = list(FuzdbCondensateRecord.__dataclass_fields__)
REFERENCE_COLUMNS = list(FuzdbReferenceRecord.__dataclass_fields__)
CROSSREF_COLUMNS = list(FuzdbCrossrefRecord.__dataclass_fields__)
SEARCH_COLUMNS = list(FuzdbSearchRecord.__dataclass_fields__)

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PED_RE = re.compile(r"\bPED\d{5,}\b", re.IGNORECASE)
_DISPROT_RE = re.compile(r"\bDP\d{5,}\b", re.IGNORECASE)
_TOPOLOGY_LABELS = {
    "P": "Polymorphic",
    "C": "Clamp",
    "F": "Flanking",
    "R": "Random",
}


def normalize_fuzdb_entries(
    raw_entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Normalize raw FuzDB API payloads into curated table rows."""
    entry_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    structure_link_rows: list[dict[str, Any]] = []
    functional_site_rows: list[dict[str, Any]] = []
    ptm_rows: list[dict[str, Any]] = []
    isoform_rows: list[dict[str, Any]] = []
    condensate_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    crossref_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    for raw_entry in sorted(
        raw_entries, key=lambda row: str(row.get("entry_id") or "")
    ):
        if not raw_entry.get("entry_id"):
            continue
        normalized = normalize_fuzdb_entry(raw_entry)
        entry_rows.append(normalized["entry"])
        region_rows.extend(normalized["fuzzy_regions"])
        structure_link_rows.extend(normalized["structure_links"])
        functional_site_rows.extend(normalized["functional_sites"])
        ptm_rows.extend(normalized["ptm_sites"])
        isoform_rows.extend(normalized["isoforms"])
        condensate_rows.extend(normalized["condensates"])
        reference_rows.extend(normalized["references"])
        crossref_rows.extend(normalized["crossrefs"])
        search_rows.extend(normalized["search_index"])

    return {
        "entries": entry_rows,
        "fuzzy_regions": region_rows,
        "structure_links": structure_link_rows,
        "functional_sites": functional_site_rows,
        "ptm_sites": ptm_rows,
        "isoforms": isoform_rows,
        "condensates": condensate_rows,
        "references": reference_rows,
        "crossrefs": crossref_rows,
        "search_index": _dedupe_dict_rows(search_rows),
    }


def normalize_fuzdb_entry(raw_entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw FuzDB entry into curated row collections."""
    fc_id = clean_text(raw_entry.get("entry_id"))
    if fc_id is None:
        raise ValueError("FuzDB entry is missing `entry_id`.")
    entry_uid = f"{SOURCE_ID}:{fc_id}"
    sequence = clean_sequence(raw_entry.get("sequence"))
    sequence_hash = (
        hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else None
    )
    topology_class = first_text(
        raw_entry.get("topology_class"),
        raw_entry.get("topology"),
        raw_entry.get("fuzzy_topology"),
    )
    mechanism_category = first_text(
        raw_entry.get("mechanism_category"),
        raw_entry.get("mechanism"),
    )
    classification_text = build_classification_text(raw_entry)
    inferred_topology = infer_topology_class(classification_text)
    inferred_mechanism = infer_mechanism_category(classification_text)
    topology_class = topology_class or inferred_topology["value"]
    mechanism_category = mechanism_category or inferred_mechanism["value"]
    detection_methods = pipe_join([clean_text(raw_entry.get("detection_method"))])
    regions = parse_fuzzy_regions(raw_entry)
    region_rows = build_region_rows(
        fc_id=fc_id,
        entry_uid=entry_uid,
        regions=regions,
        topology_class=topology_class,
    )
    structure_link_rows = build_structure_links(
        fc_id=fc_id,
        entry_uid=entry_uid,
        detection_methods=detection_methods,
        raw_entry=raw_entry,
    )
    functional_site_rows = build_functional_sites(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
        regions=regions,
    )
    ptm_rows = build_ptm_rows(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
        regions=regions,
        sequence=sequence,
    )
    isoform_rows = build_isoform_rows(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
        regions=regions,
    )
    condensate_rows = build_condensate_rows(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
    )
    reference_rows = build_reference_rows(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
    )
    crossref_rows = build_crossref_rows(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
        structure_link_rows=structure_link_rows,
        condensate_rows=condensate_rows,
        reference_rows=reference_rows,
    )
    entry_row = build_entry_row(
        fc_id=fc_id,
        entry_uid=entry_uid,
        raw_entry=raw_entry,
        sequence=sequence,
        sequence_hash=sequence_hash,
        regions=regions,
        topology_class=topology_class,
        mechanism_category=mechanism_category,
        inferred_topology=inferred_topology,
        inferred_mechanism=inferred_mechanism,
        detection_methods=detection_methods,
        crossref_rows=crossref_rows,
        reference_rows=reference_rows,
    )
    search_rows = build_search_rows(
        entry_row=entry_row,
        structure_link_rows=structure_link_rows,
        crossref_rows=crossref_rows,
        reference_rows=reference_rows,
        condensate_rows=condensate_rows,
    )
    return {
        "entry": entry_row,
        "fuzzy_regions": region_rows,
        "structure_links": structure_link_rows,
        "functional_sites": functional_site_rows,
        "ptm_sites": ptm_rows,
        "isoforms": isoform_rows,
        "condensates": condensate_rows,
        "references": reference_rows,
        "crossrefs": crossref_rows,
        "search_index": search_rows,
    }


def parse_fuzzy_regions(
    raw_entry: dict[str, Any],
) -> list[tuple[int | None, int | None]]:
    """Parse the fuzzy-region boundary list."""
    regions: list[tuple[int | None, int | None]] = []
    for region in raw_entry.get("fuzzy_region") or []:
        start = first_int(region.get("start"))
        end = first_int(region.get("end"))
        regions.append((start, end))
    regions = sorted(set(regions), key=lambda item: (item[0] or -1, item[1] or -1))
    return regions


def build_entry_row(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
    sequence: str | None,
    sequence_hash: str | None,
    regions: list[tuple[int | None, int | None]],
    topology_class: str | None,
    mechanism_category: str | None,
    inferred_topology: dict[str, str | None],
    inferred_mechanism: dict[str, str | None],
    detection_methods: str | None,
    crossref_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the accession-level FuzDB metadata row."""
    xref_map = _group_xrefs(crossref_rows)
    pmids = sorted(
        {
            str(row["pmid"])
            for row in reference_rows
            if row.get("pmid") is not None and str(row.get("pmid")).strip()
        }
    )
    fuzzy_region_text = pipe_join(
        [
            f"{start}-{end}"
            for start, end in regions
            if start is not None and end is not None
        ]
    )
    entry = FuzdbEntryRecord(
        fc_id=fc_id,
        entry_uid=entry_uid,
        source_id=SOURCE_ID,
        native_id=fc_id,
        protein_name=clean_text(raw_entry.get("protein_name")),
        partner_name=clean_text(raw_entry.get("partner_name")),
        organism=first_text(
            raw_entry.get("organism"),
            raw_entry.get("species"),
            raw_entry.get("source_organism"),
        ),
        uniprot_id=uppercase_text(raw_entry.get("uniprot_acc")),
        uniprot_url=(
            f"https://identifiers.org/uniprot:{uppercase_text(raw_entry.get('uniprot_acc'))}"
            if uppercase_text(raw_entry.get("uniprot_acc"))
            else None
        ),
        sequence=sequence,
        sequence_hash=sequence_hash,
        fuzzy_region_text=fuzzy_region_text,
        topology_class=topology_class,
        topology_label=_TOPOLOGY_LABELS.get(topology_class) if topology_class else None,
        topology_confidence=inferred_topology["confidence"] if topology_class else None,
        topology_source=(
            "explicit_api"
            if first_text(
                raw_entry.get("topology_class"),
                raw_entry.get("topology"),
                raw_entry.get("fuzzy_topology"),
            )
            else inferred_topology["source"]
        ),
        mechanism_category=mechanism_category,
        mechanism_confidence=(
            inferred_mechanism["confidence"] if mechanism_category else None
        ),
        mechanism_source=(
            "explicit_api"
            if first_text(
                raw_entry.get("mechanism_category"),
                raw_entry.get("mechanism"),
            )
            else inferred_mechanism["source"]
        ),
        detection_methods=detection_methods,
        pdb_ids=pipe_join(xref_map.get("PDB", [])),
        bmrb_ids=pipe_join(xref_map.get("BMRB", [])),
        ped_ids=pipe_join(xref_map.get("PED", [])),
        disprot_ids=pipe_join(xref_map.get("DisProt", [])),
        llps_ids=pipe_join(
            [
                *xref_map.get("PhaSepDB", []),
                *xref_map.get("PhaSePro", []),
                *xref_map.get("LLPSDB", []),
                *xref_map.get("PhaseP", []),
            ]
        ),
        pmids=pipe_join(pmids),
        biological_function_text=clean_description(
            raw_entry.get("biological_activity", {}).get("description")
        ),
        structural_evidence_text=clean_description(
            raw_entry.get("structure", {}).get("description")
        ),
        biochemical_evidence_text=clean_description(
            raw_entry.get("biochemical_evidence", {}).get("description")
        ),
        structure_mechanism_text=clean_description(
            raw_entry.get("structure_mechanism", {}).get("description")
        ),
        significance_text=clean_description(
            raw_entry.get("significance", {}).get("description")
        ),
        medical_relevance_text=clean_description(
            raw_entry.get("medical_relevance", {}).get("description")
        ),
        context_dependence_text=clean_description(
            raw_entry.get("context_dependence", {}).get("description")
        ),
        further_reading_text=pipe_join(
            [
                row["citation_text"]
                for row in reference_rows
                if row.get("reference_scope") == "further_reading"
                and row.get("citation_text")
            ]
        ),
        condensate_text=clean_description(
            raw_entry.get("condensate", {}).get("description")
        ),
        timestamp=clean_text(raw_entry.get("timestamp")),
        entry_url=ENTRY_URL_TEMPLATE.format(fc_id=fc_id),
    )
    return entry.as_dict()


def build_region_rows(
    fc_id: str,
    entry_uid: str,
    regions: list[tuple[int | None, int | None]],
    topology_class: str | None,
) -> list[dict[str, Any]]:
    """Build one long-form fuzzy-region table."""
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(regions, start=1):
        rows.append(
            FuzdbRegionRecord(
                region_uid=f"{entry_uid}:region:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                region_index=index,
                region_start=start,
                region_end=end,
                region_text=(
                    f"{start}-{end}" if start is not None and end is not None else None
                ),
                topology_class=topology_class,
            ).as_dict()
        )
    return rows


def build_structure_links(
    fc_id: str,
    entry_uid: str,
    detection_methods: str | None,
    raw_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the structure or disorder cross-link table."""
    rows: list[dict[str, Any]] = []
    index = 0
    for reference in raw_entry.get("pdb_reference") or []:
        index += 1
        publication = reference.get("reference") or {}
        rows.append(
            FuzdbStructureLinkRecord(
                link_uid=f"{entry_uid}:structure:pdb:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                database_name="PDB",
                database_id=uppercase_text(reference.get("id")) or "",
                pmid=clean_text(publication.get("pmid")),
                method=detection_methods,
                description=clean_text(publication.get("title")),
            ).as_dict()
        )
    for reference in raw_entry.get("bmrb_reference") or []:
        index += 1
        publication = reference.get("reference") or {}
        rows.append(
            FuzdbStructureLinkRecord(
                link_uid=f"{entry_uid}:structure:bmrb:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                database_name="BMRB",
                database_id=clean_text(reference.get("id")) or "",
                pmid=clean_text(publication.get("pmid")),
                method=detection_methods,
                description=clean_text(publication.get("title")),
            ).as_dict()
        )

    extracted_ids = extract_text_crossrefs(raw_entry)
    for database_name, ids in [
        ("PED", extracted_ids["ped_ids"]),
        ("DisProt", extracted_ids["disprot_ids"]),
    ]:
        for database_id in ids:
            index += 1
            rows.append(
                FuzdbStructureLinkRecord(
                    link_uid=(
                        f"{entry_uid}:structure:{database_name.lower()}:{database_id}"
                    ),
                    entry_uid=entry_uid,
                    source_id=SOURCE_ID,
                    native_id=fc_id,
                    parent_uid=entry_uid,
                    fc_id=fc_id,
                    database_name=database_name,
                    database_id=database_id,
                    pmid=None,
                    method=detection_methods,
                    description="Mentioned in curated FuzDB prose",
                ).as_dict()
            )
    return rows


def build_functional_sites(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
    regions: list[tuple[int | None, int | None]],
) -> list[dict[str, Any]]:
    """Build domain, motif, and functional-site rows."""
    rows: list[dict[str, Any]] = []
    index = 0
    for domain in raw_entry.get("interpro_pfam_domain") or []:
        start = first_int(domain.get("begin"))
        end = first_int(domain.get("end"))
        if not overlaps_any_region(start, end, regions):
            continue
        index += 1
        rows.append(
            FuzdbFunctionalSiteRecord(
                site_uid=f"{entry_uid}:site:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                site_type="domain",
                site_name=clean_text(domain.get("description")),
                site_start=start,
                site_end=end,
                source_db="Pfam",
                external_source_id=uppercase_text(domain.get("acccession")),
                pmid=None,
            ).as_dict()
        )
    for feature in raw_entry.get("uniprot_features_domain") or []:
        start = first_int(feature.get("begin"))
        end = first_int(feature.get("end"))
        if not overlaps_any_region(start, end, regions):
            continue
        feature_type = clean_text(feature.get("type"))
        category = clean_text(feature.get("category"))
        site_type = (
            "domain" if (category or "").lower() == "domain" else "functional_site"
        )
        index += 1
        rows.append(
            FuzdbFunctionalSiteRecord(
                site_uid=f"{entry_uid}:site:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                site_type=site_type,
                site_name=first_text(feature.get("description"), feature_type),
                site_start=start,
                site_end=end,
                source_db="UniProt",
                external_source_id=feature_type,
                pmid=pipe_join(extract_feature_pmids(feature)),
            ).as_dict()
        )
    for feature in raw_entry.get("elm_reference") or []:
        start = first_int(feature.get("start"))
        end = first_int(feature.get("end"))
        if not overlaps_any_region(start, end, regions):
            continue
        index += 1
        rows.append(
            FuzdbFunctionalSiteRecord(
                site_uid=f"{entry_uid}:site:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                site_type="slim",
                site_name=first_text(feature.get("name"), feature.get("id")),
                site_start=start,
                site_end=end,
                source_db="ELM",
                external_source_id=uppercase_text(feature.get("elm_acc")),
                pmid=pipe_join(
                    [
                        clean_text(reference.get("pmid"))
                        for reference in feature.get("reference") or []
                        if clean_text(reference.get("pmid"))
                    ]
                ),
            ).as_dict()
        )
    return rows


def build_ptm_rows(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
    regions: list[tuple[int | None, int | None]],
    sequence: str | None,
) -> list[dict[str, Any]]:
    """Build PTM rows overlapping the annotated fuzzy regions."""
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(
        raw_entry.get("uniprot_features_ptm") or [], start=1
    ):
        start = first_int(feature.get("begin"))
        end = first_int(feature.get("end"))
        if not overlaps_any_region(start, end, regions):
            continue
        residue = None
        if sequence and start is not None and 1 <= start <= len(sequence):
            residue = sequence[start - 1]
        rows.append(
            FuzdbPtmRecord(
                ptm_uid=f"{entry_uid}:ptm:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                position=start,
                residue=residue,
                ptm_type=first_text(feature.get("description"), feature.get("type")),
                functional_effect=None,
                source_db="UniProt",
            ).as_dict()
        )
    return rows


def build_isoform_rows(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
    regions: list[tuple[int | None, int | None]],
) -> list[dict[str, Any]]:
    """Build isoform or context rows from feature-level molecule annotations."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    index = 0
    for feature_list in [
        raw_entry.get("uniprot_features_domain") or [],
        raw_entry.get("uniprot_features_ptm") or [],
    ]:
        for feature in feature_list:
            start = first_int(feature.get("begin"))
            end = first_int(feature.get("end"))
            molecule = clean_text(feature.get("molecule"))
            if molecule is None or not overlaps_any_region(start, end, regions):
                continue
            effect = first_text(feature.get("description"), feature.get("type"))
            key = (molecule, effect or "", f"{start}-{end}")
            if key in seen:
                continue
            seen.add(key)
            index += 1
            rows.append(
                FuzdbIsoformRecord(
                    isoform_uid=f"{entry_uid}:isoform:{index}",
                    entry_uid=entry_uid,
                    source_id=SOURCE_ID,
                    native_id=fc_id,
                    parent_uid=entry_uid,
                    fc_id=fc_id,
                    isoform_name=molecule,
                    isoform_effect=effect,
                    context_text=(
                        f"Feature region {start}-{end}"
                        if start is not None and end is not None
                        else None
                    ),
                ).as_dict()
            )
    return rows


def build_condensate_rows(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build LLPS or condensate cross-reference rows."""
    rows: list[dict[str, Any]] = []
    index = 0
    condensate = raw_entry.get("condensate") or {}
    condensate_id = clean_text(condensate.get("id"))
    if condensate_id:
        index += 1
        rows.append(
            FuzdbCondensateRecord(
                condensate_uid=f"{entry_uid}:condensate:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                llps_role=None,
                database_name=clean_text(condensate.get("db")),
                database_id=condensate_id,
            ).as_dict()
        )

    for llps_item in raw_entry.get("llps_cross_reference") or []:
        database_name = clean_text(llps_item.get("db"))
        base_database_id = clean_text(llps_item.get("id"))
        entries = llps_item.get("entries") or []
        if entries:
            for nested in entries:
                index += 1
                rows.append(
                    FuzdbCondensateRecord(
                        condensate_uid=f"{entry_uid}:condensate:{index}",
                        entry_uid=entry_uid,
                        source_id=SOURCE_ID,
                        native_id=fc_id,
                        parent_uid=entry_uid,
                        fc_id=fc_id,
                        llps_role=clean_text(nested.get("type")),
                        database_name=database_name,
                        database_id=first_text(nested.get("id"), base_database_id),
                    ).as_dict()
                )
        elif base_database_id:
            index += 1
            rows.append(
                FuzdbCondensateRecord(
                    condensate_uid=f"{entry_uid}:condensate:{index}",
                    entry_uid=entry_uid,
                    source_id=SOURCE_ID,
                    native_id=fc_id,
                    parent_uid=entry_uid,
                    fc_id=fc_id,
                    llps_role=None,
                    database_name=database_name,
                    database_id=base_database_id,
                ).as_dict()
            )
    return rows


def build_reference_rows(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one deduplicated citation table from structured reference blocks."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    index = 0

    def add_reference(scope: str, pmid: str | None, citation_text: str | None) -> None:
        nonlocal index
        key = (scope, pmid or "", citation_text or "")
        if key in seen:
            return
        seen.add(key)
        index += 1
        rows.append(
            FuzdbReferenceRecord(
                reference_uid=f"{entry_uid}:reference:{index}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                reference_scope=scope,
                pmid=pmid,
                citation_text=citation_text,
            ).as_dict()
        )

    for reference in raw_entry.get("list_fuzzy_references") or []:
        add_reference(
            "key",
            clean_text(reference.get("pmid")),
            clean_text(reference.get("title")),
        )
    for reference in raw_entry.get("pdb_reference") or []:
        publication = reference.get("reference") or {}
        add_reference(
            "structure",
            clean_text(publication.get("pmid")),
            clean_text(publication.get("title")),
        )
    for reference in raw_entry.get("bmrb_reference") or []:
        publication = reference.get("reference") or {}
        add_reference(
            "structure",
            clean_text(publication.get("pmid")),
            clean_text(publication.get("title")),
        )
    for feature in raw_entry.get("elm_reference") or []:
        for reference in feature.get("reference") or []:
            add_reference(
                "function",
                clean_text(reference.get("pmid")),
                clean_text(reference.get("title")),
            )
    for feature_list in [
        raw_entry.get("uniprot_features_domain") or [],
        raw_entry.get("uniprot_features_ptm") or [],
    ]:
        for feature in feature_list:
            for pmid in extract_feature_pmids(feature):
                add_reference("function", pmid, None)
    for llps_item in raw_entry.get("llps_cross_reference") or []:
        for publication in llps_item.get("publications") or []:
            add_reference(
                "further_reading",
                clean_text(publication.get("pmid")),
                clean_text(publication.get("title")),
            )
    return rows


def build_crossref_rows(
    fc_id: str,
    entry_uid: str,
    raw_entry: dict[str, Any],
    structure_link_rows: list[dict[str, Any]],
    condensate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build canonical external identifier rows."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_xref(namespace: str, identifier: str | None) -> None:
        identifier = clean_text(identifier)
        if identifier is None:
            return
        key = (namespace, identifier)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            FuzdbCrossrefRecord(
                xref_uid=f"{entry_uid}:xref:{namespace.lower()}:{sanitize_identifier(identifier)}",
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=fc_id,
                parent_uid=entry_uid,
                fc_id=fc_id,
                xref_namespace=namespace,
                xref_id=identifier,
            ).as_dict()
        )

    add_xref("UniProt", uppercase_text(raw_entry.get("uniprot_acc")))
    for structure_link in structure_link_rows:
        add_xref(
            str(structure_link["database_name"]),
            str(structure_link["database_id"]),
        )
    extracted = extract_text_crossrefs(raw_entry)
    for identifier in extracted["ped_ids"]:
        add_xref("PED", identifier)
    for identifier in extracted["disprot_ids"]:
        add_xref("DisProt", identifier)
    for reference in reference_rows:
        add_xref("PMID", reference.get("pmid"))
    for row in condensate_rows:
        namespace = clean_text(row.get("database_name"))
        if namespace is None:
            continue
        add_xref(namespace, row.get("database_id"))
    return rows


def build_classification_text(raw_entry: dict[str, Any]) -> str:
    """Build one normalized prose string for classification inference."""
    parts: list[str] = []
    for key in [
        "biological_activity",
        "structure",
        "significance",
        "condensate",
    ]:
        value = raw_entry.get(key) or {}
        if isinstance(value, dict):
            description = clean_description(value.get("description"))
            if description:
                parts.append(description)
    for reference in raw_entry.get("list_fuzzy_references") or []:
        title = clean_text(reference.get("title"))
        if title:
            parts.append(title)
    return search_normalize(" ".join(parts)) or ""


def infer_topology_class(text: str) -> dict[str, str | None]:
    """Infer one topology class from structured FuzDB prose."""
    if not text:
        return {"value": None, "confidence": None, "source": None}

    scored_rules = {
        "P": [
            "alternative mode",
            "alternative modes",
            "alternative register",
            "alternative registers",
            "alternative structure",
            "alternative structures",
            "alternative conformation",
            "alternative conformations",
            "same partner",
            "simultaneously in alternative",
        ],
        "C": [
            "dynamic linker between two structured binding regions",
            "linker between two structured binding regions",
            "two structured binding regions",
            "between two structured binding",
            "bipartite",
            "two basic clusters",
            "two binding regions",
        ],
        "F": [
            "flanking",
            "flanks the structured binding site",
            "flanks the structured binding",
            "sequences flanking",
            "flanking regions",
            "n-terminal tail",
            "c-terminal tail",
            "tail remains dynamic",
        ],
        "R": [
            "interchange in the bound form",
            "interchange in the bound",
            "binding motifs",
            "alternative pattern",
            "different pattern of",
            "linked by fuzzy regions interchange",
        ],
    }
    scores = {
        label: sum(2 if phrase in text else 0 for phrase in phrases)
        for label, phrases in scored_rules.items()
    }
    if "bipartite" in text and "alternative register" in text:
        scores["P"] += 1
        scores["C"] += 1
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ordered[0]
    if best_score == 0:
        return {"value": None, "confidence": None, "source": None}
    second_score = ordered[1][1]
    confidence = (
        "high" if best_score >= 4 and best_score >= second_score + 2 else "medium"
    )
    return {
        "value": best_label,
        "confidence": confidence,
        "source": "help_definition_keyword_inference",
    }


def infer_mechanism_category(text: str) -> dict[str, str | None]:
    """Infer one mechanistic category from structured FuzDB prose."""
    if not text:
        return {"value": None, "confidence": None, "source": None}

    scored_rules = {
        "Conformational selection": [
            "conformational selection",
            "secondary structure element",
            "formation of a secondary structure",
            "biased for binding",
            "preformed",
            "fold into",
            "register",
        ],
        "Flexibility modulation": [
            "binding entropy",
            "transient interactions with the interface",
            "modulates recognition",
            "tolerant to mutations",
            "flexible contacts",
            "remains dynamic",
            "dynamic change",
        ],
        "Tethering": [
            "local concentration",
            "serves as an anchor",
            "anchor",
            "tether",
            "proximity of the partner",
            "dynamic linker",
            "linked by",
        ],
        "Competitive binding": [
            "competitive binding",
            "compete with the intermolecular interactions",
            "compete with",
            "intramolecular interactions",
            "autoinhib",
        ],
    }
    scores = {
        label: sum(2 if phrase in text else 0 for phrase in phrases)
        for label, phrases in scored_rules.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ordered[0]
    if best_score == 0:
        return {"value": None, "confidence": None, "source": None}
    second_score = ordered[1][1]
    confidence = (
        "high" if best_score >= 4 and best_score >= second_score + 2 else "medium"
    )
    return {
        "value": best_label,
        "confidence": confidence,
        "source": "help_definition_keyword_inference",
    }


def build_search_rows(
    entry_row: dict[str, Any],
    structure_link_rows: list[dict[str, Any]],
    crossref_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    condensate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the long-form search index rows."""
    rows: list[dict[str, Any]] = []

    def add_search(scope: str, namespace: str, value: str | None) -> None:
        value = clean_text(value)
        normalized = search_normalize(value)
        if value is None or normalized is None:
            return
        rows.append(
            FuzdbSearchRecord(
                entry_uid=entry_row["entry_uid"],
                source_id=SOURCE_ID,
                native_id=entry_row["native_id"],
                scope=scope,
                search_namespace=namespace,
                search_value_raw=value,
                search_value_normalized=normalized,
            ).as_dict()
        )

    add_search("entry", "fc_id", entry_row.get("fc_id"))
    add_search("entry", "protein_name", entry_row.get("protein_name"))
    add_search("entry", "partner_name", entry_row.get("partner_name"))
    add_search("entry", "organism", entry_row.get("organism"))
    add_search("entry", "uniprot_id", entry_row.get("uniprot_id"))
    add_search("entry", "topology_class", entry_row.get("topology_class"))
    add_search("entry", "mechanism_category", entry_row.get("mechanism_category"))
    for method in split_pipe(entry_row.get("detection_methods")):
        add_search("entry", "detection_method", method)

    namespace_map = {
        "PDB": "pdb_id",
        "BMRB": "bmrb_id",
        "PED": "ped_id",
        "DisProt": "disprot_id",
        "PMID": "pmid",
        "UniProt": "uniprot_id",
    }
    for crossref in crossref_rows:
        namespace = namespace_map.get(crossref["xref_namespace"])
        if namespace is None:
            continue
        add_search("structure", namespace, crossref["xref_id"])
    for reference in reference_rows:
        add_search("publication", "pmid", reference.get("pmid"))
        add_search("publication", "publication_title", reference.get("citation_text"))
    for structure_link in structure_link_rows:
        add_search(
            "structure",
            structure_link["database_name"].lower(),
            structure_link["database_id"],
        )
    for condensate in condensate_rows:
        add_search("condensate", "llps_role", condensate.get("llps_role"))
        add_search("condensate", "llps_id", condensate.get("database_id"))
    return _dedupe_dict_rows(rows)


def extract_text_crossrefs(raw_entry: dict[str, Any]) -> dict[str, list[str]]:
    """Extract PED and DisProt identifiers from the raw entry payload."""
    blob = json.dumps(raw_entry, sort_keys=True)
    return {
        "ped_ids": sorted({match.upper() for match in _PED_RE.findall(blob)}),
        "disprot_ids": sorted({match.upper() for match in _DISPROT_RE.findall(blob)}),
    }


def extract_feature_pmids(feature: dict[str, Any]) -> list[str]:
    """Extract PMID values from UniProt feature evidence blocks."""
    pmids: list[str] = []
    for evidence in feature.get("evidences") or []:
        source = evidence.get("source") or {}
        if clean_text(source.get("name")) != "PubMed":
            continue
        pmid = clean_text(source.get("id"))
        if pmid:
            pmids.append(pmid)
    return sorted(set(pmids))


def overlaps_any_region(
    start: int | None,
    end: int | None,
    regions: list[tuple[int | None, int | None]],
) -> bool:
    """Return whether one residue interval overlaps any fuzzy region."""
    if start is None or end is None:
        return False
    return any(
        region_start is not None
        and region_end is not None
        and start <= region_end
        and end >= region_start
        for region_start, region_end in regions
    )


def clean_text(value: Any) -> str | None:
    """Normalize one text field while preserving inline markup."""
    if value is None:
        return None
    text = html.unescape(str(value))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def clean_description(value: Any) -> str | None:
    """Normalize one prose description while preserving semantic markup."""
    return clean_text(value)


def clean_sequence(value: Any) -> str | None:
    """Normalize one protein sequence to a compact uppercase string."""
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value)).strip().upper()
    return text or None


def uppercase_text(value: Any) -> str | None:
    """Return one normalized uppercase identifier string."""
    text = clean_text(value)
    return text.upper() if text else None


def first_text(*values: Any) -> str | None:
    """Return the first non-empty normalized text value."""
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return None


def first_int(*values: Any) -> int | None:
    """Return the first value that can be parsed as integer."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return int(float(text))
        except ValueError:
            continue
    return None


def pipe_join(values: list[str | None]) -> str | None:
    """Return one sorted, pipe-joined string of unique non-null values."""
    normalized = sorted(
        {str(value).strip() for value in values if value and str(value).strip()}
    )
    return "|".join(normalized) if normalized else None


def split_pipe(value: Any) -> list[str]:
    """Split one pipe-delimited string into normalized values."""
    text = clean_text(value)
    if text is None:
        return []
    return [part for part in text.split("|") if part]


def search_normalize(value: Any) -> str | None:
    """Normalize one search value to lowercase plain text."""
    text = clean_text(value)
    if text is None:
        return None
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text or None


def sanitize_identifier(value: str) -> str:
    """Return one identifier safe for compound row IDs."""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


def _group_xrefs(crossref_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group cross-reference rows by namespace."""
    grouped: dict[str, list[str]] = {}
    for row in crossref_rows:
        grouped.setdefault(str(row["xref_namespace"]), []).append(str(row["xref_id"]))
    return {key: sorted(set(values)) for key, values in grouped.items()}


def _dedupe_dict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate rows while preserving insertion order."""
    seen: set[tuple[tuple[str, Any], ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
