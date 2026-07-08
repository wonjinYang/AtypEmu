"""Record types for curated FuzDB source tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class FuzdbEntryRecord:
    """Curated accession-level FuzDB metadata."""

    fc_id: str
    entry_uid: str
    source_id: str
    native_id: str
    protein_name: str | None
    partner_name: str | None
    organism: str | None
    uniprot_id: str | None
    uniprot_url: str | None
    sequence: str | None
    sequence_hash: str | None
    fuzzy_region_text: str | None
    topology_class: str | None
    topology_label: str | None
    topology_confidence: str | None
    topology_source: str | None
    mechanism_category: str | None
    mechanism_confidence: str | None
    mechanism_source: str | None
    detection_methods: str | None
    pdb_ids: str | None
    bmrb_ids: str | None
    ped_ids: str | None
    disprot_ids: str | None
    llps_ids: str | None
    pmids: str | None
    biological_function_text: str | None
    structural_evidence_text: str | None
    biochemical_evidence_text: str | None
    structure_mechanism_text: str | None
    significance_text: str | None
    medical_relevance_text: str | None
    context_dependence_text: str | None
    further_reading_text: str | None
    condensate_text: str | None
    timestamp: str | None
    entry_url: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbRegionRecord:
    """Curated fuzzy-region row."""

    region_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    region_index: int
    region_start: int | None
    region_end: int | None
    region_text: str | None
    topology_class: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbStructureLinkRecord:
    """Curated structure or disorder cross-reference row."""

    link_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    database_name: str
    database_id: str
    pmid: str | None
    method: str | None
    description: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbFunctionalSiteRecord:
    """Curated functional-site row."""

    site_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    site_type: str
    site_name: str | None
    site_start: int | None
    site_end: int | None
    source_db: str | None
    external_source_id: str | None
    pmid: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbPtmRecord:
    """Curated PTM-site row."""

    ptm_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    position: int | None
    residue: str | None
    ptm_type: str | None
    functional_effect: str | None
    source_db: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbIsoformRecord:
    """Curated isoform or context row."""

    isoform_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    isoform_name: str | None
    isoform_effect: str | None
    context_text: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbCondensateRecord:
    """Curated LLPS or condensate cross-reference row."""

    condensate_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    llps_role: str | None
    database_name: str | None
    database_id: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbReferenceRecord:
    """Curated citation row."""

    reference_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    reference_scope: str
    pmid: str | None
    citation_text: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbCrossrefRecord:
    """Canonical long-form external identifier row."""

    xref_uid: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    fc_id: str
    xref_namespace: str
    xref_id: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FuzdbSearchRecord:
    """Long-form search-index row for FuzDB."""

    entry_uid: str
    source_id: str
    native_id: str
    scope: str
    search_namespace: str
    search_value_raw: str
    search_value_normalized: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)
