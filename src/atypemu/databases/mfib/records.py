"""Record types for curated MFIB source tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MfibEntryRecord:
    """Curated accession-level MFIB metadata."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    name: str | None
    pdb_id: str | None
    exp_method: str | None
    resolution: float | None
    assembly: str | None
    total_number_of_chains: int | None
    number_of_unique_proteins: int | None
    pdb_note: str | None
    class_name: str | None
    subclass_name: str | None
    sequence_domain: str | None
    evidence_level: str | None
    evidence_coverage: str | None
    evidence_text: str | None
    source_organism: str | None
    publication_pmid: str | None
    publication_authors: str | None
    publication_title: str | None
    publication_journal: str | None
    publication_year: str | None
    publication_volume: str | None
    publication_issue: str | None
    publication_pages: str | None
    publication_abstract: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibChainRecord:
    """Curated MFIB chain-level metadata."""

    accession: str
    entry_uid: str
    chain_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    chain_id: str
    chain_name: str | None
    source_organism: str | None
    uniprot_id: str | None
    uniprot_start: int | None
    uniprot_end: int | None
    uniprot_coverage: str | None
    uniprot_sequence: str | None
    uniprot_length: int | None
    sequence_hash: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibRegionRecord:
    """Curated MFIB chain-region metadata."""

    accession: str
    entry_uid: str
    chain_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    chain_id: str
    region_type: str | None
    region_name: str | None
    region_id: str | None
    region_start: int | None
    region_end: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibEvidenceRecord:
    """Curated MFIB evidence metadata."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    scope: str
    chain_id: str | None
    evidence_level: str | None
    evidence_coverage: str | None
    sequence_domain: str | None
    complex_evidence: str | None
    support: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibGoRecord:
    """Curated GO-term metadata."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    namespace: str
    go_accession: str | None
    go_name: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibRelatedRecord:
    """Curated MFIB related-structure edge."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    related_accession: str
    related_entry_uid: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibCrossrefRecord:
    """Curated MFIB cross-reference row."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    chain_id: str | None
    namespace: str
    xref_value: str
    source_field: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibSearchRecord:
    """Long-form search-index row for MFIB."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    scope: str
    chain_id: str | None
    search_namespace: str
    search_value_raw: str
    search_value_normalized: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibAssetRecord:
    """Curated MFIB asset inventory row."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    asset_uid: str
    pdb_id: str | None
    asset_kind: str
    asset_path: str | None
    available: bool
    shared_across_entries: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibCifEntryFeatureRecord:
    """Entry-level deterministic structure features derived from CIF."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    pdb_id: str
    asset_path: str
    model_count: int | None
    chain_count_in_cif: int | None
    polymer_chain_count: int | None
    atom_count: int | None
    polymer_atom_count: int | None
    residue_count: int | None
    ligand_residue_count: int | None
    has_multiple_models: bool | None
    experimental_method_cif: str | None
    resolution_cif: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibCifChainFeatureRecord:
    """Chain-level deterministic structure features derived from CIF."""

    accession: str
    entry_uid: str
    chain_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    pdb_id: str
    chain_id: str
    atom_count: int | None
    polymer_atom_count: int | None
    residue_count: int | None
    hetero_residue_count: int | None
    is_polypeptide: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibInterfacePairRecord:
    """Curated MFIB chain-pair geometry summary row."""

    accession: str
    entry_uid: str
    pair_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    pdb_id: str
    chain_id_a: str
    chain_id_b: str
    model_selection: str
    contact_level: str
    distance_cutoff: float
    interface_area: float | None
    buried_surface_area: float | None
    contact_count: int
    interface_residue_count_a: int
    interface_residue_count_b: int
    contact_map_path: str | None
    interface_residue_set_path: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibInterfaceResidueRecord:
    """Curated MFIB interface-residue row."""

    accession: str
    entry_uid: str
    pair_uid: str
    chain_uid: str
    source_id: str
    native_id: str
    parent_uid: str
    chain_id: str
    residue_id: str
    resname: str
    is_interface: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class MfibGeometryEntryFeatureRecord:
    """Accession-level summary features derived from chain-pair geometry."""

    accession: str
    entry_uid: str
    source_id: str
    native_id: str
    geometry_available: bool
    pair_count: int
    max_interface_area: float | None
    total_interface_area: float | None
    max_contact_count: int | None
    model_selection: str
    contact_level: str
    distance_cutoff: float

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)
