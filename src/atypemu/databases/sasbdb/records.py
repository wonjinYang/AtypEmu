"""Record types for curated SASBDB source tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SasbdbEntryRecord:
    """Curated accession-level SASBDB metadata."""

    code: str
    entry_uid: str
    source_id: str
    native_id: str
    status: str | None
    type_of_curve: str | None
    angular_unit: str | None
    intensity_unit: str | None
    guinier_rg: float | None
    guinier_rg_error: float | None
    pddf_rg: float | None
    pddf_rg_error: float | None
    pddf_dmax: float | None
    pddf_dmax_error: float | None
    experimental_mw: float | None
    experimental_mw_error: float | None
    guinier_i0_mw: float | None
    guinier_i0_mw_error: float | None
    porod_mw: float | None
    porod_mw_error: float | None
    porod_volume: float | None
    porod_volume_error: float | None
    project_title: str | None
    publication_title: str | None
    publication_journal: str | None
    publication_doi: str | None
    publication_pmid: str | None
    publication_published_date: str | None
    experiment_description: str | None
    intensities_url: str | None
    pddf_url: str | None
    sascif_url: str | None
    intensities_path: str | None
    pddf_path: str | None
    sascif_path: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SasbdbMoleculeRecord:
    """Curated SASBDB accession-molecule metadata."""

    code: str
    entry_uid: str
    molecule_uid: str
    source_id: str
    native_id: str
    entity_id: str | None
    long_name: str | None
    short_name: str | None
    molecular_type: str | None
    organism: str | None
    oligomerization: str | None
    uniprot_code: str | None
    mw: float | None
    total_mw: float | None
    number_molecules: int | None
    fasta_path: str | None
    sequence: str | None
    sequence_hash: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SasbdbValidationRecord:
    """Curated SASBDB validation-condition metadata."""

    code: str
    entry_uid: str
    source_id: str
    native_id: str
    wavelength: float | None
    cell_temperature: float | None
    storage_temperature: float | None
    beamline_name: str | None
    instrument_name: str | None
    detector_name: str | None
    type_of_source: str | None
    sample_detector_distance: float | None
    concentration_min: float | None
    concentration_max: float | None
    concentration_unit: str | None
    buffer_name: str | None
    buffer_ph: float | None
    buffer_additive: str | None
    s_min: float | None
    s_max: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SasbdbAssetRecord:
    """Curated SASBDB asset inventory row."""

    code: str
    entry_uid: str
    source_id: str
    native_id: str
    asset_uid: str
    asset_kind: str
    entity_id: str | None
    asset_url: str | None
    asset_path: str | None
    available: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SasbdbLlmRecord:
    """Fallback extraction row for unresolved validation prose."""

    code: str
    entry_uid: str
    raw_text: str
    extracted_json: str | None
    model_name: str
    status: str
    confidence: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return asdict(self)
