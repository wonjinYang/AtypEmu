"""Deterministic normalization helpers for SASBDB REST and asset payloads."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from atypemu.databases.sasbdb.config import (
    FASTA_URL_TEMPLATE,
    LLM_PRIMARY_MODEL,
    LLM_RETRY_MODEL,
    SOURCE_ID,
)
from atypemu.databases.sasbdb.records import (
    SasbdbAssetRecord,
    SasbdbEntryRecord,
    SasbdbLlmRecord,
    SasbdbMoleculeRecord,
    SasbdbValidationRecord,
)


_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$")

ENTRY_COLUMNS = list(SasbdbEntryRecord.__dataclass_fields__)
MOLECULE_COLUMNS = list(SasbdbMoleculeRecord.__dataclass_fields__)
VALIDATION_COLUMNS = list(SasbdbValidationRecord.__dataclass_fields__)
ASSET_COLUMNS = list(SasbdbAssetRecord.__dataclass_fields__)
LLM_COLUMNS = list(SasbdbLlmRecord.__dataclass_fields__)


def normalize_entry_bundle(
    code: str,
    summary: dict[str, Any],
    sascif_path: Path | None,
    intensity_path: Path | None,
    pddf_path: Path | None,
    html_text: str | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Normalize one SASBDB accession into curated table rows.

    Args:
        code: SASBDB accession code.
        summary: REST summary payload.
        sascif_path: Optional downloaded SASCIF path.
        intensity_path: Optional downloaded intensity-curve path.
        pddf_path: Optional downloaded p(r) path.
        html_text: Optional entry HTML used only when structured parsing leaves
            fields unresolved.

    Returns:
        Tuple of entry row, molecule rows, validation row, asset rows, and
        fallback LLM rows.
    """
    entry_uid = f"{SOURCE_ID}:{code}"
    sascif_payload = (
        parse_sascif_payload(sascif_path)
        if sascif_path is not None and sascif_path.exists()
        else {}
    )

    entry_row = build_entry_row(
        code=code,
        summary=summary,
        entry_uid=entry_uid,
        intensity_path=intensity_path,
        pddf_path=pddf_path,
        sascif_path=sascif_path,
    )
    validation_row = build_validation_row(
        code=code,
        summary=summary,
        entry_uid=entry_uid,
        sascif_payload=sascif_payload,
    )
    llm_rows: list[dict[str, Any]] = []

    if html_text:
        extracted = extract_validation_from_html(html_text)
        validation_row = fill_missing_validation_fields(validation_row, extracted)
        if extracted.get("_llm_candidate_text"):
            llm_rows.append(
                SasbdbLlmRecord(
                    code=code,
                    entry_uid=entry_uid,
                    raw_text=extracted["_llm_candidate_text"],
                    extracted_json=None,
                    model_name=f"{LLM_PRIMARY_MODEL}|retry:{LLM_RETRY_MODEL}",
                    status="not_invoked",
                    confidence=None,
                ).as_dict()
            )

    molecule_rows = build_molecule_rows(
        code=code,
        summary=summary,
        entry_uid=entry_uid,
        sascif_payload=sascif_payload,
        fasta_root=(
            sascif_path.parent.parent / "fasta" / code
            if sascif_path is not None
            else None
        ),
    )
    asset_rows = build_asset_rows(
        code=code,
        summary=summary,
        entry_uid=entry_uid,
        sascif_path=sascif_path,
        intensity_path=intensity_path,
        pddf_path=pddf_path,
        molecule_rows=molecule_rows,
    )

    return entry_row, molecule_rows, validation_row, asset_rows, llm_rows


def build_entry_row(
    code: str,
    summary: dict[str, Any],
    entry_uid: str,
    intensity_path: Path | None,
    pddf_path: Path | None,
    sascif_path: Path | None,
) -> dict[str, Any]:
    """Build the accession-level row from the summary payload."""
    project = summary.get("project") or {}
    publication = project.get("publication") or {}
    return SasbdbEntryRecord(
        code=code,
        entry_uid=entry_uid,
        source_id=SOURCE_ID,
        native_id=code,
        status=summary.get("status"),
        type_of_curve=summary.get("type_of_curve"),
        angular_unit=summary.get("angular_unit"),
        intensity_unit=summary.get("intensity_unit"),
        guinier_rg=to_float(summary.get("guinier_rg")),
        guinier_rg_error=to_float(summary.get("guinier_rg_error")),
        pddf_rg=to_float(summary.get("pddf_rg")),
        pddf_rg_error=to_float(summary.get("pddf_rg_error")),
        pddf_dmax=to_float(summary.get("pddf_dmax")),
        pddf_dmax_error=to_float(summary.get("pddf_dmax_error")),
        experimental_mw=to_float(summary.get("experimental_mw")),
        experimental_mw_error=to_float(summary.get("experimental_mw_error")),
        guinier_i0_mw=to_float(summary.get("guinier_i0_mw")),
        guinier_i0_mw_error=to_float(summary.get("guinier_i0_mw_error")),
        porod_mw=to_float(summary.get("porod_mw")),
        porod_mw_error=to_float(summary.get("porod_mw_error")),
        porod_volume=to_float(summary.get("porod_volume")),
        porod_volume_error=to_float(summary.get("porod_volume_error")),
        project_title=project.get("title"),
        publication_title=publication.get("title"),
        publication_journal=publication.get("journal"),
        publication_doi=publication.get("doi"),
        publication_pmid=publication.get("pmid"),
        publication_published_date=publication.get("published_date"),
        experiment_description=summary.get("experiment_description"),
        intensities_url=summary.get("intensities_data"),
        pddf_url=summary.get("pddf_data"),
        sascif_url=summary.get("sascif_data"),
        intensities_path=relative_source_path(intensity_path),
        pddf_path=relative_source_path(pddf_path),
        sascif_path=relative_source_path(sascif_path),
    ).as_dict()


def build_validation_row(
    code: str,
    summary: dict[str, Any],
    entry_uid: str,
    sascif_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the validation-condition row from structured payloads."""
    experiment = summary.get("experiment") or {}
    instrument = experiment.get("instrument") or {}
    sample = experiment.get("sample") or {}
    buffer_payload = sample.get("buffer") or {}

    return SasbdbValidationRecord(
        code=code,
        entry_uid=entry_uid,
        source_id=SOURCE_ID,
        native_id=code,
        wavelength=first_float(
            experiment.get("wavelength"),
            sascif_payload.get("wavelength"),
        ),
        cell_temperature=first_float(
            experiment.get("cell_temperature"),
            sascif_payload.get("cell_temperature"),
        ),
        storage_temperature=first_float(
            experiment.get("storage_temperature"),
            sascif_payload.get("storage_temperature"),
        ),
        beamline_name=first_text(
            instrument.get("beamline_name"),
            sascif_payload.get("beamline_name"),
        ),
        instrument_name=first_text(
            instrument.get("name"),
            sascif_payload.get("instrument_name"),
        ),
        detector_name=first_text(
            instrument.get("detector"),
            sascif_payload.get("detector_name"),
        ),
        type_of_source=first_text(
            instrument.get("type_of_source"),
            sascif_payload.get("type_of_source"),
        ),
        sample_detector_distance=first_float(
            experiment.get("sample_detector_distance"),
            sascif_payload.get("sample_detector_distance"),
        ),
        concentration_min=to_float(experiment.get("concentration_min")),
        concentration_max=to_float(experiment.get("concentration_max")),
        concentration_unit=first_text(
            experiment.get("concentration_unit"),
            buffer_payload.get("concentration_unit"),
        ),
        buffer_name=first_text(
            buffer_payload.get("name"),
            sascif_payload.get("buffer_name"),
        ),
        buffer_ph=first_float(
            buffer_payload.get("ph"),
            sascif_payload.get("buffer_ph"),
        ),
        buffer_additive=first_text(
            buffer_payload.get("additive"),
            sascif_payload.get("buffer_additive"),
        ),
        s_min=to_float(experiment.get("s_min")),
        s_max=to_float(experiment.get("s_max")),
    ).as_dict()


def build_molecule_rows(
    code: str,
    summary: dict[str, Any],
    entry_uid: str,
    sascif_payload: dict[str, Any],
    fasta_root: Path | None,
) -> list[dict[str, Any]]:
    """Build the accession-molecule rows using FASTA as canonical sequence."""
    summary_molecules = list(
        (summary.get("experiment") or {}).get("sample", {}).get("molecule", [])
    )
    entity_rows = list(sascif_payload.get("entities", []))
    matched = match_summary_molecules_to_entities(summary_molecules, entity_rows)
    rows: list[dict[str, Any]] = []

    for index, pair in enumerate(matched, start=1):
        molecule = pair["molecule"]
        entity = pair["entity"]
        entity_id = entity.get("entity_id") if entity else None
        fasta_path = (
            (fasta_root / f"{entity_id}.fasta")
            if fasta_root is not None and entity_id is not None
            else None
        )
        sequence = (
            parse_fasta_sequence(fasta_path)
            if fasta_path and fasta_path.exists()
            else None
        )
        molecule_uid = (
            f"{SOURCE_ID}:{code}:{entity_id}"
            if entity_id is not None
            else f"{SOURCE_ID}:{code}:m{index}"
        )
        rows.append(
            SasbdbMoleculeRecord(
                code=code,
                entry_uid=entry_uid,
                molecule_uid=molecule_uid,
                source_id=SOURCE_ID,
                native_id=code,
                entity_id=entity_id,
                long_name=first_text(
                    molecule.get("long_name"),
                    entity.get("long_name") if entity else None,
                ),
                short_name=first_text(
                    molecule.get("short_name"),
                    entity.get("short_name") if entity else None,
                ),
                molecular_type=first_text(
                    molecule.get("molecular_type"),
                    entity.get("molecular_type") if entity else None,
                ),
                organism=first_text(
                    molecule.get("organism"),
                    entity.get("organism") if entity else None,
                ),
                oligomerization=first_text(molecule.get("oligomerization")),
                uniprot_code=first_text(
                    molecule.get("uniprot_code"),
                    entity.get("uniprot_code") if entity else None,
                ),
                mw=to_float(molecule.get("mw")),
                total_mw=to_float(molecule.get("total_mw")),
                number_molecules=to_int(molecule.get("number_molecules")),
                fasta_path=relative_source_path(fasta_path),
                sequence=sequence,
                sequence_hash=sequence_hash(sequence),
            ).as_dict()
        )

    return rows


def build_asset_rows(
    code: str,
    summary: dict[str, Any],
    entry_uid: str,
    sascif_path: Path | None,
    intensity_path: Path | None,
    pddf_path: Path | None,
    molecule_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the source-owned asset inventory rows."""
    rows = [
        SasbdbAssetRecord(
            code=code,
            entry_uid=entry_uid,
            source_id=SOURCE_ID,
            native_id=code,
            asset_uid=f"sasbdb:{code}:intensity",
            asset_kind="intensity_dat",
            entity_id=None,
            asset_url=summary.get("intensities_data"),
            asset_path=relative_source_path(intensity_path),
            available=bool(intensity_path and intensity_path.exists()),
        ).as_dict(),
        SasbdbAssetRecord(
            code=code,
            entry_uid=entry_uid,
            source_id=SOURCE_ID,
            native_id=code,
            asset_uid=f"sasbdb:{code}:pddf",
            asset_kind="pddf_out",
            entity_id=None,
            asset_url=summary.get("pddf_data"),
            asset_path=relative_source_path(pddf_path),
            available=bool(pddf_path and pddf_path.exists()),
        ).as_dict(),
        SasbdbAssetRecord(
            code=code,
            entry_uid=entry_uid,
            source_id=SOURCE_ID,
            native_id=code,
            asset_uid=f"sasbdb:{code}:sascif",
            asset_kind="sascif",
            entity_id=None,
            asset_url=summary.get("sascif_data"),
            asset_path=relative_source_path(sascif_path),
            available=bool(sascif_path and sascif_path.exists()),
        ).as_dict(),
    ]

    for molecule in molecule_rows:
        rows.append(
            SasbdbAssetRecord(
                code=code,
                entry_uid=entry_uid,
                source_id=SOURCE_ID,
                native_id=code,
                asset_uid=f"sasbdb:{code}:fasta:{molecule.get('entity_id')}",
                asset_kind="fasta",
                entity_id=molecule.get("entity_id"),
                asset_url=(
                    FASTA_URL_TEMPLATE.format(
                        code=code,
                        entity_id=molecule.get("entity_id"),
                    )
                    if molecule.get("entity_id")
                    else None
                ),
                asset_path=molecule.get("fasta_path"),
                available=bool(molecule.get("fasta_path")),
            ).as_dict()
        )
    return rows


def parse_sascif_payload(path: str | Path) -> dict[str, Any]:
    """Parse a SASCIF file into structured helper payloads."""
    payload = MMCIF2Dict(str(path))
    entity_rows = _parse_sascif_entities(payload)
    return {
        "entities": entity_rows,
        "wavelength": first_text_from_dict(payload, "_sas_beam.radiation_wavelength"),
        "cell_temperature": first_text_from_dict(payload, "_sas_scan.cell_temperature"),
        "storage_temperature": first_text_from_dict(
            payload, "_sas_scan.storage_temperature"
        ),
        "sample_detector_distance": first_text_from_dict(
            payload, "_sas_detc.sample_to_detector_distance"
        ),
        "beamline_name": first_text_from_dict(payload, "_sas_beam.instrument_name"),
        "instrument_name": first_text_from_dict(payload, "_sas_beam.instrument_name"),
        "detector_name": first_text_from_dict(payload, "_sas_detc.name"),
        "type_of_source": first_text_from_dict(payload, "_sas_beam.type"),
        "buffer_name": first_text_from_dict(payload, "_sas_buffer.name"),
        "buffer_ph": first_text_from_dict(payload, "_sas_buffer.pH"),
        "buffer_additive": first_text_from_dict(payload, "_sas_buffer.comment"),
    }


def _parse_sascif_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse molecule-level entity metadata from a SASCIF dictionary."""
    entity_rows = _dict_rows(
        payload,
        {
            "entity_id": "_entity.id",
            "molecular_type": "_entity.type",
            "long_name": "_entity.pdbx_description",
        },
    )
    short_name_rows = _dict_rows(
        payload,
        {
            "entity_id": "_entity_name_com.entity_id",
            "short_name": "_entity_name_com.name",
        },
    )
    uniprot_rows = _dict_rows(
        payload,
        {
            "entity_id": "_struct_ref.entity_id",
            "uniprot_code": "_struct_ref.db_code",
        },
    )

    short_names = {row["entity_id"]: row.get("short_name") for row in short_name_rows}
    uniprots = {row["entity_id"]: row.get("uniprot_code") for row in uniprot_rows}

    rows: list[dict[str, Any]] = []
    for row in entity_rows:
        rows.append(
            {
                "entity_id": row.get("entity_id"),
                "molecular_type": row.get("molecular_type"),
                "long_name": row.get("long_name"),
                "short_name": short_names.get(row.get("entity_id")),
                "uniprot_code": uniprots.get(row.get("entity_id")),
                "organism": None,
            }
        )
    return rows


def match_summary_molecules_to_entities(
    summary_molecules: list[dict[str, Any]],
    entity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match summary molecule payloads to SASCIF entities."""
    unmatched = list(entity_rows)
    pairs: list[dict[str, Any]] = []
    for molecule in summary_molecules:
        match = _pop_matching_entity(molecule, unmatched)
        pairs.append({"molecule": molecule, "entity": match})

    for entity in unmatched:
        pairs.append({"molecule": {}, "entity": entity})
    return pairs


def _pop_matching_entity(
    molecule: dict[str, Any],
    unmatched: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Remove and return the best matching SASCIF entity for one molecule."""
    if not unmatched:
        return None

    uniprot = first_text(molecule.get("uniprot_code"))
    short_name = first_text(molecule.get("short_name"))
    long_name = first_text(molecule.get("long_name"))

    def _match(field: str, value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        normalized = normalize_text(value)
        for index, entity in enumerate(unmatched):
            candidate = normalize_text(entity.get(field))
            if candidate and candidate == normalized:
                return unmatched.pop(index)
        return None

    for field, value in [
        ("uniprot_code", uniprot),
        ("short_name", short_name),
        ("long_name", long_name),
    ]:
        match = _match(field, value)
        if match is not None:
            return match

    if len(unmatched) == 1:
        return unmatched.pop(0)
    return None


def parse_fasta_sequence(path: str | Path) -> str | None:
    """Parse a FASTA sequence into one normalized one-letter string."""
    lines = Path(path).read_text().splitlines()
    sequence = "".join(
        line.strip() for line in lines if line.strip() and not line.startswith(">")
    )
    sequence = sequence.replace(" ", "").replace("\r", "").replace("\n", "")
    return sequence or None


def parse_intensity_profile(path: str | Path) -> list[dict[str, Any]]:
    """Parse one SASBDB intensity ``.dat`` file into long-form points."""
    points: list[dict[str, Any]] = []
    for point_index, numbers in enumerate(
        _numeric_lines(Path(path).read_text()), start=0
    ):
        if len(numbers) < 2:
            continue
        points.append(
            {
                "point_index": point_index,
                "x_value": numbers[0],
                "y_value": numbers[1],
                "error_value": numbers[2] if len(numbers) >= 3 else None,
            }
        )
    return points


def parse_pddf_profile(path: str | Path) -> list[dict[str, Any]]:
    """Parse one SASBDB p(r) ``.out`` file into long-form points."""
    segments = _numeric_segments(Path(path).read_text())
    if not segments:
        return []
    segment = max(segments, key=len)
    return [
        {
            "point_index": point_index,
            "x_value": numbers[0],
            "y_value": numbers[1],
            "error_value": numbers[2] if len(numbers) >= 3 else None,
        }
        for point_index, numbers in enumerate(segment, start=0)
    ]


def extract_validation_from_html(html_text: str) -> dict[str, Any]:
    """Extract a few validation fields from entry HTML as a deterministic fallback."""
    fields: dict[str, Any] = {}
    lower = html_text.lower()
    match = re.search(r"data validation(.{0,4000})", lower, flags=re.DOTALL)
    if match:
        fields["_llm_candidate_text"] = match.group(1).strip()

    label_map = {
        "buffer ph": "buffer_ph",
        "wavelength": "wavelength",
        "cell temperature": "cell_temperature",
        "storage temperature": "storage_temperature",
    }
    for label, field_name in label_map.items():
        label_match = re.search(
            rf"{re.escape(label)}\s*</[^>]+>\s*<[^>]+>\s*([^<]+)",
            html_text,
            flags=re.IGNORECASE,
        )
        if label_match:
            fields[field_name] = label_match.group(1).strip()
    return fields


def fill_missing_validation_fields(
    validation_row: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing validation values from a fallback payload."""
    updated = dict(validation_row)
    for field_name, value in fallback.items():
        if field_name.startswith("_"):
            continue
        if updated.get(field_name) in {None, pd.NA, ""}:
            updated[field_name] = (
                to_float(value)
                if field_name.endswith(("temperature", "distance", "ph", "wavelength"))
                else value
            )
    return updated


def relative_source_path(path: Path | None) -> str | None:
    """Return a source-relative path string when possible."""
    if path is None:
        return None
    parts = list(path.parts)
    for index, value in enumerate(parts):
        if value == SOURCE_ID:
            return "/".join(parts[index + 1 :])
    return str(path)


def sequence_hash(sequence: str | None) -> str | None:
    """Return a deterministic SHA-256 hash for one sequence."""
    if not sequence:
        return None
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None or value == "" or value == "?":
        return None
    try:
        return float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    if value is None or value == "" or value == "?":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def first_float(*values: Any) -> float | None:
    """Return the first value that cleanly converts to float."""
    for value in values:
        converted = to_float(value)
        if converted is not None:
            return converted
    return None


def first_text(*values: Any) -> str | None:
    """Return the first non-empty text value."""
    for value in values:
        if value is None or value is pd.NA:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalize_text(value: Any) -> str | None:
    """Return lowercased text for tolerant matching."""
    text = first_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip().lower()


def _numeric_lines(text: str) -> list[list[float]]:
    """Return fully numeric lines from a text payload."""
    rows: list[list[float]] = []
    for line in text.splitlines():
        numbers = _line_numbers(line)
        if numbers is not None and len(numbers) >= 2:
            rows.append(numbers)
    return rows


def _numeric_segments(text: str) -> list[list[list[float]]]:
    """Return contiguous numeric segments from a text payload."""
    segments: list[list[list[float]]] = []
    current: list[list[float]] = []
    last_x: float | None = None

    for line in text.splitlines():
        numbers = _line_numbers(line)
        if numbers is None or len(numbers) < 2:
            if current:
                segments.append(current)
                current = []
                last_x = None
            continue
        if last_x is not None and numbers[0] < last_x:
            if current:
                segments.append(current)
            current = [numbers]
            last_x = numbers[0]
            continue
        current.append(numbers)
        last_x = numbers[0]

    if current:
        segments.append(current)
    return [segment for segment in segments if len(segment) >= 3]


def _line_numbers(line: str) -> list[float] | None:
    """Parse one fully numeric whitespace-delimited line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    tokens = stripped.split()
    if not tokens or not all(_FLOAT_RE.match(token) for token in tokens):
        return None
    return [float(token.replace("D", "E").replace("d", "e")) for token in tokens]


def _dict_rows(
    payload: dict[str, Any],
    columns: dict[str, str],
) -> list[dict[str, Any]]:
    """Extract row dictionaries from MMCIF-style loop columns."""
    lengths = [
        len(_ensure_list(payload.get(tag)))
        for tag in columns.values()
        if payload.get(tag) is not None
    ]
    if not lengths:
        return []
    row_count = min(lengths)
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row = {}
        for field_name, tag in columns.items():
            values = _ensure_list(payload.get(tag))
            row[field_name] = values[index] if index < len(values) else None
        rows.append(row)
    return rows


def _ensure_list(value: Any) -> list[Any]:
    """Normalize MMCIF scalar-or-list values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_text_from_dict(payload: dict[str, Any], tag: str) -> str | None:
    """Return the first text value for one MMCIF tag."""
    values = _ensure_list(payload.get(tag))
    return first_text(values[0] if values else None)
