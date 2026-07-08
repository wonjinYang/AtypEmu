"""Sample-condition helpers for BMRB NMR-STAR files.

The candidate-free posterior path needs a few experiment-condition features
that strongly affect amide protons, especially pH and temperature.  This module
keeps the parser deliberately small: it extracts scalar rows from common
NMR-STAR loops and returns neutral values when files or fields are absent.
"""

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any


SAMPLE_CONDITION_FEATURE_NAMES = [
    "sample_ph_centered",
    "sample_temperature_centered",
    "sample_ionic_strength_log",
    "sample_d2o_fraction",
    "sample_protein_concentration_log",
    "sample_condition_available",
    "sample_hn_exchange_proxy",
    "sample_condition_risk",
]

_CONDITION_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def extract_bmrb_sample_conditions(
    data_root: str | Path,
    bmrb_id: str,
) -> dict[str, float]:
    """Extract compact sample-condition scalars for one BMRB entry.

    Args:
        data_root: Repository ``data`` root.
        bmrb_id: BMRB identifier such as ``bmr11128``.

    Returns:
        Dictionary with raw scalar values and availability flags.  Missing
        fields are represented by ``nan`` plus neutral feature values downstream.
    """

    normalized_id = str(bmrb_id).lower().replace("bmrb:", "")
    if normalized_id and not normalized_id.startswith("bmr"):
        normalized_id = f"bmr{normalized_id}"
    cache_key = (str(Path(data_root).resolve()), normalized_id)
    if cache_key in _CONDITION_CACHE:
        return dict(_CONDITION_CACHE[cache_key])

    nmrstar_path = Path(data_root) / "bmrb" / "downloads" / "nmrstar" / f"{normalized_id}.str"
    if not nmrstar_path.exists():
        result = _neutral_condition_payload(status="missing_nmrstar")
        _CONDITION_CACHE[cache_key] = result
        return dict(result)

    text = nmrstar_path.read_text(errors="ignore")
    condition_rows = _star_loop_rows(text, "_Sample_condition_variable.")
    component_rows = _star_loop_rows(text, "_Sample_component.")

    values: dict[str, float] = {}
    for row in condition_rows:
        condition_type = str(row.get("Type", "")).strip("'\"").lower()
        value = _safe_float(row.get("Val"))
        if not math.isfinite(value):
            continue
        if condition_type == "ph":
            values["ph"] = value
        elif "temp" in condition_type:
            values["temperature_k"] = value
        elif "ionic" in condition_type:
            values["ionic_strength_mm"] = _to_millimolar(
                value,
                row.get("Val_units"),
            )
        elif "pressure" in condition_type:
            values["pressure_atm"] = value

    d2o_fraction = float("nan")
    h2o_fraction = float("nan")
    protein_concentration_mm = float("nan")
    for row in component_rows:
        name = str(row.get("Mol_common_name", "")).strip("'\"").lower()
        component_type = str(row.get("Type", "")).strip("'\"").lower()
        concentration = _safe_float(row.get("Concentration_val"))
        units = row.get("Concentration_val_units")
        if not math.isfinite(concentration):
            continue
        if "d2o" in name:
            d2o_fraction = _fraction_from_concentration(concentration, units)
        elif "h2o" in name:
            h2o_fraction = _fraction_from_concentration(concentration, units)
        elif component_type == "protein":
            protein_concentration_mm = _to_millimolar(concentration, units)

    values["d2o_fraction"] = d2o_fraction
    values["h2o_fraction"] = h2o_fraction
    values["protein_concentration_mm"] = protein_concentration_mm
    values["condition_available"] = float(
        any(
            math.isfinite(float(values.get(name, float("nan"))))
            for name in ["ph", "temperature_k", "ionic_strength_mm", "d2o_fraction"]
        )
    )
    values["status"] = 1.0
    _CONDITION_CACHE[cache_key] = values
    return dict(values)


def sample_condition_feature_tail(
    condition: dict[str, float] | None,
    *,
    family: str,
) -> list[float]:
    """Return normalized sample-condition features for one target row.

    Args:
        condition: Payload from :func:`extract_bmrb_sample_conditions`.
        family: Atom family.  HN receives a nonzero exchange-risk feature.

    Returns:
        Eight finite feature values matching ``SAMPLE_CONDITION_FEATURE_NAMES``.
    """

    condition = condition or {}
    ph = _finite_or_nan(condition.get("ph"))
    temperature = _finite_or_nan(condition.get("temperature_k"))
    ionic = _finite_or_nan(condition.get("ionic_strength_mm"))
    d2o = _finite_or_nan(condition.get("d2o_fraction"))
    protein = _finite_or_nan(condition.get("protein_concentration_mm"))

    ph_centered = _clip((ph - 7.0) / 2.0, -2.0, 2.0) if math.isfinite(ph) else 0.0
    temp_centered = (
        _clip((temperature - 298.0) / 20.0, -2.0, 2.0)
        if math.isfinite(temperature)
        else 0.0
    )
    ionic_log = (
        _clip(math.log1p(max(ionic, 0.0)) / math.log1p(1000.0), 0.0, 1.5)
        if math.isfinite(ionic)
        else 0.0
    )
    d2o_fraction = _clip(d2o, 0.0, 1.0) if math.isfinite(d2o) else 0.0
    protein_log = (
        _clip(math.log1p(max(protein, 0.0)) / math.log1p(5.0), 0.0, 1.5)
        if math.isfinite(protein)
        else 0.0
    )
    available = float(
        any(math.isfinite(value) for value in [ph, temperature, ionic, d2o, protein])
    )
    ph_exchange = 0.0
    if math.isfinite(ph):
        ph_exchange = max(0.0, ph - 7.2) / 2.8 + max(0.0, 5.8 - ph) / 2.0
    temp_exchange = max(0.0, temperature - 298.0) / 25.0 if math.isfinite(temperature) else 0.0
    solvent_exchange = d2o_fraction if math.isfinite(d2o) else 0.0
    hn_exchange_proxy = _clip(
        0.55 * ph_exchange + 0.30 * temp_exchange + 0.15 * solvent_exchange,
        0.0,
        1.0,
    )
    condition_risk = hn_exchange_proxy if family == "HN" else 0.25 * hn_exchange_proxy
    return [
        float(ph_centered),
        float(temp_centered),
        float(ionic_log),
        float(d2o_fraction),
        float(protein_log),
        float(available),
        float(hn_exchange_proxy if family == "HN" else 0.0),
        float(condition_risk),
    ]


def _neutral_condition_payload(*, status: str) -> dict[str, float]:
    """Return a neutral condition payload."""

    return {
        "ph": float("nan"),
        "temperature_k": float("nan"),
        "ionic_strength_mm": float("nan"),
        "pressure_atm": float("nan"),
        "d2o_fraction": float("nan"),
        "h2o_fraction": float("nan"),
        "protein_concentration_mm": float("nan"),
        "condition_available": 0.0,
        "status": 0.0 if status else 1.0,
    }


def _star_loop_rows(text: str, category_prefix: str) -> list[dict[str, str]]:
    """Extract simple one-line NMR-STAR loop rows for one category prefix."""

    rows: list[dict[str, str]] = []
    headers: list[str] = []
    in_loop = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "loop_":
            headers = []
            in_loop = True
            continue
        if in_loop and line == "stop_":
            headers = []
            in_loop = False
            continue
        if not in_loop:
            continue
        if line.startswith("_"):
            headers.append(line.split()[0])
            continue
        if not headers or not all(header.startswith(category_prefix) for header in headers):
            continue
        try:
            fields = shlex.split(line, comments=False, posix=True)
        except ValueError:
            fields = line.split()
        if len(fields) < len(headers):
            continue
        row = {
            header.removeprefix(category_prefix): value
            for header, value in zip(headers, fields, strict=False)
        }
        rows.append(row)
    return rows


def _safe_float(value: Any) -> float:
    """Convert a STAR scalar into a float."""

    text = str(value).strip().strip("'\"")
    if text in {"", ".", "?", "n/a", "N/A"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _finite_or_nan(value: Any) -> float:
    """Return a finite float or NaN."""

    number = _safe_float(value)
    return number if math.isfinite(number) else float("nan")


def _to_millimolar(value: float, units: Any) -> float:
    """Convert a concentration-like value into mM when possible."""

    unit = str(units).strip().lower()
    if unit in {"m", "mol/l", "mol"}:
        return value * 1000.0
    if unit in {"um", "µm"}:
        return value / 1000.0
    return value


def _fraction_from_concentration(value: float, units: Any) -> float:
    """Convert percent/fraction solvent concentration into a unit fraction."""

    unit = str(units).strip().lower()
    if unit == "%":
        return value / 100.0
    return value if 0.0 <= value <= 1.0 else float("nan")


def _clip(value: float, lower: float, upper: float) -> float:
    """Clip a float into a bounded interval."""

    return min(max(float(value), lower), upper)
