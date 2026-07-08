"""Human-facing table writers for SASBDB curated outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def rows_to_frame(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> pd.DataFrame:
    """Convert rows to a nullable dataframe with a fixed column order."""
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe = dataframe.replace({r"^\s*$": pd.NA}, regex=True)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def write_tsv(
    path: str | Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> pd.DataFrame:
    """Write a curated TSV table and return the dataframe."""
    dataframe = rows_to_frame(rows, columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, sep="\t", index=False, na_rep="")
    return dataframe
