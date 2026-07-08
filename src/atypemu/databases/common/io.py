"""IO helpers shared by database-source normalization code."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> None:
    """Write one JSON payload to disk."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON Lines file."""
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    Path(path).write_text(content + ("\n" if rows else ""))


def write_tsv(
    path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Write one TSV table with a stable field order."""
    path_obj = Path(path)
    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: str | Path) -> Any:
    """Read one JSON payload from disk."""
    return json.loads(Path(path).read_text())
