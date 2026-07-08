"""Archive extraction helpers for source-candidate corpora."""

from __future__ import annotations

import zipfile
from pathlib import Path


def unpack_zip_archives(
    source_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Unpack all zip archives in one source directory.

    Args:
        source_dir: Directory containing zip chunks.
        output_dir: Destination directory for extracted contents.
        overwrite: Whether to overwrite existing files.

    Returns:
        Small summary dictionary with archive, file, and accession counts.
    """
    source_root = Path(source_dir)
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)

    archive_count = 0
    member_count = 0
    for archive_path in sorted(source_root.glob("*.zip")):
        archive_count += 1
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                destination_path = destination_root / member.filename
                if destination_path.exists() and not overwrite:
                    continue
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination_path.open("wb") as dst:
                    dst.write(src.read())
                member_count += 1

    accession_count = len(
        [path for path in destination_root.iterdir() if path.is_dir()]
    )
    return {
        "archives": archive_count,
        "members_written": member_count,
        "accessions": accession_count,
    }
