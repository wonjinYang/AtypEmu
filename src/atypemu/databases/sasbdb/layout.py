"""Layout helpers for the source-owned SASBDB data root."""

from __future__ import annotations

from pathlib import Path


def sasbdb_root(data_root: str | Path) -> Path:
    """Return the canonical SASBDB source root below ``data/``."""
    root = Path(data_root)
    if root.name == "sasbdb":
        return root
    return root / "sasbdb"


def sasbdb_download_root(root: str | Path) -> Path:
    """Return the raw-download root for SASBDB."""
    return Path(root) / "downloads"


def sasbdb_protein_codes_path(root: str | Path) -> Path:
    """Return the JSON file containing the discovered protein code list."""
    return sasbdb_download_root(root) / "protein_codes.json"


def sasbdb_summary_root(root: str | Path) -> Path:
    """Return the summary JSON directory."""
    return sasbdb_download_root(root) / "summaries"


def sasbdb_sascif_root(root: str | Path) -> Path:
    """Return the SASCIF download directory."""
    return sasbdb_download_root(root) / "sascif"


def sasbdb_fasta_root(root: str | Path) -> Path:
    """Return the FASTA download directory."""
    return sasbdb_download_root(root) / "fasta"


def sasbdb_assets_root(root: str | Path) -> Path:
    """Return the reusable asset root."""
    return Path(root) / "assets"


def sasbdb_intensity_root(root: str | Path) -> Path:
    """Return the downloaded intensity-curve asset root."""
    return sasbdb_assets_root(root) / "intensities"


def sasbdb_pddf_root(root: str | Path) -> Path:
    """Return the downloaded p(r) asset root."""
    return sasbdb_assets_root(root) / "pddf"


def sasbdb_tables_root(root: str | Path) -> Path:
    """Return the human-facing table root."""
    return Path(root) / "tables"


def sasbdb_dataset_root(root: str | Path) -> Path:
    """Return the source-owned dataset root."""
    return Path(root) / "datasets"


def sasbdb_workspace_dataset_root(root: str | Path, workspace: str = "default") -> Path:
    """Return the dataset workspace root."""
    return sasbdb_dataset_root(root) / workspace


def sasbdb_debug_root(root: str | Path) -> Path:
    """Return the debug/provenance root."""
    return Path(root) / "_debug"


def sasbdb_html_debug_root(root: str | Path) -> Path:
    """Return the HTML fallback snapshot root."""
    return sasbdb_debug_root(root) / "html"


def ensure_sasbdb_workspace(root: str | Path) -> None:
    """Create the canonical SASBDB source subtree."""
    for path in [
        sasbdb_download_root(root),
        sasbdb_summary_root(root),
        sasbdb_sascif_root(root),
        sasbdb_fasta_root(root),
        sasbdb_assets_root(root),
        sasbdb_intensity_root(root),
        sasbdb_pddf_root(root),
        sasbdb_tables_root(root),
        sasbdb_dataset_root(root),
        sasbdb_workspace_dataset_root(root),
        sasbdb_debug_root(root),
        sasbdb_html_debug_root(root),
    ]:
        path.mkdir(parents=True, exist_ok=True)
