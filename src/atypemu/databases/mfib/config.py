"""Configuration constants for the MFIB source package."""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_ID = "mfib"
DATASET_NAME = "mfib"
DATASET_VERSION = "1.0.0"
DEFAULT_WORKSPACE = "default"
USER_AGENT = "AtypEmu/0.1 MFIB collector"

HOME_URL = "https://mfib.pbrg.hu/"
DOWNLOADS_PAGE_URL = "https://mfib.pbrg.hu/downloads.php"
COMPLETE_JSON_ZIP_URL = "https://mfib.pbrg.hu/downloads/MFIB_complete_json.zip"
COMPLETE_XML_ZIP_URL = "https://mfib.pbrg.hu/downloads/MFIB_complete_xml.zip"
COMPLETE_TXT_URL = "https://mfib.pbrg.hu/downloads/MFIB_complete.txt"
FORMAT_DEFINITION_URL = "https://mfib.pbrg.hu/downloads/MFIB_format_definition.txt"
INDIVIDUAL_JSON_URL = "https://mfib.pbrg.hu/downloads/MFIB_individual.json"
INDIVIDUAL_XSD_URL = "https://mfib.pbrg.hu/downloads/MFIB_individual.xsd"
CIF_ARCHIVE_URL = "https://mfib.pbrg.hu/downloads/MFIB_download_pdb_cif_ALL.zip"


@dataclass(slots=True)
class MfibCrawlConfig:
    """Runtime configuration for MFIB crawling.

    Attributes:
        refresh: Whether to overwrite cached downloads and curated outputs.
        extract_cif: Whether to download and extract the CIF archive and build
            basic structure features.
        timeout_seconds: Per-request timeout in seconds.
    """

    refresh: bool = False
    extract_cif: bool = True
    timeout_seconds: int = 60
