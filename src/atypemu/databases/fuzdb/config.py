"""Configuration constants for the FuzDB source package."""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_ID = "fuzdb"
DATASET_NAME = "fuzdb"
DATASET_VERSION = "1.0.0"
DEFAULT_WORKSPACE = "default"

HOME_URL = "https://fuzdb.org"
BROWSE_URL = "https://fuzdb.org/browse"
ENTRY_URL_TEMPLATE = "https://fuzdb.org/entry/{fc_id}"
ENTRIES_API_URL = "https://fuzdb.org/api/entries"
ENTRIES_FORMAT_URL_TEMPLATE = "https://fuzdb.org/api/entries?format={format_name}"
ORIGINAL_PAPER_URL = "https://academic.oup.com/nar/article/45/D1/D228/2333923"
V4_PAPER_URL = "https://academic.oup.com/nar/article/50/D1/D509/6430485"
USER_AGENT = "AtypEmu/0.1.0"


@dataclass(slots=True)
class FuzdbCrawlConfig:
    """Runtime configuration for FuzDB crawling."""

    refresh: bool = False
    timeout_seconds: int = 60
    max_entries: int | None = None
