"""Configuration constants for the SASBDB source package."""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_ID = "sasbdb"
DATASET_NAME = "sasbdb"
DATASET_VERSION = "1.0.0"
DEFAULT_WORKSPACE = "default"
USER_AGENT = "AtypEmu/0.1 SASBDB collector"

SWAGGER_URL = "https://www.sasbdb.org/rest-api/swagger/swagger.json"
PROTEIN_CODES_URL = (
    "https://www.sasbdb.org/rest-api/entry/codes/molecular_type/protein/"
)
SUMMARY_URL_TEMPLATE = "https://www.sasbdb.org/rest-api/entry/summary/{code}/"
SUMMARY_BATCH_URL = "https://www.sasbdb.org/rest-api/entry/summary/list/"
ENTRY_HTML_URL_TEMPLATE = "https://www.sasbdb.org/data/{code}/"
FASTA_URL_TEMPLATE = "https://www.sasbdb.org/molecule/{code}/{entity_id}.fasta"

LLM_PRIMARY_MODEL = "GPT-5.3-Codex-Spark"
LLM_RETRY_MODEL = "gpt-5.4-mini"


@dataclass(slots=True)
class SasbdbCrawlConfig:
    """Runtime configuration for SASBDB crawling.

    Attributes:
        protein_only: Whether to restrict the crawl universe to protein entries.
        refresh: Whether to overwrite cached downloads and derived tables.
        include_html_fallback: Whether to fetch HTML only when structured fields
            remain unresolved after REST and SASCIF parsing.
        batch_size: Number of accession codes per summary-list request.
        timeout_seconds: Per-request timeout in seconds.
    """

    protein_only: bool = True
    refresh: bool = False
    include_html_fallback: bool = True
    batch_size: int = 100
    timeout_seconds: int = 60
