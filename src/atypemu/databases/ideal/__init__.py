"""IDEAL source package for AtypEmu annotation-oriented benchmark ingestion."""

from atypemu.databases.ideal.export import export_ideal_dataframes
from atypemu.databases.ideal.ingest import crawl_ideal

__all__ = [
    "crawl_ideal",
    "export_ideal_dataframes",
]
