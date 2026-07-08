"""Ingestion entrypoints for PED collection."""

from atypemu.databases.ped.crawl import crawl_ped

__all__ = ["crawl_ped"]
