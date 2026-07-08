"""Normalization entrypoints for PED curated tables."""

from atypemu.databases.ped.bridge import prepare_ped_bridge
from atypemu.databases.ped.classify import classify_ped_entries

__all__ = ["classify_ped_entries", "prepare_ped_bridge"]
