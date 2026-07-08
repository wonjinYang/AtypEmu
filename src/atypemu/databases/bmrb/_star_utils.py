"""Shared STAR parsing helpers used by AtypEmu parsers."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Iterable

from atypemu.errors import ParseError


def as_none(value: str | None) -> str | None:
    """Convert STAR null markers to ``None``.

    Args:
        value: Raw STAR token.

    Returns:
        Normalized optional string.
    """
    if value is None or value in {".", "?"}:
        return None
    return value


def as_float(value: str | None) -> float | None:
    """Convert a STAR token to ``float`` when possible.

    Args:
        value: Raw STAR token.

    Returns:
        Parsed float or ``None``.
    """
    normalized = as_none(value)
    if normalized is None:
        return None
    return float(normalized)


def as_int(value: str | None) -> int | None:
    """Convert a STAR token to ``int`` when possible.

    Args:
        value: Raw STAR token.

    Returns:
        Parsed integer or ``None``.
    """
    normalized = as_none(value)
    if normalized is None:
        return None
    return int(float(normalized))


@dataclass(slots=True)
class LoopTable:
    """Simple loop representation for the fallback STAR parser."""

    tags: list[str]
    rows: list[list[str]]

    def category(self) -> str:
        """Return the shared category prefix for the loop."""
        if not self.tags:
            return ""
        tag = self.tags[0].lstrip("_")
        return tag.split(".", 1)[0]

    def terminal_names(self) -> list[str]:
        """Return tag names without the category prefix."""
        return [tag.split(".", 1)[-1] for tag in self.tags]

    def iter_dicts(self) -> Iterable[dict[str, str]]:
        """Yield rows as dictionaries keyed by terminal tag names."""
        keys = self.terminal_names()
        for row in self.rows:
            yield dict(zip(keys, row, strict=True))


def _tokenize(line: str) -> list[str]:
    """Tokenize a STAR loop line conservatively."""
    return shlex.split(line, posix=False)


def parse_fallback_loops(text: str) -> list[LoopTable]:
    """Parse STAR loops using a limited text-only fallback parser.

    Args:
        text: Raw file contents.

    Returns:
        Parsed loop tables.
    """
    lines = text.splitlines()
    index = 0
    loops: list[LoopTable] = []

    while index < len(lines):
        line = lines[index].strip()
        if line != "loop_":
            index += 1
            continue

        index += 1
        tags: list[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith("_"):
                tags.append(stripped)
                index += 1
                continue
            break

        if not tags:
            raise ParseError("Encountered loop_ without tag headers.")

        rows: list[list[str]] = []
        buffer: list[str] = []
        width = len(tags)
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped == "stop_":
                index += 1
                break
            if stripped == "loop_" or stripped.startswith("save_"):
                break

            buffer.extend(_tokenize(stripped))
            while len(buffer) >= width:
                row = buffer[:width]
                buffer = buffer[width:]
                rows.append(row)
            index += 1

        loops.append(LoopTable(tags=tags, rows=rows))

    return loops
