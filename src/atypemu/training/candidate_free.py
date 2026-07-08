"""Compatibility loader for the archived candidate-free training path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ARCHIVED_CANDIDATE_FREE = (
    Path(__file__).resolve().parents[3]
    / "archive"
    / "ponytail_20260708"
    / "legacy_candidate_free_viewer"
    / "src"
    / "atypemu"
    / "training"
    / "candidate_free.py"
)


def _load_archived_candidate_free() -> object:
    if not _ARCHIVED_CANDIDATE_FREE.exists():
        raise ImportError(
            "candidate_free was archived and is unavailable at "
            f"{_ARCHIVED_CANDIDATE_FREE}"
        )
    spec = importlib.util.spec_from_file_location(
        "_atypemu_archived_candidate_free",
        _ARCHIVED_CANDIDATE_FREE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load archived candidate_free: {_ARCHIVED_CANDIDATE_FREE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ponytail: tiny shim keeps old imports alive while the real legacy code stays archived.
_legacy = _load_archived_candidate_free()
for _name, _value in vars(_legacy).items():
    if _name not in {"__builtins__", "__cached__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}:
        globals()[_name] = _value

__all__ = sorted(name for name in globals() if not name.startswith("_"))
