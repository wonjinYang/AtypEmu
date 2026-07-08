"""Generic dataframe manifest types and lazy registries for AtypEmu data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_WORKSPACE = "default"


@dataclass(slots=True)
class DataFrameTableMeta:
    """Metadata for one exported Parquet table."""

    name: str
    relative_path: str
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int
    primary_key: list[str]
    join_keys: list[str]
    workspace: str
    level: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize the table metadata to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "relative_path": self.relative_path,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "row_count": int(self.row_count),
            "primary_key": list(self.primary_key),
            "join_keys": list(self.join_keys),
            "workspace": self.workspace,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataFrameTableMeta":
        """Create metadata from one manifest dictionary."""
        return cls(
            name=str(payload["name"]),
            relative_path=str(payload["relative_path"]),
            columns=list(payload.get("columns", [])),
            dtypes=dict(payload.get("dtypes", {})),
            row_count=int(payload.get("row_count", 0)),
            primary_key=list(payload.get("primary_key", [])),
            join_keys=list(payload.get("join_keys", [])),
            workspace=str(payload.get("workspace", DEFAULT_WORKSPACE)),
            level=str(payload.get("level", "entry")),
        )


@dataclass(slots=True)
class DataFrameWorkspaceMeta:
    """Workspace-level metadata for one dataset root."""

    workspace: str
    backend: str
    tables: dict[str, DataFrameTableMeta] = field(default_factory=dict)
    wide_views: dict[str, str] = field(default_factory=dict)

    @property
    def row_counts(self) -> dict[str, int]:
        """Return recorded row counts keyed by table name."""
        return {name: meta.row_count for name, meta in sorted(self.tables.items())}

    @property
    def primary_keys(self) -> dict[str, list[str]]:
        """Return recorded primary keys keyed by table name."""
        return {name: meta.primary_key for name, meta in sorted(self.tables.items())}

    @property
    def join_keys(self) -> dict[str, list[str]]:
        """Return recorded join keys keyed by table name."""
        return {name: meta.join_keys for name, meta in sorted(self.tables.items())}

    @property
    def paths(self) -> dict[str, str]:
        """Return relative Parquet paths keyed by table name."""
        payload = {
            name: meta.relative_path for name, meta in sorted(self.tables.items())
        }
        payload.update(dict(sorted(self.wide_views.items())))
        return payload

    def as_dict(self) -> dict[str, Any]:
        """Serialize workspace metadata to a JSON-compatible dictionary."""
        return {
            "workspace": self.workspace,
            "backend": self.backend,
            "tables": {
                name: meta.as_dict() for name, meta in sorted(self.tables.items())
            },
            "row_counts": self.row_counts,
            "primary_keys": self.primary_keys,
            "join_keys": self.join_keys,
            "paths": self.paths,
            "wide_views": dict(sorted(self.wide_views.items())),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataFrameWorkspaceMeta":
        """Create workspace metadata from one manifest dictionary."""
        tables = {
            name: DataFrameTableMeta.from_dict(meta)
            for name, meta in dict(payload.get("tables", {})).items()
        }
        return cls(
            workspace=str(payload.get("workspace", DEFAULT_WORKSPACE)),
            backend=str(payload.get("backend", "pandas+pyarrow")),
            tables=tables,
            wide_views=dict(payload.get("wide_views", {})),
        )


@dataclass(slots=True)
class DataFrameDatasetManifest:
    """Top-level manifest for one source, meta, or integrated dataset bundle."""

    dataset_name: str
    dataset_version: str
    created_at_utc: str
    root_kind: str
    data_root: str
    dataset_root: str
    backend: str
    source_id: str | None = None
    default_workspace: str = DEFAULT_WORKSPACE
    workspaces: dict[str, DataFrameWorkspaceMeta] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-compatible dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "created_at_utc": self.created_at_utc,
            "root_kind": self.root_kind,
            "data_root": self.data_root,
            "dataset_root": self.dataset_root,
            "backend": self.backend,
            "source_id": self.source_id,
            "default_workspace": self.default_workspace,
            "workspaces": {
                name: meta.as_dict() for name, meta in sorted(self.workspaces.items())
            },
        }

    def to_json(self, path: str | Path) -> None:
        """Write the manifest to disk."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataFrameDatasetManifest":
        """Create a manifest from one dictionary."""
        return cls(
            dataset_name=str(payload["dataset_name"]),
            dataset_version=str(payload["dataset_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            root_kind=str(payload["root_kind"]),
            data_root=str(payload["data_root"]),
            dataset_root=str(payload["dataset_root"]),
            backend=str(payload.get("backend", "pandas+pyarrow")),
            source_id=(
                None
                if payload.get("source_id") is None
                else str(payload.get("source_id"))
            ),
            default_workspace=str(payload.get("default_workspace", DEFAULT_WORKSPACE)),
            workspaces={
                name: DataFrameWorkspaceMeta.from_dict(meta)
                for name, meta in dict(payload.get("workspaces", {})).items()
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DataFrameDatasetManifest":
        """Load a manifest from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text()))


class _BaseDataFrameRegistry:
    """Shared lazy-loading implementation for Parquet dataset registries."""

    def __init__(
        self,
        manifest: DataFrameDatasetManifest,
        manifest_path: str | Path,
        workspace: str | None = None,
    ) -> None:
        """Initialize the registry from one parsed manifest."""
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent
        self.workspace = _resolve_workspace_name(
            requested=workspace,
            available=list(self.manifest.workspaces),
            default=self.manifest.default_workspace,
        )
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def workspace_meta(self) -> DataFrameWorkspaceMeta:
        """Return metadata for the active workspace."""
        return self.manifest.workspaces[self.workspace]

    def available_tables(self) -> list[str]:
        """Return canonical table names for the active workspace."""
        return sorted(self.workspace_meta.tables)

    def table_path(self, name: str) -> Path:
        """Resolve one table or derived view path."""
        if name in self.workspace_meta.tables:
            return self.dataset_root / self.workspace_meta.tables[name].relative_path
        if name in self.workspace_meta.wide_views:
            return self.dataset_root / self.workspace_meta.wide_views[name]
        raise KeyError(
            f"Table '{name}' is not available in workspace '{self.workspace}'."
        )

    def load_table(self, name: str, use_cache: bool = True) -> pd.DataFrame:
        """Load one Parquet table or derived view."""
        if use_cache and name in self._cache:
            return self._cache[name]
        dataframe = pd.read_parquet(self.table_path(name))
        if use_cache:
            self._cache[name] = dataframe
        return dataframe

    def schema(self, name: str) -> dict[str, str]:
        """Return the recorded schema for one table or derived view."""
        if name in self.workspace_meta.tables:
            return dict(self.workspace_meta.tables[name].dtypes)
        dataframe = self.load_table(name=name, use_cache=True)
        return {column: str(dtype) for column, dtype in dataframe.dtypes.items()}

    def row_count(self, name: str) -> int:
        """Return the recorded row count for one table or derived view."""
        if name in self.workspace_meta.tables:
            return self.workspace_meta.tables[name].row_count
        return len(self.load_table(name=name, use_cache=True))

    def clear_cache(self) -> None:
        """Clear all cached dataframes."""
        self._cache.clear()

    def workspace_summary(self) -> dict[str, Any]:
        """Return a compact workspace summary dictionary."""
        return {
            "dataset_name": self.manifest.dataset_name,
            "root_kind": self.manifest.root_kind,
            "source_id": self.manifest.source_id,
            "workspace": self.workspace,
            "tables": self.available_tables(),
            "row_counts": self.workspace_meta.row_counts,
            "paths": self.workspace_meta.paths,
        }


class SourceDataFrameRegistry(_BaseDataFrameRegistry):
    """Lazy dataframe registry for one source-owned dataset root."""

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path,
        source_id: str,
        workspace: str | None = None,
    ) -> "SourceDataFrameRegistry":
        """Create a registry from ``data/<source>/datasets/manifest.json``."""
        manifest_path = Path(data_root) / source_id / "datasets" / "manifest.json"
        return cls.from_manifest(manifest_path=manifest_path, workspace=workspace)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        workspace: str | None = None,
    ) -> "SourceDataFrameRegistry":
        """Create a source registry from one manifest path."""
        manifest = DataFrameDatasetManifest.from_json(manifest_path)
        if manifest.root_kind != "source":
            raise ValueError(
                f"Manifest '{manifest_path}' is not a source dataset manifest."
            )
        return cls(manifest=manifest, manifest_path=manifest_path, workspace=workspace)

    def load_entry_view(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the canonical entry-level trainer view."""
        if "entry_view" in self.workspace_meta.tables:
            return self.load_table("entry_view", use_cache=use_cache)
        return self.load_table("entries", use_cache=use_cache)

    def load_validation_view(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the canonical validation-facing entry view."""
        return self.load_entry_view(use_cache=use_cache)

    def load_assets(self, use_cache: bool = True) -> pd.DataFrame:
        """Load one source-level asset table when available."""
        for table_name in ["assets", "models", "profiles"]:
            if table_name in self.workspace_meta.tables:
                return self.load_table(table_name, use_cache=use_cache)
        raise ValueError(
            f"Workspace '{self.workspace}' does not expose an asset-level table."
        )

    def load_model_view(self, use_cache: bool = True) -> pd.DataFrame:
        """Load one model-level dataframe when the source exposes it."""
        if "models" not in self.workspace_meta.tables:
            raise ValueError(
                "This workspace does not expose model rows. "
                "Use a workspace with a `models` table such as PED benchmark."
            )
        return self.load_table("models", use_cache=use_cache)

    def load_links(self, use_cache: bool = True) -> pd.DataFrame:
        """Load one source-local link or bridge table when available."""
        for table_name in ["links", "bridges"]:
            if table_name in self.workspace_meta.tables:
                return self.load_table(table_name, use_cache=use_cache)
        raise ValueError(
            f"Workspace '{self.workspace}' does not expose a link-level table."
        )


class MetaDataFrameRegistry(_BaseDataFrameRegistry):
    """Lazy dataframe registry for cross-source metadata tables."""

    @classmethod
    def from_data_root(cls, data_root: str | Path) -> "MetaDataFrameRegistry":
        """Create a metadata registry from ``data/meta/datasets/manifest.json``."""
        manifest_path = Path(data_root) / "meta" / "datasets" / "manifest.json"
        return cls.from_manifest(manifest_path)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MetaDataFrameRegistry":
        """Create a metadata registry from one manifest path."""
        manifest = DataFrameDatasetManifest.from_json(manifest_path)
        if manifest.root_kind != "meta":
            raise ValueError(
                f"Manifest '{manifest_path}' is not a meta dataset manifest."
            )
        return cls(
            manifest=manifest,
            manifest_path=manifest_path,
            workspace=DEFAULT_WORKSPACE,
        )

    def load_entity_index(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the canonical cross-source entity index."""
        return self.load_table("entities", use_cache=use_cache)

    def load_validation_index(self, use_cache: bool = True) -> pd.DataFrame:
        """Load a validation-oriented entity view."""
        cache_key = "__validation_index__"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        view = self.load_table("entities", use_cache=use_cache).copy()
        for table_name in ["links", "splits"]:
            if table_name not in self.workspace_meta.tables:
                continue
            view = _merge_frames(view, self.load_table(table_name), "entity_uid")
        if use_cache:
            self._cache[cache_key] = view
        return view

    def load_source_links(
        self,
        source_id_a: str,
        source_id_b: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load curated link rows between two source identifiers."""
        links = self.load_table("links", use_cache=use_cache)
        required = {"left_source_id", "right_source_id"}
        if not required <= set(links.columns):
            raise ValueError("Meta links table does not expose source pair columns.")
        mask = (
            (links["left_source_id"] == source_id_a)
            & (links["right_source_id"] == source_id_b)
        ) | (
            (links["left_source_id"] == source_id_b)
            & (links["right_source_id"] == source_id_a)
        )
        return links.loc[mask].reset_index(drop=True)


class IntegratedDataRegistry(_BaseDataFrameRegistry):
    """Lazy dataframe registry for cross-source integrated artifacts."""

    @classmethod
    def from_data_root(cls, data_root: str | Path) -> "IntegratedDataRegistry":
        """Create an integrated-data registry from ``data/integrated``."""
        manifest_path = Path(data_root) / "integrated" / "manifest.json"
        return cls.from_manifest(manifest_path)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "IntegratedDataRegistry":
        """Create an integrated-data registry from one manifest path."""
        manifest = DataFrameDatasetManifest.from_json(manifest_path)
        if manifest.root_kind != "integrated":
            raise ValueError(
                f"Manifest '{manifest_path}' is not an integrated dataset manifest."
            )
        return cls(
            manifest=manifest,
            manifest_path=manifest_path,
            workspace=DEFAULT_WORKSPACE,
        )

    def load_teacher_examples(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the canonical teacher-example table when available."""
        return self.load_table("teacher_examples", use_cache=use_cache)

    def load_observable_supervision(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the canonical observable-supervision table when available."""
        return self.load_table("observable_supervision", use_cache=use_cache)

    def load_benchmark_entries(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the canonical cross-source benchmark manifest."""
        return self.load_table("benchmark_entries", use_cache=use_cache)

    def load_training_entry_view(self, use_cache: bool = True) -> pd.DataFrame:
        """Load the trainer-facing accession view when available."""
        return self.load_table("training_entry_view", use_cache=use_cache)


def _resolve_workspace_name(
    requested: str | None,
    available: list[str],
    default: str,
) -> str:
    """Resolve an optional workspace name against the manifest payload."""
    if not available:
        raise ValueError("The dataset manifest does not define any workspaces.")
    if requested is None:
        if default in available:
            return default
        if len(available) == 1:
            return available[0]
        raise ValueError(
            "The dataset exposes multiple workspaces and no default workspace."
        )
    normalized = requested.strip().lower()
    if normalized not in available:
        raise ValueError(
            f"Workspace '{requested}' is not available. "
            f"Expected one of: {', '.join(sorted(available))}."
        )
    return normalized


def _merge_frames(
    left: pd.DataFrame, right: pd.DataFrame, join_key: str
) -> pd.DataFrame:
    """Merge one dataframe into another on a shared key with suffix handling."""
    payload = right.copy()
    rename_map: dict[str, str] = {}
    for column in payload.columns:
        if column == join_key:
            continue
        if column in left.columns:
            rename_map[column] = (
                f"{payload.attrs.get('source_table', 'joined')}__{column}"
            )
    if rename_map:
        payload = payload.rename(columns=rename_map)
    return left.merge(payload, on=join_key, how="left")
