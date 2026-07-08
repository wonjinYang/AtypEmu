"""Training-plan config and dataframe export helpers for AtypEmu."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback for local tooling.
    fcntl = None

import pandas as pd

from atypemu.datasets import (
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    IntegratedDataRegistry,
    MetaDataFrameRegistry,
    SourceDataFrameRegistry,
)
from atypemu.datasets.registry import DEFAULT_WORKSPACE


TRAINING_PLAN_VERSION = "1.0.0"
TRAINING_PLAN_FILENAME = "training_plan.json"
BMRB_MEASUREMENT_TO_ROLE = {
    "chemical_shift": ("primary", "stage_c_primary"),
    "j_coupling": ("secondary", "stage_c_secondary"),
    "noe_restraint": ("secondary", "stage_c_secondary"),
}
SAXS_SUMMARY_COLUMNS = [
    "guinier_rg",
    "pddf_rg",
    "pddf_dmax",
    "experimental_mw",
    "guinier_i0_mw",
    "porod_mw",
    "porod_volume",
]
OBSERVABLE_SUPERVISION_COLUMNS = [
    "supervision_uid",
    "entity_uid",
    "source_id",
    "native_id",
    "linked_bmrb_entity_uids",
    "linked_bmrb_count",
    "supervision_family",
    "supervision_kind",
    "supervision_role",
    "training_phase",
    "measurement_count",
    "split",
    "asset_path",
    "source_table",
]
BENCHMARK_ENTRY_COLUMNS = [
    "benchmark_uid",
    "entity_uid",
    "source_id",
    "native_id",
    "display_name",
    "benchmark_role",
    "training_policy",
    "evaluation_track",
    "linked_bmrb_entity_uids",
    "linked_bmrb_count",
    "sequence_hash",
    "status",
    "notes",
]


@dataclass(slots=True)
class TrainingSourcePolicy:
    """One source-level role policy for staged AtypEmu training."""

    source_id: str
    default_role: str
    train_usage: str
    eval_usage: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize the source policy to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingSourcePolicy":
        """Create a source policy from one dictionary."""
        return cls(
            source_id=str(payload["source_id"]),
            default_role=str(payload["default_role"]),
            train_usage=str(payload["train_usage"]),
            eval_usage=str(payload["eval_usage"]),
            notes=str(payload["notes"]),
        )


@dataclass(slots=True)
class TrainingPlanConfig:
    """Canonical staged training plan for AtypEmu."""

    plan_name: str
    plan_version: str
    objective: str
    density_mode: str
    teacher_strategy: str
    primary_targets: list[str] = field(default_factory=list)
    secondary_targets: list[str] = field(default_factory=list)
    auxiliary_targets: list[str] = field(default_factory=list)
    benchmark_priority: list[str] = field(default_factory=list)
    source_policies: dict[str, TrainingSourcePolicy] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the training plan to a JSON-compatible dictionary."""
        return {
            "plan_name": self.plan_name,
            "plan_version": self.plan_version,
            "objective": self.objective,
            "density_mode": self.density_mode,
            "teacher_strategy": self.teacher_strategy,
            "primary_targets": list(self.primary_targets),
            "secondary_targets": list(self.secondary_targets),
            "auxiliary_targets": list(self.auxiliary_targets),
            "benchmark_priority": list(self.benchmark_priority),
            "source_policies": {
                source_id: policy.as_dict()
                for source_id, policy in sorted(self.source_policies.items())
            },
        }

    def to_json(self, path: str | Path) -> None:
        """Write the canonical training plan to disk."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingPlanConfig":
        """Create a training plan from one dictionary."""
        return cls(
            plan_name=str(payload["plan_name"]),
            plan_version=str(payload["plan_version"]),
            objective=str(payload["objective"]),
            density_mode=str(payload["density_mode"]),
            teacher_strategy=str(payload["teacher_strategy"]),
            primary_targets=list(payload.get("primary_targets", [])),
            secondary_targets=list(payload.get("secondary_targets", [])),
            auxiliary_targets=list(payload.get("auxiliary_targets", [])),
            benchmark_priority=list(payload.get("benchmark_priority", [])),
            source_policies={
                source_id: TrainingSourcePolicy.from_dict(item)
                for source_id, item in dict(payload.get("source_policies", {})).items()
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingPlanConfig":
        """Load a training plan from one JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def default(cls) -> "TrainingPlanConfig":
        """Return the repository-default staged training plan."""
        return cls(
            plan_name="teacher_first_hybrid_density",
            plan_version=TRAINING_PLAN_VERSION,
            objective="teacher_first",
            density_mode="hybrid_bridge",
            teacher_strategy="offline_core_empirical_density",
            primary_targets=["chemical_shifts"],
            secondary_targets=["j_couplings", "noe_restraints"],
            auxiliary_targets=["saxs_summary", "saxs_pddf_profile", "saxs_intensity"],
            benchmark_priority=[
                "ped_free_state",
                "sasbdb_saxs",
                "mfib_bound_state",
                "fuzdb_fuzzy_context",
            ],
            source_policies={
                "bmrb": TrainingSourcePolicy(
                    source_id="bmrb",
                    default_role="core_supervision",
                    train_usage="primary_train",
                    eval_usage="in_domain_validation",
                    notes="Primary NMR supervision anchor for teacher generation and distillation.",
                ),
                "ped": TrainingSourcePolicy(
                    source_id="ped",
                    default_role="external_benchmark",
                    train_usage="disabled_v1",
                    eval_usage="free_state_validation",
                    notes="Catalog and benchmark workspaces remain holdout-only in v1.",
                ),
                "sasbdb": TrainingSourcePolicy(
                    source_id="sasbdb",
                    default_role="auxiliary_observable",
                    train_usage="auxiliary_only",
                    eval_usage="saxs_validation",
                    notes="SAXS is used for auxiliary forward supervision and evaluation, not teacher density generation.",
                ),
                "mfib": TrainingSourcePolicy(
                    source_id="mfib",
                    default_role="ood_benchmark",
                    train_usage="disabled_v1",
                    eval_usage="bound_complex_benchmark",
                    notes="MFIB remains an out-of-domain benchmark with optional geometry-aware slicing.",
                ),
                "fuzdb": TrainingSourcePolicy(
                    source_id="fuzdb",
                    default_role="context_benchmark",
                    train_usage="disabled_v1",
                    eval_usage="fuzzy_context_benchmark",
                    notes="FuzDB remains a fuzzy-context benchmark and metadata source in v1.",
                ),
                "mobidb": TrainingSourcePolicy(
                    source_id="mobidb",
                    default_role="future_auxiliary",
                    train_usage="future_v2",
                    eval_usage="future_benchmark",
                    notes="Reserved for disorder-region supervision once the source package is implemented.",
                ),
                "dibs": TrainingSourcePolicy(
                    source_id="dibs",
                    default_role="future_benchmark",
                    train_usage="future_v2",
                    eval_usage="future_bound_benchmark",
                    notes="Reserved for bound-disorder benchmark expansion.",
                ),
                "ideal": TrainingSourcePolicy(
                    source_id="ideal",
                    default_role="future_benchmark",
                    train_usage="future_v2",
                    eval_usage="future_protean_benchmark",
                    notes="Reserved for protean and induced-folding benchmark expansion.",
                ),
            },
        )


def export_training_dataframes(
    data_root: str | Path,
    integrated_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Export trainer-facing integrated dataframes and the canonical plan.

    Args:
        data_root: Repository ``data`` root.
        integrated_root: Cross-source integrated root such as ``data/integrated``.
        refresh: Whether to overwrite existing Parquet tables.

    Returns:
        Summary dictionary describing the written training bundle.
    """
    data_root_path = Path(data_root)
    integrated_root_path = Path(integrated_root)
    datasets_dir = integrated_root_path / "datasets"
    configs_dir = integrated_root_path / "configs"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    plan = TrainingPlanConfig.default()
    plan_path = configs_dir / TRAINING_PLAN_FILENAME
    plan.to_json(plan_path)

    integrated_registry = IntegratedDataRegistry.from_manifest(
        integrated_root_path / "manifest.json"
    )
    bmrb_registry = SourceDataFrameRegistry.from_data_root(data_root_path, "bmrb")
    meta_registry = _load_meta_registry(data_root_path)

    accessions = integrated_registry.load_table("accessions")
    splits = integrated_registry.load_table("splits")
    bmrb_entries = bmrb_registry.load_table("entries")
    bmrb_targets = bmrb_registry.load_table("targets")
    meta_links = (
        meta_registry.load_table("links")
        if meta_registry is not None and "links" in meta_registry.available_tables()
        else pd.DataFrame()
    )
    meta_entities = (
        meta_registry.load_table("entities")
        if meta_registry is not None and "entities" in meta_registry.available_tables()
        else pd.DataFrame()
    )

    teacher_examples = _build_teacher_examples(
        accessions=accessions,
        splits=splits,
        bmrb_entries=bmrb_entries,
        data_root=data_root_path,
        integrated_root=integrated_root_path,
    )
    observable_supervision = _build_observable_supervision(
        data_root=data_root_path,
        teacher_examples=teacher_examples,
        bmrb_targets=bmrb_targets,
        meta_links=meta_links,
    )
    benchmark_entries = _build_benchmark_entries(
        data_root=data_root_path,
        meta_links=meta_links,
        meta_entities=meta_entities,
    )
    training_entry_view = _build_training_entry_view(
        teacher_examples=teacher_examples,
        observable_supervision=observable_supervision,
        benchmark_entries=benchmark_entries,
    )

    tables = {
        "teacher_examples": teacher_examples,
        "observable_supervision": observable_supervision,
        "benchmark_entries": benchmark_entries,
        "training_entry_view": training_entry_view,
    }
    metadata = {
        name: _write_training_table(
            integrated_root=integrated_root_path,
            table_name=name,
            dataframe=dataframe,
            refresh=refresh,
        )
        for name, dataframe in tables.items()
    }
    _write_training_manifest(
        data_root=data_root_path,
        integrated_root=integrated_root_path,
        metadata=metadata,
    )
    _update_training_workspace_config(
        config_path=configs_dir / "training_workspace.json",
        plan_path=plan_path,
        datasets_dir=datasets_dir,
    )

    return {
        "plan_path": str(plan_path),
        "tables_written": len(tables),
        "teacher_examples": len(teacher_examples),
        "observable_supervision": len(observable_supervision),
        "benchmark_entries": len(benchmark_entries),
    }


def _build_teacher_examples(
    accessions: pd.DataFrame,
    splits: pd.DataFrame,
    bmrb_entries: pd.DataFrame,
    data_root: Path,
    integrated_root: Path,
) -> pd.DataFrame:
    """Build the canonical teacher-example dataframe."""
    split_map = dict(zip(splits["entity_uid"], splits["split"], strict=True))
    entry_map = {
        row["entry_uid"]: row for row in bmrb_entries.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for record in accessions.to_dict(orient="records"):
        bmrb_id = str(record["bmrb_id"])
        entity_uid = f"bmrb:{bmrb_id}"
        source_paths = _normalize_mapping(record.get("source_paths"))
        rel_source_paths = {
            source: _data_relative_path(path, data_root)
            for source, path in sorted(source_paths.items())
        }
        source_entry = entry_map.get(entity_uid, {})
        pool_rel = Path("integrated") / "pools" / bmrb_id / "candidate_pool.jsonl"
        observable_rel = (
            Path("integrated") / "observables" / bmrb_id / "observables.npz"
        )
        solution_rel = (
            Path("integrated") / "teachers" / bmrb_id / "weight_solution.json"
        )
        breakdown_rel = (
            Path("integrated") / "teachers" / bmrb_id / "energy_breakdown.json"
        )
        target_path = _source_relative_path(
            "bmrb", source_entry.get("bundle_path")
        ) or _data_relative_path(record.get("bmrb_bundle_path"), data_root)
        rows.append(
            {
                "entity_uid": entity_uid,
                "source_id": "bmrb",
                "native_id": bmrb_id,
                "bmrb_id": bmrb_id,
                "sequence": record.get("sequence"),
                "sequence_hash": record.get("sequence_hash"),
                "split": split_map.get(entity_uid),
                "candidate_sources": "|".join(_normalize_list(record.get("sources"))),
                "candidate_source_count": len(_normalize_list(record.get("sources"))),
                "candidate_structure_roots": json.dumps(
                    rel_source_paths, sort_keys=True
                ),
                "target_bundle_path": target_path,
                "chemical_shift_count": int(record.get("chemical_shift_count") or 0),
                "j_coupling_count": int(record.get("j_coupling_count") or 0),
                "noe_count": int(record.get("noe_count") or 0),
                "candidate_pool_path": str(pool_rel),
                "observable_bundle_path": str(observable_rel),
                "teacher_solution_path": str(solution_rel),
                "teacher_breakdown_path": str(breakdown_rel),
                "target_ready": bool(
                    target_path and (data_root / target_path).exists()
                ),
                "pool_ready": (data_root / pool_rel).exists(),
                "observable_ready": (data_root / observable_rel).exists(),
                "teacher_ready": (data_root / solution_rel).exists(),
                "teacher_method": _load_teacher_method(data_root / solution_rel),
                "density_support": "discrete_support",
                "density_bridge_mode": "hybrid_bridge",
            }
        )
    return (
        pd.DataFrame(rows)
        .replace({r"^\s*$": pd.NA}, regex=True)
        .convert_dtypes(dtype_backend="pyarrow")
    )


def _build_observable_supervision(
    data_root: Path,
    teacher_examples: pd.DataFrame,
    bmrb_targets: pd.DataFrame,
    meta_links: pd.DataFrame,
) -> pd.DataFrame:
    """Build aggregated supervision availability rows for training."""
    split_map = dict(
        zip(teacher_examples["entity_uid"], teacher_examples["split"], strict=True)
    )
    bundle_map = dict(
        zip(
            teacher_examples["entity_uid"],
            teacher_examples["target_bundle_path"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    grouped = bmrb_targets.groupby(
        ["entry_uid", "native_id", "measurement_kind"], dropna=False
    )
    for (entry_uid, native_id, measurement_kind), frame in grouped:
        role, phase = BMRB_MEASUREMENT_TO_ROLE.get(
            str(measurement_kind),
            ("secondary", "stage_c_secondary"),
        )
        rows.append(
            {
                "supervision_uid": f"{entry_uid}:{measurement_kind}",
                "entity_uid": entry_uid,
                "source_id": "bmrb",
                "native_id": native_id,
                "linked_bmrb_entity_uids": entry_uid,
                "linked_bmrb_count": 1,
                "supervision_family": "nmr",
                "supervision_kind": measurement_kind,
                "supervision_role": role,
                "training_phase": phase,
                "measurement_count": len(frame),
                "split": split_map.get(entry_uid),
                "asset_path": bundle_map.get(entry_uid),
                "source_table": "targets",
            }
        )

    sasbdb_registry = _load_source_registry(data_root, "sasbdb")
    if sasbdb_registry is not None:
        sasbdb_entries = sasbdb_registry.load_table("entries")
        for row in sasbdb_entries.to_dict(orient="records"):
            entry_uid = row["entry_uid"]
            linked_ids = _linked_bmrb_entities(meta_links, entry_uid, "sasbdb")
            rows.append(
                {
                    "supervision_uid": f"{entry_uid}:saxs_summary",
                    "entity_uid": entry_uid,
                    "source_id": "sasbdb",
                    "native_id": row["native_id"],
                    "linked_bmrb_entity_uids": (
                        "|".join(linked_ids) if linked_ids else pd.NA
                    ),
                    "linked_bmrb_count": len(linked_ids),
                    "supervision_family": "saxs",
                    "supervision_kind": "saxs_summary",
                    "supervision_role": "auxiliary",
                    "training_phase": "stage_c_auxiliary",
                    "measurement_count": sum(
                        0 if pd.isna(row.get(column)) else 1
                        for column in SAXS_SUMMARY_COLUMNS
                    ),
                    "split": "external_validation",
                    "asset_path": row.get("sascif_path"),
                    "source_table": "entries",
                }
            )
        if "profiles" in sasbdb_registry.available_tables():
            profiles = sasbdb_registry.load_table("profiles")
            grouped_profiles = profiles.groupby(
                ["entry_uid", "native_id", "profile_kind", "asset_path"],
                dropna=False,
            )
            for (
                entry_uid,
                native_id,
                profile_kind,
                asset_path,
            ), frame in grouped_profiles:
                linked_ids = _linked_bmrb_entities(meta_links, entry_uid, "sasbdb")
                rows.append(
                    {
                        "supervision_uid": f"{entry_uid}:{profile_kind}",
                        "entity_uid": entry_uid,
                        "source_id": "sasbdb",
                        "native_id": native_id,
                        "linked_bmrb_entity_uids": (
                            "|".join(linked_ids) if linked_ids else pd.NA
                        ),
                        "linked_bmrb_count": len(linked_ids),
                        "supervision_family": "saxs",
                        "supervision_kind": f"saxs_{profile_kind}",
                        "supervision_role": "auxiliary",
                        "training_phase": "stage_c_auxiliary",
                        "measurement_count": len(frame),
                        "split": "external_validation",
                        "asset_path": _source_relative_path("sasbdb", asset_path),
                        "source_table": "profiles",
                    }
                )
    return _frame_from_rows(rows, OBSERVABLE_SUPERVISION_COLUMNS)


def _build_benchmark_entries(
    data_root: Path,
    meta_links: pd.DataFrame,
    meta_entities: pd.DataFrame,
) -> pd.DataFrame:
    """Build one benchmark manifest table across external sources."""
    entity_map = {
        row["entity_uid"]: row for row in meta_entities.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []

    ped_registry = _load_source_registry(data_root, "ped", workspace="catalog")
    if ped_registry is not None:
        for row in ped_registry.load_table("validation_manifest").to_dict(
            orient="records"
        ):
            entity_uid = row["entry_uid"]
            linked_ids = _linked_bmrb_entities(meta_links, entity_uid, "ped")
            rows.append(
                _benchmark_row(
                    entity_uid=entity_uid,
                    source_id="ped",
                    native_id=row["native_id"],
                    display_name=row.get("protein_name"),
                    benchmark_role="external_free_state_benchmark",
                    training_policy=_ped_training_policy(row.get("recommended_role")),
                    evaluation_track=row.get("validation_group"),
                    linked_bmrb_entity_uids=linked_ids,
                    sequence_hash=entity_map.get(entity_uid, {}).get("sequence_hash"),
                    status=row.get("recommended_role"),
                    notes=row.get("generation_family"),
                )
            )

    sasbdb_registry = _load_source_registry(data_root, "sasbdb")
    if sasbdb_registry is not None:
        for row in sasbdb_registry.load_table("entries").to_dict(orient="records"):
            entity_uid = row["entry_uid"]
            linked_ids = _linked_bmrb_entities(meta_links, entity_uid, "sasbdb")
            evaluation_track = (
                "saxs_profile"
                if row.get("intensities_path") or row.get("pddf_path")
                else "saxs_summary"
            )
            rows.append(
                _benchmark_row(
                    entity_uid=entity_uid,
                    source_id="sasbdb",
                    native_id=row["native_id"],
                    display_name=row.get("project_title") or row.get("code"),
                    benchmark_role="saxs_benchmark",
                    training_policy="auxiliary_validation_only",
                    evaluation_track=evaluation_track,
                    linked_bmrb_entity_uids=linked_ids,
                    sequence_hash=entity_map.get(entity_uid, {}).get("sequence_hash"),
                    status=row.get("status"),
                    notes=row.get("type_of_curve"),
                )
            )

    mfib_registry = _load_source_registry(data_root, "mfib")
    if mfib_registry is not None:
        geometry_map = {}
        if "geometry_entry_features" in mfib_registry.available_tables():
            geometry_map = {
                row["entry_uid"]: row
                for row in mfib_registry.load_table("geometry_entry_features").to_dict(
                    orient="records"
                )
            }
        for row in mfib_registry.load_table("entries").to_dict(orient="records"):
            entity_uid = row["entry_uid"]
            linked_ids = _linked_bmrb_entities(meta_links, entity_uid, "mfib")
            geometry_row = geometry_map.get(entity_uid, {})
            rows.append(
                _benchmark_row(
                    entity_uid=entity_uid,
                    source_id="mfib",
                    native_id=row["native_id"],
                    display_name=row.get("name"),
                    benchmark_role="bound_complex_benchmark",
                    training_policy="benchmark_only",
                    evaluation_track=row.get("evidence_level"),
                    linked_bmrb_entity_uids=linked_ids,
                    sequence_hash=entity_map.get(entity_uid, {}).get("sequence_hash"),
                    status=row.get("exp_method"),
                    notes=_coalesce(
                        geometry_row.get("pair_count"),
                        row.get("class_name"),
                        row.get("subclass_name"),
                    ),
                )
            )

    fuzdb_registry = _load_source_registry(data_root, "fuzdb")
    if fuzdb_registry is not None:
        for row in fuzdb_registry.load_table("entries").to_dict(orient="records"):
            entity_uid = row["entry_uid"]
            linked_ids = _linked_bmrb_entities(meta_links, entity_uid, "fuzdb")
            rows.append(
                _benchmark_row(
                    entity_uid=entity_uid,
                    source_id="fuzdb",
                    native_id=row["native_id"],
                    display_name=row.get("protein_name"),
                    benchmark_role="fuzzy_context_benchmark",
                    training_policy="benchmark_only",
                    evaluation_track=_coalesce(
                        row.get("topology_class"),
                        row.get("mechanism_category"),
                        row.get("detection_methods"),
                    ),
                    linked_bmrb_entity_uids=linked_ids,
                    sequence_hash=entity_map.get(entity_uid, {}).get("sequence_hash"),
                    status=row.get("partner_name"),
                    notes=row.get("mechanism_category"),
                )
            )

    return _frame_from_rows(rows, BENCHMARK_ENTRY_COLUMNS)


def _build_training_entry_view(
    teacher_examples: pd.DataFrame,
    observable_supervision: pd.DataFrame,
    benchmark_entries: pd.DataFrame,
) -> pd.DataFrame:
    """Build a trainer-facing entry view from the exported bundle tables."""
    view = teacher_examples.copy()
    nmr_rows = observable_supervision.loc[observable_supervision["source_id"] == "bmrb"]
    if not nmr_rows.empty:
        summary = (
            nmr_rows.pivot_table(
                index="entity_uid",
                columns="supervision_kind",
                values="measurement_count",
                aggfunc="sum",
                fill_value=0,
            )
            .rename(
                columns={
                    "chemical_shift": "chemical_shift_targets",
                    "j_coupling": "j_coupling_targets",
                    "noe_restraint": "noe_targets",
                }
            )
            .reset_index()
        )
        view = view.merge(summary, on="entity_uid", how="left")

    exploded = _explode_linked_bmrb_ids(benchmark_entries)
    if not exploded.empty:
        benchmark_summary = (
            exploded.pivot_table(
                index="linked_bmrb_entity_uid",
                columns="source_id",
                values="benchmark_uid",
                aggfunc="count",
                fill_value=0,
            )
            .rename(
                columns={
                    "ped": "ped_benchmark_links",
                    "sasbdb": "sasbdb_benchmark_links",
                    "mfib": "mfib_benchmark_links",
                    "fuzdb": "fuzdb_benchmark_links",
                }
            )
            .reset_index()
            .rename(columns={"linked_bmrb_entity_uid": "entity_uid"})
        )
        view = view.merge(benchmark_summary, on="entity_uid", how="left")
        count_columns = [
            column for column in benchmark_summary.columns if column != "entity_uid"
        ]
        if count_columns:
            view["total_external_benchmark_links"] = (
                view[count_columns].fillna(0).sum(axis=1)
            )
    return view.convert_dtypes(dtype_backend="pyarrow")


def _write_training_table(
    integrated_root: Path,
    table_name: str,
    dataframe: pd.DataFrame,
    refresh: bool,
) -> DataFrameTableMeta:
    """Write one training dataframe and return manifest metadata."""
    relative_path = Path("datasets") / f"{table_name}.parquet"
    target_path = integrated_root / relative_path
    if refresh or not target_path.exists():
        dataframe.to_parquet(target_path, index=False)
    return DataFrameTableMeta(
        name=table_name,
        relative_path=str(relative_path),
        columns=list(dataframe.columns),
        dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        row_count=len(dataframe),
        primary_key=_primary_key(table_name),
        join_keys=_join_keys(table_name),
        workspace=DEFAULT_WORKSPACE,
        level="entry" if table_name != "observable_supervision" else "measurement",
    )


def _write_training_manifest(
    data_root: Path,
    integrated_root: Path,
    metadata: dict[str, DataFrameTableMeta],
) -> None:
    """Merge training tables into the integrated dataframe manifest."""
    manifest_path = integrated_root / "manifest.json"
    lock_path = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            manifest = DataFrameDatasetManifest.from_json(manifest_path)
            workspace_meta = manifest.workspaces[DEFAULT_WORKSPACE]
            workspace_meta.tables.update(metadata)
            manifest.created_at_utc = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
            manifest.data_root = str(data_root.resolve())
            manifest.dataset_root = str(integrated_root.resolve())
            tmp_path = manifest_path.with_name(
                f".{manifest_path.name}.{os.getpid()}.tmp"
            )
            tmp_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
            tmp_path.replace(manifest_path)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _update_training_workspace_config(
    config_path: Path,
    plan_path: Path,
    datasets_dir: Path,
) -> None:
    """Update the integrated workspace config with training artifact hints."""
    if not config_path.exists():
        return
    payload = json.loads(config_path.read_text())
    payload.setdefault("artifacts", {})
    payload["artifacts"]["datasets_dir"] = str(datasets_dir.resolve())
    payload["training"] = {
        "plan_path": str(plan_path.resolve()),
        "teacher_examples_table": str(
            (datasets_dir / "teacher_examples.parquet").resolve()
        ),
        "observable_supervision_table": str(
            (datasets_dir / "observable_supervision.parquet").resolve()
        ),
        "benchmark_entries_table": str(
            (datasets_dir / "benchmark_entries.parquet").resolve()
        ),
        "training_entry_view_table": str(
            (datasets_dir / "training_entry_view.parquet").resolve()
        ),
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_source_registry(
    data_root: Path,
    source_id: str,
    workspace: str | None = None,
) -> SourceDataFrameRegistry | None:
    """Load one source registry when the manifest exists."""
    manifest_path = data_root / source_id / "datasets" / "manifest.json"
    if not manifest_path.exists():
        return None
    return SourceDataFrameRegistry.from_manifest(manifest_path, workspace=workspace)


def _load_meta_registry(data_root: Path) -> MetaDataFrameRegistry | None:
    """Load the meta registry when available."""
    manifest_path = data_root / "meta" / "datasets" / "manifest.json"
    if not manifest_path.exists():
        return None
    return MetaDataFrameRegistry.from_manifest(manifest_path)


def _primary_key(table_name: str) -> list[str]:
    """Return the canonical primary key for one training table."""
    return {
        "teacher_examples": ["entity_uid"],
        "observable_supervision": ["supervision_uid"],
        "benchmark_entries": ["benchmark_uid"],
        "training_entry_view": ["entity_uid"],
    }[table_name]


def _join_keys(table_name: str) -> list[str]:
    """Return the canonical join keys for one training table."""
    return {
        "teacher_examples": ["entity_uid"],
        "observable_supervision": ["entity_uid"],
        "benchmark_entries": ["entity_uid"],
        "training_entry_view": ["entity_uid"],
    }[table_name]


def _normalize_list(value: Any) -> list[str]:
    """Normalize a list-like or pipe-joined value to a string list."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        normalized = value.tolist()
        if isinstance(normalized, list):
            return [str(item) for item in normalized if str(item).strip()]
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item for item in text.split("|") if item]


def _normalize_mapping(value: Any) -> dict[str, str]:
    """Normalize a mapping-like field to a string-keyed dictionary."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def _data_relative_path(path_value: Any, data_root: Path) -> str | None:
    """Convert one path-like value to a ``data/``-relative path when possible."""
    if path_value is None or pd.isna(path_value):
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path)


def _source_relative_path(source_id: str, path_value: Any) -> str | None:
    """Convert one source-local path into a ``data/<source>/...``-relative path."""
    if path_value is None or pd.isna(path_value):
        return None
    path = Path(str(path_value))
    return str(Path(source_id) / path) if not path.is_absolute() else str(path)


def _load_teacher_method(solution_path: Path) -> str | None:
    """Read the teacher method from one saved weight solution when present."""
    if not solution_path.exists():
        return None
    payload = json.loads(solution_path.read_text())
    method = payload.get("method")
    return None if method is None else str(method)


def _linked_bmrb_entities(
    meta_links: pd.DataFrame,
    entity_uid: str,
    source_id: str,
) -> list[str]:
    """Return linked BMRB entities for one external entity."""
    if meta_links.empty:
        return []
    mask = (
        (meta_links["left_entity_uid"] == entity_uid)
        & (meta_links["right_source_id"] == "bmrb")
        & (meta_links["left_source_id"] == source_id)
    ) | (
        (meta_links["right_entity_uid"] == entity_uid)
        & (meta_links["left_source_id"] == "bmrb")
        & (meta_links["right_source_id"] == source_id)
    )
    frame = meta_links.loc[mask]
    linked = {
        (
            row["left_entity_uid"]
            if row["left_source_id"] == "bmrb"
            else row["right_entity_uid"]
        )
        for row in frame.to_dict(orient="records")
    }
    return sorted(linked)


def _ped_training_policy(recommended_role: Any) -> str:
    """Map PED validation roles to the staged training policy labels."""
    mapping = {
        "external_benchmark_only": "benchmark_only",
        "training_augmentation_candidate": "documented_but_inactive",
        "validation_only": "holdout_validation_only",
    }
    return mapping.get(str(recommended_role), "benchmark_only")


def _benchmark_row(
    entity_uid: str,
    source_id: str,
    native_id: str,
    display_name: Any,
    benchmark_role: str,
    training_policy: str,
    evaluation_track: Any,
    linked_bmrb_entity_uids: list[str],
    sequence_hash: Any,
    status: Any,
    notes: Any,
) -> dict[str, Any]:
    """Build one normalized benchmark-entry row."""
    return {
        "benchmark_uid": f"{source_id}:{native_id}:benchmark",
        "entity_uid": entity_uid,
        "source_id": source_id,
        "native_id": native_id,
        "display_name": display_name,
        "benchmark_role": benchmark_role,
        "training_policy": training_policy,
        "evaluation_track": evaluation_track,
        "linked_bmrb_entity_uids": (
            "|".join(linked_bmrb_entity_uids) if linked_bmrb_entity_uids else pd.NA
        ),
        "linked_bmrb_count": len(linked_bmrb_entity_uids),
        "sequence_hash": sequence_hash,
        "status": status,
        "notes": notes,
    }


def _explode_linked_bmrb_ids(benchmark_entries: pd.DataFrame) -> pd.DataFrame:
    """Explode one benchmark table by linked BMRB entity identifiers."""
    if benchmark_entries.empty:
        return _frame_from_rows(
            [],
            ["benchmark_uid", "source_id", "linked_bmrb_entity_uid"],
        )
    rows: list[dict[str, Any]] = []
    for row in benchmark_entries.to_dict(orient="records"):
        linked_ids = _normalize_list(row.get("linked_bmrb_entity_uids"))
        for linked_id in linked_ids:
            rows.append(
                {
                    "benchmark_uid": row["benchmark_uid"],
                    "source_id": row["source_id"],
                    "linked_bmrb_entity_uid": linked_id,
                }
            )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _coalesce(*values: Any) -> Any:
    """Return the first non-empty value in order."""
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return pd.NA


def _frame_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build one nullable dataframe with a stable empty-schema fallback."""
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=columns)
    return dataframe.replace({r"^\s*$": pd.NA}, regex=True).convert_dtypes(
        dtype_backend="pyarrow"
    )
