"""Generate UCBShift2.0/CS-reweighting teachers from BioEmu conformer probes.

This module connects the detached BioEmu conformer ensemble probe to the
teacher-student NMR training path:

1. export sampled BioEmu conformers to per-sample PDB files,
2. run UCBShift2.0 on each conformer,
3. fit CS-reweighting population weights against experimental chemical shifts, and
4. write teacher weights/teacher posterior means for latent-student training.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atypemu.datasets import IntegratedDataRegistry
from atypemu.training.bioemu_ucbshift_cnnls_teacher import (
    run_bioemu_ucbshift_cnnls_teacher,
)
from atypemu.training.config import UCBShiftGenerationConfig
from atypemu.training.materialize import resolve_existing_path
from atypemu.training.ucbshift import (
    UCBShiftRuntime,
    _build_ucbshift_env,
    _run_candidate_sidecars,
    _ucbshift_sidecar_usable,
    resolve_ucbshift_runtime,
)


MIN_TEACHER_CONFORMERS_PER_ENTITY = 512
_UCBSHIFT_PDB_REPAIR_MARKER_VERSION = "v1"
_BIOEMU_EXPORT_MAX_SPAN_NM = 20.0
_BIOEMU_EXPORT_MAX_ABS_COORD_ANGSTROM = 250.0
_SAMPLE_SIDECAR_RE = re.compile(r"sample_(\d+)\.csv$")


_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


@dataclass(slots=True)
class BioEmuProbeTeacherSummary:
    """Summary for the BioEmu probe -> UCBShift -> CS-reweighting pipeline."""

    status: str
    message: str
    probe_input_rows: int
    selected_entity_count: int
    shard_count: int
    shard_index: int
    sidecar_only: bool
    ready_probe_outputs_only: bool
    skipped_not_ready_entity_count: int
    requested_conformers: int
    exported_pdb_count: int
    sidecar_count: int
    failed_sidecar_count: int
    artifacts: dict[str, str]
    teacher_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "probe_input_rows": self.probe_input_rows,
            "selected_entity_count": self.selected_entity_count,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
            "sidecar_only": self.sidecar_only,
            "ready_probe_outputs_only": self.ready_probe_outputs_only,
            "skipped_not_ready_entity_count": self.skipped_not_ready_entity_count,
            "requested_conformers": self.requested_conformers,
            "exported_pdb_count": self.exported_pdb_count,
            "sidecar_count": self.sidecar_count,
            "failed_sidecar_count": self.failed_sidecar_count,
            "artifacts": dict(self.artifacts),
            "teacher_summary": dict(self.teacher_summary),
        }


def run_bioemu_probe_ucbshift_cnnls_teacher(
    *,
    data_root: str | Path,
    run_dir: str | Path,
    probe_name: str,
    output_dir: str | Path | None = None,
    pdb_root: str | Path | None = None,
    sidecar_root: str | Path | None = None,
    ucbshift_config_path: str | Path | None = None,
    method: str = "cs_reweighting",
    lambda_reg: float = 0.01,
    default_cs_sigma: float = 1.0,
    max_entities: int | None = None,
    max_conformers_per_entity: int | None = None,
    refresh_outputs: bool = False,
    worker_count: int | None = None,
    require_all_sidecars: bool = False,
    skip_pdb_export: bool = False,
    allow_topology_fallback: bool = False,
    ready_probe_outputs_only: bool = False,
    auto_fill_missing_sidecars: bool = False,
    sidecar_only: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
) -> BioEmuProbeTeacherSummary:
    """Build a CS-reweighting teacher from detached BioEmu probe outputs."""

    data_root_path = Path(data_root)
    run_dir_path = Path(run_dir)
    report_dir = Path(output_dir) if output_dir else run_dir_path / "reports"
    arrays_dir = report_dir / "arrays"
    figures_dir = report_dir / "figures"
    pdb_root = _resolve_teacher_cache_root(
        pdb_root,
        default=report_dir / "bioemu_ucbshift_teacher_structures" / probe_name,
        base_dir=report_dir,
    )
    sidecar_root = _resolve_teacher_cache_root(
        sidecar_root,
        default=report_dir / "bioemu_ucbshift_teacher_sidecars" / probe_name,
        base_dir=report_dir,
    )
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdb_root.mkdir(parents=True, exist_ok=True)
    sidecar_root.mkdir(parents=True, exist_ok=True)

    probe_inputs_path = (
        run_dir_path / "reports" / "arrays" / "bioemu_detached_conformer_probe_inputs.parquet"
    )
    if not probe_inputs_path.exists():
        raise FileNotFoundError(f"Missing BioEmu probe inputs: {probe_inputs_path}")
    probe_inputs = pd.read_parquet(probe_inputs_path)
    if probe_inputs.empty:
        raise ValueError(f"BioEmu probe input table is empty: {probe_inputs_path}")

    selected_inputs = _select_probe_inputs(
        probe_inputs,
        probe_name=probe_name,
        max_entities=max_entities,
    )
    selected_inputs = _apply_entity_shard(
        selected_inputs,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    selected_entity_count_before_ready_filter = int(
        selected_inputs["entity_uid"].astype(str).nunique()
    )
    skipped_not_ready_entity_count = 0
    if ready_probe_outputs_only:
        selected_inputs = _filter_ready_probe_outputs(
            selected_inputs,
            run_dir=run_dir_path,
            allow_topology_fallback=allow_topology_fallback,
        )
        skipped_not_ready_entity_count = (
            selected_entity_count_before_ready_filter
            - int(selected_inputs["entity_uid"].astype(str).nunique())
        )
    if selected_inputs.empty:
        manifest_path = arrays_dir / "bioemu_probe_ucbshift_sidecar_manifest.parquet"
        target_manifest_path = arrays_dir / "bioemu_probe_ucbshift_target_manifest.parquet"
        failures_path = arrays_dir / "bioemu_probe_ucbshift_sidecar_failures.parquet"
        _write_table(pd.DataFrame(), manifest_path)
        _write_table(pd.DataFrame(), target_manifest_path)
        _write_table(pd.DataFrame(), failures_path)
        summary = BioEmuProbeTeacherSummary(
            status="no_ready_probe_outputs",
            message="No detached BioEmu probe outputs are ready for UCBShift sidecars.",
            probe_input_rows=int(len(probe_inputs)),
            selected_entity_count=0,
            shard_count=int(shard_count),
            shard_index=int(shard_index),
            sidecar_only=bool(sidecar_only),
            ready_probe_outputs_only=bool(ready_probe_outputs_only),
            skipped_not_ready_entity_count=int(skipped_not_ready_entity_count),
            requested_conformers=0,
            exported_pdb_count=0,
            sidecar_count=0,
            failed_sidecar_count=0,
            artifacts={
                "sidecar_manifest": str(manifest_path),
                "target_manifest": str(target_manifest_path),
                "sidecar_failures": str(failures_path),
                "pdb_root": str(pdb_root),
                "sidecar_root": str(sidecar_root),
            },
            teacher_summary={"status": "skipped_no_ready_probe_outputs"},
        )
        summary_path = arrays_dir / "bioemu_probe_ucbshift_cnnls_teacher_summary.json"
        summary_path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return summary
    target_manifest, sequence_by_entity = _target_manifest_from_registry(
        data_root_path,
        selected_inputs["entity_uid"].astype(str).tolist(),
    )

    ucbshift_config = (
        UCBShiftGenerationConfig.from_json(ucbshift_config_path)
        if ucbshift_config_path
        else UCBShiftGenerationConfig()
    )
    if worker_count is not None:
        ucbshift_config.worker_count = int(worker_count)
    ucbshift_config.refresh_outputs = bool(refresh_outputs)
    runtime = resolve_ucbshift_runtime(
        config=ucbshift_config,
        data_root=data_root_path,
        repo_root=data_root_path.parent,
    )
    runtime = _prepare_bioemu_ucbshift_runtime(runtime, report_dir)
    env = _build_ucbshift_env(runtime)

    manifest_rows: list[dict[str, Any]] = []
    sidecar_failures: list[dict[str, Any]] = []
    requested_conformers = 0
    exported_pdb_count = 0
    sidecar_count = 0

    for row in selected_inputs.itertuples(index=False):
        entity_uid = str(getattr(row, "entity_uid"))
        entity_label = _safe_label(f"{getattr(row, 'rank', 0):02d}_{entity_uid}")
        output_path = Path(str(getattr(row, "output_dir")))
        if not output_path.is_absolute():
            output_path = run_dir_path / output_path
        sequence = str(getattr(row, "sequence_one_letter", "") or "")
        requested = int(getattr(row, "sample_count", 0) or 0)
        if max_conformers_per_entity is not None:
            requested = min(
                requested,
                max(int(max_conformers_per_entity), MIN_TEACHER_CONFORMERS_PER_ENTITY),
            )
        requested = max(requested, MIN_TEACHER_CONFORMERS_PER_ENTITY)
        requested_conformers += requested

        entity_pdb_dir = pdb_root / entity_label
        entity_sidecar_dir = sidecar_root / entity_label
        entity_pdb_dir.mkdir(parents=True, exist_ok=True)
        entity_sidecar_dir.mkdir(parents=True, exist_ok=True)

        expected_pdb_paths = [
            entity_pdb_dir / f"sample_{sample_index:05d}.pdb"
            for sample_index in range(requested)
        ]
        expected_sidecar_paths = [
            entity_sidecar_dir / f"sample_{sample_index:05d}.csv"
            for sample_index in range(requested)
        ]
        all_sidecars_ready = (
            not refresh_outputs
            and all(
                _ucbshift_sidecar_current_for_pdb(sidecar_path, pdb_path)
                for sidecar_path, pdb_path in zip(
                    expected_sidecar_paths,
                    expected_pdb_paths,
                    strict=False,
                )
            )
        )
        pdb_paths = (
            expected_pdb_paths
            if all_sidecars_ready
            else _ensure_probe_sample_pdbs(
                probe_output_dir=output_path,
                output_dir=entity_pdb_dir,
                sequence=sequence,
                requested_count=requested,
                skip_export=skip_pdb_export,
                allow_topology_fallback=allow_topology_fallback,
            )
        )
        exported_pdb_count += len(pdb_paths)

        pending_specs: list[tuple[str, Path, Path]] = []
        entity_manifest_rows: list[dict[str, Any]] = []
        for sample_index, pdb_path in enumerate(pdb_paths):
            candidate_id = f"{entity_uid}:bioemu_sample_{sample_index:05d}"
            sidecar_path = entity_sidecar_dir / f"sample_{sample_index:05d}.csv"
            sidecar_current = _ucbshift_sidecar_current_for_pdb(sidecar_path, pdb_path)
            if sidecar_current and not refresh_outputs:
                sidecar_count += 1
            else:
                if sidecar_path.exists() and not refresh_outputs:
                    _discard_stale_probe_sidecar(sidecar_path)
                pending_specs.append(("BioEmu", pdb_path, sidecar_path))
            entity_manifest_rows.append(
                {
                    "entity_uid": entity_uid,
                    "bmrb_id": str(getattr(row, "bmrb_id", "")),
                    "candidate_id": candidate_id,
                    "sample_index": sample_index,
                    "prior_weight": 1.0 / max(len(pdb_paths), 1),
                    "structure_path": str(pdb_path),
                    "chemical_shift_path": str(sidecar_path),
                    "chemical_shift_format": "ucbshift",
                    "probe_name": probe_name,
                }
            )

        if pending_specs:
            results = _run_candidate_sidecars(
                pending_specs=pending_specs,
                runtime=runtime,
                config=ucbshift_config,
                env=env,
                sequence_text=sequence_by_entity.get(entity_uid, ""),
                max_workers=min(max(int(ucbshift_config.worker_count), 1), len(pending_specs)),
            )
            for result in results:
                if result.status == "written":
                    sidecar_count += 1
                else:
                    sidecar_failures.append(
                        {
                            "entity_uid": entity_uid,
                            "pdb_path": str(result.pdb_path),
                            "chemical_shift_path": str(result.output_path),
                            "status": result.status,
                            "message": result.message,
                        }
                    )
        manifest_rows.extend(entity_manifest_rows)

    shift_manifest = pd.DataFrame(manifest_rows)
    failures = pd.DataFrame(sidecar_failures)
    manifest_path = arrays_dir / "bioemu_probe_ucbshift_sidecar_manifest.parquet"
    target_manifest_path = arrays_dir / "bioemu_probe_ucbshift_target_manifest.parquet"
    failures_path = arrays_dir / "bioemu_probe_ucbshift_sidecar_failures.parquet"
    raw_failures_path = arrays_dir / "bioemu_probe_ucbshift_sidecar_failures_raw.parquet"
    replacements_path = arrays_dir / "bioemu_ucbshift_sidecar_failure_replacements.parquet"

    replacement_artifacts: dict[str, str] = {}
    if auto_fill_missing_sidecars and not failures.empty:
        _write_table(failures, raw_failures_path)
        replacement_report = _fill_missing_probe_sidecars_from_neighbors(
            failures=failures,
            report_path=replacements_path,
        )
        replacement_artifacts["sidecar_raw_failures"] = str(raw_failures_path)
        replacement_artifacts["sidecar_failure_replacements"] = str(replacements_path)
        resolved_paths = set(
            replacement_report.loc[
                replacement_report["status"].isin(["filled", "already_present"]),
                "target_sidecar_path",
            ].astype(str)
        )
        if resolved_paths:
            sidecar_count += len(resolved_paths)
            failures = failures.loc[
                ~failures["chemical_shift_path"].astype(str).isin(resolved_paths)
            ].reset_index(drop=True)
    elif replacements_path.exists():
        replacement_artifacts["sidecar_failure_replacements"] = str(replacements_path)

    _write_table(shift_manifest, manifest_path)
    _write_table(target_manifest, target_manifest_path)
    _write_table(failures, failures_path)

    if sidecar_only:
        teacher = None
        status = "ok" if sidecar_count > 0 else "failed"
        message = (
            "BioEmu probe UCBShift sidecars generated; teacher fitting skipped."
            if status == "ok"
            else "BioEmu probe UCBShift sidecar warmstart did not produce sidecars."
        )
        teacher_summary: dict[str, Any] = {
            "status": "skipped_sidecar_only",
            "message": "CS-reweighting teacher fit was intentionally skipped.",
        }
        teacher_artifacts: dict[str, str] = {}
    else:
        teacher = run_bioemu_ucbshift_cnnls_teacher(
            data_root=data_root_path,
            run_dir=run_dir_path,
            shift_manifest_path=manifest_path,
            output_dir=report_dir,
            target_manifest_path=target_manifest_path,
            method=method,
            lambda_reg=lambda_reg,
            default_cs_sigma=default_cs_sigma,
            max_entities=None,
            require_all_sidecars=require_all_sidecars,
        )
        status = "ok" if teacher.status == "ok" else "failed"
        message = (
            "BioEmu probe UCBShift/CS-reweighting teacher generated."
            if status == "ok"
            else "BioEmu probe teacher generation did not fit any entity."
        )
        teacher_summary = teacher.as_dict()
        teacher_artifacts = teacher.artifacts
    artifacts = {
        "sidecar_manifest": str(manifest_path),
        "target_manifest": str(target_manifest_path),
        "sidecar_failures": str(failures_path),
        "pdb_root": str(pdb_root),
        "sidecar_root": str(sidecar_root),
        **replacement_artifacts,
        **teacher_artifacts,
    }
    summary = BioEmuProbeTeacherSummary(
        status=status,
        message=message,
        probe_input_rows=int(len(probe_inputs)),
        selected_entity_count=int(selected_inputs["entity_uid"].nunique()),
        shard_count=int(shard_count),
        shard_index=int(shard_index),
        sidecar_only=bool(sidecar_only),
        ready_probe_outputs_only=bool(ready_probe_outputs_only),
        skipped_not_ready_entity_count=int(skipped_not_ready_entity_count),
        requested_conformers=int(requested_conformers),
        exported_pdb_count=int(exported_pdb_count),
        sidecar_count=int(sidecar_count),
        failed_sidecar_count=int(len(failures)),
        artifacts=artifacts,
        teacher_summary=teacher_summary,
    )
    summary_path = arrays_dir / "bioemu_probe_ucbshift_cnnls_teacher_summary.json"
    summary_path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return summary


def _resolve_teacher_cache_root(
    value: str | Path | None,
    *,
    default: Path,
    base_dir: Path,
) -> Path:
    if value is None or str(value).strip() == "":
        return default
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _select_probe_inputs(
    frame: pd.DataFrame,
    *,
    probe_name: str,
    max_entities: int | None,
) -> pd.DataFrame:
    selected = frame.copy()
    if "ready" in selected.columns:
        selected = selected.loc[selected["ready"].astype(bool)].copy()
    if "probe_name" in selected.columns:
        selected = selected.loc[selected["probe_name"].astype(str).eq(str(probe_name))].copy()
    if "rank" in selected.columns:
        selected = selected.sort_values("rank", kind="stable")
    if max_entities is not None:
        keep = list(dict.fromkeys(selected["entity_uid"].astype(str)))[: int(max_entities)]
        selected = selected.loc[selected["entity_uid"].astype(str).isin(keep)].copy()
    if selected.empty:
        raise ValueError(f"No ready BioEmu probe inputs found for probe_name={probe_name!r}.")
    return selected.reset_index(drop=True)


def _filter_ready_probe_outputs(
    frame: pd.DataFrame,
    *,
    run_dir: Path,
    allow_topology_fallback: bool,
) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        output_path = Path(str(getattr(row, "output_dir")))
        if not output_path.is_absolute():
            output_path = run_dir / output_path
        if (output_path / "synthetic_context_samples.npz").exists() or (
            allow_topology_fallback and (output_path / "topology.pdb").exists()
        ):
            rows.append(row._asdict())
    if not rows:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def _apply_entity_shard(
    frame: pd.DataFrame,
    *,
    shard_count: int,
    shard_index: int,
) -> pd.DataFrame:
    """Select a deterministic entity shard without splitting one entity."""

    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")
    if shard_count == 1:
        return frame.reset_index(drop=True)
    entity_order = list(dict.fromkeys(frame["entity_uid"].astype(str)))
    selected_entities = set(entity_order[shard_index::shard_count])
    selected = frame.loc[frame["entity_uid"].astype(str).isin(selected_entities)].copy()
    if selected.empty:
        raise ValueError(
            f"No BioEmu probe entries selected for shard {shard_index}/{shard_count}."
        )
    return selected.reset_index(drop=True)


def _target_manifest_from_registry(
    data_root: Path,
    entity_uids: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    registry = IntegratedDataRegistry.from_data_root(data_root)
    teacher_examples = registry.load_teacher_examples()
    entity_set = {str(entity_uid) for entity_uid in entity_uids}
    rows: list[dict[str, str]] = []
    sequence_by_entity: dict[str, str] = {}
    for row in teacher_examples.itertuples(index=False):
        entity_uid = str(getattr(row, "entity_uid"))
        if entity_uid not in entity_set:
            continue
        target_bundle_path = getattr(row, "target_bundle_path", None)
        if target_bundle_path is None or pd.isna(target_bundle_path):
            continue
        resolved_target = resolve_existing_path(
            target_bundle_path,
            data_root=data_root,
            repo_root=data_root.parent,
        )
        if resolved_target is None:
            continue
        rows.append(
            {
                "entity_uid": entity_uid,
                "target_bundle_path": str(resolved_target),
            }
        )
        sequence_by_entity[entity_uid] = str(getattr(row, "sequence", "") or "")
    if not rows:
        raise ValueError("No target bundles were found for the selected BioEmu probe entries.")
    return pd.DataFrame(rows).drop_duplicates(), sequence_by_entity


def _prepare_bioemu_ucbshift_runtime(
    runtime: UCBShiftRuntime,
    report_dir: Path,
) -> UCBShiftRuntime:
    """Create a job-local UCBShift runtime that tolerates BioEmu probe PDBs.

    Official BioEmu samples are backbone-heavy-atom conformers. UCBShift's
    upstream SPARTA+ feature builder was written for conventional PDB files and
    otherwise drops every residue when DSSP/PPBuilder cannot form a clean chain.
    We patch only this copied runtime so the official checkout stays untouched.
    """

    patched_root = report_dir / "ucbshift_dssp_tolerant_runtime"
    patched_root.mkdir(parents=True, exist_ok=True)
    for source in runtime.ucbshift_root.iterdir():
        target = patched_root / source.name
        if source.is_dir():
            if source.name not in {".git", "__pycache__"}:
                shutil.copytree(source, target, dirs_exist_ok=True)
            continue
        if source.suffix != ".pyc":
            shutil.copy2(source, target)

    patched_models_dir = patched_root / runtime.models_dir.name
    if runtime.models_dir.exists() and runtime.models_dir.resolve() != patched_models_dir.resolve():
        shutil.copytree(runtime.models_dir, patched_models_dir, dirs_exist_ok=True)
    if not patched_models_dir.exists():
        patched_models_dir = runtime.models_dir

    patch_warnings: list[str] = []
    spartap_path = patched_root / "spartap_features.py"
    if spartap_path.exists():
        _patch_spartap_features_for_bioemu(spartap_path)
    else:
        patch_warnings.append(f"Missing copied spartap_features.py: {spartap_path}")

    model_file_count = (
        len(list(patched_models_dir.glob("*.sav")))
        if patched_models_dir.exists()
        else runtime.model_file_count
    )
    return replace(
        runtime,
        ucbshift_root=patched_root,
        ucbshift_script=patched_root / "CSpred.py",
        models_dir=patched_models_dir,
        model_file_count=model_file_count,
        warnings=[
            *runtime.warnings,
            *patch_warnings,
            f"BioEmu DSSP-tolerant UCBShift runtime: {patched_root}",
        ],
    )


def _patch_spartap_features_for_bioemu(spartap_path: Path) -> None:
    """Patch a copied SPARTA+ feature reader to keep BioEmu residues usable."""

    text = spartap_path.read_text()
    marker = "# AtypEmu BioEmu probe patch: DSSP/PPBuilder tolerant."
    if marker not in text:
        text = marker + "\n" + text

    original_poly_line = (
        "                poly_residues = [res for poly in polys for res in poly "
        "if res.id[2] == ' ']\n"
    )
    patched_poly_block = (
        "                poly_residues = [res for poly in polys for res in poly "
        "if res.id[2] == ' ']\n"
        "                all_standard_residues = [\n"
        "                    res for res in chain.get_unpacked_list()\n"
        "                    if res.id[0] == ' ' and res.id[2] == ' '\n"
        "                    and self._fix_res_name(res.resname) in AAlet3\n"
        "                ]\n"
        "                if len(poly_residues) < len(all_standard_residues):\n"
        "                    poly_residues = all_standard_residues\n"
        "                    dihedrals = [[0, 0, 0, 0] for _ in poly_residues]\n"
    )
    if patched_poly_block not in text and original_poly_line in text:
        text = text.replace(original_poly_line, patched_poly_block)

    # UCBShift writes zero feature blocks in these branches, but the upstream
    # code then continues and drops the residue. For BioEmu teacher generation
    # we keep the row so the teacher fit can still use the backbone CS matrix.
    # The upstream sidechain hbond fallback width is 185 while the SideChain
    # schema has 165 columns; that mismatch is harmless only because upstream
    # skips the row. Use schema-derived widths before removing the continue.
    text = text.replace(
        "                        row_data += 40*[0]\n"
        "                        continue",
        "                        row_data += [0] * len(efield_column_names)",
    )
    text = text.replace(
        "                        row_data += 185*[0]\n"
        "                        continue",
        "                        row_data += [0] * len(hbond_schain_column_names)",
    )
    text = text.replace(
        "                        row_data += 16*[0]\n"
        "                        continue",
        "                        row_data += 16*[0]",
    )
    original_append = "                    data.append(row_data)\n"
    patched_append = (
        "                    if len(row_data) != len(col_names):\n"
        "                        raise ValueError(\n"
        "                            'BioEmu UCBShift feature schema mismatch: '\n"
        "                            + str(len(row_data)) + ' values for '\n"
        "                            + str(len(col_names)) + ' columns at '\n"
        "                            + str((chain.id, res.id))\n"
        "                        )\n"
        "                    data.append(row_data)\n"
    )
    if patched_append not in text and original_append in text:
        text = text.replace(original_append, patched_append)
    text = text.replace(".  Skipping this residue", ".  Using zero fallback")
    spartap_path.write_text(text)


def _ensure_probe_sample_pdbs(
    *,
    probe_output_dir: Path,
    output_dir: Path,
    sequence: str,
    requested_count: int,
    skip_export: bool,
    allow_topology_fallback: bool,
) -> list[Path]:
    existing = sorted(output_dir.glob("sample_*.pdb"))
    if skip_export and len(existing) >= requested_count:
        return _repair_probe_pdbs_for_ucbshift(existing[:requested_count], sequence)
    npz_path = probe_output_dir / "synthetic_context_samples.npz"
    if npz_path.exists():
        paths = _export_npz_samples_to_pdbs(
            npz_path=npz_path,
            output_dir=output_dir,
            sequence=sequence,
            requested_count=requested_count,
        )
        return _repair_probe_pdbs_for_ucbshift(paths, sequence)
    if allow_topology_fallback:
        topology = probe_output_dir / "topology.pdb"
        if topology.exists():
            fallback_path = output_dir / "sample_00000.pdb"
            if not fallback_path.exists():
                fallback_path.write_text(topology.read_text())
            return _repair_probe_pdbs_for_ucbshift([fallback_path], sequence)
    raise FileNotFoundError(
        f"Missing BioEmu sample NPZ for probe output {probe_output_dir}. "
        "Run the detached BioEmu conformer probe first."
    )


def _export_npz_samples_to_pdbs(
    *,
    npz_path: Path,
    output_dir: Path,
    sequence: str,
    requested_count: int,
) -> list[Path]:
    try:
        import torch
        from bioemu.convert_chemgraph import save_pdb_and_xtc
    except Exception as exc:  # pragma: no cover - depends on external BioEmu.
        raise RuntimeError(
            "Exporting BioEmu sampled conformers to PDB requires the official "
            "bioemu package in PYTHONPATH."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path) as data:
        pos = np.asarray(data["pos"], dtype=np.float32)
        orientations = np.asarray(data["node_orientations"], dtype=np.float32)
        npz_sequence = str(data["sequence"]) if "sequence" in data.files else ""
    if pos.ndim != 3 or orientations.ndim != 4:
        raise ValueError(f"Unexpected BioEmu sample shapes in {npz_path}.")
    sample_count = min(int(requested_count), int(pos.shape[0]), int(orientations.shape[0]))
    if sample_count <= 0:
        raise ValueError(f"No BioEmu samples are available in {npz_path}.")
    sample_sequence = sequence or npz_sequence
    if not sample_sequence:
        raise ValueError(f"Missing sequence for BioEmu sample export: {npz_path}")
    source_indices, replacement_rows = _bioemu_export_source_indices(pos[:sample_count])
    _write_bioemu_export_replacement_report(output_dir, replacement_rows)
    paths: list[Path] = []
    for sample_index in range(sample_count):
        pdb_path = output_dir / f"sample_{sample_index:05d}.pdb"
        xtc_path = output_dir / f"sample_{sample_index:05d}.xtc"
        source_index = int(source_indices[sample_index])
        force_reexport = source_index != sample_index
        if pdb_path.exists() and (
            force_reexport or _probe_pdb_needs_safe_reexport(pdb_path)
        ):
            pdb_path.unlink()
            _probe_pdb_repair_marker(pdb_path).unlink(missing_ok=True)
            xtc_path.unlink(missing_ok=True)
        if not pdb_path.exists():
            sample_pos = np.asarray(
                pos[source_index : source_index + 1],
                dtype=np.float32,
            ).copy()
            sample_pos -= sample_pos.mean(axis=1, keepdims=True)
            save_pdb_and_xtc(
                pos_nm=torch.as_tensor(sample_pos),
                node_orientations=torch.as_tensor(
                    orientations[source_index : source_index + 1]
                ),
                topology_path=pdb_path,
                xtc_path=xtc_path,
                sequence=sample_sequence,
                filter_samples=False,
            )
            if xtc_path.exists():
                xtc_path.unlink()
        paths.append(pdb_path)
    return paths


def _bioemu_export_source_indices(pos: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Map unstable BioEmu conformers to the nearest stable sample for PDB export."""

    sample_count = int(pos.shape[0])
    finite = np.isfinite(pos).all(axis=(1, 2))
    spans = np.ptp(pos, axis=1)
    max_span = np.max(spans, axis=1)
    max_span_nm = _bioemu_export_max_span_nm()
    valid = finite & (max_span <= max_span_nm)
    if not np.any(valid):
        raise ValueError(
            "No BioEmu conformer has finite coordinates within the export span "
            f"threshold ({max_span_nm:g} nm)."
        )
    valid_indices = np.flatnonzero(valid)
    source_indices = np.arange(sample_count, dtype=np.int64)
    replacement_rows: list[dict[str, Any]] = []
    for sample_index in np.flatnonzero(~valid):
        nearest = int(valid_indices[np.argmin(np.abs(valid_indices - sample_index))])
        source_indices[int(sample_index)] = nearest
        reason = "nonfinite_coordinates" if not bool(finite[int(sample_index)]) else "span_exceeded"
        replacement_rows.append(
            {
                "sample_index": int(sample_index),
                "replacement_sample_index": nearest,
                "reason": reason,
                "max_span_nm": float(max_span[int(sample_index)]),
                "threshold_nm": float(max_span_nm),
            }
        )
    return source_indices, replacement_rows


def _bioemu_export_max_span_nm() -> float:
    raw_value = os.environ.get("BIOEMU_UCBSHIFT_MAX_EXPORT_SPAN_NM", "").strip()
    if raw_value:
        try:
            value = float(raw_value)
            if np.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
    return _BIOEMU_EXPORT_MAX_SPAN_NM


def _write_bioemu_export_replacement_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    path = output_dir / "bioemu_export_replacements.tsv"
    if not rows:
        path.unlink(missing_ok=True)
        return
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _probe_pdb_needs_safe_reexport(path: Path) -> bool:
    atom_count = 0
    max_abs_coord = 0.0
    try:
        for line in path.read_text().splitlines():
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            parsed = _parse_pdb_atom_line(line)
            if parsed is None:
                return False
            xyz = np.asarray(parsed["xyz"], dtype=float)
            max_abs_coord = max(max_abs_coord, float(np.max(np.abs(xyz))))
            atom_count += 1
    except OSError:
        return True
    if atom_count <= 0:
        return True
    return max_abs_coord > _bioemu_export_max_abs_coord_angstrom()


def _bioemu_export_max_abs_coord_angstrom() -> float:
    raw_value = os.environ.get("BIOEMU_UCBSHIFT_MAX_EXPORT_ABS_COORD_A", "").strip()
    if raw_value:
        try:
            value = float(raw_value)
            if np.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
    return _BIOEMU_EXPORT_MAX_ABS_COORD_ANGSTROM


def _ucbshift_sidecar_current_for_pdb(sidecar_path: Path, pdb_path: Path) -> bool:
    if not _ucbshift_sidecar_usable(sidecar_path) or not pdb_path.exists():
        return False
    if _probe_pdb_needs_safe_reexport(pdb_path):
        return False
    try:
        reference_mtime = pdb_path.stat().st_mtime
        marker = _probe_pdb_repair_marker(pdb_path)
        if marker.exists():
            reference_mtime = max(reference_mtime, marker.stat().st_mtime)
        return sidecar_path.stat().st_mtime >= reference_mtime
    except OSError:
        return False


def _discard_stale_probe_sidecar(sidecar_path: Path) -> None:
    """Remove an outdated sidecar so the generic runner cannot reuse it."""

    try:
        sidecar_path.unlink()
    except FileNotFoundError:
        pass


def _fill_missing_probe_sidecars_from_neighbors(
    *,
    failures: pd.DataFrame,
    report_path: Path,
) -> pd.DataFrame:
    """Conservatively backfill failed BioEmu sidecars from same-entity neighbors."""

    rows: list[dict[str, object]] = []
    for row in failures.to_dict(orient="records"):
        target = Path(str(row.get("chemical_shift_path", "")))
        target_index = _sample_sidecar_index(target)
        if target_index is None:
            rows.append(
                {
                    "entity_uid": row.get("entity_uid", ""),
                    "target_sidecar_path": str(target),
                    "target_sample_index": "",
                    "source_sidecar_path": "",
                    "source_sample_index": "",
                    "status": "skipped_unrecognized_name",
                    "message": "Could not parse sample index from target sidecar.",
                }
            )
            continue
        if _ucbshift_sidecar_usable(target):
            rows.append(
                {
                    "entity_uid": row.get("entity_uid", ""),
                    "target_sidecar_path": str(target),
                    "target_sample_index": target_index,
                    "source_sidecar_path": "",
                    "source_sample_index": "",
                    "status": "already_present",
                    "message": "",
                }
            )
            continue
        nearest = _nearest_usable_probe_sidecar(target)
        if nearest is None:
            rows.append(
                {
                    "entity_uid": row.get("entity_uid", ""),
                    "target_sidecar_path": str(target),
                    "target_sample_index": target_index,
                    "source_sidecar_path": "",
                    "source_sample_index": "",
                    "status": "failed_no_neighbor",
                    "message": "No usable same-entity sidecar was available.",
                }
            )
            continue
        source_path, source_index = nearest
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.name}.neighbor_tmp")
        shutil.copyfile(source_path, tmp_path)
        tmp_path.replace(target)
        os.utime(target, None)
        rows.append(
            {
                "entity_uid": row.get("entity_uid", ""),
                "target_sidecar_path": str(target),
                "target_sample_index": target_index,
                "source_sidecar_path": str(source_path),
                "source_sample_index": source_index,
                "status": "filled",
                "message": "",
            }
        )
    report = pd.DataFrame(rows)
    _write_table(report, report_path)
    return report


def _sample_sidecar_index(path: Path) -> int | None:
    match = _SAMPLE_SIDECAR_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _nearest_usable_probe_sidecar(target: Path) -> tuple[Path, int] | None:
    target_index = _sample_sidecar_index(target)
    if target_index is None:
        return None
    candidates: list[tuple[int, int, Path]] = []
    for candidate in target.parent.glob("sample_*.csv"):
        if candidate == target or not _ucbshift_sidecar_usable(candidate):
            continue
        candidate_index = _sample_sidecar_index(candidate)
        if candidate_index is None:
            continue
        candidates.append((abs(candidate_index - target_index), candidate_index, candidate))
    if not candidates:
        return None
    _, source_index, source_path = min(candidates, key=lambda item: (item[0], item[1]))
    return source_path, source_index


def _repair_probe_pdbs_for_ucbshift(paths: list[Path], sequence: str) -> list[Path]:
    """Normalize BioEmu-exported PDBs for UCBShift sidecar generation."""

    for path in paths:
        if _probe_pdb_repair_marker_current(path):
            continue
        _repair_probe_pdb_for_ucbshift(path, sequence)
        _write_probe_pdb_repair_marker(path)
    return paths


def _probe_pdb_repair_marker(path: Path) -> Path:
    return path.with_name(f"{path.name}.ucbshift_repaired")


def _probe_pdb_repair_marker_current(path: Path) -> bool:
    marker = _probe_pdb_repair_marker(path)
    if not path.exists() or not marker.exists():
        return False
    try:
        return (
            marker.read_text().strip() == _UCBSHIFT_PDB_REPAIR_MARKER_VERSION
            and marker.stat().st_mtime >= path.stat().st_mtime
        )
    except OSError:
        return False


def _write_probe_pdb_repair_marker(path: Path) -> None:
    _probe_pdb_repair_marker(path).write_text(_UCBSHIFT_PDB_REPAIR_MARKER_VERSION)


def _repair_probe_pdb_for_ucbshift(path: Path, sequence: str) -> None:
    """Rewrite one BioEmu PDB with 1-based residues and pseudo H/HA atoms."""

    atom_rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        parsed = _parse_pdb_atom_line(line)
        if parsed is not None:
            atom_rows.append(parsed)
    if not atom_rows:
        raise ValueError(f"No parseable ATOM/HETATM records in exported PDB: {path}")

    residue_numbers = [int(row["resseq"]) for row in atom_rows]
    offset = 1 - min(residue_numbers) if min(residue_numbers) <= 0 else 0
    residues: dict[tuple[str, int, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str]] = []
    sequence_resnames = [_ONE_TO_THREE.get(letter.upper()) for letter in sequence.strip()]
    for row in atom_rows:
        row["chain_id"] = str(row.get("chain_id") or "A")[:1] or "A"
        row["resseq"] = int(row["resseq"]) + offset
        key = (str(row["chain_id"]), int(row["resseq"]), str(row.get("icode") or " "))
        if key not in residues:
            residue_index = len(order)
            resname = str(row["resname"]).upper()
            if residue_index < len(sequence_resnames) and sequence_resnames[residue_index]:
                resname = str(sequence_resnames[residue_index])
            residues[key] = {"resname": resname, "atoms": {}, "atom_order": []}
            order.append(key)
        atom_name = str(row["atom_name"]).strip()
        residues[key]["atoms"][atom_name] = row
        residues[key]["atom_order"].append(atom_name)

    for index, key in enumerate(order):
        previous_atoms = residues[order[index - 1]]["atoms"] if index > 0 else {}
        _add_pseudo_backbone_protons(residues[key], previous_atoms)

    output_lines: list[str] = []
    serial = 1
    for chain_id, resseq, icode in order:
        residue = residues[(chain_id, resseq, icode)]
        resname = str(residue["resname"])
        atom_names = _ucbshift_atom_order(list(residue["atom_order"]), residue["atoms"])
        for atom_name in atom_names:
            atom = residue["atoms"][atom_name]
            output_lines.append(
                _format_pdb_atom_line(
                    serial=serial,
                    atom_name=atom_name,
                    resname=resname,
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    xyz=np.asarray(atom["xyz"], dtype=float),
                    occupancy=float(atom.get("occupancy", 1.0)),
                    bfactor=float(atom.get("bfactor", 0.0)),
                    element=str(atom.get("element") or _infer_element(atom_name)),
                )
            )
            serial += 1
    output_lines.append("TER\n")
    output_lines.append("END\n")
    path.write_text("".join(output_lines))


def _parse_pdb_atom_line(line: str) -> dict[str, Any] | None:
    try:
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain_id = line[21:22].strip() or "A"
        resseq = int(line[22:26])
        icode = line[26:27] or " "
        xyz = np.array(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ],
            dtype=float,
        )
        occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
        bfactor = float(line[60:66]) if line[60:66].strip() else 0.0
        element = line[76:78].strip() or _infer_element(atom_name)
    except Exception:
        return None
    if not atom_name or not resname:
        return None
    return {
        "atom_name": atom_name,
        "resname": resname,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "xyz": xyz,
        "occupancy": occupancy,
        "bfactor": bfactor,
        "element": element,
    }


def _add_pseudo_backbone_protons(
    residue: dict[str, Any],
    previous_atoms: dict[str, dict[str, Any]],
) -> None:
    atoms: dict[str, dict[str, Any]] = residue["atoms"]
    resname = str(residue["resname"])
    if resname != "PRO" and "H" not in atoms and "N" in atoms:
        n_xyz = np.asarray(atoms["N"]["xyz"], dtype=float)
        if "C" in previous_atoms:
            direction = n_xyz - np.asarray(previous_atoms["C"]["xyz"], dtype=float)
        elif "CA" in atoms:
            direction = n_xyz - np.asarray(atoms["CA"]["xyz"], dtype=float)
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
        atoms["H"] = _new_pseudo_atom("H", n_xyz + _unit_vector(direction) * 1.0)
        residue["atom_order"].append("H")

    if "HA" not in atoms and "CA" in atoms:
        ca_xyz = np.asarray(atoms["CA"]["xyz"], dtype=float)
        direction = np.array([0.0, 1.0, 0.0], dtype=float)
        if "N" in atoms:
            direction = direction + ca_xyz - np.asarray(atoms["N"]["xyz"], dtype=float)
        if "C" in atoms:
            direction = direction + ca_xyz - np.asarray(atoms["C"]["xyz"], dtype=float)
        atoms["HA"] = _new_pseudo_atom("HA", ca_xyz + _unit_vector(direction) * 1.09)
        residue["atom_order"].append("HA")


def _new_pseudo_atom(atom_name: str, xyz: np.ndarray) -> dict[str, Any]:
    return {
        "atom_name": atom_name,
        "xyz": np.asarray(xyz, dtype=float),
        "occupancy": 1.0,
        "bfactor": 0.0,
        "element": "H",
    }


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _ucbshift_atom_order(atom_order: list[str], atoms: dict[str, dict[str, Any]]) -> list[str]:
    preferred = ["N", "H", "CA", "HA", "C", "O", "CB"]
    ordered: list[str] = []
    for atom_name in preferred + atom_order + sorted(atoms):
        if atom_name in atoms and atom_name not in ordered:
            ordered.append(atom_name)
    return ordered


def _format_pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    icode: str,
    xyz: np.ndarray,
    occupancy: float,
    bfactor: float,
    element: str,
) -> str:
    atom_field = f"{atom_name:>4}"[:4]
    return (
        f"ATOM  {serial:5d} {atom_field} {resname:>3} {chain_id[:1]}"
        f"{resseq:4d}{(icode or ' ')[:1]}   "
        f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {_infer_element(element):>2}\n"
    )


def _infer_element(atom_name: str) -> str:
    stripped = "".join(ch for ch in str(atom_name).strip() if ch.isalpha())
    return (stripped[:1] or "C").upper()


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    frame.to_csv(path.with_suffix(".tsv"), sep="\t", index=False)


def _safe_label(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "entity"
