#!/usr/bin/env python3
"""Run the bounded, target-unread OpenMM protonation smoke in an isolated SIF."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CANDIDATE_ID = "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_recovery_v3"
PLAN_CONTRACT = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_"
    "recovery_plan_v3"
)
COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_"
    "recovery_source_commitment_v3"
)
PLAN_PATH = Path("/work/plan.json")
COMMITMENT_PATH = Path("/work/source_commitment.json")
SOURCE_PATH = Path("/work/generator.py")
CHECKER_PATH = Path("/work/checker.py")
LAUNCHER_PATH = Path("/work/launcher.py")
FAILURE_EVIDENCE_PATH = Path("/work/failure_evidence.json")
OUTPUT_PARENT = Path("/out")
OUTPUT_NAME = "condition_uncertainty_protonation_smoke_recovery_v3"
STANDARD_RESIDUES = frozenset(
    (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    )
)
CLOSED_FIELDS = frozenset(
    (
        "authorization_consumed",
        "outer_or_formal_metrics_opened",
        "science_executed",
        "source_construction_executed",
        "source_scores_read",
        "target_atom_identities_read",
        "target_values_read",
    )
)


class SmokeError(ValueError):
    """Fail-closed smoke error."""


def _json_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise SmokeError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"), object_pairs_hook=_json_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SmokeError) as error:
        raise SmokeError("invalid %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise SmokeError("%s must be a JSON object" % label)
    return value


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        raise SmokeError(
            "%s schema drifted: %s" % (label, sorted(set(value) ^ expected))
        )


def _closed(value: Mapping[str, Any], label: str) -> None:
    _schema(value, CLOSED_FIELDS, label)
    if any(value[field] is not False for field in CLOSED_FIELDS):
        raise SmokeError("%s opened a forbidden capability" % label)


def _regular_file(path: Path, expected: Path, label: str) -> None:
    if path != expected or path.is_symlink() or not path.is_file():
        raise SmokeError("%s path is not the exact regular file" % label)


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _state_file(base: Path, extension: str) -> Path:
    return base.parent / (base.name + extension)


def _seed(
    candidate_id: str, entity_uid: str, parent_support_id: str, branch_id: str
) -> int:
    raw = "\0".join((candidate_id, entity_uid, parent_support_id, branch_id)).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _support_id(
    entity: Mapping[str, Any], branch: Mapping[str, Any], runtime_sha: str
) -> str:
    raw = "\0".join(
        (
            CANDIDATE_ID,
            entity["entity_uid"],
            entity["parent_support_id"],
            branch["condition_branch_id"],
            entity["parent_pdb"]["sha256"],
            runtime_sha,
        )
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    _schema(
        plan,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "entities",
            "execution",
            "failed_predecessor",
            "interpretation_limits",
            "output_contract",
            "parent_policy",
            "source_commitment",
            "state",
            "unresolved_blockers",
        },
        "smoke plan",
    )
    if (
        plan["artifact_kind"]
        != "hold_only_target_unread_bounded_protonation_smoke_recovery_plan_not_support_not_authorization"
        or plan["candidate_id"] != CANDIDATE_ID
        or plan["contract"] != PLAN_CONTRACT
        or plan["state"] != "HOLD_SMOKE_RECOVERY_V3_SOURCE_FROZEN_UNRUN"
    ):
        raise SmokeError("smoke plan identity/state drifted")
    _closed(plan["closed_capabilities"], "smoke plan capabilities")
    execution = plan["execution"]
    _schema(
        execution,
        {
            "fresh_python_process_per_entity_branch_repeat",
            "isolated_mounts",
            "pythonhashseed_required",
            "repeat_count",
            "runtime",
            "seed_derivation",
        },
        "execution",
    )
    if (
        execution["fresh_python_process_per_entity_branch_repeat"] is not True
        or execution["pythonhashseed_required"] is not True
        or execution["repeat_count"] != 2
    ):
        raise SmokeError("smoke process/repeat policy drifted")
    mounts = execution["isolated_mounts"]
    _schema(
        mounts,
        {"network", "read_only", "target_or_score_storage_mounted", "writable"},
        "mounts",
    )
    if mounts != {
        "network": "none",
        "read_only": [
            "/work/generator.py",
            "/work/checker.py",
            "/work/launcher.py",
            "/work/failure_evidence.json",
            "/work/plan.json",
            "/work/source_commitment.json",
            "/inputs/bmr10109_BioEmu_1.pdb",
            "/inputs/bmr4333_BioEmu_1.pdb",
        ],
        "target_or_score_storage_mounted": False,
        "writable": ["/out"],
    }:
        raise SmokeError("smoke isolation policy drifted")
    if execution["seed_derivation"] != (
        "first 64 bits of SHA256(candidate_id NUL entity_uid NUL parent_support_id "
        "NUL condition_branch_id)"
    ):
        raise SmokeError("seed derivation drifted")
    runtime = execution["runtime"]
    _schema(
        runtime,
        {
            "force_field_relative_path",
            "force_field_sha256",
            "hydrogen_definitions_relative_path",
            "hydrogen_definitions_sha256",
            "modeller_source_relative_path",
            "modeller_source_sha256",
            "openmm_exact_version",
            "openmm_version_accessor",
            "platform",
            "sif_host_path",
            "sif_sha256",
        },
        "runtime",
    )
    if runtime != {
        "force_field_relative_path": "amber14/protein.ff14SB.xml",
        "force_field_sha256": "d9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8",
        "hydrogen_definitions_relative_path": "openmm/app/data/hydrogens.xml",
        "hydrogen_definitions_sha256": "413096cd3005ca5a638180e9cf623a8f6d574c81acf0e2c9d92b2bd26bb7658d",
        "modeller_source_relative_path": "openmm/app/modeller.py",
        "modeller_source_sha256": "f61e61f1419fcc3c24e7096ab96e10f87f70951085a83941d6040390e8819ca3",
        "openmm_exact_version": "8.6.0.dev-c6173db",
        "openmm_version_accessor": "openmm.version.full_version",
        "platform": "Reference",
        "sif_host_path": "/home/yang07/.cache/atypemu_openmm86_runtime/openmm86_protonation_8.6.0.sif",
        "sif_sha256": "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5",
    }:
        raise SmokeError("runtime identity drifted")
    entities = plan["entities"]
    if not isinstance(entities, list) or len(entities) != 2:
        raise SmokeError("smoke entity roster drifted")
    expected = {
        "bmrb:10109:entity:1": {
            "bmrb_id": "bmr10109",
            "branches": [
                {
                    "branch_id": "observed_pH_6.0",
                    "condition_branch_id": "observed:6.0",
                    "pH": "6.0",
                }
            ],
            "condition_state": "observed",
            "entity_uid": "bmrb:10109:entity:1",
            "parent_pdb": {
                "isolated_path": "/inputs/bmr10109_BioEmu_1.pdb",
                "repository_path": "data/k32_complete_coordinate_supports_v4/bmr10109/bmr10109_BioEmu_1.pdb",
                "sha256": "c14c2c3cd80b783d8addaf219ef2e6d13c059e4e7860f2e254488f0961ea066f",
            },
            "parent_support_id": "1",
        },
        "bmrb:4333:entity:1": {
            "bmrb_id": "bmr4333",
            "branches": [
                {
                    "branch_id": "regime_0_pH_2.2",
                    "condition_branch_id": "regime:0:2.2",
                    "pH": "2.2",
                },
                {
                    "branch_id": "regime_1_pH_5.45",
                    "condition_branch_id": "regime:1:5.45",
                    "pH": "5.45",
                },
                {
                    "branch_id": "regime_2_pH_7.5",
                    "condition_branch_id": "regime:2:7.5",
                    "pH": "7.5",
                },
                {
                    "branch_id": "regime_3_pH_9.25",
                    "condition_branch_id": "regime:3:9.25",
                    "pH": "9.25",
                },
                {
                    "branch_id": "regime_4_pH_12.0",
                    "condition_branch_id": "regime:4:12.0",
                    "pH": "12.0",
                },
            ],
            "condition_state": "state_missing",
            "entity_uid": "bmrb:4333:entity:1",
            "parent_pdb": {
                "isolated_path": "/inputs/bmr4333_BioEmu_1.pdb",
                "repository_path": "data/k32_complete_coordinate_supports_v4/bmr4333/bmr4333_BioEmu_1.pdb",
                "sha256": "0d8ac3f45e6e82e153f86bf91c164a016d4b1280ad229820ba2f9a10f6695146",
            },
            "parent_support_id": "1",
        },
    }
    result = []  # type: List[Dict[str, Any]]
    for entity in entities:
        _schema(
            entity,
            {
                "bmrb_id",
                "branches",
                "condition_state",
                "entity_uid",
                "parent_pdb",
                "parent_support_id",
            },
            "entity",
        )
        expected_entity = expected.get(entity["entity_uid"])
        if expected_entity is None or entity != expected_entity:
            raise SmokeError("smoke entity identity drifted")
        _schema(
            entity["parent_pdb"],
            {"isolated_path", "repository_path", "sha256"},
            "parent PDB",
        )
        branches = entity["branches"]
        if not isinstance(branches, list):
            raise SmokeError("smoke branch roster drifted")
        for branch in branches:
            _schema(branch, {"branch_id", "condition_branch_id", "pH"}, "branch")
            try:
                ph = float(branch["pH"])
            except (TypeError, ValueError) as error:
                raise SmokeError("invalid branch pH") from error
            if not math.isfinite(ph) or not 0.0 <= ph <= 14.0:
                raise SmokeError("out-of-range branch pH")
        result.append(dict(entity))
    if sum(len(entity["branches"]) for entity in result) != 6:
        raise SmokeError("smoke must contain exactly six states")
    if plan["interpretation_limits"] != {
        "all_atom_feasibility_established": False,
        "all_support_replay_completed": False,
        "bounded_smoke_only": True,
        "condition_metadata_recovered_for_unresolved_entity": False,
        "sampling_sufficiency_established": False,
        "science_or_score_evidence": False,
    }:
        raise SmokeError("interpretation limits drifted")
    if plan["parent_policy"] != {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_protonation_plan_v1.json",
        "raw_sha256": "37af7bd203fbe105178657663879f1737c905aebb12d4d9115d61773dc1e8843",
    }:
        raise SmokeError("parent policy binding drifted")
    if plan["failed_predecessor"] != {
        "candidate_id": "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_recovery_v2",
        "failure_evidence_path": "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_failure_evidence_v2.json",
        "failure_evidence_sha256": "6080541d6e9806ee15921944e3d4494a6b5c2347e1102a18f18d13503d169c9a",
        "isolated_path": "/work/failure_evidence.json",
        "repair": "use a broad 1.1 to 1.5 angstrom S-H bound that admits the exact 1.336 angstrom frozen ff14SB equilibrium length",
    }:
        raise SmokeError("failed predecessor binding drifted")
    if plan["source_commitment"] != {
        "contract": COMMITMENT_CONTRACT,
        "isolated_path": "/work/source_commitment.json",
        "required_before_execution": True,
    }:
        raise SmokeError("source commitment contract drifted")
    if plan["output_contract"] != {
        "all_distinct_atom_minimum_distance_angstrom": "0.5",
        "atom_identity": "chain_id/residue_id/insertion_code/residue_name/atom_name/element",
        "broad_valence_and_covalent_geometry_required": True,
        "canonical_protonation_signature": "ordered residue identity plus sorted hydrogen-name-to-heavy-parent-name bonds",
        "deterministic_pdb_bytes": True,
        "existing_parent_heavy_identities_and_coordinates_byte_exact": True,
        "finite_potential_energy_required": True,
        "force_field_system_construction_required": True,
        "new_hydrogen_exactly_one_heavy_parent_required": True,
        "output_directory_must_not_exist": True,
        "returned_variant_vector_required": True,
        "support_id_derivation": "SHA256(candidate_id NUL entity_uid NUL parent_support_id NUL condition_branch_id NUL parent_pdb_sha256 NUL runtime_sif_sha256)",
        "two_repeat_pdb_and_metadata_bytes_must_match": True,
    }:
        raise SmokeError("output contract drifted")
    if plan["unresolved_blockers"] != [
        "the bounded smoke recovery v3 has not yet run",
        "this smoke cannot qualify any support beyond its two exact parent examples",
        "production protonation generator source commitment and all-support independent checker are absent",
        "all-support deterministic replay physicality and effective-distinct-state audits are absent",
        "support order diversity thresholds and downstream one-shared-q inference remain unqualified",
    ]:
        raise SmokeError("unresolved blocker inventory drifted")
    return result


def _validate_commitment(
    commitment: Mapping[str, Any], commitment_raw: bytes, plan_raw: bytes
) -> str:
    _schema(
        commitment,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "runtime_sif_sha256",
            "scope",
            "source_git_commit",
            "sources",
            "state",
        },
        "source commitment",
    )
    if (
        commitment["artifact_kind"]
        != "hold_only_target_unread_bounded_smoke_source_commitment"
        or commitment["candidate_id"] != CANDIDATE_ID
        or commitment["contract"] != COMMITMENT_CONTRACT
        or commitment["runtime_sif_sha256"]
        != "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
        or commitment["scope"]
        != (
            "bounded target-unread smoke source only; not all-support, science, score, "
            "feasibility, or authorization evidence"
        )
        or commitment["state"] != "HOLD_SMOKE_RECOVERY_V3_SOURCE_FROZEN_UNRUN"
    ):
        raise SmokeError("source commitment identity drifted")
    commit = commitment["source_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise SmokeError("source Git commit is malformed")
    _closed(commitment["closed_capabilities"], "source commitment capabilities")
    _schema(
        commitment["sources"],
        {
            "checker",
            "generator",
            "launcher",
            "plan",
            "predecessor_failure_evidence",
        },
        "committed sources",
    )
    mounted = {
        "generator": SOURCE_PATH,
        "checker": CHECKER_PATH,
        "launcher": LAUNCHER_PATH,
        "predecessor_failure_evidence": FAILURE_EVIDENCE_PATH,
        "plan": PLAN_PATH,
    }
    for name, path in mounted.items():
        record = commitment["sources"][name]
        _schema(record, {"repository_path", "sha256"}, "%s commitment" % name)
        if name == "plan":
            actual = _sha256_bytes(plan_raw)
        else:
            _regular_file(path, path, name)
            actual = _sha256_file(path)
        if actual != record["sha256"]:
            raise SmokeError("%s source hash drifted" % name)
    if _canonical(commitment) != commitment_raw:
        raise SmokeError("source commitment is not canonical JSON")
    return _sha256_bytes(commitment_raw)


def _atom_identity(atom: Any) -> Tuple[str, str, str, str, str, str]:
    residue = atom.residue
    element = "" if atom.element is None else atom.element.symbol
    return (
        residue.chain.id,
        residue.id,
        residue.insertionCode,
        residue.name,
        atom.name,
        element,
    )


def _heavy_records(topology: Any, positions: Any) -> List[List[str]]:
    from openmm import unit

    xyz = positions.value_in_unit(unit.nanometer)
    records = []  # type: List[List[str]]
    for atom in topology.atoms():
        if atom.element is None or atom.element.atomic_number == 1:
            continue
        identity = list(_atom_identity(atom))
        position = xyz[atom.index]
        records.append(
            identity
            + [
                float(position[0]).hex(),
                float(position[1]).hex(),
                float(position[2]).hex(),
            ]
        )
    return records


def _signature(topology: Any) -> List[Dict[str, Any]]:
    atoms = list(topology.atoms())
    bonded = {atom: [] for atom in atoms}
    for atom1, atom2 in topology.bonds():
        bonded[atom1].append(atom2)
        bonded[atom2].append(atom1)
    result = []  # type: List[Dict[str, Any]]
    for residue in topology.residues():
        hydrogens = []  # type: List[List[str]]
        for atom in residue.atoms():
            if atom.element is None or atom.element.atomic_number != 1:
                continue
            parents = [
                other
                for other in bonded[atom]
                if other.element is not None and other.element.atomic_number != 1
            ]
            if len(parents) != 1 or len(bonded[atom]) != 1:
                raise SmokeError("hydrogen does not have exactly one heavy parent")
            hydrogens.append([atom.name, parents[0].name])
        result.append(
            {
                "chain_id": residue.chain.id,
                "hydrogens": sorted(hydrogens),
                "insertion_code": residue.insertionCode,
                "residue_id": residue.id,
                "residue_name": residue.name,
            }
        )
    return result


def _physicality(
    topology: Any, positions: Any, forcefield: Any, platform: Any
) -> Dict[str, str]:
    import numpy as np
    from openmm import Context, VerletIntegrator, unit
    from openmm.app import CutoffNonPeriodic

    atoms = list(topology.atoms())
    identities = [_atom_identity(atom) for atom in atoms]
    if len(identities) != len(set(identities)):
        raise SmokeError("duplicate atom identity")
    xyz = np.asarray(positions.value_in_unit(unit.angstrom), dtype=np.float64)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise SmokeError("nonfinite or malformed coordinates")
    minimum2 = math.inf
    for start in range(0, len(atoms), 256):
        delta = xyz[start : start + 256, None, :] - xyz[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        for local, absolute in enumerate(range(start, min(start + 256, len(atoms)))):
            distance2[local, absolute] = math.inf
        minimum2 = min(minimum2, float(distance2.min()))
    minimum = math.sqrt(minimum2)
    if not math.isfinite(minimum) or minimum < 0.5:
        raise SmokeError("all-distinct-atom minimum distance is below 0.5 A")
    bonds = {frozenset((a.index, b.index)) for a, b in topology.bonds()}
    neighbors = {atom.index: [] for atom in atoms}
    bond_lengths = []
    hydrogen_angles = []
    maximum_neighbors = {1: 1, 6: 4, 7: 4, 8: 2, 16: 6}
    for left, right in topology.bonds():
        if left.element is None or right.element is None:
            raise SmokeError("bonded atom has no element")
        neighbors[left.index].append(right)
        neighbors[right.index].append(left)
        length = float(np.linalg.norm(xyz[left.index] - xyz[right.index]))
        elements = {left.element.atomic_number, right.element.atomic_number}
        if elements == {1, 16}:
            lower, upper = 1.1, 1.5
        elif 1 in elements:
            lower, upper = 0.7, 1.3
        else:
            lower, upper = 1.0, 2.3
        if not lower <= length <= upper:
            raise SmokeError("covalent bond length is outside the broad physical range")
        bond_lengths.append(length)
    for atom in atoms:
        if atom.element is None:
            raise SmokeError("atom has no element")
        atomic_number = atom.element.atomic_number
        if (
            atomic_number not in maximum_neighbors
            or len(neighbors[atom.index]) > maximum_neighbors[atomic_number]
        ):
            raise SmokeError("atom valence exceeds the broad element limit")
        if atomic_number != 1:
            continue
        parent = neighbors[atom.index][0]
        for other in neighbors[parent.index]:
            if other.index == atom.index or other.element.atomic_number == 1:
                continue
            first = xyz[atom.index] - xyz[parent.index]
            second = xyz[other.index] - xyz[parent.index]
            cosine = float(
                np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if not 45.0 <= angle <= 180.0:
                raise SmokeError(
                    "hydrogen-parent-heavy angle is outside the broad physical range"
                )
            hydrogen_angles.append(angle)
    if not bond_lengths or not hydrogen_angles:
        raise SmokeError("covalent geometry audit had no observations")
    for chain in topology.chains():
        residues = list(chain.residues())
        for previous, current in zip(residues, residues[1:]):
            previous_c = [atom for atom in previous.atoms() if atom.name == "C"]
            current_n = [atom for atom in current.atoms() if atom.name == "N"]
            if (
                len(previous_c) != 1
                or len(current_n) != 1
                or frozenset((previous_c[0].index, current_n[0].index)) not in bonds
            ):
                raise SmokeError("backbone C-N continuity failed")
    system = forcefield.createSystem(
        topology, rigidWater=False, nonbondedMethod=CutoffNonPeriodic
    )
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator, platform)
    try:
        context.setPositions(positions)
        energy = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context
        del integrator
    if not math.isfinite(float(energy)):
        raise SmokeError("potential energy is nonfinite")
    return {
        "maximum_covalent_bond_angstrom_hex": max(bond_lengths).hex(),
        "minimum_distance_angstrom_hex": float(minimum).hex(),
        "minimum_covalent_bond_angstrom_hex": min(bond_lengths).hex(),
        "minimum_hydrogen_angle_degrees_hex": min(hydrogen_angles).hex(),
        "potential_energy_kj_mol_hex": float(energy).hex(),
    }


def _runtime_identity(plan: Mapping[str, Any]) -> Tuple[Any, Any, Dict[str, str]]:
    import openmm
    from openmm import Platform
    from openmm.app import ForceField, modeller

    runtime = plan["execution"]["runtime"]
    if openmm.version.full_version != runtime["openmm_exact_version"]:
        raise SmokeError("OpenMM version drifted")
    platform = Platform.getPlatformByName("Reference")
    if platform.getName() != runtime["platform"]:
        raise SmokeError("OpenMM platform drifted")
    modeller_path = Path(modeller.__file__).resolve(strict=True)
    hydrogen_path = modeller_path.parent / "data" / "hydrogens.xml"
    forcefield_path = (
        modeller_path.parent / "data" / runtime["force_field_relative_path"]
    )
    hashes = {
        "force_field_sha256": _sha256_file(forcefield_path),
        "hydrogen_definitions_sha256": _sha256_file(hydrogen_path),
        "modeller_source_sha256": _sha256_file(modeller_path),
    }
    for field, actual in hashes.items():
        if runtime[field] != actual:
            raise SmokeError("runtime %s drifted" % field)
    return ForceField(runtime["force_field_relative_path"]), platform, hashes


def _find_state(
    entities: Sequence[Mapping[str, Any]], entity_uid: str, branch_id: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    for entity in entities:
        if entity["entity_uid"] == entity_uid:
            for branch in entity["branches"]:
                if branch["branch_id"] == branch_id:
                    return entity, branch
    raise SmokeError("unknown child entity/branch")


def _child(
    plan: Mapping[str, Any],
    commitment_sha: str,
    entity: Mapping[str, Any],
    branch: Mapping[str, Any],
    output: Path,
) -> None:
    import numpy as np
    from openmm.app import Modeller, PDBFile

    parent = Path(entity["parent_pdb"]["isolated_path"])
    _regular_file(parent, parent, "parent PDB")
    if _sha256_file(parent) != entity["parent_pdb"]["sha256"]:
        raise SmokeError("parent PDB hash drifted")
    seed = _seed(
        CANDIDATE_ID,
        entity["entity_uid"],
        entity["parent_support_id"],
        branch["condition_branch_id"],
    )
    expected_hashseed = str(seed & 0xFFFFFFFF)
    if os.environ.get("PYTHONHASHSEED") != expected_hashseed:
        raise SmokeError("PYTHONHASHSEED drifted")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    pdb = PDBFile(str(parent))
    if any(
        residue.name not in STANDARD_RESIDUES for residue in pdb.topology.residues()
    ):
        raise SmokeError("unsupported or nonstandard residue")
    modeller = Modeller(pdb.topology, pdb.positions)
    hydrogens = [
        atom
        for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element.atomic_number == 1
    ]
    modeller.delete(hydrogens)
    parent_heavy = _heavy_records(modeller.topology, modeller.positions)
    forcefield, platform, runtime_hashes = _runtime_identity(plan)
    variants = modeller.addHydrogens(
        forcefield,
        pH=float(branch["pH"]),
        variants=None,
        platform=platform,
    )
    emitted_heavy = _heavy_records(modeller.topology, modeller.positions)
    if emitted_heavy != parent_heavy:
        raise SmokeError("existing parent heavy identities or coordinates changed")
    signature = _signature(modeller.topology)
    diagnostics = _physicality(
        modeller.topology, modeller.positions, forcefield, platform
    )
    stream = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, stream, keepIds=True)
    pdb_raw = stream.getvalue().encode("utf-8")
    support_id = _support_id(entity, branch, plan["execution"]["runtime"]["sif_sha256"])
    metadata = {
        "artifact_kind": "hold_only_target_unread_bounded_protonation_smoke_state",
        "atom_count": len(list(modeller.topology.atoms())),
        "branch_id": branch["branch_id"],
        "candidate_id": CANDIDATE_ID,
        "condition_branch_id": branch["condition_branch_id"],
        "condition_state": entity["condition_state"],
        "entity_uid": entity["entity_uid"],
        "heavy_records_sha256": _sha256_bytes(_canonical(parent_heavy)),
        "hydrogen_count": sum(len(row["hydrogens"]) for row in signature),
        "parent_pdb_sha256": entity["parent_pdb"]["sha256"],
        "parent_support_id": entity["parent_support_id"],
        "ph": branch["pH"],
        "physicality": diagnostics,
        "protonation_signature": signature,
        "protonation_signature_sha256": _sha256_bytes(_canonical(signature)),
        "returned_variant_vector": variants,
        "runtime_hashes": runtime_hashes,
        "seed64": str(seed),
        "source_commitment_sha256": commitment_sha,
        "support_id": support_id,
    }
    _write_exclusive(_state_file(output, ".pdb"), pdb_raw)
    _write_exclusive(_state_file(output, ".json"), _canonical(metadata))


def _parent(
    plan: Mapping[str, Any],
    commitment_sha: str,
    entities: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    if output != OUTPUT_PARENT / OUTPUT_NAME or output.exists() or output.is_symlink():
        raise SmokeError("output directory must be the exact new no-clobber path")
    output.mkdir(mode=0o700)
    records = []  # type: List[Dict[str, Any]]
    for entity in entities:
        for branch in entity["branches"]:
            bases = []  # type: List[Path]
            seed = _seed(
                CANDIDATE_ID,
                entity["entity_uid"],
                entity["parent_support_id"],
                branch["condition_branch_id"],
            )
            for repeat in range(2):
                base = output / (
                    "%s__%s__repeat%d"
                    % (entity["bmrb_id"], branch["branch_id"], repeat)
                )
                environment = os.environ.copy()
                environment["PYTHONHASHSEED"] = str(seed & 0xFFFFFFFF)
                subprocess.run(
                    [
                        sys.executable,
                        str(SOURCE_PATH),
                        "--child",
                        entity["entity_uid"],
                        branch["branch_id"],
                        str(base),
                    ],
                    check=True,
                    env=environment,
                    cwd="/work",
                )
                bases.append(base)
            first_pdb = _state_file(bases[0], ".pdb").read_bytes()
            second_pdb = _state_file(bases[1], ".pdb").read_bytes()
            first_json = _state_file(bases[0], ".json").read_bytes()
            second_json = _state_file(bases[1], ".json").read_bytes()
            if first_pdb != second_pdb or first_json != second_json:
                raise SmokeError("fresh-process repeat bytes drifted")
            records.append(
                {
                    "branch_id": branch["branch_id"],
                    "entity_uid": entity["entity_uid"],
                    "metadata_sha256": _sha256_bytes(first_json),
                    "pdb_sha256": _sha256_bytes(first_pdb),
                }
            )
    result = {
        "artifact_kind": "hold_only_target_unread_bounded_protonation_smoke_result",
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": {field: False for field in sorted(CLOSED_FIELDS)},
        "source_commitment_sha256": commitment_sha,
        "state_count": len(records),
        "states": records,
        "status": "HOLD_BOUNDED_PROTONATION_SMOKE_GENERATED_NOT_INDEPENDENTLY_CHECKED",
    }
    _write_exclusive(output / "result.json", _canonical(result))


def _self_test() -> None:
    checks = 0
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=_json_pairs)
        raise AssertionError("duplicate JSON test did not fail")
    except SmokeError:
        checks += 1
    first = _seed("a", "b", "c", "d")
    if first != _seed("a", "b", "c", "d") or first == _seed("a", "b", "c", "e"):
        raise AssertionError("seed derivation is not stable/separated")
    checks += 1
    if _canonical({"b": 1, "a": 2}) != b'{"a":2,"b":1}\n':
        raise AssertionError("canonical JSON drifted")
    checks += 1
    print("METRIC condition_uncertainty_protonation_smoke_self_checks=%d" % checks)
    print("STATUS HOLD_SELF_TEST_ONLY_SMOKE_UNRUN")


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--child", nargs=3, metavar=("ENTITY_UID", "BRANCH_ID", "OUTPUT_BASE")
    )
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    _regular_file(PLAN_PATH, PLAN_PATH, "smoke plan")
    _regular_file(COMMITMENT_PATH, COMMITMENT_PATH, "source commitment")
    _regular_file(SOURCE_PATH, SOURCE_PATH, "generator")
    _regular_file(CHECKER_PATH, CHECKER_PATH, "checker")
    _regular_file(LAUNCHER_PATH, LAUNCHER_PATH, "launcher")
    _regular_file(
        FAILURE_EVIDENCE_PATH, FAILURE_EVIDENCE_PATH, "predecessor failure evidence"
    )
    plan_raw = PLAN_PATH.read_bytes()
    plan = _load_json(PLAN_PATH, "smoke plan")
    entities = _validate_plan(plan)
    commitment_raw = COMMITMENT_PATH.read_bytes()
    commitment = _load_json(COMMITMENT_PATH, "source commitment")
    commitment_sha = _validate_commitment(commitment, commitment_raw, plan_raw)
    if args.child:
        entity, branch = _find_state(entities, args.child[0], args.child[1])
        output = Path(args.child[2])
        if (
            output.parent != OUTPUT_PARENT / OUTPUT_NAME
            or output.is_absolute() is False
        ):
            raise SmokeError("child output path escaped exact smoke directory")
        _child(plan, commitment_sha, entity, branch, output)
    else:
        _parent(plan, commitment_sha, entities, OUTPUT_PARENT / OUTPUT_NAME)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SmokeError, subprocess.SubprocessError) as error:
        print("ERROR %s" % error, file=sys.stderr)
        raise SystemExit(2)
