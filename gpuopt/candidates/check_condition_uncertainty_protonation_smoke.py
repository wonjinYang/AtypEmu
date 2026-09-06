#!/usr/bin/env python3
"""Independently replay the bounded target-unread protonation smoke."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_v1"
)
PLAN_CONTRACT = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_plan_v1"
)
COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_"
    "source_commitment_v1"
)
PLAN_PATH = Path("/work/plan.json")
COMMITMENT_PATH = Path("/work/source_commitment.json")
GENERATOR_PATH = Path("/work/generator.py")
CHECKER_PATH = Path("/work/checker.py")
LAUNCHER_PATH = Path("/work/launcher.py")
OUTPUT = Path("/out/condition_uncertainty_protonation_smoke_v1")
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


class CheckError(ValueError):
    """Fail-closed independent check error."""


def _pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value = {}  # type: Dict[str, Any]
    for key, item in pairs:
        if key in value:
            raise CheckError("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _load(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, CheckError) as error:
        raise CheckError("invalid %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise CheckError("%s must be an object" % label)
    return value


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CheckError("missing, indirect, or nonregular file: %s" % path)
    raw = path.read_bytes()
    if len(raw) > limit:
        raise CheckError("oversize file: %s" % path)
    return raw


def _schema(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        raise CheckError(
            "%s schema drifted: %s" % (label, sorted(set(value) ^ expected))
        )


def _closed(value: Mapping[str, Any], label: str) -> None:
    _schema(value, CLOSED_FIELDS, label)
    if any(value[field] is not False for field in CLOSED_FIELDS):
        raise CheckError("%s opened a forbidden capability" % label)


def _seed(entity_uid: str, support: str, branch_id: str) -> int:
    raw = "\0".join((CANDIDATE_ID, entity_uid, support, branch_id)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _support(
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


def _plan(raw: bytes) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    plan = _load(raw, "plan")
    _schema(
        plan,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "entities",
            "execution",
            "interpretation_limits",
            "output_contract",
            "parent_policy",
            "source_commitment",
            "state",
            "unresolved_blockers",
        },
        "plan",
    )
    if (
        plan["artifact_kind"]
        != "hold_only_target_unread_bounded_protonation_smoke_plan_not_support_not_authorization"
        or plan["candidate_id"] != CANDIDATE_ID
        or plan["contract"] != PLAN_CONTRACT
        or plan["state"] != "HOLD_SMOKE_SOURCE_FROZEN_UNRUN"
    ):
        raise CheckError("plan identity/state drifted")
    _closed(plan["closed_capabilities"], "plan capabilities")
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
    mounts = execution["isolated_mounts"]
    _schema(
        mounts,
        {"network", "read_only", "target_or_score_storage_mounted", "writable"},
        "mounts",
    )
    if (
        execution["fresh_python_process_per_entity_branch_repeat"] is not True
        or execution["pythonhashseed_required"] is not True
        or execution["repeat_count"] != 2
        or mounts["network"] != "none"
        or mounts["target_or_score_storage_mounted"] is not False
    ):
        raise CheckError("execution isolation drifted")
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
            "platform",
            "sif_host_path",
            "sif_sha256",
        },
        "runtime",
    )
    if (
        runtime["force_field_relative_path"] != "amber14/protein.ff14SB.xml"
        or runtime["openmm_exact_version"] != "8.6.0.dev-c6173db"
        or runtime["platform"] != "Reference"
        or runtime["sif_sha256"]
        != "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
    ):
        raise CheckError("runtime drifted")
    entities = plan["entities"]
    expected = {
        "bmrb:10109:entity:1": ("bmr10109", "observed", ["6.0"]),
        "bmrb:4333:entity:1": (
            "bmr4333",
            "state_missing",
            ["2.2", "5.45", "7.5", "9.25", "12.0"],
        ),
    }
    if not isinstance(entities, list) or len(entities) != 2:
        raise CheckError("entity roster drifted")
    states = 0
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
        wanted = expected.get(entity["entity_uid"])
        if (
            wanted is None
            or (entity["bmrb_id"], entity["condition_state"]) != wanted[:2]
            or entity["parent_support_id"] != "1"
        ):
            raise CheckError("entity identity drifted")
        _schema(
            entity["parent_pdb"],
            {"isolated_path", "repository_path", "sha256"},
            "parent PDB",
        )
        if [branch.get("pH") for branch in entity["branches"]] != wanted[2]:
            raise CheckError("branch pH roster drifted")
        for branch in entity["branches"]:
            _schema(branch, {"branch_id", "condition_branch_id", "pH"}, "branch")
            states += 1
    if states != 6:
        raise CheckError("state count drifted")
    return plan, entities


def _commitment(raw: bytes, plan_raw: bytes) -> str:
    value = _load(raw, "commitment")
    _schema(
        value,
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
        "commitment",
    )
    if (
        value["artifact_kind"]
        != "hold_only_target_unread_bounded_smoke_source_commitment"
        or value["candidate_id"] != CANDIDATE_ID
        or value["contract"] != COMMITMENT_CONTRACT
        or value["runtime_sif_sha256"]
        != "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
        or value["scope"]
        != (
            "bounded target-unread smoke source only; not all-support, science, score, "
            "feasibility, or authorization evidence"
        )
        or value["state"] != "HOLD_SMOKE_SOURCE_FROZEN_UNRUN"
    ):
        raise CheckError("commitment identity drifted")
    commit = value["source_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise CheckError("source Git commit is malformed")
    _closed(value["closed_capabilities"], "commitment capabilities")
    _schema(value["sources"], {"checker", "generator", "launcher", "plan"}, "sources")
    actual = {
        "checker": _file(CHECKER_PATH, 1_000_000),
        "generator": _file(GENERATOR_PATH, 1_000_000),
        "launcher": _file(LAUNCHER_PATH, 1_000_000),
        "plan": plan_raw,
    }
    for name, body in actual.items():
        record = value["sources"][name]
        _schema(record, {"repository_path", "sha256"}, "%s source" % name)
        if _sha(body) != record["sha256"]:
            raise CheckError("%s commitment hash drifted" % name)
    if _canonical(value) != raw:
        raise CheckError("source commitment is not canonical JSON")
    return _sha(raw)


def _identity(atom: Any) -> Tuple[str, str, str, str, str, str]:
    residue = atom.residue
    return (
        residue.chain.id,
        residue.id,
        residue.insertionCode,
        residue.name,
        atom.name,
        "" if atom.element is None else atom.element.symbol,
    )


def _heavy(topology: Any, positions: Any) -> List[List[str]]:
    from openmm import unit

    xyz = positions.value_in_unit(unit.nanometer)
    rows = []  # type: List[List[str]]
    for atom in topology.atoms():
        if atom.element is not None and atom.element.atomic_number != 1:
            point = xyz[atom.index]
            rows.append(
                list(_identity(atom))
                + [float(point[0]).hex(), float(point[1]).hex(), float(point[2]).hex()]
            )
    return rows


def _protonation(topology: Any) -> List[Dict[str, Any]]:
    atoms = list(topology.atoms())
    neighbors = {atom: [] for atom in atoms}
    for left, right in topology.bonds():
        neighbors[left].append(right)
        neighbors[right].append(left)
    rows = []  # type: List[Dict[str, Any]]
    for residue in topology.residues():
        hydrogen_bonds = []  # type: List[List[str]]
        for atom in residue.atoms():
            if atom.element is None or atom.element.atomic_number != 1:
                continue
            heavy = [
                other
                for other in neighbors[atom]
                if other.element is not None and other.element.atomic_number != 1
            ]
            if len(neighbors[atom]) != 1 or len(heavy) != 1:
                raise CheckError("hydrogen parent valence failed")
            hydrogen_bonds.append([atom.name, heavy[0].name])
        rows.append(
            {
                "chain_id": residue.chain.id,
                "hydrogens": sorted(hydrogen_bonds),
                "insertion_code": residue.insertionCode,
                "residue_id": residue.id,
                "residue_name": residue.name,
            }
        )
    return rows


def _audit(
    topology: Any, positions: Any, forcefield: Any, platform: Any
) -> Dict[str, str]:
    import numpy as np
    from openmm import Context, VerletIntegrator, unit
    from openmm.app import CutoffNonPeriodic

    atoms = list(topology.atoms())
    identities = [_identity(atom) for atom in atoms]
    if len(set(identities)) != len(identities):
        raise CheckError("atom identities are not unique")
    xyz = np.asarray(positions.value_in_unit(unit.angstrom), dtype=np.float64)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise CheckError("coordinates are malformed or nonfinite")
    smallest2 = math.inf
    for begin in range(0, len(atoms), 257):
        delta = xyz[begin : begin + 257, None, :] - xyz[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        for local, absolute in enumerate(range(begin, min(begin + 257, len(atoms)))):
            squared[local, absolute] = math.inf
        smallest2 = min(smallest2, float(np.min(squared)))
    minimum = math.sqrt(smallest2)
    if not math.isfinite(minimum) or minimum < 0.5:
        raise CheckError("minimum distance failed")
    edges = {
        tuple(sorted((left.index, right.index))) for left, right in topology.bonds()
    }
    for chain in topology.chains():
        residues = list(chain.residues())
        for left, right in zip(residues, residues[1:]):
            carbons = [atom for atom in left.atoms() if atom.name == "C"]
            nitrogens = [atom for atom in right.atoms() if atom.name == "N"]
            if (
                len(carbons) != 1
                or len(nitrogens) != 1
                or tuple(sorted((carbons[0].index, nitrogens[0].index))) not in edges
            ):
                raise CheckError("backbone continuity failed")
    system = forcefield.createSystem(
        topology, rigidWater=False, nonbondedMethod=CutoffNonPeriodic
    )
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator, platform)
    try:
        context.setPositions(positions)
        energy = float(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context
        del integrator
    if not math.isfinite(energy):
        raise CheckError("potential energy is nonfinite")
    return {
        "minimum_distance_angstrom_hex": float(minimum).hex(),
        "potential_energy_kj_mol_hex": energy.hex(),
    }


def _runtime(plan: Mapping[str, Any]) -> Tuple[Any, Any, Dict[str, str]]:
    import openmm
    from openmm import Platform
    from openmm.app import ForceField, modeller

    expected = plan["execution"]["runtime"]
    if openmm.__version__ != expected["openmm_exact_version"]:
        raise CheckError("OpenMM version drifted")
    platform = Platform.getPlatformByName("Reference")
    source = Path(modeller.__file__).resolve(strict=True)
    hashes = {
        "force_field_sha256": _sha(
            _file(
                source.parent / "data" / expected["force_field_relative_path"],
                2_000_000,
            )
        ),
        "hydrogen_definitions_sha256": _sha(
            _file(source.parent / "data" / "hydrogens.xml", 1_000_000)
        ),
        "modeller_source_sha256": _sha(_file(source, 1_000_000)),
    }
    if (
        any(hashes[key] != expected[key] for key in hashes)
        or platform.getName() != "Reference"
    ):
        raise CheckError("runtime source identity drifted")
    return ForceField(expected["force_field_relative_path"]), platform, hashes


def _regenerate(
    plan: Mapping[str, Any],
    entity: Mapping[str, Any],
    branch: Mapping[str, Any],
    forcefield: Any,
    platform: Any,
) -> Tuple[bytes, Dict[str, Any]]:
    import numpy as np
    from openmm.app import Modeller, PDBFile

    parent_path = Path(entity["parent_pdb"]["isolated_path"])
    parent_raw = _file(parent_path, 2_000_000)
    if _sha(parent_raw) != entity["parent_pdb"]["sha256"]:
        raise CheckError("parent PDB hash drifted")
    seed = _seed(
        entity["entity_uid"], entity["parent_support_id"], branch["condition_branch_id"]
    )
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    parent = PDBFile(io.StringIO(parent_raw.decode("utf-8")))
    if any(
        residue.name not in STANDARD_RESIDUES for residue in parent.topology.residues()
    ):
        raise CheckError("nonstandard parent residue")
    model = Modeller(parent.topology, parent.positions)
    model.delete(
        [
            atom
            for atom in model.topology.atoms()
            if atom.element is not None and atom.element.atomic_number == 1
        ]
    )
    initial_heavy = _heavy(model.topology, model.positions)
    variants = model.addHydrogens(
        forcefield, pH=float(branch["pH"]), variants=None, platform=platform
    )
    if _heavy(model.topology, model.positions) != initial_heavy:
        raise CheckError("regenerated heavy coordinates changed")
    signature = _protonation(model.topology)
    physicality = _audit(model.topology, model.positions, forcefield, platform)
    handle = io.StringIO()
    PDBFile.writeFile(model.topology, model.positions, handle, keepIds=True)
    runtime_hashes = {
        key: plan["execution"]["runtime"][key]
        for key in (
            "force_field_sha256",
            "hydrogen_definitions_sha256",
            "modeller_source_sha256",
        )
    }
    metadata = {
        "artifact_kind": "hold_only_target_unread_bounded_protonation_smoke_state",
        "atom_count": len(list(model.topology.atoms())),
        "branch_id": branch["branch_id"],
        "candidate_id": CANDIDATE_ID,
        "condition_branch_id": branch["condition_branch_id"],
        "condition_state": entity["condition_state"],
        "entity_uid": entity["entity_uid"],
        "heavy_records_sha256": _sha(_canonical(initial_heavy)),
        "hydrogen_count": sum(len(row["hydrogens"]) for row in signature),
        "parent_pdb_sha256": entity["parent_pdb"]["sha256"],
        "parent_support_id": entity["parent_support_id"],
        "ph": branch["pH"],
        "physicality": physicality,
        "protonation_signature": signature,
        "protonation_signature_sha256": _sha(_canonical(signature)),
        "returned_variant_vector": variants,
        "runtime_hashes": runtime_hashes,
        "seed64": str(seed),
        "source_commitment_sha256": "",
        "support_id": _support(
            entity, branch, plan["execution"]["runtime"]["sif_sha256"]
        ),
    }
    return handle.getvalue().encode("utf-8"), metadata


def verify() -> int:
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise CheckError("smoke output directory is missing or indirect")
    plan_raw = _file(PLAN_PATH, 100_000)
    plan, entities = _plan(plan_raw)
    commitment_raw = _file(COMMITMENT_PATH, 100_000)
    commitment_sha = _commitment(commitment_raw, plan_raw)
    forcefield, platform, runtime_hashes = _runtime(plan)
    expected_names = {"result.json"}
    expected_states = []  # type: List[Dict[str, str]]
    checks = 0
    for entity in entities:
        for branch in entity["branches"]:
            independent_pdb, independent_metadata = _regenerate(
                plan, entity, branch, forcefield, platform
            )
            independent_metadata["source_commitment_sha256"] = commitment_sha
            independent_json = _canonical(independent_metadata)
            for repeat in range(2):
                stem = "%s__%s__repeat%d" % (
                    entity["bmrb_id"],
                    branch["branch_id"],
                    repeat,
                )
                expected_names.update({stem + ".pdb", stem + ".json"})
                actual_pdb = _file(OUTPUT / (stem + ".pdb"), 2_000_000)
                actual_json = _file(OUTPUT / (stem + ".json"), 2_000_000)
                if actual_pdb != independent_pdb or actual_json != independent_json:
                    raise CheckError(
                        "emitted state does not match independent replay: %s" % stem
                    )
                checks += 2
            expected_states.append(
                {
                    "branch_id": branch["branch_id"],
                    "entity_uid": entity["entity_uid"],
                    "metadata_sha256": _sha(independent_json),
                    "pdb_sha256": _sha(independent_pdb),
                }
            )
    names = {entry.name for entry in os.scandir(str(OUTPUT))}
    if names != expected_names or any(
        entry.is_symlink() for entry in os.scandir(str(OUTPUT))
    ):
        raise CheckError("output inventory drifted")
    result_raw = _file(OUTPUT / "result.json", 100_000)
    result = _load(result_raw, "result")
    expected_result = {
        "artifact_kind": "hold_only_target_unread_bounded_protonation_smoke_result",
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": {field: False for field in sorted(CLOSED_FIELDS)},
        "source_commitment_sha256": commitment_sha,
        "state_count": 6,
        "states": expected_states,
        "status": "HOLD_BOUNDED_PROTONATION_SMOKE_GENERATED_NOT_INDEPENDENTLY_CHECKED",
    }
    if result != expected_result or _canonical(result) != result_raw:
        raise CheckError("result summary drifted")
    checks += len(expected_names) + len(runtime_hashes) + 1
    return checks


def _self_test() -> int:
    checks = 0
    try:
        _load(b'{"x":1,"x":2}', "duplicate test")
        raise AssertionError("duplicate JSON accepted")
    except CheckError:
        checks += 1
    if _seed("u", "1", "b") != _seed("u", "1", "b") or _seed("u", "1", "b") == _seed(
        "u", "1", "c"
    ):
        raise AssertionError("seed separation failed")
    checks += 1
    if _canonical({"z": 0, "a": 1}) != b'{"a":1,"z":0}\n':
        raise AssertionError("canonical JSON failed")
    checks += 1
    return checks


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.self_test:
        checks = _self_test()
        print(
            "METRIC condition_uncertainty_protonation_smoke_independent_checks=%d"
            % checks
        )
        print("STATUS HOLD_SELF_TEST_ONLY_SMOKE_UNRUN")
        return 0
    checks = verify()
    print(
        "METRIC condition_uncertainty_protonation_smoke_independent_checks=%d" % checks
    )
    for field in sorted(CLOSED_FIELDS):
        print("METRIC %s=0" % field)
    print("STATUS HOLD_BOUNDED_PROTONATION_SMOKE_PASSED_NOT_ALL_SUPPORT_EVIDENCE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError, ValueError) as error:
        print("ERROR %s" % error, file=sys.stderr)
        raise SystemExit(2)
