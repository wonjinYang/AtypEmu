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
SOURCE_PATH = Path("/work/generator.py")
CHECKER_PATH = Path("/work/checker.py")
LAUNCHER_PATH = Path("/work/launcher.py")
OUTPUT_PARENT = Path("/out")
OUTPUT_NAME = "condition_uncertainty_protonation_smoke_v1"
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
        != "hold_only_target_unread_bounded_protonation_smoke_plan_not_support_not_authorization"
        or plan["candidate_id"] != CANDIDATE_ID
        or plan["contract"] != PLAN_CONTRACT
        or plan["state"]
        not in {"HOLD_SMOKE_SOURCE_UNFROZEN_UNRUN", "HOLD_SMOKE_SOURCE_FROZEN_UNRUN"}
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
    if (
        mounts["network"] != "none"
        or mounts["target_or_score_storage_mounted"] is not False
    ):
        raise SmokeError("smoke isolation policy drifted")
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
        raise SmokeError("runtime identity drifted")
    entities = plan["entities"]
    if not isinstance(entities, list) or len(entities) != 2:
        raise SmokeError("smoke entity roster drifted")
    expected = {
        "bmrb:10109:entity:1": ("observed", ["6.0"]),
        "bmrb:4333:entity:1": ("state_missing", ["2.2", "5.45", "7.5", "9.25", "12.0"]),
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
        state = expected.get(entity["entity_uid"])
        if (
            state is None
            or entity["condition_state"] != state[0]
            or entity["parent_support_id"] != "1"
        ):
            raise SmokeError("smoke entity identity drifted")
        _schema(
            entity["parent_pdb"],
            {"isolated_path", "repository_path", "sha256"},
            "parent PDB",
        )
        branches = entity["branches"]
        if (
            not isinstance(branches, list)
            or [branch.get("pH") for branch in branches] != state[1]
        ):
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
        or commitment["state"] != "HOLD_SMOKE_SOURCE_FROZEN_UNRUN"
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
        {"checker", "generator", "launcher", "plan"},
        "committed sources",
    )
    mounted = {
        "generator": SOURCE_PATH,
        "checker": CHECKER_PATH,
        "launcher": LAUNCHER_PATH,
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
        "minimum_distance_angstrom_hex": float(minimum).hex(),
        "potential_energy_kj_mol_hex": float(energy).hex(),
    }


def _runtime_identity(plan: Mapping[str, Any]) -> Tuple[Any, Any, Dict[str, str]]:
    import openmm
    from openmm import Platform
    from openmm.app import ForceField, modeller

    runtime = plan["execution"]["runtime"]
    if openmm.__version__ != runtime["openmm_exact_version"]:
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
    _write_exclusive(output.with_suffix(".pdb"), pdb_raw)
    _write_exclusive(output.with_suffix(".json"), _canonical(metadata))


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
            first_pdb = bases[0].with_suffix(".pdb").read_bytes()
            second_pdb = bases[1].with_suffix(".pdb").read_bytes()
            first_json = bases[0].with_suffix(".json").read_bytes()
            second_json = bases[1].with_suffix(".json").read_bytes()
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
