#!/usr/bin/env python3
"""Independent checker for the fixed-slot K=32 complete-coordinate asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_CONTRACT = "k32_support_asset_source_commitment_v1"
AUTHORIZATION_CONTRACT = "k32_support_asset_authorization_v1"
PREFLIGHT_CONTRACT = "k32_complete_coordinate_support_preflight_v1"
RECEIPT_CONTRACT = "k32_complete_coordinate_support_asset_v1"
CONSUMED_CONTRACT = "k32_support_asset_consumed_authorization_v1"
CLAIM_CONTRACT = "k32_support_asset_external_claim_v1"
DECISION_CONTRACT = "k32_complete_coordinate_support_independent_decision_v1"
EXPECTED_SUPPORTS = (
    1,
    32,
    63,
    94,
    126,
    157,
    188,
    221,
    251,
    281,
    312,
    344,
    376,
    407,
    438,
    469,
    501,
    533,
    565,
    595,
    626,
    656,
    687,
    719,
    751,
    781,
    811,
    843,
    876,
    906,
    937,
    968,
)
RETAINED_K8 = frozenset({1, 126, 251, 376, 501, 626, 751, 876})
EXPECTED_SOURCE_FILES = frozenset(
    {
        "gpuopt/materialize_k32_complete_coordinate_supports.py",
        "gpuopt/check_k32_complete_coordinate_supports.py",
        ".auto/preunblind/k32_complete_coordinate_support_qualification_v4.json",
        "data/all_atom_observer_v1/commitment.json",
        ".auto/staging/k32_topology_physicality_audit_v4_summary.json",
    }
)
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_coordinates(path: Path) -> dict[str, Any]:
    identities: list[tuple[str, str, str, int, str, int, str, str]] = []
    coordinates: list[tuple[float, float, float]] = []
    residues: dict[tuple[int, str, int, str], dict[str, np.ndarray]] = {}
    order: list[tuple[int, str, int, str]] = []
    sequence: list[str] = []
    seen_ca: set[tuple[int, str, int, str]] = set()
    raw_lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    segment = 0
    for line in raw_lines:
        if line.startswith("ENDMDL"):
            break
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or line[16:17] not in {" ", "A"}:
            continue
        name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        seq_id = int(line[22:26])
        insertion = line[26:27].strip()
        element = line[76:78].strip().upper() or next(
            (letter for letter in name if letter.isalpha()), ""
        )
        xyz = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        key = (segment, chain, seq_id, insertion)
        identity = (
            line[:6].strip(),
            name,
            resname,
            segment,
            chain,
            seq_id,
            insertion,
            element,
        )
        if key not in residues:
            residues[key] = {}
            order.append(key)
        if name in residues[key]:
            raise ValueError(f"duplicate atom identity: {path}:{key}:{name}")
        identities.append(identity)
        coordinates.append(tuple(float(value) for value in xyz))
        residues[key][name] = xyz
        if name == "CA" and key not in seen_ca and resname in AA3_TO_1:
            sequence.append(AA3_TO_1[resname])
            seen_ca.add(key)
    array = np.asarray(coordinates, dtype=np.float64)
    if not identities or not np.isfinite(array).all():
        raise ValueError(f"empty or nonfinite PDB: {path}")
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate full atom identity: {path}")
    return {
        "path": path,
        "identities": identities,
        "coordinates": array,
        "residues": residues,
        "order": order,
        "sequence": "".join(sequence),
    }


def minimum_distance(coordinates: np.ndarray) -> float:
    width = 0.5
    occupied: dict[tuple[int, int, int], list[int]] = {}
    minimum = math.inf
    for index, coordinate in enumerate(coordinates):
        cell = tuple(int(math.floor(float(value) / width)) for value in coordinate)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for previous in occupied.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
                    ):
                        minimum = min(
                            minimum,
                            float(np.linalg.norm(coordinate - coordinates[previous])),
                        )
        occupied.setdefault(cell, []).append(index)
    if math.isfinite(minimum):
        return minimum
    return min(
        float(np.linalg.norm(coordinates[index] - coordinates[index - 1]))
        for index in range(1, len(coordinates))
    )


def physicality(records: dict[str, Any]) -> tuple[bool, float, list[str]]:
    failures: list[str] = []
    residues = records["residues"]
    order = records["order"]
    for key in order:
        atoms = residues[key]
        for first, second, low, high in (
            ("N", "CA", 1.20, 1.70),
            ("CA", "C", 1.25, 1.80),
            ("C", "O", 1.05, 1.45),
        ):
            if first in atoms and second in atoms:
                distance = float(np.linalg.norm(atoms[first] - atoms[second]))
                if distance < low or distance > high:
                    failures.append(f"{key}:{first}-{second}={distance:.8f}")
    for current, following in zip(order[:-1], order[1:], strict=True):
        if (
            current[0] != following[0]
            or current[1] != following[1]
            or following[2] != current[2] + 1
        ):
            continue
        if "C" not in residues[current] or "N" not in residues[following]:
            continue
        distance = float(
            np.linalg.norm(residues[current]["C"] - residues[following]["N"])
        )
        if distance < 1.15 or distance > 1.55:
            failures.append(f"{current}->{following}:C-N={distance:.8f}")
    minimum = minimum_distance(records["coordinates"])
    from openmm.app import PDBFile

    normalized_identities = tuple(
        (name, resname, segment, chain, seq_id, insertion, element)
        for _, name, resname, segment, chain, seq_id, insertion, element in records[
            "identities"
        ]
    )
    pdb = PDBFile(str(records["path"]))
    atoms = tuple(pdb.topology.atoms())
    topology_identities = tuple(
        (
            atom.name.upper(),
            atom.residue.name.upper(),
            atom.residue.chain.index,
            atom.residue.chain.id or "_",
            int(atom.residue.id),
            atom.residue.insertionCode.strip(),
            atom.element.symbol.upper(),
        )
        for atom in atoms
    )
    bonds = tuple(
        (
            first.index,
            second.index,
            first.element.symbol == "H" or second.element.symbol == "H",
        )
        for first, second in pdb.topology.bonds()
    )
    if normalized_identities != topology_identities:
        raise ValueError(f"OpenMM atom-order mismatch: {records['path']}")
    for first, second, has_hydrogen in bonds:
        distance = float(
            np.linalg.norm(
                records["coordinates"][first] - records["coordinates"][second]
            )
        )
        lower, upper = (0.65, 1.50) if has_hydrogen else (1.00, 2.20)
        if distance < lower or distance > upper:
            failures.append(f"{first}-{second}={distance:.8f}")
    return minimum >= 0.5 and not failures, minimum, failures


def candidate_indices(slot: int) -> list[int]:
    candidates = [slot]
    fixed = set(EXPECTED_SUPPORTS)
    for distance in range(1, 1000):
        for candidate in (slot - distance, slot + distance):
            if 1 <= candidate <= 1000 and candidate not in fixed:
                candidates.append(candidate)
    return candidates


def coordinate_path(root: Path, bmrb_id: str, source_index: int) -> Path:
    return root / "data" / "BioEmu" / bmrb_id / f"{bmrb_id}_BioEmu_{source_index}.pdb"


def expected_selected_index(
    root: Path, bmrb_id: str, slot: int, expected_sequence: str
) -> tuple[int, float]:
    for candidate in candidate_indices(slot):
        path = coordinate_path(root, bmrb_id, candidate)
        if not path.is_file():
            if candidate == slot:
                raise FileNotFoundError(f"fixed support missing: {bmrb_id}:{slot}")
            continue
        records = read_coordinates(path)
        if records["sequence"] != expected_sequence:
            if candidate == slot:
                raise ValueError(f"fixed support sequence mismatch: {bmrb_id}:{slot}")
            continue
        passed, minimum, _ = physicality(records)
        if passed:
            return candidate, minimum
    raise ValueError(f"no deterministic valid replacement: {bmrb_id}:{slot}")


def verify_external_consumption(
    git_dir: Path,
    source_sha256: str,
    preflight_sha256: str,
    consumed: dict[str, Any],
    claim: dict[str, Any],
) -> dict[str, Any]:
    authorization_blob = str(consumed["authorization_git_blob"])
    authorization_bytes = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "blob", authorization_blob],
        check=True,
        capture_output=True,
    ).stdout
    observed_authorization_blob = hashlib.sha1(  # noqa: S324 - Git identity.
        f"blob {len(authorization_bytes)}\0".encode() + authorization_bytes
    ).hexdigest()
    if observed_authorization_blob != authorization_blob:
        raise ValueError("authorization Git blob identity mismatch")
    authorization_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    if any(
        payload.get("authorization_sha256") != authorization_sha256
        for payload in (consumed, claim)
    ):
        raise ValueError("authorization content hash mismatch")
    authorization = json.loads(authorization_bytes)
    expected_authorization = {
        "contract": AUTHORIZATION_CONTRACT,
        "authorized": True,
        "authorization_ref": str(consumed["authorization_ref"]),
        "output_relative_path": "data/k32_complete_coordinate_supports_v4",
        "preflight_receipt_sha256": preflight_sha256,
        "slurm_job_id": str(consumed["slurm_job_id"]),
        "source_commitment_sha256": source_sha256,
    }
    if authorization != expected_authorization:
        raise ValueError("authorization payload mismatch")
    expected_claim = {
        "contract": CLAIM_CONTRACT,
        "authorization_git_blob": authorization_blob,
        "authorization_ref": str(consumed["authorization_ref"]),
        "authorization_sha256": authorization_sha256,
        "consumption_ref": (
            "refs/atypemu-consumptions/k32-support/" + authorization_blob
        ),
        "preflight_receipt_sha256": preflight_sha256,
        "slurm_job_id": str(consumed["slurm_job_id"]),
        "source_commitment_sha256": source_sha256,
    }
    if claim != expected_claim:
        raise ValueError("external claim payload mismatch")
    claim_bytes = json.dumps(claim, indent=2, sort_keys=True).encode() + b"\n"
    claim_blob = hashlib.sha1(  # noqa: S324 - Git identity.
        f"blob {len(claim_bytes)}\0".encode() + claim_bytes
    ).hexdigest()
    if claim_blob != str(consumed["external_claim_git_blob"]):
        raise ValueError("external claim Git blob identity mismatch")
    if consumed.get("consumption_ref") != claim.get("consumption_ref"):
        raise ValueError("external consumption marker identity mismatch")
    consumed_ref_blob = subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "rev-parse",
            f"{consumed['consumption_ref']}^{{blob}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if consumed_ref_blob != claim_blob:
        raise ValueError("append-only consumption marker mismatch")
    observed = subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "rev-parse",
            f"{consumed['authorization_ref']}^{{blob}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != str(consumed["external_claim_git_blob"]):
        raise ValueError("external authorization ref does not point to consumed claim")
    return authorization


def validate_source_inventory(
    rows: list[dict[str, Any]], parent_entities: list[dict[str, Any]]
) -> None:
    if len(rows) != 162 * 32:
        raise ValueError("source pair inventory length mismatch")
    parent_by_entity = {
        str(row["entity_uid"]): str(row["bmrb_id"]) for row in parent_entities
    }
    if len(parent_entities) != 162 or len(parent_by_entity) != 162:
        raise ValueError("parent entity roster mismatch")
    observed: dict[str, set[int]] = {}
    selected: dict[str, set[int]] = {}
    for row in rows:
        entity_uid = str(row["entity_uid"])
        if entity_uid not in parent_by_entity:
            raise ValueError(f"unknown source entity: {entity_uid}")
        bmrb_id = parent_by_entity[entity_uid]
        support_index = int(row["support_index"])
        if str(row["bmrb_id"]) != bmrb_id:
            raise ValueError(f"source BMRB identity mismatch: {entity_uid}")
        if str(row["support_id"]) != f"BioEmu_{support_index}":
            raise ValueError(f"source support identity mismatch: {entity_uid}")
        observed.setdefault(entity_uid, set()).add(support_index)
        selected.setdefault(entity_uid, set()).add(int(row["selected_source_index"]))
    if set(observed) != set(parent_by_entity):
        raise ValueError("source entity roster differs from parent")
    expected_supports = set(EXPECTED_SUPPORTS)
    for entity_uid in sorted(observed):
        if observed[entity_uid] != expected_supports:
            raise ValueError(f"per-entity support roster mismatch: {entity_uid}")
        if len(selected[entity_uid]) != 32:
            raise ValueError(f"duplicate selected source within entity: {entity_uid}")


def check(args: argparse.Namespace) -> dict[str, Any]:
    import openmm

    root = args.root.resolve()
    source = json.loads(args.source_commitment.read_text())
    preflight = json.loads(args.preflight_receipt.read_text())
    receipt = json.loads(args.receipt.read_text())
    consumed = json.loads(args.consumed_authorization.read_text())
    claim = json.loads(args.external_claim.read_text())
    expected_contracts = (
        (source, SOURCE_CONTRACT),
        (preflight, PREFLIGHT_CONTRACT),
        (receipt, RECEIPT_CONTRACT),
        (consumed, CONSUMED_CONTRACT),
        (claim, CLAIM_CONTRACT),
    )
    for payload, contract in expected_contracts:
        if payload.get("contract") != contract:
            raise ValueError(f"contract mismatch: {contract}")
    for payload in (source, preflight, receipt):
        if payload.get("target_values_read") is not False:
            raise ValueError("support artifact claims target-value access")
        if payload.get("source_gate_authorized") is not False:
            raise ValueError("support artifact claims source-gate authorization")
        if payload.get("formal_evaluation_authorized") is not False:
            raise ValueError("support artifact claims formal authorization")
    unhashed = dict(source)
    stored = unhashed.pop("commitment_sha256", None)
    if canonical_sha256(unhashed) != stored:
        raise ValueError("source commitment canonical hash mismatch")
    source_sha256 = sha256_file(args.source_commitment)
    for payload in (preflight, receipt, consumed, claim):
        if payload.get("source_commitment_sha256") != source_sha256:
            raise ValueError("source commitment binding mismatch")
    preflight_sha256 = sha256_file(args.preflight_receipt)
    for payload in (receipt, consumed, claim):
        if payload.get("preflight_receipt_sha256") != preflight_sha256:
            raise ValueError("preflight binding mismatch")
    if receipt.get("consumed_authorization_sha256") != sha256_file(
        args.consumed_authorization
    ):
        raise ValueError("consumed authorization binding mismatch")
    if receipt.get("external_claim_sha256") != sha256_file(args.external_claim):
        raise ValueError("external claim binding mismatch")
    if consumed.get("external_claim_sha256") != sha256_file(args.external_claim):
        raise ValueError("consumed claim binding mismatch")
    for key in (
        "authorization_git_blob",
        "authorization_ref",
        "authorization_sha256",
        "slurm_job_id",
    ):
        if consumed.get(key) != claim.get(key):
            raise ValueError(f"external consumption identity mismatch: {key}")
    authorization = verify_external_consumption(
        args.authorization_git_dir.resolve(),
        source_sha256,
        preflight_sha256,
        consumed,
        claim,
    )
    if tuple(int(value) for value in source.get("support_indices", ())) != EXPECTED_SUPPORTS:
        raise ValueError("source support roster mismatch")
    if tuple(int(value) for value in source.get("legacy_k8_indices", ())) != tuple(
        sorted(RETAINED_K8)
    ):
        raise ValueError("source retained-K8 roster mismatch")
    if source.get("openmm_version") != str(openmm.__version__):
        raise ValueError("OpenMM topology version mismatch")
    if source.get("entity_count") != 162 or source.get("pair_count") != 162 * 32:
        raise ValueError("source cardinality mismatch")
    if source.get("support_count") != 32:
        raise ValueError("source support count mismatch")
    if (
        preflight.get("pair_count") != 162 * 32
        or receipt.get("pair_count") != 162 * 32
        or receipt.get("entity_count") != 162
        or receipt.get("support_count") != 32
    ):
        raise ValueError("preflight or receipt cardinality mismatch")
    files = source.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_SOURCE_FILES:
        raise ValueError("source file manifest mismatch")
    for relative, expected in files.items():
        path = (root / str(relative)).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"source path escapes root: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"source file hash mismatch: {relative}")
    parent = json.loads((root / "data/all_atom_observer_v1/commitment.json").read_text())
    validate_source_inventory(source.get("pairs", []), parent.get("entities", []))
    sequences = {str(row["entity_uid"]): str(row["sequence"]) for row in parent["entities"]}
    source_rows = {
        (str(row["entity_uid"]), int(row["support_index"])): row
        for row in source.get("pairs", [])
    }
    raw_outputs = receipt.get("outputs", [])
    if len(raw_outputs) != 162 * 32:
        raise ValueError("output inventory length mismatch")
    output_rows = {
        (str(row["entity_uid"]), int(row["support_index"])): row
        for row in raw_outputs
    }
    if len(source_rows) != 162 * 32 or source_rows.keys() != output_rows.keys():
        raise ValueError("source/output identity mismatch")
    expected_replacements = {
        (str(row["entity_uid"]), int(row["support_index"]), int(row["selected_source_index"]))
        for row in source_rows.values()
        if row["replacement_required"]
    }
    preflight_replacements = {
        (str(row["entity_uid"]), int(row["support_index"]), int(row["selected_source_index"]))
        for row in preflight.get("replacements", [])
    }
    if len(preflight.get("replacements", [])) != len(preflight_replacements):
        raise ValueError("duplicate preflight replacement entry")
    if expected_replacements != preflight_replacements:
        raise ValueError("preflight replacement inventory mismatch")
    minimum = math.inf
    for identity, source_row in source_rows.items():
        output_row = output_rows[identity]
        bmrb_id = str(source_row["bmrb_id"])
        slot = int(source_row["support_index"])
        fixed_relative = f"data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{slot}.pdb"
        selected_index = int(source_row["selected_source_index"])
        selected_relative = (
            f"data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{selected_index}.pdb"
        )
        output_relative = (
            f"{authorization['output_relative_path']}/{bmrb_id}/"
            f"{bmrb_id}_BioEmu_{slot}.pdb"
        )
        if source_row.get("fixed_input_relative_path") != fixed_relative:
            raise ValueError(f"noncanonical fixed coordinate path text: {identity}")
        if source_row.get("selected_input_relative_path") != selected_relative:
            raise ValueError(f"noncanonical selected coordinate path text: {identity}")
        if output_row.get("output_relative_path") != output_relative:
            raise ValueError(f"noncanonical output coordinate path text: {identity}")
        fixed_path = (root / fixed_relative).resolve()
        selected_path = (root / selected_relative).resolve()
        output_path = (root / output_relative).resolve()
        if any(root not in path.parents for path in (fixed_path, selected_path, output_path)):
            raise ValueError(f"coordinate path escapes root: {identity}")
        expected_fixed_path = coordinate_path(root, bmrb_id, slot).resolve()
        expected_selected_path = coordinate_path(
            root, bmrb_id, int(source_row["selected_source_index"])
        ).resolve()
        expected_output_path = (
            root
            / str(authorization["output_relative_path"])
            / bmrb_id
            / f"{bmrb_id}_BioEmu_{slot}.pdb"
        ).resolve()
        if fixed_path != expected_fixed_path or selected_path != expected_selected_path:
            raise ValueError(f"noncanonical source coordinate path: {identity}")
        if output_path != expected_output_path:
            raise ValueError(f"noncanonical output coordinate path: {identity}")
        expected_output_metadata = {
            "bmrb_id": bmrb_id,
            "entity_uid": str(source_row["entity_uid"]),
            "support_id": str(source_row["support_id"]),
            "support_index": slot,
            "selected_source_index": int(source_row["selected_source_index"]),
            "selected_input_sha256": str(source_row["selected_input_sha256"]),
        }
        for field, expected_value in expected_output_metadata.items():
            if output_row.get(field) != expected_value:
                raise ValueError(f"output metadata mismatch: {identity}:{field}")
        if sha256_file(fixed_path) != source_row["fixed_input_sha256"]:
            raise ValueError(f"fixed input hash mismatch: {identity}")
        if sha256_file(selected_path) != source_row["selected_input_sha256"]:
            raise ValueError(f"selected input hash mismatch: {identity}")
        if sha256_file(output_path) != output_row["output_sha256"]:
            raise ValueError(f"output hash mismatch: {identity}")
        if output_row["output_sha256"] != source_row["selected_input_sha256"]:
            raise ValueError(f"output is not byte-identical to selected input: {identity}")
        expected_index, observed_minimum = expected_selected_index(
            root, bmrb_id, slot, sequences[identity[0]]
        )
        minimum = min(minimum, observed_minimum)
        if expected_index != int(source_row["selected_source_index"]):
            raise ValueError(f"replacement is not deterministic nearest valid: {identity}")
        replacement = expected_index != slot
        if replacement != bool(source_row["replacement_required"]):
            raise ValueError(f"replacement flag mismatch: {identity}")
        expected_disposition = (
            "deterministic_target_unread_replacement"
            if replacement
            else "byte_identical_fixed_support"
        )
        if output_row.get("disposition") != expected_disposition:
            raise ValueError(f"output disposition mismatch: {identity}")
    replacement_count = len(expected_replacements)
    for payload in (source, preflight, receipt):
        if int(payload.get("replacement_count", -1)) != replacement_count:
            raise ValueError("replacement count mismatch")
    legacy_k8_replacement_count = sum(
        support_index in RETAINED_K8
        for _entity_uid, support_index, _source_index in expected_replacements
    )
    for payload in (source, preflight, receipt):
        if int(payload.get("legacy_k8_replacement_count", -1)) != legacy_k8_replacement_count:
            raise ValueError("historical K8 replacement count mismatch")
    return {
        "passed": True,
        "pair_count": len(output_rows),
        "entity_count": 162,
        "support_count": 32,
        "replacement_count": replacement_count,
        "minimum_distinct_atom_distance_angstrom": minimum,
        "legacy_k8_replacement_count": legacy_k8_replacement_count,
        "valid_fixed_support_bytes_preserved": True,
        "all_outputs_byte_identical_to_selected_source": True,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--source-commitment", type=Path, required=True)
    result.add_argument("--preflight-receipt", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    result.add_argument("--consumed-authorization", type=Path, required=True)
    result.add_argument("--external-claim", type=Path, required=True)
    result.add_argument("--authorization-git-dir", type=Path, required=True)
    result.add_argument("--decision", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        summary = check(args)
        decision = {"contract": DECISION_CONTRACT, **summary, "errors": []}
        return_code = 0
    except Exception as error:  # Fail closed with a durable decision.
        decision = {
            "contract": DECISION_CONTRACT,
            "passed": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "target_values_read": False,
            "source_gate_authorized": False,
            "formal_evaluation_authorized": False,
        }
        return_code = 1
    write_json_new(args.decision, decision)
    print(json.dumps(decision, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
