#!/usr/bin/env python3
"""Freeze and materialize a target-unread fixed-slot K=32 coordinate asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT = "k32_support_asset_source_commitment_v1"
PREFLIGHT_CONTRACT = "k32_complete_coordinate_support_preflight_v1"
AUTHORIZATION_CONTRACT = "k32_support_asset_authorization_v1"
CONSUMED_CONTRACT = "k32_support_asset_consumed_authorization_v1"
EXTERNAL_CLAIM_CONTRACT = "k32_support_asset_external_claim_v1"
RECEIPT_CONTRACT = "k32_complete_coordinate_support_asset_v1"
PLAN_CONTRACT = "k32_complete_coordinate_support_qualification_plan_v4"
MATERIALIZER_RELATIVE = "gpuopt/materialize_k32_complete_coordinate_supports.py"
CHECKER_RELATIVE = "gpuopt/check_k32_complete_coordinate_supports.py"
PLAN_RELATIVE = ".auto/preunblind/k32_complete_coordinate_support_qualification_v4.json"
PARENT_RELATIVE = "data/all_atom_observer_v1/commitment.json"
TOPOLOGY_AUDIT_RELATIVE = ".auto/staging/k32_topology_physicality_audit_v4_summary.json"
RETAINED_K8 = frozenset({1, 126, 251, 376, 501, 626, 751, 876})
EXPECTED_SOURCE_FILES = frozenset(
    {
        MATERIALIZER_RELATIVE,
        CHECKER_RELATIVE,
        PLAN_RELATIVE,
        PARENT_RELATIVE,
        TOPOLOGY_AUDIT_RELATIVE,
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


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if plan.get("contract") != PLAN_CONTRACT:
        raise ValueError("K32 support plan contract mismatch")
    supports = [int(value) for value in plan.get("support_indices", [])]
    legacy = [int(value) for value in plan.get("legacy_k8_indices", [])]
    historical_k16 = [int(value) for value in plan.get("historical_k16_indices", [])]
    midpoint16 = [int(value) for value in plan.get("fixed_midpoint16_indices", [])]
    if len(supports) != 32 or supports != sorted(set(supports)):
        raise ValueError("K32 plan needs 32 ordered unique slot indices")
    if len(legacy) != 8 or not set(legacy).issubset(supports):
        raise ValueError("K32 plan has an invalid retained-K8 roster")
    if (
        len(historical_k16) != 16
        or len(midpoint16) != 16
        or set(historical_k16).intersection(midpoint16)
        or sorted(historical_k16 + midpoint16) != supports
    ):
        raise ValueError("historical K16 and midpoint-16 do not partition K32")
    if plan.get("replacement_policy", {}).get("candidate_order") != (
        "fixed slot, then increasing absolute index distance with lower-index ties first"
    ):
        raise ValueError("K32 replacement order mismatch")
    expected_ranges = {
        "N_CA": [1.2, 1.7],
        "CA_C": [1.25, 1.8],
        "C_O": [1.05, 1.45],
        "peptide_C_N": [1.15, 1.55],
    }
    raw = plan.get("raw_preflight", {})
    if raw.get("backbone_bond_ranges_angstrom") != expected_ranges:
        raise ValueError("K32 plan backbone ranges mismatch")
    if raw.get("minimum_distinct_atom_distance_angstrom") != 0.5:
        raise ValueError("K32 plan minimum-distance threshold mismatch")
    if plan.get("scope", {}).get("source_gate_authorized") is not False:
        raise ValueError("support plan may not authorize a source gate")
    if plan.get("scope", {}).get("formal_evaluation_authorized") is not False:
        raise ValueError("support plan may not authorize formal evaluation")
    if tuple(int(value) for value in plan.get("legacy_k8_indices", ())) != tuple(
        sorted(RETAINED_K8)
    ):
        raise ValueError("K32 retained-K8 roster mismatch")
    if plan.get("outputs", {}).get("coordinate_root") != (
        "data/k32_complete_coordinate_supports_v4"
    ):
        raise ValueError("K32 output root mismatch")
    return plan


def pdb_records(path: Path) -> dict[str, Any]:
    identities: list[tuple[str, str, str, int, str, int, str, str]] = []
    coordinates: list[tuple[float, float, float]] = []
    residues: dict[tuple[int, str, int, str], dict[str, np.ndarray]] = {}
    residue_order: list[tuple[int, str, int, str]] = []
    sequence: list[str] = []
    seen_ca: set[tuple[int, str, int, str]] = set()
    raw_bytes = path.read_bytes()
    raw_lines = raw_bytes.decode("ascii", errors="replace").splitlines()
    segment = 0
    for line in raw_lines:
        if line.startswith("ENDMDL"):
            break
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[16:17] not in {" ", "A"}:
            continue
        name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        seq_id = int(line[22:26])
        insertion = line[26:27].strip()
        element = line[76:78].strip().upper() or next(
            (character for character in name if character.isalpha()), ""
        )
        coordinate = np.asarray(
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
            residue_order.append(key)
        if name in residues[key]:
            raise ValueError(f"duplicate atom identity: {path}:{key}:{name}")
        identities.append(identity)
        coordinates.append(tuple(float(value) for value in coordinate))
        residues[key][name] = coordinate
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
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "identities": identities,
        "coordinates": array,
        "residues": residues,
        "residue_order": residue_order,
        "sequence": "".join(sequence),
    }


def minimum_distinct_distance(coordinates: np.ndarray, cutoff: float = 0.5) -> float:
    cells: dict[tuple[int, int, int], list[int]] = {}
    minimum = math.inf
    for index, coordinate in enumerate(coordinates):
        cell = tuple(np.floor(coordinate / cutoff).astype(np.int64).tolist())
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for previous in cells.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
                    ):
                        minimum = min(
                            minimum,
                            float(np.linalg.norm(coordinate - coordinates[previous])),
                        )
        cells.setdefault(cell, []).append(index)
    if math.isfinite(minimum):
        return minimum
    return min(
        float(np.linalg.norm(coordinates[index] - coordinates[index - 1]))
        for index in range(1, len(coordinates))
    )


def physicality_audit(records: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    residues = records["residues"]
    order = records["residue_order"]
    for key in order:
        atoms = residues[key]
        for first, second, low, high in (
            ("N", "CA", 1.20, 1.70),
            ("CA", "C", 1.25, 1.80),
            ("C", "O", 1.05, 1.45),
        ):
            if first in atoms and second in atoms:
                distance = float(np.linalg.norm(atoms[first] - atoms[second]))
                if not low <= distance <= high:
                    violations.append(f"{key}:{first}-{second}={distance:.8f}")
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
        if not 1.15 <= distance <= 1.55:
            violations.append(f"{current}->{following}:C-N={distance:.8f}")
    minimum = minimum_distinct_distance(records["coordinates"])
    covalent_violations: list[str] = []
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
        if not lower <= distance <= upper:
            covalent_violations.append(f"{first}-{second}={distance:.8f}")
    return {
        "pdb_sha256": records["sha256"],
        "atom_count": len(records["identities"]),
        "minimum_distinct_atom_distance_angstrom": minimum,
        "backbone_violation_count": len(violations),
        "backbone_violation_examples": violations[:8],
        "covalent_bond_violation_count": len(covalent_violations),
        "covalent_bond_violation_examples": covalent_violations[:8],
        "pass": minimum >= 0.5 and not violations and not covalent_violations,
    }


def candidate_indices(slot: int, fixed_slots: set[int]) -> list[int]:
    candidates = [slot]
    for distance in range(1, 1000):
        for candidate in (slot - distance, slot + distance):
            if 1 <= candidate <= 1000 and candidate not in fixed_slots:
                candidates.append(candidate)
    return candidates


def support_path(primary_root: Path, bmrb_id: str, index: int) -> Path:
    return primary_root / bmrb_id / f"{bmrb_id}_BioEmu_{index}.pdb"


def select_source(
    primary_root: Path,
    bmrb_id: str,
    slot: int,
    fixed_slots: set[int],
    expected_sequence: str,
) -> tuple[Path, int, dict[str, Any], dict[str, Any]]:
    fixed_path = support_path(primary_root, bmrb_id, slot)
    if not fixed_path.is_file():
        raise FileNotFoundError(f"missing fixed support: {bmrb_id}:{slot}")
    fixed_records = pdb_records(fixed_path)
    if fixed_records["sequence"] != expected_sequence:
        raise ValueError(f"fixed support sequence mismatch: {bmrb_id}:{slot}")
    fixed_audit = physicality_audit(fixed_records)
    if fixed_audit["pass"]:
        return fixed_path, slot, fixed_audit, fixed_audit
    for candidate in candidate_indices(slot, fixed_slots)[1:]:
        path = support_path(primary_root, bmrb_id, candidate)
        if not path.is_file():
            continue
        records = pdb_records(path)
        if records["sequence"] != expected_sequence:
            continue
        audit = physicality_audit(records)
        if audit["pass"]:
            return path, candidate, fixed_audit, audit
    raise ValueError(f"no valid deterministic replacement: {bmrb_id}:{slot}")


def freeze(args: argparse.Namespace) -> int:
    import openmm

    root = args.root.resolve()
    plan_path = (root / args.plan).resolve()
    parent_path = (root / args.parent_commitment).resolve()
    primary_root = (root / args.primary_root).resolve()
    if primary_root != (root / "data/BioEmu").resolve():
        raise ValueError("K32 primary root is not canonical")
    plan = load_plan(plan_path)
    audit_binding = plan.get("topology_physicality_audit", {})
    if audit_binding.get("path") != TOPOLOGY_AUDIT_RELATIVE or sha256_file(
        root / TOPOLOGY_AUDIT_RELATIVE
    ) != audit_binding.get("sha256"):
        raise ValueError("K32 topology physicality audit binding mismatch")
    parent = json.loads(parent_path.read_text())
    entities = sorted(parent.get("entities", []), key=lambda row: str(row["bmrb_id"]))
    if len(entities) != 162:
        raise ValueError("K32 support freeze requires the frozen 162-entity roster")
    supports = [int(value) for value in plan["support_indices"]]
    fixed_slots = set(supports)
    retained_k8 = {int(value) for value in plan["legacy_k8_indices"]}
    files = {
        MATERIALIZER_RELATIVE: sha256_file(root / MATERIALIZER_RELATIVE),
        CHECKER_RELATIVE: sha256_file(root / CHECKER_RELATIVE),
        PLAN_RELATIVE: sha256_file(root / PLAN_RELATIVE),
        PARENT_RELATIVE: sha256_file(root / PARENT_RELATIVE),
        TOPOLOGY_AUDIT_RELATIVE: sha256_file(root / TOPOLOGY_AUDIT_RELATIVE),
    }
    def freeze_entity(entity: dict[str, Any]) -> list[dict[str, Any]]:
        entity_rows: list[dict[str, Any]] = []
        bmrb_id = str(entity["bmrb_id"])
        entity_uid = str(entity["entity_uid"])
        selected_for_entity: set[int] = set()
        for slot in supports:
            selected_path, selected_index, fixed_audit, selected_audit = select_source(
                primary_root,
                bmrb_id,
                slot,
                fixed_slots,
                str(entity["sequence"]),
            )
            if selected_index in selected_for_entity:
                raise ValueError(f"duplicate selected source: {bmrb_id}:{selected_index}")
            selected_for_entity.add(selected_index)
            fixed_path = support_path(primary_root, bmrb_id, slot)
            entity_rows.append(
                {
                    "bmrb_id": bmrb_id,
                    "entity_uid": entity_uid,
                    "support_id": f"BioEmu_{slot}",
                    "support_index": slot,
                    "fixed_input_relative_path": str(fixed_path.relative_to(root)),
                    "fixed_input_sha256": fixed_audit["pdb_sha256"],
                    "fixed_input_audit": fixed_audit,
                    "selected_source_index": selected_index,
                    "selected_input_relative_path": str(selected_path.relative_to(root)),
                    "selected_input_sha256": selected_audit["pdb_sha256"],
                    "selected_input_audit": selected_audit,
                    "replacement_required": selected_index != slot,
                }
            )
        return entity_rows

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = [
            row
            for entity_rows in executor.map(freeze_entity, entities)
            for row in entity_rows
        ]
    if len(rows) != 162 * 32:
        raise ValueError("K32 source commitment pair count mismatch")
    payload: dict[str, Any] = {
        "contract": CONTRACT,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "entity_count": 162,
        "support_count": 32,
        "pair_count": len(rows),
        "replacement_count": sum(row["replacement_required"] for row in rows),
        "legacy_k8_replacement_count": sum(
            row["replacement_required"] and row["support_index"] in retained_k8
            for row in rows
        ),
        "openmm_version": str(openmm.__version__),
        "support_indices": supports,
        "legacy_k8_indices": sorted(retained_k8),
        "plan_sha256": sha256_file(plan_path),
        "parent_commitment_sha256": sha256_file(parent_path),
        "files": dict(sorted(files.items())),
        "pairs": rows,
    }
    payload["commitment_sha256"] = canonical_sha256(payload)
    write_json_new(args.output, payload)
    print(
        json.dumps(
            {
                "pair_count": len(rows),
                "replacement_count": payload["replacement_count"],
                "source_commitment_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def verify_commitment(root: Path, path: Path) -> dict[str, Any]:
    commitment = json.loads(path.read_text())
    if commitment.get("contract") != CONTRACT:
        raise ValueError("K32 source commitment contract mismatch")
    plan = load_plan(root / PLAN_RELATIVE)
    expected_metadata = {
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "entity_count": 162,
        "support_count": 32,
        "pair_count": 162 * 32,
        "support_indices": plan["support_indices"],
        "legacy_k8_indices": sorted(RETAINED_K8),
    }
    for field, expected in expected_metadata.items():
        if commitment.get(field) != expected:
            raise ValueError(f"K32 source commitment metadata mismatch: {field}")
    files = commitment.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_SOURCE_FILES:
        raise ValueError("K32 source commitment manifest mismatch")
    without_hash = dict(commitment)
    stored = without_hash.pop("commitment_sha256", None)
    if canonical_sha256(without_hash) != stored:
        raise ValueError("K32 source commitment canonical hash mismatch")
    for relative, expected in commitment.get("files", {}).items():
        candidate = (root / str(relative)).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"source path escapes root: {relative}")
        if sha256_file(candidate) != expected:
            raise ValueError(f"source file hash mismatch: {relative}")
    for row in commitment.get("pairs", []):
        for prefix in ("fixed", "selected"):
            candidate = (root / str(row[f"{prefix}_input_relative_path"])).resolve()
            if root not in candidate.parents:
                raise ValueError("coordinate path escapes root")
            if sha256_file(candidate) != row[f"{prefix}_input_sha256"]:
                raise ValueError(
                    f"coordinate hash mismatch: {row.get('entity_uid')}:{row.get('support_index')}:{prefix}"
                )
    return commitment


def preflight(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    primary_root = (root / args.primary_root).resolve()
    if primary_root != (root / "data/BioEmu").resolve():
        raise ValueError("K32 preflight primary root is not canonical")
    commitment = verify_commitment(root, args.source_commitment)
    plan = load_plan(root / PLAN_RELATIVE)
    supports = [int(value) for value in plan["support_indices"]]
    parent = json.loads((root / PARENT_RELATIVE).read_text())
    sequences = {str(row["entity_uid"]): str(row["sequence"]) for row in parent["entities"]}
    def verify_row(row: dict[str, Any]) -> dict[str, Any] | None:
        selected_path, selected_index, fixed_audit, selected_audit = select_source(
            primary_root,
            str(row["bmrb_id"]),
            int(row["support_index"]),
            set(supports),
            sequences[str(row["entity_uid"])],
        )
        if selected_index != int(row["selected_source_index"]):
            raise ValueError("preflight replacement index mismatch")
        if sha256_file(selected_path) != row["selected_input_sha256"]:
            raise ValueError("preflight selected hash mismatch")
        if fixed_audit != row["fixed_input_audit"]:
            raise ValueError("preflight fixed audit mismatch")
        if selected_audit != row["selected_input_audit"]:
            raise ValueError("preflight selected audit mismatch")
        if not selected_audit["pass"]:
            raise ValueError("preflight selected an invalid support")
        if row["replacement_required"]:
            return {
                "entity_uid": str(row["entity_uid"]),
                "support_index": int(row["support_index"]),
                "selected_source_index": selected_index,
            }
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        observed_replacements = [
            replacement
            for replacement in executor.map(verify_row, commitment["pairs"])
            if replacement is not None
        ]
    receipt = {
        "contract": PREFLIGHT_CONTRACT,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "source_commitment_sha256": sha256_file(args.source_commitment),
        "pair_count": len(commitment["pairs"]),
        "replacement_count": len(observed_replacements),
        "legacy_k8_replacement_count": sum(
            row["support_index"] in set(plan["legacy_k8_indices"])
            for row in observed_replacements
        ),
        "replacements": observed_replacements,
    }
    write_json_new(args.receipt, receipt)
    print(
        json.dumps(
            {
                "preflight_receipt_sha256": sha256_file(args.receipt),
                "replacement_count": len(observed_replacements),
            },
            sort_keys=True,
        )
    )
    return 0


def consume_authorization(
    args: argparse.Namespace,
    *,
    source_commitment_sha256: str,
    preflight_receipt_sha256: str,
) -> dict[str, str]:
    git_dir = args.authorization_git_dir.resolve()
    authorization_bytes = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "blob", args.authorization_git_blob],
        check=True,
        capture_output=True,
    ).stdout
    actual_blob = hashlib.sha1(  # noqa: S324 - Git blob identity is the contract.
        f"blob {len(authorization_bytes)}\0".encode() + authorization_bytes
    ).hexdigest()
    if actual_blob != args.authorization_git_blob:
        raise ValueError("authorization Git blob identity mismatch")
    resolved = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", f"{args.authorization_ref}^{{blob}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != args.authorization_git_blob:
        raise ValueError("external authorization ref is not fresh")
    authorization = json.loads(authorization_bytes)
    expected = {
        "contract": AUTHORIZATION_CONTRACT,
        "authorized": True,
        "authorization_ref": args.authorization_ref,
        "output_relative_path": str(args.output_root.relative_to(args.root)),
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "slurm_job_id": str(args.slurm_job_id),
        "source_commitment_sha256": source_commitment_sha256,
    }
    if authorization != expected:
        raise ValueError("K32 support authorization payload mismatch")
    authorization_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    consumption_ref = (
        "refs/atypemu-consumptions/k32-support/" + args.authorization_git_blob
    )
    claim = {
        "contract": EXTERNAL_CLAIM_CONTRACT,
        "authorization_git_blob": args.authorization_git_blob,
        "authorization_ref": args.authorization_ref,
        "authorization_sha256": authorization_sha256,
        "consumption_ref": consumption_ref,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "slurm_job_id": str(args.slurm_job_id),
        "source_commitment_sha256": source_commitment_sha256,
    }
    write_json_new(args.external_claim, claim)
    claim_blob = subprocess.run(
        ["git", f"--git-dir={git_dir}", "hash-object", "-w", str(args.external_claim)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "update-ref",
            consumption_ref,
            claim_blob,
            "0" * 40,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", f"--git-dir={git_dir}", "update-ref", args.authorization_ref, claim_blob, args.authorization_git_blob],
        check=True,
        capture_output=True,
        text=True,
    )
    consumed = {
        "contract": CONSUMED_CONTRACT,
        "authorization_git_blob": args.authorization_git_blob,
        "authorization_ref": args.authorization_ref,
        "authorization_sha256": authorization_sha256,
        "consumption_ref": consumption_ref,
        "external_claim_git_blob": claim_blob,
        "external_claim_sha256": sha256_file(args.external_claim),
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "slurm_job_id": str(args.slurm_job_id),
        "source_commitment_sha256": source_commitment_sha256,
    }
    write_json_new(args.consumed_authorization, consumed)
    return consumed


def materialize(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    args.root = root
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    for generated in (args.receipt, args.consumed_authorization, args.external_claim):
        if generated.exists():
            raise FileExistsError(f"generated artifact already exists: {generated}")
    commitment = verify_commitment(root, args.source_commitment)
    plan = load_plan(root / PLAN_RELATIVE)
    if args.output_root != (root / plan["outputs"]["coordinate_root"]).resolve():
        raise ValueError("materialization output root differs from frozen plan")
    source_sha256 = sha256_file(args.source_commitment)
    preflight_receipt = json.loads(args.preflight_receipt.read_text())
    if preflight_receipt.get("contract") != PREFLIGHT_CONTRACT:
        raise ValueError("K32 preflight contract mismatch")
    for field in (
        "target_values_read",
        "source_gate_authorized",
        "formal_evaluation_authorized",
    ):
        if preflight_receipt.get(field) is not False:
            raise ValueError(f"K32 preflight scope mismatch: {field}")
    if preflight_receipt.get("source_commitment_sha256") != source_sha256:
        raise ValueError("K32 preflight source binding mismatch")
    if preflight_receipt.get("pair_count") != commitment["pair_count"]:
        raise ValueError("K32 preflight pair count mismatch")
    if preflight_receipt.get("replacement_count") != commitment["replacement_count"]:
        raise ValueError("K32 preflight replacement count mismatch")
    preflight_sha256 = sha256_file(args.preflight_receipt)
    consume_authorization(
        args,
        source_commitment_sha256=source_sha256,
        preflight_receipt_sha256=preflight_sha256,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    for row in commitment["pairs"]:
        source = root / row["selected_input_relative_path"]
        output = (
            args.output_root
            / str(row["bmrb_id"])
            / f"{row['bmrb_id']}_BioEmu_{row['support_index']}.pdb"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, output.open("xb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle)
        output_sha256 = sha256_file(output)
        if output_sha256 != row["selected_input_sha256"]:
            raise ValueError("materialized coordinate bytes changed")
        outputs.append(
            {
                "bmrb_id": str(row["bmrb_id"]),
                "entity_uid": str(row["entity_uid"]),
                "support_id": str(row["support_id"]),
                "support_index": int(row["support_index"]),
                "selected_source_index": int(row["selected_source_index"]),
                "selected_input_sha256": str(row["selected_input_sha256"]),
                "output_relative_path": str(output.relative_to(root)),
                "output_sha256": output_sha256,
                "disposition": (
                    "deterministic_target_unread_replacement"
                    if row["replacement_required"]
                    else "byte_identical_fixed_support"
                ),
            }
        )
    receipt = {
        "contract": RECEIPT_CONTRACT,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "source_commitment_sha256": source_sha256,
        "preflight_receipt_sha256": preflight_sha256,
        "consumed_authorization_sha256": sha256_file(args.consumed_authorization),
        "external_claim_sha256": sha256_file(args.external_claim),
        "slurm_job_id": str(args.slurm_job_id),
        "entity_count": 162,
        "support_count": 32,
        "pair_count": len(outputs),
        "replacement_count": sum(
            row["disposition"] == "deterministic_target_unread_replacement"
            for row in outputs
        ),
        "legacy_k8_replacement_count": sum(
            row["disposition"] == "deterministic_target_unread_replacement"
            and row["support_index"] in set(commitment["legacy_k8_indices"])
            for row in outputs
        ),
        "outputs": outputs,
    }
    write_json_new(args.receipt, receipt)
    print(
        json.dumps(
            {
                "pair_count": len(outputs),
                "receipt_sha256": sha256_file(args.receipt),
                "replacement_count": receipt["replacement_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--plan", type=Path, default=Path(PLAN_RELATIVE))
    freeze_parser.add_argument("--parent-commitment", type=Path, default=Path(PARENT_RELATIVE))
    freeze_parser.add_argument("--primary-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--root", type=Path, required=True)
    preflight_parser.add_argument("--source-commitment", type=Path, required=True)
    preflight_parser.add_argument("--primary-root", type=Path, required=True)
    preflight_parser.add_argument("--receipt", type=Path, required=True)
    materialize_parser = commands.add_parser("materialize")
    materialize_parser.add_argument("--root", type=Path, required=True)
    materialize_parser.add_argument("--source-commitment", type=Path, required=True)
    materialize_parser.add_argument("--preflight-receipt", type=Path, required=True)
    materialize_parser.add_argument("--output-root", type=Path, required=True)
    materialize_parser.add_argument("--receipt", type=Path, required=True)
    materialize_parser.add_argument("--authorization-git-dir", type=Path, required=True)
    materialize_parser.add_argument("--authorization-ref", required=True)
    materialize_parser.add_argument("--authorization-git-blob", required=True)
    materialize_parser.add_argument("--consumed-authorization", type=Path, required=True)
    materialize_parser.add_argument("--external-claim", type=Path, required=True)
    materialize_parser.add_argument("--slurm-job-id", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "freeze":
        return freeze(args)
    if args.command == "preflight":
        return preflight(args)
    if args.command == "materialize":
        return materialize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
