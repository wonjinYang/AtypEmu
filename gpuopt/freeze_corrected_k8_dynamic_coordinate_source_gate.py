#!/usr/bin/env python3
"""Create the target-value-unread commitment for the corrected K8 source gate."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpuopt.candidates.corrected_k8_dynamic_coordinate import (  # noqa: E402
    embedding_path,
    sha256_file,
)


STATIC_FILES = (
    ".auto/checks.sh",
    ".auto/measure_body.sh",
    "gpuopt/candidates/__init__.py",
    "gpuopt/candidates/corrected_k8_dynamic_coordinate.py",
    "gpuopt/source_gate_eligibility.py",
    "gpuopt/preflight_all_label_autoresearch.py",
    "gpuopt/freeze_corrected_k8_dynamic_coordinate_source_gate.py",
    "gpuopt/run_corrected_k8_dynamic_coordinate_source_gate.py",
    "gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py",
    "gpuopt/slurm/run_corrected_k8_dynamic_coordinate_source_gate_l40s.sbatch",
    "gpuopt/tests/test_corrected_k8_dynamic_coordinate_source_gate.py",
    ".auto/preunblind/corrected_k8_dynamic_coordinate_plan.json",
    ".auto/frozen/all_label_inventory.json",
    "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/post_handoff_audit.json",
    "data/all_atom_observer_v1/commitment.json",
    "data/all_atom_observer_v1/feature_receipt.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/receipt.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/A/predictions.parquet",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/A/summary.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/B/predictions.parquet",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/B/summary.json",
)
EXPECTED_SUPPORT_IDS = tuple(
    f"BioEmu_{index}" for index in (1, 126, 251, 376, 501, 626, 751, 876)
)
CANONICAL_INVALIDATION_SHA256 = (
    "5a84b9ea624694976e3ff31548437fae8e0f8856d19698fd77b2997d3d96152d"
)


def collect_source_paths(root: Path) -> tuple[set[Path], list[dict]]:
    data_root = root / "data/all_atom_observer_v1"
    commitment = json.loads((data_root / "commitment.json").read_text())
    entities = [
        entity
        for entity in commitment["entities"]
        if entity.get("split") == "train" and entity.get("observer_fold") in {"A", "B"}
    ]
    paths = {root / relative for relative in STATIC_FILES}
    embedding_cache = root / "data/all_atom_shared_q_e2e_v0/esm2_cache"
    structure_root = root / "data/BioEmu"
    for entity in entities:
        bmrb_id = str(entity["bmrb_id"])
        support_ids = tuple(str(value) for value in entity["support_ids"])
        if support_ids != EXPECTED_SUPPORT_IDS:
            raise ValueError(f"source support roster differs from frozen K=8: {bmrb_id}")
        paths.add(data_root / "features" / f"{bmrb_id}.parquet")
        paths.add(data_root / "targets" / f"{bmrb_id}.parquet")
        paths.add(embedding_path(embedding_cache, str(entity["sequence"])))
        for support_id in support_ids:
            paths.add(
                structure_root / bmrb_id / f"{bmrb_id}_{support_id}.pdb"
            )
    paths.update(
        (root / "data/all_atom_shared_q_e2e_v0/ucbshift_x_anchor_v0").glob("*.npz")
    )
    return paths, entities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths, entities = collect_source_paths(root)
    invalidation_path = root / (
        "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/"
        "post_handoff_audit.json"
    )
    if sha256_file(invalidation_path) != CANONICAL_INVALIDATION_SHA256:
        raise ValueError("canonical Job 134710 invalidation audit mismatch")
    ordered = sorted(paths)
    missing = [path for path in ordered if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing committed sources: {missing[:5]}")
    with ThreadPoolExecutor(max_workers=16) as executor:
        digests = list(executor.map(sha256_file, ordered))
    files = {
        str(path.relative_to(root)): digest
        for path, digest in zip(ordered, digests, strict=True)
    }
    receipt = {
        "contract": "corrected_k8_dynamic_coordinate_source_gate_commitment_v1",
        "invalidated_parent_job": "134710",
        "invalidated_parent_disposition": "INVALIDATED",
        "canonical_invalidation_sha256": CANONICAL_INVALIDATION_SHA256,
        "formal_evaluation_authorized": False,
        "target_values_opened": False,
        "source_entity_count": len(entities),
        "support_ids": list(EXPECTED_SUPPORT_IDS),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = args.output.open("x")
    with descriptor:
        json.dump(receipt, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")
    print(f"committed {len(files)} files for {len(entities)} source entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
