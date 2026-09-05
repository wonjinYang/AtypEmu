"""Validate the HOLD-only exact-nested support-count feasibility plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json"
)
LEVELS = [32, 128, 768, 1536]
CANDIDATES = {
    "32": "atypemu_nested_support_count_v1_k32_seed",
    "128": "atypemu_nested_support_count_v1_k128",
    "768": "atypemu_nested_support_count_v1_k768",
    "1536": "atypemu_nested_support_count_v1_k1536",
}
FALSE_CAPABILITIES = {
    "formal_metrics": False,
    "outer_metrics": False,
    "science_execution": False,
    "source_scores": False,
    "target_value_deserialization": False,
}
PLAN_CANONICAL_SHA256 = (
    "e00bd9f36707b09df7528e9a1cd9a17cf2f2f99a05df1d7de1ce481a41f9776e"
)
TOP_LEVEL_FIELDS = {
    "artifact_kind",
    "authorization",
    "candidate_ids",
    "capabilities",
    "contract",
    "controls",
    "directional_source_crossfit",
    "diversity_audit",
    "evidence",
    "fixed_protocol",
    "interpretation",
    "ladder",
    "provenance",
    "readiness",
    "replication",
    "source_feasibility",
    "state",
    "study_id",
    "target_unread_scope",
}
SAFE_EVIDENCE = {
    "current_k32_plan": {
        "path": "gpuopt/preunblind/k32_nested_k8_source_gate_plan_v1.json",
        "sha256": "9ce0b48e8c4246ef3faf052460a486a97d02c8728b76cb379c019e012a34f50b",
    },
    "current_selector": {
        "path": "gpuopt/materialize_k32_complete_coordinate_supports.py",
        "sha256": "1123f6a9c725c29e07f312e657481a309a9bdb759166811f1ab50ce1aa6233fe",
    },
    "jeon_thesis": {
        "path": "references/thesis_jeon.pdf",
        "sha256": "8a51ff19002dfc028ed43fe99b39c5d363024b91ce4c1398a0246d8952cd925e",
    },
}
EXPECTED_LEVEL_STATUS = {
    "32": "EXISTING_SEED_REQUIRES_NEW_PROVENANCE_BINDING",
    "128": "UNQUALIFIED",
    "768": "UNQUALIFIED",
    "1536": "BLOCKED_BY_CURRENT_SOURCE_CAPACITY",
}
EXPECTED_FIXED_PROTOCOL = {
    "assimilation_learning_rate": 0.08,
    "assimilation_steps": 100,
    "eligibility": "frozen inventory plus at least two finite rows and target variance above 1e-15, applied before normalization, fitting, and assimilation",
    "entry_shared_q": "one softmax simplex per entry and arm across every eligible label",
    "loss": "atom-balanced normalized SmoothL1 observer fit and normalized MSE assimilation; no CCC/concordance surrogate",
    "observer_batch_size": 4096,
    "observer_epochs": 1024,
    "observer_learning_rate": 0.002,
    "q_kl_weight": 0.003,
    "reference_regularizer_weight": 0.003,
    "torsion_regularizer_weight": 0.03,
    "training_normalization_only": True,
}
EXPECTED_CONTROLS = {
    "lower_k": "exact immediate-predecessor support projection with independently fitted observer and fresh assimilation state",
    "no_coordinate": "same K roster with coordinate actuation disabled and independently optimized q/reference state",
    "state_reuse_between_arms_allowed": False,
}
EXPECTED_CROSSFIT = {
    "assignment": "sort descending whole-sequence-cluster size then lexical cluster ID; assign least-loaded half then half index",
    "cluster_overlap_allowed": False,
    "directions": {
        "A_to_B": "source observer fold A only",
        "B_to_A": "source observer fold B only",
    },
    "held_halves": [0, 1],
    "independent_arithmetic_checker_required": True,
}
EXPECTED_PROVENANCE = {
    "allowed_construction_inputs": [
        "target-unread coordinate catalogs",
        "coordinate PDB bytes",
        "sequence-cluster metadata",
    ],
    "forbidden_construction_inputs": [
        "chemical-shift target values",
        "source scores",
        "development scores",
        "outer scores",
        "formal scores",
        "target-derived support selection",
    ],
    "k32_seed_must_replay_to_original_structural_sources": True,
    "mere_hash_rebinding_is_sufficient": False,
}
EXPECTED_BLOCKERS = [
    "K1536 exceeds the only currently evidenced BioEmu index namespace",
    "K128 and K768 complete-coordinate inventories are not yet qualified",
    "support construction and diversity thresholds are not frozen",
    "primary and replicate ladder roots are not committed",
]


def _require(condition: bool, message: str, checks: list[str], name: str) -> None:
    if not condition:
        raise ValueError(message)
    checks.append(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_plan(plan: dict[str, Any], root: Path) -> list[str]:
    """Return named checks; never opens targets, scores, or authorization state."""
    checks: list[str] = []
    _require(
        set(plan) == TOP_LEVEL_FIELDS
        and _canonical_sha256(plan) == PLAN_CANONICAL_SHA256,
        "plan schema or value differs from the exact HOLD contract",
        checks,
        "exact_plan_schema_and_values",
    )
    _require(
        plan.get("artifact_kind")
        == "hold_feasibility_plan_not_authorization_or_execution_receipt",
        "plan could be mistaken for an authorization or execution receipt",
        checks,
        "not_an_authorization_or_execution_receipt",
    )
    _require(
        plan.get("contract") == "atypemu_nested_support_count_feasibility_plan_v1"
        and plan.get("study_id") == "atypemu_nested_support_count_v1"
        and plan.get("state") == "HOLD_FEASIBILITY_BLOCKED",
        "invalid support-count plan identity or state",
        checks,
        "identity_and_hold_state",
    )
    _require(
        plan.get("capabilities") == FALSE_CAPABILITIES,
        "support-count HOLD capabilities changed",
        checks,
        "all_science_capabilities_disabled",
    )
    ladder = plan.get("ladder", {})
    _require(
        ladder.get("levels") == LEVELS,
        "support-count ladder must be exactly 32/128/768/1536",
        checks,
        "exact_ladder",
    )
    _require(
        ladder.get("larger_contains_lower")
        == "ordered support IDs and coordinate bytes exactly",
        "support nesting is not exact and byte-identical",
        checks,
        "exact_nesting_rule",
    )
    _require(
        plan.get("candidate_ids") == CANDIDATES
        and len(set(CANDIDATES.values())) == len(LEVELS),
        "candidate IDs are not unique and K-specific",
        checks,
        "k_specific_candidates",
    )
    feasibility = plan.get("source_feasibility", {})
    _require(
        feasibility.get("currently_evidenced_coordinate_methods") == ["BioEmu"]
        and feasibility.get("currently_evidenced_index_min") == 1
        and feasibility.get("currently_evidenced_index_max") == 1000
        and feasibility.get("level_status") == EXPECTED_LEVEL_STATUS,
        "current target-unread coordinate-source evidence changed",
        checks,
        "current_source_capacity_and_all_level_statuses",
    )
    _require(
        feasibility.get("level_status", {}).get("1536")
        == "BLOCKED_BY_CURRENT_SOURCE_CAPACITY"
        and feasibility.get(
            "new_target_unread_generation_or_coordinate_source_required_for_k1536"
        )
        is True,
        "K1536 must remain blocked by the current 1000-frame source namespace",
        checks,
        "k1536_capacity_block",
    )
    _require(
        plan.get("fixed_protocol") == EXPECTED_FIXED_PROTOCOL,
        "fixed observer/loss/assimilation protocol drifted",
        checks,
        "fixed_protocol",
    )
    _require(
        plan.get("controls") == EXPECTED_CONTROLS,
        "matched controls are not independently optimized",
        checks,
        "independent_matched_controls",
    )
    diversity = plan.get("diversity_audit", {})
    _require(
        diversity.get("state") == "BLOCKED_PENDING_TARGET_UNREAD_THRESHOLDS"
        and diversity.get("near_duplicate_and_coverage_thresholds") == "UNFROZEN"
        and diversity.get("exact_coordinate_duplicate_count_max_per_entity") == 0
        and diversity.get("all_supports_audited") is True,
        "diversity readiness must fail closed until thresholds are frozen",
        checks,
        "diversity_hold",
    )
    required_metrics = {
        "source_method_composition",
        "coordinate_sha256",
        "rigid_aligned_ca_rmsd",
        "backbone_torsion_occupancy",
        "sidechain_torsion_occupancy",
        "contact_map_occupancy",
        "radius_of_gyration",
        "secondary_structure_occupancy",
    }
    _require(
        set(diversity.get("metrics", [])) == required_metrics,
        "target-unread diversity audit roster changed",
        checks,
        "diversity_metric_roster",
    )
    _require(
        plan.get("provenance") == EXPECTED_PROVENANCE,
        "K32 transitive provenance or target-selection exclusion weakened",
        checks,
        "transitive_target_unread_provenance",
    )
    _require(
        plan.get("directional_source_crossfit") == EXPECTED_CROSSFIT,
        "direction-specific cluster-disjoint crossfit changed",
        checks,
        "source_crossfit",
    )
    replication = plan.get("replication", {})
    _require(
        replication.get(
            "independently_generated_target_unread_replicate_required_for_general_sufficiency"
        )
        is True
        and replication.get("primary_and_replicate_commitments_must_differ") is True
        and replication.get("replicate_manifest_state") == "ABSENT",
        "general-sufficiency replicate requirement weakened",
        checks,
        "replicate_required",
    )
    authorization = plan.get("authorization", {})
    readiness = plan.get("readiness", {})
    _require(
        authorization.get("request_allowed_by_this_plan") is False
        and authorization.get("existing_k32_authorization_inherited") is False
        and authorization.get(
            "fresh_external_once_only_authorization_per_k_required"
        )
        is True
        and {readiness.get(key) for key in (
            "formal_evaluation",
            "materialization",
            "science_authorization",
            "science_execution",
        )}
        == {"HOLD"},
        "HOLD or fresh-authorization boundary weakened",
        checks,
        "authorization_and_execution_hold",
    )
    _require(
        readiness.get("blocking_conditions") == EXPECTED_BLOCKERS,
        "feasibility blocking conditions changed",
        checks,
        "exact_feasibility_blockers",
    )
    interpretation = plan.get("interpretation", {})
    _require(
        interpretation.get("k32_role")
        == "low-K coordinate/Jacobian causal-infrastructure smoke only"
        and interpretation.get("jeon_transfer_claim_allowed") is False
        and interpretation.get("general_sampling_sufficiency_established") is False
        and interpretation.get("one_ladder_establishes_general_sufficiency") is False,
        "K32, Jeon, or sampling-sufficiency interpretation drifted",
        checks,
        "interpretation_limits",
    )
    _require(
        "No chemical-shift target value or score is deserialized"
        in plan.get("target_unread_scope", ""),
        "target-unread planning boundary is missing",
        checks,
        "target_unread_boundary",
    )
    serialized = json.dumps(plan, sort_keys=True).lower()
    _require(
        ".parquet" not in serialized
        and "authorization_ref" not in serialized
        and "authorized\": true" not in serialized,
        "plan includes a forbidden data or authorization binding",
        checks,
        "forbidden_bindings_absent",
    )
    evidence = plan.get("evidence", {})
    _require(
        evidence == SAFE_EVIDENCE,
        "evidence path is outside the exact target-unread allowlist",
        checks,
        "safe_evidence_allowlist",
    )
    evidence_ok = True
    for binding in SAFE_EVIDENCE.values():
        relative = Path(str(binding.get("path", "")))
        path = (root / relative).resolve()
        evidence_ok &= (
            not relative.is_absolute()
            and root.resolve() in path.parents
            and path.is_file()
            and _sha256(path) == binding.get("sha256")
        )
    _require(
        evidence_ok and set(evidence) == {
            "current_k32_plan",
            "current_selector",
            "jeon_thesis",
        },
        "bound target-unread planning evidence mismatch",
        checks,
        "evidence_hashes",
    )
    return checks


def self_test(plan: dict[str, Any], root: Path) -> int:
    tamper_cases = (
        (("capabilities", "source_scores"), True),
        (("ladder", "levels"), [32, 128, 768]),
        (("source_feasibility", "currently_evidenced_index_max"), 1536),
        (("source_feasibility", "level_status", "128"), "QUALIFIED"),
        (("diversity_audit", "near_duplicate_and_coverage_thresholds"), {}),
        (("authorization", "request_allowed_by_this_plan"), True),
        (("replication", "replicate_manifest_state"), "PRESENT"),
        (("fixed_protocol", "q_kl_weight"), 0.0),
        (("directional_source_crossfit", "cluster_overlap_allowed"), True),
        (("evidence", "current_selector", "path"), "data/targets/source.parquet"),
    )
    for keys, value in tamper_cases:
        tampered = copy.deepcopy(plan)
        cursor: dict[str, Any] = tampered
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        try:
            validate_plan(tampered, root)
        except ValueError as error:
            if "schema or value" not in str(error):
                raise AssertionError(str(error)) from error
        else:
            raise AssertionError(f"tampered plan accepted: {keys}")
    extra = copy.deepcopy(plan)
    extra["science_ready"] = True
    try:
        validate_plan(extra, root)
    except ValueError as error:
        if "schema or value" not in str(error):
            raise AssertionError(str(error)) from error
    else:
        raise AssertionError("unknown top-level field accepted")
    return len(tamper_cases) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=PLAN_RELATIVE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = json.loads(path.read_text())
    checks = validate_plan(plan, root)
    negative_checks = self_test(plan, root) if args.self_test else 0
    if not args.acknowledge_hold_only:
        print("STATUS HOLD_FEASIBILITY_BLOCKED")
        print("REFUSAL this artifact is not science or authorization clearance")
        return 3
    print(f"METRIC support_count_plan_checks={len(checks) + negative_checks}")
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_FEASIBILITY_BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
