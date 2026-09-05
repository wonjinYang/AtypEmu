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
    "3c73e1de62c2b125b89da62c7a36dd36803aaef094145afa5584caa0ccd416c7"
)
CATALOG_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_catalog_feasibility_receipt.json"
)
CATALOG_RECEIPT_SHA256 = (
    "9069f0eea7a4db213babbfc02d09512b5bfe00051a46d7ae723fa90c65c58828"
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
    "catalog_feasibility_receipt": {
        "path": str(CATALOG_RECEIPT_RELATIVE),
        "sha256": CATALOG_RECEIPT_SHA256,
    },
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
    "32": "COUNT_FEASIBLE_EXISTING_SEED_REQUIRES_NEW_PROVENANCE_BINDING",
    "128": "COUNT_FEASIBLE_UNQUALIFIED_DIVERSITY_PHYSICALITY_NESTING",
    "768": "COUNT_FEASIBLE_UNQUALIFIED_DIVERSITY_PHYSICALITY_NESTING",
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
    "checker_scope": "future source-score arithmetic only, distinct from catalog integrity checking",
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
    "K128 and K768 are count-feasible but diversity, complete-coordinate physicality, and exact nesting are unqualified",
    "K32 original structural-source provenance replay is not established for this study",
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
        and feasibility.get("count_feasibility_definition")
        == "per entity using heavy-topology-compatible exact-heavy-coordinate-unique supports"
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
        and diversity.get("audit_scope") == "every support at every ladder level",
        "diversity readiness must fail closed until thresholds are frozen",
        checks,
        "diversity_hold",
    )
    _require(
        diversity.get("canonical_heavy_atom_identity_must_match_within_entity")
        is True
        and diversity.get(
            "hydrogen_topology_variation_requires_explicit_per_support_receipt"
        )
        is True,
        "heavy-atom topology or hydrogen-variant audit rule changed",
        checks,
        "topology_audit_contract",
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
            "catalog_feasibility_receipt",
            "current_k32_plan",
            "current_selector",
            "jeon_thesis",
        },
        "bound target-unread planning evidence mismatch",
        checks,
        "evidence_hashes",
    )
    receipt = json.loads((root / CATALOG_RECEIPT_RELATIVE).read_text())
    _require(
        receipt.get("contract")
        == "atypemu_nested_support_count_v1_catalog_feasibility_receipt_v1"
        and receipt.get("study_id") == "atypemu_nested_support_count_v1"
        and receipt.get("status") == "COUNT_FEASIBILITY_ONLY_HOLD"
        and receipt.get("artifact_kind")
        == "target_unread_count_feasibility_receipt_not_authorization",
        "catalog receipt identity or HOLD status changed",
        checks,
        "catalog_receipt_identity",
    )
    _require(
        receipt.get("target_values_read") is False
        and receipt.get("source_scores_read") is False
        and receipt.get("outer_or_formal_metrics_opened") is False
        and receipt.get("science_executed") is False
        and receipt.get("authorization_consumed") is False,
        "catalog receipt crossed a forbidden science boundary",
        checks,
        "catalog_receipt_target_score_authorization_unread",
    )
    catalog = receipt.get("catalog", {})
    _require(
        catalog.get("entity_count") == 135
        and catalog.get("count_feasibility_definition")
        == "per entity using heavy-topology-compatible exact-heavy-coordinate-unique supports"
        and catalog.get("canonical_filename_count_minimum") == 991
        and catalog.get("canonical_filename_count_maximum") == 1000
        and catalog.get("total_canonical_filename_count") == 134850
        and catalog.get("heavy_topology_compatible_unique_coordinate_count")
        == 134850
        and catalog.get("unexpected_entry_count") == 0,
        "catalog count or topology inventory changed",
        checks,
        "catalog_count_inventory",
    )
    _require(
        catalog.get("entities_with_hydrogen_topology_variation") == 108
        and catalog.get("all_atom_topology_variant_maximum") == 89,
        "hydrogen-topology variation evidence changed",
        checks,
        "catalog_hydrogen_variation_inventory",
    )
    expected_feasibility = {
        "32": {
            "all_entities_count_feasible": True,
            "entity_count_feasible": 135,
            "maximum_entity_shortfall": 0,
            "total_shortfall": 0,
        },
        "128": {
            "all_entities_count_feasible": True,
            "entity_count_feasible": 135,
            "maximum_entity_shortfall": 0,
            "total_shortfall": 0,
        },
        "768": {
            "all_entities_count_feasible": True,
            "entity_count_feasible": 135,
            "maximum_entity_shortfall": 0,
            "total_shortfall": 0,
        },
        "1536": {
            "all_entities_count_feasible": False,
            "entity_count_feasible": 0,
            "maximum_entity_shortfall": 545,
            "total_shortfall": 72510,
        },
    }
    _require(
        receipt.get("level_count_feasibility") == expected_feasibility,
        "support-level count feasibility changed",
        checks,
        "exact_level_count_feasibility",
    )
    claims = receipt.get("claims", {})
    _require(
        claims.get("count_feasible_levels") == [32, 128, 768]
        and claims.get("source_method_composition") == {"BioEmu": 1.0}
        and claims.get("adequate_conformer_sampling_established") is False
        and claims.get("diversity_qualified") is False
        and claims.get("k32_original_structural_source_replay_established") is False
        and claims.get("physicality_qualified") is False
        and claims.get("support_materialized") is False
        and claims.get("scientific_gate_authorized") is False,
        "count-only claim limitations changed",
        checks,
        "catalog_claim_limits",
    )
    summary = receipt.get("catalog_summary", {})
    executions = receipt.get("executions", {})
    summary_sha256 = "737aaff560041ab7b64a750d011469a43563ab8a307af518eea3b98903f99b41"
    _require(
        summary.get("sha256") == summary_sha256
        and executions.get("yulab_mac_studio", {}).get(
            "aggregate_summary_sha256"
        )
        == summary_sha256
        and executions.get("iremb_slurm", {}).get("aggregate_summary_sha256")
        == summary_sha256,
        "independent catalog executions do not agree byte-for-byte",
        checks,
        "independent_execution_byte_agreement",
    )
    _require(
        executions.get("yulab_mac_studio", {}).get("completed_shard_tasks") == 27
        and executions.get("iremb_slurm", {}).get("completed_shard_tasks") == 27
        and executions.get("netbird_gateway_checker", {}).get("status") == "PASS"
        and executions.get("netbird_gateway_checker", {}).get("scope")
        == "sealed catalog integrity and count arithmetic only, not a scientific gate checker"
        and executions.get("netbird_gateway_checker", {}).get("checker_checks")
        == 2029620,
        "distributed execution or independent checker evidence incomplete",
        checks,
        "distributed_execution_and_checker_complete",
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
