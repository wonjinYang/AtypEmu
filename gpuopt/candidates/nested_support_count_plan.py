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


def validate_plan(plan: dict[str, Any], root: Path) -> list[str]:
    """Return named checks; never opens targets, scores, or authorization state."""
    checks: list[str] = []
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
        and feasibility.get("currently_evidenced_index_max") == 1000,
        "current target-unread coordinate-source evidence changed",
        checks,
        "current_source_capacity",
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
    fixed = plan.get("fixed_protocol", {})
    _require(
        fixed.get("observer_epochs") == 1024
        and fixed.get("observer_learning_rate") == 0.002
        and fixed.get("observer_batch_size") == 4096
        and fixed.get("assimilation_steps") == 100
        and fixed.get("assimilation_learning_rate") == 0.08
        and fixed.get("training_normalization_only") is True
        and "no CCC" in fixed.get("loss", ""),
        "fixed observer/loss/assimilation protocol drifted",
        checks,
        "fixed_protocol",
    )
    controls = plan.get("controls", {})
    _require(
        controls.get("state_reuse_between_arms_allowed") is False
        and "independently" in controls.get("no_coordinate", "")
        and "independently" in controls.get("lower_k", ""),
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
    provenance = plan.get("provenance", {})
    _require(
        provenance.get("k32_seed_must_replay_to_original_structural_sources") is True
        and provenance.get("mere_hash_rebinding_is_sufficient") is False
        and "target-derived support selection"
        in provenance.get("forbidden_construction_inputs", []),
        "K32 transitive provenance or target-selection exclusion weakened",
        checks,
        "transitive_target_unread_provenance",
    )
    crossfit = plan.get("directional_source_crossfit", {})
    _require(
        crossfit.get("cluster_overlap_allowed") is False
        and crossfit.get("held_halves") == [0, 1]
        and crossfit.get("independent_arithmetic_checker_required") is True
        and set(crossfit.get("directions", {})) == {"A_to_B", "B_to_A"},
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
    evidence = plan.get("evidence", {})
    evidence_ok = True
    for binding in evidence.values():
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
    serialized = json.dumps(plan, sort_keys=True).lower()
    _require(
        ".parquet" not in serialized
        and "authorization_ref" not in serialized
        and "authorized\": true" not in serialized,
        "plan includes a forbidden data or authorization binding",
        checks,
        "forbidden_bindings_absent",
    )
    return checks


def self_test(plan: dict[str, Any], root: Path) -> int:
    expected_messages = (
        (("capabilities", "source_scores"), True, "capabilities changed"),
        (("ladder", "levels"), [32, 128, 768], "ladder"),
        (("source_feasibility", "currently_evidenced_index_max"), 1536, "evidence changed"),
        (("diversity_audit", "near_duplicate_and_coverage_thresholds"), {}, "thresholds"),
        (("authorization", "request_allowed_by_this_plan"), True, "HOLD"),
        (("replication", "replicate_manifest_state"), "PRESENT", "replicate"),
    )
    for keys, value, message in expected_messages:
        tampered = copy.deepcopy(plan)
        cursor: dict[str, Any] = tampered
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        try:
            validate_plan(tampered, root)
        except ValueError as error:
            if message.lower() not in str(error).lower():
                raise AssertionError((message, str(error))) from error
        else:
            raise AssertionError(f"tampered plan accepted: {keys}")
    return len(expected_messages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=PLAN_RELATIVE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = json.loads(path.read_text())
    checks = validate_plan(plan, root)
    negative_checks = self_test(plan, root) if args.self_test else 0
    print(f"METRIC support_count_plan_checks={len(checks) + negative_checks}")
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_FEASIBILITY_BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
