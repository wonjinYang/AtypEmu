"""Validate the HOLD-only solution-state protonation support plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_solution_state_protonation_support_plan_v1.json"
)
CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_solution_state_protonation_support_plan.py"
)
PLAN_SHA256 = "292a2d7f793286e5317cca34e41a1aed214be608da17f687dc101435a7b3f4b0"
PLAN_RAW_SHA256 = "16618f36d5979b7facbc9803d23fc7152c967af14be202c485e5f974d4f63422"
TOP_LEVEL_FIELDS = {
    "artifact_kind",
    "authorization",
    "candidate_id",
    "closed_capabilities",
    "cohort_policy",
    "condition_manifest",
    "contract",
    "exact_nesting",
    "forbidden_source_builder_inputs",
    "microstate_policy",
    "parent_inputs",
    "physicality_gate",
    "required_stage_order",
    "state",
    "study_id",
    "unresolved_blockers",
}
FALSE_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read_by_source_builder": False,
    "target_values_read": False,
}
REQUIRED_FORBIDDEN_INPUTS = {
    "assigned_atom_inventory",
    "chemical_shift_targets",
    "eligibility_rows",
    "existing_all_atom_blocker_receipt",
    "source_or_outer_scores",
    "target_atom_identities",
    "target_values",
}
EXPECTED_STAGE_ORDER = [
    "freeze_condition_manifest",
    "freeze_policy_generator_checker_environment_and_parent_inputs",
    "independent_cold_review",
    "create_source_commitment",
    "construct_complete_supports",
    "independent_raw_coordinate_and_physicality_replay",
    "freeze_target_unread_support_order_and_exact_nesting",
    "identity_only_coverage_recount",
    "separate_science_commitment_review_and_authorization",
]
EXPECTED_CONDITION_MANIFEST = {
    "allowed_fields": [
        "entity_uid",
        "experiment_id",
        "sample_id",
        "sample_condition_list_id",
        "solution_ph",
        "solution_temperature_k",
        "solution_ionic_strength_mm",
        "source_relative_path",
        "source_sha256",
    ],
    "forbidden_fields": [
        "atom_id",
        "atom_name",
        "chemical_shift",
        "target_uid",
        "target_value",
    ],
    "linkage": (
        "Chem_shift_experiment.Experiment_ID through the entry-level Experiment "
        "loop to sample and sample-condition records"
    ),
    "missing_numeric_condition": "FAIL_ENTIRE_CANDIDATE_NO_FALLBACK",
    "path": "ABSENT_UNQUALIFIED",
    "sha256": "ABSENT_UNQUALIFIED",
    "state": "ABSENT_BLOCKS_SOURCE_CONSTRUCTION",
}
EXPECTED_BLOCKERS = [
    "condition manifest absent",
    "protonation algorithm force field version pruning and seed absent",
    "parent input manifests absent",
    "support ordering and diversity thresholds absent",
    "independent physicality checker absent",
    "K1536 remains separately blocked by current source capacity",
]
EXPECTED_PARENT_INPUTS = {
    "parent_structure_manifest": "ABSENT_UNQUALIFIED",
    "protein_sequence_manifest": "ABSENT_UNQUALIFIED",
    "required_parent_heavy_atom_identity": (
        "exact atom identity and coordinate hash preservation"
    ),
    "source_namespace": "data/solution_state_protonation_support_v1",
}
EXPECTED_MICROSTATE_POLICY = {
    "algorithm": "ABSENT_UNQUALIFIED",
    "complete_explicit_hydrogens_required": True,
    "force_field_and_version": "ABSENT_UNQUALIFIED",
    "post_recount_reweighting_allowed": False,
    "probabilities": (
        "finite positive proposal probabilities normalized across retained states "
        "before recount"
    ),
    "pruning_rule": "ABSENT_UNQUALIFIED",
    "seed_derivation": "ABSENT_UNQUALIFIED",
    "selection_inputs": [
        "parent_structure",
        "protein_sequence",
        "solution_conditions",
    ],
    "target_conditioned_state_selection_allowed": False,
}
EXPECTED_NESTING = {
    "count_unit": (
        "one completed coordinate support with one explicitly declared "
        "protonation microstate"
    ),
    "levels": [32, 128, 768],
    "prefix_identity": (
        "support IDs and coordinate bytes at lower K are exact prefixes of every "
        "higher K"
    ),
    "support_ordering": "ABSENT_UNQUALIFIED",
    "support_ordering_must_precede_target_inventory_access": True,
}
EXPECTED_PHYSICALITY = {
    "all_distinct_atom_minimum_distance_angstrom": 0.5,
    "checks": [
        "finite coordinates",
        "unique atom identities",
        "complete microstate hydrogens",
        "valid valence and covalent geometry",
        "backbone continuity",
        "parent heavy-atom identity and coordinate preservation",
    ],
    "independent_checker": "ABSENT_UNQUALIFIED",
    "must_cover_every_emitted_support": True,
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require(condition: bool, message: str, checks: list[str], name: str) -> None:
    if not condition:
        raise ValueError(message)
    checks.append(name)


def validate_plan(plan: dict[str, Any], *, enforce_hash: bool = True) -> list[str]:
    checks: list[str] = []
    _require(
        set(plan) == TOP_LEVEL_FIELDS,
        "plan top-level schema drifted",
        checks,
        "exact_top_level_schema",
    )
    if enforce_hash:
        _require(
            _canonical_sha256(plan) == PLAN_SHA256,
            "plan canonical hash drifted",
            checks,
            "whole_plan_hash",
        )
    _require(
        plan["artifact_kind"]
        == "hold_only_target_unread_solution_state_support_plan_not_authorization"
        and plan["contract"]
        == "atypemu_nested_support_count_v1_solution_state_protonation_support_plan_v1"
        and plan["candidate_id"]
        == "atypemu_nested_support_count_v1_solution_state_protonation_support_v1"
        and plan["study_id"] == "atypemu_nested_support_count_v1",
        "candidate identity drifted",
        checks,
        "candidate_identity",
    )
    _require(
        plan["state"] == "HOLD_PLAN_ONLY_REQUIRED_BINDINGS_ABSENT",
        "plan is not HOLD-only",
        checks,
        "hold_state",
    )
    _require(
        plan["closed_capabilities"] == FALSE_CAPABILITIES,
        "a protected capability was opened",
        checks,
        "closed_capabilities",
    )
    authorization = plan["authorization"]
    _require(
        authorization == {
            "consumption_allowed": False,
            "existing_authorizations_reusable": False,
            "fresh_external_authorization_required_after_all_gates": True,
        },
        "authorization boundary drifted",
        checks,
        "authorization_closed",
    )
    cohort = plan["cohort_policy"]
    _require(
        cohort == {
            "entity_count": 135,
            "entity_or_residue_exceptions": [],
            "missing_condition_record": "FAIL_ENTIRE_CANDIDATE",
            "uniform_policy_required": True,
        },
        "cohort-wide no-exception policy drifted",
        checks,
        "uniform_cohort_policy",
    )
    conditions = plan["condition_manifest"]
    _require(
        conditions == EXPECTED_CONDITION_MANIFEST,
        "condition-manifest fail-closed contract drifted",
        checks,
        "condition_manifest_absent_and_target_free",
    )
    _require(
        set(plan["forbidden_source_builder_inputs"]) == REQUIRED_FORBIDDEN_INPUTS,
        "source-builder forbidden inputs drifted",
        checks,
        "builder_target_isolation",
    )
    _require(
        plan["parent_inputs"] == EXPECTED_PARENT_INPUTS,
        "parent-input provenance drifted",
        checks,
        "parent_inputs_absent_and_separate",
    )
    _require(
        plan["microstate_policy"] == EXPECTED_MICROSTATE_POLICY,
        "microstate target-isolation contract drifted",
        checks,
        "microstate_policy_unqualified_and_target_free",
    )
    _require(
        plan["exact_nesting"] == EXPECTED_NESTING,
        "exact-nesting planning contract drifted",
        checks,
        "exact_nesting_unqualified",
    )
    _require(
        plan["physicality_gate"] == EXPECTED_PHYSICALITY,
        "physicality gate drifted",
        checks,
        "physicality_unqualified",
    )
    _require(
        plan["required_stage_order"] == EXPECTED_STAGE_ORDER,
        "fail-closed stage order drifted",
        checks,
        "stage_order",
    )
    _require(
        plan["unresolved_blockers"] == EXPECTED_BLOCKERS,
        "required HOLD blockers drifted",
        checks,
        "unresolved_blockers",
    )
    return checks


def self_test(plan: dict[str, Any]) -> int:
    cases: list[tuple[str, Any]] = [
        ("closed_capabilities.target_values_read", True),
        ("cohort_policy.entity_or_residue_exceptions", ["bmrb:50238"]),
        ("condition_manifest.path", "conditions.json"),
        ("condition_manifest.allowed_fields", ["target_values"]),
        ("condition_manifest.linkage", "target_values allowed"),
        ("condition_manifest.forbidden_fields", []),
        ("forbidden_source_builder_inputs", ["target_values"]),
        ("microstate_policy.algorithm", "entity_specific_GLUE2_repair"),
        ("microstate_policy.post_recount_reweighting_allowed", True),
        ("microstate_policy.target_conditioned_state_selection_allowed", True),
        ("exact_nesting.levels", [32, 128, 768, 1536]),
        ("exact_nesting.support_ordering", "observed_result_rank"),
        ("physicality_gate.independent_checker", "ready.py"),
        ("required_stage_order", list(reversed(EXPECTED_STAGE_ORDER))),
        ("state", "READY_TO_MATERIALIZE"),
        ("unresolved_blockers", ["condition manifest absent"] * 6),
    ]
    for dotted, value in cases:
        tampered = copy.deepcopy(plan)
        cursor: dict[str, Any] = tampered
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
        try:
            validate_plan(tampered, enforce_hash=False)
        except ValueError:
            continue
        raise AssertionError(f"tampering accepted: {dotted}")
    extra = copy.deepcopy(plan)
    extra["unexpected"] = True
    try:
        validate_plan(extra, enforce_hash=False)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown top-level field accepted")
    try:
        _reject_duplicate_keys([("state", "HOLD"), ("state", "READY")])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    return len(cases) + 2


def _bound_paths() -> tuple[Path, Path]:
    checker = Path(__file__).absolute()
    root = checker.parents[2]
    plan = root / PLAN_RELATIVE
    if not (
        checker == root / CHECKER_RELATIVE
        and checker.is_file()
        and not checker.is_symlink()
        and checker.resolve(strict=True) == checker
        and plan.is_file()
        and not plan.is_symlink()
        and plan.resolve(strict=True) == plan
    ):
        raise ValueError("checker or plan path is indirect, noncanonical, or misplaced")
    return root, plan


def verify(*, run_self_test: bool) -> int:
    _, path = _bound_paths()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PLAN_RAW_SHA256:
        raise ValueError("plan raw-byte hash drifted")
    plan = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    checks = validate_plan(plan)
    negative = self_test(plan) if run_self_test else 0
    return len(checks) + negative + 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    args = parser.parse_args()
    checks = verify(run_self_test=args.self_test)
    if not args.acknowledge_hold_only:
        print("STATUS HOLD_PLAN_ONLY_REQUIRED_BINDINGS_ABSENT")
        print("REFUSAL this plan does not authorize source construction or science")
        return 3
    print(f"METRIC solution_state_support_plan_checks={checks}")
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_PLAN_ONLY_REQUIRED_BINDINGS_ABSENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
