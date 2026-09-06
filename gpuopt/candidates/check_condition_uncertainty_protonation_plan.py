"""Validate the target-unread HOLD-only condition-uncertainty plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_plan_v1.json"
)
CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_condition_uncertainty_protonation_plan.py"
)
V1_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_solution_state_protonation_support_plan_v1.json"
)
RECEIPT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3/receipt.json"
)
INDEPENDENT_CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_openmm86_unique_assigned_ph_recovery_v3_independent.py"
)
PLAN_RAW_SHA256 = "37d1cb52c2bd904d3dde8104c8ee3b5194da6c31d42c36c0a78e6b72cc93b7a1"
PLAN_CANONICAL_SHA256 = (
    "e0ec8761a2031979521017f0a49275207c949ec5ccd62b345097befb0290ebe6"
)
V1_PLAN_RAW_SHA256 = "597086a617935f458b25be52190616b291ee3d6f6d8b92eaefc053a05cf8d67d"
RECEIPT_RAW_SHA256 = "2d8a255add821950e0e381401afc8c27a97d37cffb8ead827f5d8bd500bc3b8b"
INDEPENDENT_CHECKER_RAW_SHA256 = (
    "7f3a574a72a8413449c205fef5624b710e3fff2ab18623dae64fb0f3fa38bea8"
)
TOP_LEVEL_FIELDS = {
    "artifact_kind",
    "authorization",
    "candidate_id",
    "closed_capabilities",
    "condition_policy",
    "contract",
    "forbidden_source_builder_inputs",
    "interpretation_limits",
    "openmm_policy",
    "physicality_and_replay",
    "proposal_and_posterior",
    "provenance",
    "required_stage_order",
    "state",
    "study_id",
    "support_mapping",
    "unresolved_blockers",
}
FALSE_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
FORBIDDEN_INPUTS = {
    "assigned_atom_inventory",
    "chemical_shift_targets",
    "eligibility_rows",
    "existing_all_atom_blocker_receipt",
    "fold_or_evaluation_role",
    "source_or_outer_scores",
    "target_atom_identities",
    "target_values",
}
EXPECTED_STAGE_ORDER = [
    "freeze_this_policy_before_unresolved_proposal_branch_values_or_support_bytes_are_materialized",
    "freeze_exact_condition_environment_force_field_hydrogen_definitions_and_parent_manifests",
    "independent_cold_review",
    "run_bounded_target_unread_one_observed_one_unresolved_smoke",
    "freeze_generator_checker_and_source_commitment",
    "construct_complete_supports",
    "independently_replay_every_coordinate_and_physicality_check",
    "freeze_exact_support_order_and_nesting",
    "run_identity_only_coverage_recount",
    "separate_science_commitment_review_and_fresh_authorization",
]
EXPECTED_BLOCKERS = [
    "protonation generator source commitment and independent generator checker are absent",
    "bounded one-observed one-unresolved target-unread smoke is absent",
    "all-support deterministic replay physicality and effective-distinct-state audits are absent",
    "support order and diversity thresholds remain unqualified",
    "downstream proposal-to-one-shared-q inference contract remains unqualified",
    "K1536 remains separately blocked by current source capacity",
]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256(raw)


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


def _read_regular(
    root: Path, relative: Path, *, maximum_bytes: int = 2_000_000
) -> bytes:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"invalid relative input path: {relative}")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory = os.open(root, directory_flags)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"non-regular input: {relative}")
            if before.st_size > maximum_bytes:
                raise ValueError(f"input exceeds byte limit: {relative}")
            chunks = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(1_048_576, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)

            def identity(value: os.stat_result) -> tuple[int, ...]:
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_mode,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            if len(raw) != before.st_size or identity(before) != identity(after):
                raise ValueError(f"input changed while reading: {relative}")
            return raw
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _bound_root() -> Path:
    checker = Path(__file__).absolute()
    root = checker.parents[2]
    canonical = root / CHECKER_RELATIVE
    if (
        checker != canonical
        or checker.is_symlink()
        or checker.resolve(strict=True) != checker
    ):
        raise ValueError(f"noncanonical or indirect checker path: {checker}")
    return root


def _condition_regime_midpoints(thresholds: list[Decimal]) -> tuple[Decimal, ...]:
    if thresholds != sorted(set(thresholds)) or any(
        threshold <= 0 or threshold >= 14 for threshold in thresholds
    ):
        raise ValueError(
            "thresholds must be unique, sorted, and strictly inside [0,14]"
        )
    boundaries = [Decimal("0"), *thresholds, Decimal("14")]
    return tuple((left + right) / 2 for left, right in zip(boundaries, boundaries[1:]))


def _branch_schedule(entity_uid: str, branch_count: int, count: int) -> tuple[int, ...]:
    if not entity_uid or branch_count < 1 or count < 1:
        raise ValueError("invalid branch schedule input")
    key = (
        "atypemu_nested_support_count_v1_condition_uncertainty_protonation_v1"
        "\0" + entity_uid
    ).encode()
    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % branch_count
    return tuple((offset + index) % branch_count for index in range(count))


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
            _canonical_sha256(plan) == PLAN_CANONICAL_SHA256,
            "plan canonical hash drifted",
            checks,
            "whole_plan_hash",
        )
    _require(
        plan["artifact_kind"]
        == "hold_only_target_unread_condition_uncertainty_protonation_plan_not_support_not_authorization"
        and plan["candidate_id"]
        == "atypemu_nested_support_count_v1_condition_uncertainty_protonation_v1"
        and plan["contract"]
        == "atypemu_nested_support_count_v1_condition_uncertainty_protonation_plan_v1"
        and plan["study_id"] == "atypemu_nested_support_count_v1",
        "plan identity drifted",
        checks,
        "identity",
    )
    _require(
        plan["state"] == "HOLD_INPUTS_QUALIFIED_PROTONATION_SMOKE_UNRUN",
        "plan escaped HOLD",
        checks,
        "hold_state",
    )
    _require(
        plan["closed_capabilities"] == FALSE_CAPABILITIES,
        "a protected capability opened",
        checks,
        "closed_capabilities",
    )
    _require(
        plan["authorization"]
        == {
            "consumption_allowed": False,
            "existing_authorizations_reusable": False,
            "fresh_external_authorization_required_after_all_gates": True,
            "request_allowed_by_this_plan": False,
        },
        "authorization boundary drifted",
        checks,
        "authorization_closed",
    )
    condition = plan["condition_policy"]
    unresolved = condition["unresolved_rule"]
    _require(
        condition["entity_count"] == 135
        and condition["observed_entity_count"] == 119
        and condition["unresolved_entity_count"] == 16
        and condition["classification_states"]
        == ["observed", "state_missing", "state_ambiguous"]
        and condition["same_conditional_algorithm_for_every_entity"] is True
        and condition["unresolved_reason_codes_preserved"] is True,
        "condition roster or state semantics drifted",
        checks,
        "condition_states",
    )
    _require(
        condition["observed_rule"]
        == "use the one exact sealed numeric pH as a delta condition proposal"
        and unresolved
        == {
            "ambiguous_numeric_candidates_affect_proposal": False,
            "branch_mass": (
                "uniform across frozen OpenMM hydrogen-definition pH regimes; "
                "design proposal only"
            ),
            "lower_ph_bound_inclusive": "0.0",
            "method": (
                "derive one midpoint representative for every interval induced by all "
                "sorted unique finite pH transition thresholds in the frozen Modeller "
                "source and hydrogen definitions"
            ),
            "single_ph_imputation_allowed": False,
            "upper_ph_bound_inclusive": "14.0",
        },
        "observed or unresolved condition rule drifted",
        checks,
        "condition_rules",
    )
    _require(
        set(plan["forbidden_source_builder_inputs"]) == FORBIDDEN_INPUTS,
        "target-unread input boundary drifted",
        checks,
        "forbidden_inputs",
    )
    interpretation = plan["interpretation_limits"]
    _require(
        interpretation
        == {
            "all_atom_feasibility_established": False,
            "condition_metadata_recovered_for_unresolved_entities": False,
            "openmm_output_is_a_probabilistic_microstate_ensemble": False,
            "openmm_output_is_one_dominant_variant_realization_at_a_proposal_ph": True,
            "ph_regime_mass_is_a_physical_prior": False,
            "policy_self_checker_is_independent_science_evidence": False,
            "sampling_sufficiency_established": False,
        },
        "interpretation limits drifted",
        checks,
        "interpretation_limits",
    )
    openmm = plan["openmm_policy"]
    environment = openmm["environment"]
    _require(
        openmm["add_hydrogens_call"]
        == "Modeller.addHydrogens(forcefield, pH=branch_ph, variants=None, platform=Reference)"
        and openmm["variants_argument"] is None
        and openmm["existing_atom_positions_must_remain_exact"] is True
        and openmm["standard_residue_policy"]
        == "fail the entire candidate on an unsupported or nonstandard residue; no entity or residue exception"
        and "remove every input hydrogen-isotope atom"
        in openmm["input_hydrogen_policy"],
        "OpenMM invocation or input policy drifted",
        checks,
        "openmm_policy",
    )
    _require(
        environment
        == {
            "container_sha256": (
                "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
            ),
            "force_field_relative_path": "amber14/protein.ff14SB.xml",
            "force_field_sha256": (
                "d9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8"
            ),
            "hydrogen_definitions_relative_path": "openmm/app/data/hydrogens.xml",
            "hydrogen_definitions_sha256": (
                "413096cd3005ca5a638180e9cf623a8f6d574c81acf0e2c9d92b2bd26bb7658d"
            ),
            "modeller_source_relative_path": "openmm/app/modeller.py",
            "modeller_source_sha256": (
                "f61e61f1419fcc3c24e7096ab96e10f87f70951085a83941d6040390e8819ca3"
            ),
            "openmm_exact_version": "8.6.0.dev-c6173db",
            "platform": "Reference",
        },
        "qualified OpenMM environment drifted",
        checks,
        "environment_qualified",
    )
    _require(
        openmm["randomness"]
        == {
            "execution_isolation": "one fresh process per support",
            "seed_derivation": (
                "first 64 bits of SHA256(candidate_id NUL entity_uid NUL "
                "parent_support_id NUL condition_branch_id)"
            ),
            "seed_every_exposed_rng": True,
            "two_run_byte_identical_replay_required": True,
        },
        "randomness and replay contract drifted",
        checks,
        "deterministic_replay",
    )
    proposal = plan["proposal_and_posterior"]
    _require(
        proposal["proposal_weights_are_posterior_weights"] is False
        and proposal[
            "later_one_shared_q_may_change_support_membership_order_or_coordinates"
        ]
        is False
        and proposal[
            "later_one_shared_q_must_remain_one_entity_level_distribution_across_all_labels"
        ]
        is True
        and proposal["downstream_inference_contract"] == "ABSENT_UNQUALIFIED",
        "proposal and posterior were conflated or prematurely qualified",
        checks,
        "proposal_posterior_separation",
    )
    mapping = plan["support_mapping"]
    _require(
        mapping["exact_prefix_levels"] == [32, 128, 768]
        and mapping["parent_support_order"] == "ABSENT_UNQUALIFIED"
        and mapping["condition_branch_assignment"]
        == (
            "observed entities always use branch 0; unresolved "
            "branch=(entity_sha256_offset+zero_based_parent_rank) mod regime_count"
        )
        and "K-specific design weights are separate metadata"
        in mapping["prefix_contract"],
        "exact-prefix support mapping drifted",
        checks,
        "support_mapping",
    )
    provenance = plan["provenance"]
    _require(
        provenance
        == {
            "condition_evidence": {
                "independent_checker_path": INDEPENDENT_CHECKER_RELATIVE.as_posix(),
                "independent_checker_sha256": INDEPENDENT_CHECKER_RAW_SHA256,
                "receipt_path": RECEIPT_RELATIVE.as_posix(),
                "receipt_sha256": RECEIPT_RAW_SHA256,
            },
            "exact_condition_manifest": {
                "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_manifest_v1.json",
                "sha256": "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618",
            },
            "input_boundary_check": {
                "checker_path": "gpuopt/candidates/check_condition_uncertainty_inputs.py",
                "checker_sha256": "bc657d843c15ffc05cf482cb38401e425511a283760ef05a4aba9a3430ff9aee",
                "receipt_path": "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_inputs_check_receipt_v1.json",
                "receipt_sha256": "3a95bb683385079a46d921b0e17aa56cd684578b25b34ee13e9e1fd541356f8e",
            },
            "openmm_environment_manifest": {
                "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_environment_manifest_v1.json",
                "sha256": "6a5f3ff4a041d0fc055ee0a3b855ef2715a826812d11d6b2fdf1744a56fecd88",
            },
            "parent_heavy_coordinate_manifest": {
                "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1.json",
                "sha256": "77d52e663e80e55d178ffa7994596292f1753ffe4a0b4e5fd75005f826a119b3",
            },
            "protein_sequence_manifest": {
                "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_protein_sequence_manifest_v1.json",
                "sha256": "4faf799877e0387a90c9f00c641e958bd02585dde6b1b99bbdcaeb9f2e82012b",
            },
        },
        "qualified input provenance drifted",
        checks,
        "provenance",
    )
    physicality = plan["physicality_and_replay"]
    _require(
        physicality["independent_checker"] == "ABSENT_UNQUALIFIED"
        and physicality["must_cover_every_emitted_support"] is True
        and physicality["post_generation_identity_only_inventory_recount_required"]
        is True
        and physicality["effective_distinct_state_definition"]
        == (
            "unique canonical protonation signature; coordinates and returned null "
            "variants do not define the signature"
        )
        and physicality["new_hydrogen_parent_bond_check"]
        == (
            "every added hydrogen has exactly one bonded heavy parent and the force-field "
            "System construction and finite-energy checks pass"
        )
        and "exact ordered parent heavy-atom identity and coordinate preservation"
        in physicality["checks"]
        and "byte-identical deterministic replay" in physicality["checks"],
        "physicality or independent replay boundary drifted",
        checks,
        "physicality_replay",
    )
    _require(
        plan["required_stage_order"] == EXPECTED_STAGE_ORDER,
        "stage order drifted",
        checks,
        "stage_order",
    )
    _require(
        plan["unresolved_blockers"] == EXPECTED_BLOCKERS,
        "HOLD blockers drifted",
        checks,
        "unresolved_blockers",
    )
    return checks


def self_test(plan: dict[str, Any]) -> int:
    cases: list[tuple[str, Any]] = [
        ("closed_capabilities.target_values_read", True),
        ("authorization.request_allowed_by_this_plan", True),
        ("condition_policy.observed_entity_count", 135),
        ("condition_policy.unresolved_entity_count", 0),
        ("condition_policy.classification_states", ["observed"]),
        ("condition_policy.unresolved_rule.single_ph_imputation_allowed", True),
        (
            "condition_policy.unresolved_rule.ambiguous_numeric_candidates_affect_proposal",
            True,
        ),
        ("forbidden_source_builder_inputs", ["target_values"]),
        (
            "interpretation_limits.openmm_output_is_a_probabilistic_microstate_ensemble",
            True,
        ),
        ("interpretation_limits.ph_regime_mass_is_a_physical_prior", True),
        (
            "interpretation_limits.policy_self_checker_is_independent_science_evidence",
            True,
        ),
        ("openmm_policy.variants_argument", ["GLH"]),
        ("openmm_policy.environment.platform", "CUDA"),
        ("openmm_policy.environment.openmm_exact_version", "8.6"),
        ("openmm_policy.randomness.two_run_byte_identical_replay_required", False),
        ("proposal_and_posterior.proposal_weights_are_posterior_weights", True),
        ("proposal_and_posterior.downstream_inference_contract", "reuse proposal"),
        ("support_mapping.exact_prefix_levels", [128, 768]),
        ("support_mapping.parent_support_order", "score_rank"),
        ("provenance.exact_condition_manifest", "mutable.json"),
        ("physicality_and_replay.independent_checker", "unchecked.py"),
        ("required_stage_order", list(reversed(EXPECTED_STAGE_ORDER))),
        ("state", "READY_TO_CONSTRUCT"),
        ("unresolved_blockers", EXPECTED_BLOCKERS[:-1]),
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
    thresholds = [Decimal("4.4"), Decimal("6.5"), Decimal("8.5"), Decimal("10.4")]
    assert _condition_regime_midpoints(thresholds) == (
        Decimal("2.2"),
        Decimal("5.45"),
        Decimal("7.5"),
        Decimal("9.45"),
        Decimal("12.2"),
    )
    try:
        _condition_regime_midpoints([Decimal("6.5"), Decimal("4.4")])
    except ValueError:
        pass
    else:
        raise AssertionError("unsorted thresholds accepted")
    for branches in range(1, 17):
        full = _branch_schedule("bmrb:synthetic:entity:1", branches, 768)
        for level in (32, 128, 768):
            prefix = full[:level]
            counts = [prefix.count(branch) for branch in range(branches)]
            assert max(counts) - min(counts) <= 1
            assert prefix == _branch_schedule(
                "bmrb:synthetic:entity:1", branches, level
            )
    with tempfile.TemporaryDirectory(prefix="condition-policy-self-test-") as directory:
        root = Path(directory)
        (root / "real").mkdir()
        (root / "real" / "input.json").write_bytes(b"{}")
        assert _read_regular(root, Path("real/input.json")) == b"{}"
        (root / "indirect").symlink_to(root / "real", target_is_directory=True)
        (root / "leaf.json").symlink_to(root / "real" / "input.json")
        for relative in (Path("indirect/input.json"), Path("leaf.json"), Path("../x")):
            try:
                _read_regular(root, relative)
            except (OSError, ValueError):
                continue
            raise AssertionError(f"indirect input accepted: {relative}")
    return len(cases) + 9 + 16 * 3


def verify(*, run_self_test: bool) -> int:
    root = _bound_root()
    raw = _read_regular(root, PLAN_RELATIVE)
    if _sha256(raw) != PLAN_RAW_SHA256:
        raise ValueError("condition-uncertainty plan raw hash drifted")
    plan = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    checks = validate_plan(plan)
    bound_inputs = (
        (V1_PLAN_RELATIVE, V1_PLAN_RAW_SHA256, "v1 HOLD plan"),
        (RECEIPT_RELATIVE, RECEIPT_RAW_SHA256, "recovery-v3 receipt"),
        (
            INDEPENDENT_CHECKER_RELATIVE,
            INDEPENDENT_CHECKER_RAW_SHA256,
            "recovery-v3 independent checker",
        ),
    )
    verified_raw: dict[str, bytes] = {}
    for relative, expected, label in bound_inputs:
        verified_raw[label] = _read_regular(root, relative)
        if _sha256(verified_raw[label]) != expected:
            raise ValueError(f"{label} hash drifted")
        checks.append(label.replace(" ", "_") + "_raw_binding")
    v1 = json.loads(
        verified_raw["v1 HOLD plan"], object_pairs_hook=_reject_duplicate_keys
    )
    receipt = json.loads(
        verified_raw["recovery-v3 receipt"], object_pairs_hook=_reject_duplicate_keys
    )
    _require(
        v1["state"] == "HOLD_CONDITION_MANIFEST_INCOMPLETE"
        and v1["condition_manifest"]["recovery_v3"]["metadata_resolved_entity_count"]
        == 119
        and v1["condition_manifest"]["recovery_v3"][
            "state_missing_or_ambiguous_entity_count"
        ]
        == 16,
        "v1 HOLD evidence semantics drifted",
        checks,
        "v1_hold_semantics",
    )
    _require(
        receipt["entity_count"] == 135
        and receipt["combined_metadata_resolved_entity_count"] == 119
        and receipt["status"] == "HOLD_METADATA_REINTERPRETATION_ONLY"
        and not any(receipt["closed_capabilities"].values()),
        "recovery-v3 receipt semantics drifted",
        checks,
        "recovery_v3_hold_semantics",
    )
    return len(checks) + (self_test(plan) if run_self_test else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    args = parser.parse_args()
    checks = verify(run_self_test=args.self_test)
    if not args.acknowledge_hold_only:
        print("STATUS HOLD_INPUTS_QUALIFIED_PROTONATION_SMOKE_UNRUN")
        print("REFUSAL this plan is not support construction or authorization")
        return 3
    print(f"METRIC condition_uncertainty_support_plan_checks={checks}")
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_INPUTS_QUALIFIED_PROTONATION_SMOKE_UNRUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
