"""Validate the HOLD-only exact-nested support-count feasibility plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PLAN_RELATIVE = Path("gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json")
VALIDATOR_RELATIVE = Path("gpuopt/candidates/nested_support_count_plan.py")
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
    "22d91380baf6032cc708216781cee62060b0518c235066a91f7c37dc0c31e085"
)
OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_INTENT = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v3_launch_intent.json"
)
OPENMM86_DEPOSITED_PH_RECOVERY_V3_CONSUMED_INTENT = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v3_launch_intent.consumed.json"
)
OPENMM86_DEPOSITED_PH_RECOVERY_V3_OUTPUT = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v3_recovery"
)
OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_FIELDS = {
    "artifact_kind": "target_unread_metadata_fetch_launch_intent_not_authorization",
    "candidate_id": "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v3",
    "contract": "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v3_launch_v1",
    "git_commit": None,
    "plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v3_plan.json",
        "sha256": "f8eadb6a4dde415d1af016c2843f867ee32e9546a7e4a272ff211fbfe531ba1e",
    },
    "producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v3.py",
        "sha256": "8c2549c18b60e6eb69812102a14f40e063458540ef0abf9191bdf7c3f4a59a49",
    },
    "checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v3.py",
        "sha256": "ecb5cc73bfc09be99ab3891fbd0d67d091925c0b8be13abe4f720ec11a64bf7d",
    },
    "v1_failure_receipt_sha256": "a8cbd988789c6bde3e774d625ab4e6583476dfd7ae1a1f4a641b9548edbda8c5",
    "v2_launch_intent_sha256": "ffdcebcaa4cf73d1e6b3d0834c8775836ba2c0f14edcd1bd10f21c2e5ab7aa9f",
    "run_122_raw_line_sha256": "df01b57177613b0779b6b3519e43112ae021c29c58b5f8ac83a929c695402389",
    "output": OPENMM86_DEPOSITED_PH_RECOVERY_V3_OUTPUT.as_posix(),
    "target_values_read": False,
    "target_atom_identities_read": False,
    "source_scores_read": False,
    "science_executed": False,
    "source_construction_executed": False,
    "authorization_consumed": False,
}
CATALOG_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_catalog_feasibility_receipt.json"
)
CATALOG_RECEIPT_SHA256 = (
    "c820e19bbf309f05427453f97f070162d394e3992e8583314df76309cf895c94"
)
ALL_ATOM_POLICY_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_policy.json"
)
ALL_ATOM_POLICY_SHA256 = (
    "f792ea47639f19d0376088b2c2e2fd5e9a161f33e4871205664bc9972e1b35ae"
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
    "solution_state_condition_catalog_producer": {
        "path": "gpuopt/candidates/solution_state_condition_catalog.py",
        "sha256": "0a4f55973c3eadddb7d4d9ac5d2c463b1df53da7f89f7c5f672997ca5253fc36",
    },
    "solution_state_condition_catalog_checker": {
        "path": "gpuopt/candidates/check_solution_state_condition_catalog.py",
        "sha256": "2962ff781b8a44f3e6e971545a46d47f74c11d563f78a39792ea88ff4dc2296d",
    },
    "solution_state_condition_catalog_seal": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_solution_state_condition_catalog_v1.json",
        "sha256": "3f8056aba2d5efaed0a42baf6b67411f50cd6849f4a78103cea853bd862ceca8",
    },
    "solution_state_condition_catalog_evidence_archive": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip",
        "sha256": "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70",
    },
    "openmm86_deposited_ph_catalog_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_plan_v1.json",
        "sha256": "e3bc877da2c3a0c3cac90399a99aac93817bd16646f29abc15ed575b0a95d913",
    },
    "openmm86_deposited_ph_catalog_producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog.py",
        "sha256": "1a449a2e966347d5bd607dcb22c5b52a59cfb2034aa6db49ae295232ed38f0ef",
    },
    "openmm86_deposited_ph_catalog_checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog.py",
        "sha256": "dcbd457f33d66f2fe51367db42fe760b614196ba81d5a8fdb3f0990320387ff7",
    },
    "openmm86_deposited_ph_catalog_v1_failure_receipt": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v1/failure_receipt.json",
        "sha256": "a8cbd988789c6bde3e774d625ab4e6583476dfd7ae1a1f4a641b9548edbda8c5",
    },
    "openmm86_deposited_ph_catalog_v1_start_marker": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v1/started_at_utc.txt",
        "sha256": "d741789e11f995c3c034152fd7e1594a34faecdf6dd970ce4bc73309fee29141",
    },
    "openmm86_deposited_ph_catalog_recovery_v2_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v2_plan.json",
        "sha256": "5d6547c21cdf4c1dc6380340375084c53de0e137f10cee47f74f6683150227e3",
    },
    "openmm86_deposited_ph_catalog_recovery_v2_producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v2.py",
        "sha256": "aad9a1b2f20a46853b2b78dfcf46f85b2912ac8b67540dec8492764e74d1e4ae",
    },
    "openmm86_deposited_ph_catalog_recovery_v2_checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v2.py",
        "sha256": "d43093869b37f29a521cdc102008a10ab87da933a12cc9417d5d8160ab79842f",
    },
    "openmm86_deposited_ph_catalog_recovery_v2_launch_intent": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v2_launch_intent.json",
        "sha256": "ffdcebcaa4cf73d1e6b3d0834c8775836ba2c0f14edcd1bd10f21c2e5ab7aa9f",
    },
    "openmm86_deposited_ph_catalog_recovery_v3_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v3_plan.json",
        "sha256": "f8eadb6a4dde415d1af016c2843f867ee32e9546a7e4a272ff211fbfe531ba1e",
    },
    "openmm86_deposited_ph_catalog_recovery_v3_producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v3.py",
        "sha256": "8c2549c18b60e6eb69812102a14f40e063458540ef0abf9191bdf7c3f4a59a49",
    },
    "openmm86_deposited_ph_catalog_recovery_v3_checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v3.py",
        "sha256": "ecb5cc73bfc09be99ab3891fbd0d67d091925c0b8be13abe4f720ec11a64bf7d",
    },
    "openmm86_deposited_ph_catalog_recovery_v3_failed_intent": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v3_launch_intent.failed_run126.json",
        "sha256": "c75986bea0ce11684fec2d0aae6aa594ec65527a9814cd89525083f1bd9d7a54",
    },
    "openmm86_deposited_ph_catalog_recovery_v4_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v4_plan.json",
        "sha256": "079f426a454cbfaacc8bdc86dd1e7a5b577fd0cfec6ae90f725dd6158738bff3",
    },
    "openmm86_deposited_ph_catalog_recovery_v4_producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v4.py",
        "sha256": "3242f79ffff689ce8b6c10a9d69c53dde31f5f004c44931b47f8068d3cc07a9a",
    },
    "openmm86_deposited_ph_catalog_recovery_v4_checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v4.py",
        "sha256": "d417cdd32608bc1cfdf3a37d3fc4c7fae70a553fcdf896278e09f8dae42d145c",
    },
    "openmm86_deposited_ph_catalog_recovery_v4_handler": {
        "path": "gpuopt/candidates/run_openmm86_deposited_ph_recovery_v4_if_intent.py",
        "sha256": "9d89c21b6babdca0ba1fd45c78103829a26fcf49723557e6a6d1c543975a03b2",
    },
    "openmm86_deposited_ph_catalog_recovery_v5_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v5_plan.json",
        "sha256": "194b77cd92a4c81a19ccffb2a2bcc397de85116639107feb4eb62d5afe749d1d",
    },
    "openmm86_deposited_ph_catalog_recovery_v5_producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v5.py",
        "sha256": "ae3e9ca11b44e5196a6dc89c3146c7e27a0e71b8e38d41ed107745fd85edb916",
    },
    "openmm86_deposited_ph_catalog_recovery_v5_checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v5.py",
        "sha256": "39186db67622cb11cf911291fa02659eee00fe977d86226623ec9f9f4240369e",
    },
    "openmm86_deposited_ph_catalog_recovery_v5_handler": {
        "path": "gpuopt/candidates/run_openmm86_deposited_ph_recovery_v5_if_intent.py",
        "sha256": "6471b9fc28e90989fc87febcee64f75e81ac08c309ae009fd8ce0aa3520fb8ac",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v6_plan.json",
        "sha256": "c12873f74e56db94d0a44d14b1edb339331d8cf4870c1361bcb353bfc538a81f",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v6.py",
        "sha256": "6a7fff25f6cfb520511b269ca0c06e80b092eed35107b7e8cd740c8122e0d623",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v6.py",
        "sha256": "e841d7c81fb823b1939dd0b74e5ae0f610dd387fd17c4456d286fc9a2aaa99c5",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_handler": {
        "path": "gpuopt/candidates/run_openmm86_deposited_ph_recovery_v6_if_intent.py",
        "sha256": "c077226516b3b0aa97029d9ac453b57134e7fc04ece69805305460d09caf4034",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_evidence_checker": {
        "path": "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_evidence.py",
        "sha256": "2fa35326d934f81e8e1b75f76fda1e5240961e19d5a8bee000b883df50232532",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_evidence_receipt": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_receipt.json",
        "sha256": "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36",
    },
    "openmm86_deposited_ph_catalog_recovery_v6_semantic_checker": {
        "path": "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_semantic_independent.py",
        "sha256": "f12eb66b24c4696562e9c1833836c240cac4c30299b963761614b70903e46d84",
    },
    "all_atom_recount_policy": {
        "path": str(ALL_ATOM_POLICY_RELATIVE),
        "sha256": ALL_ATOM_POLICY_SHA256,
    },
    "all_atom_recount_producer": {
        "path": "gpuopt/candidates/nested_support_all_atom_recount.py",
        "sha256": "41cd2261292d47c724138f8a671d50423245fa19eff0e71f1a5a95b19215e90c",
    },
    "all_atom_recount_checker": {
        "path": "gpuopt/candidates/check_nested_support_all_atom_recount.py",
        "sha256": "a2bcfe2e204dbfd248062d44dd6475aa313a55c906923385b364ef6fe3bdbe4e",
    },
    "all_atom_raw_replay_v3_candidate": {
        "path": "gpuopt/candidates/check_nested_support_all_atom_recount_raw_v3.py",
        "sha256": "080258250c1d15950b7f9a4dad3e00391b1d644bae005882f56f4f7b3feda19d",
    },
    "all_atom_raw_evidence_archive": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_all_atom_raw_evidence_v2.tar.gz",
        "sha256": "e0df66a4046b9aef255f0d4992a732fdc2c7ebb4e1e5dccb7132a0ba511584f2",
    },
    "all_atom_raw_evidence_archive_v3": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_all_atom_raw_evidence_v3.tar.gz",
        "sha256": "57500a316e2e9036ecf3414244e36419013e801977f9fad60e1a3294796fc20d",
    },
    "all_atom_raw_evidence_checker": {
        "path": "gpuopt/candidates/check_nested_support_all_atom_raw_evidence.py",
        "sha256": "50343442484781f3e5780ed8eca59a7daa53432bbcbdfa2b063b632da8fad944",
    },
    "solution_state_protonation_support_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_solution_state_protonation_support_plan_v1.json",
        "sha256": "f6443dee60c0b7bec024420726adf8d7fdce56d6ee0ce056c9c593f547a0335a",
    },
    "solution_state_protonation_support_checker": {
        "path": "gpuopt/candidates/check_solution_state_protonation_support_plan.py",
        "sha256": "415f7883507e929712b1ce122d31ad061703367c81e8ee8872a2770592d06031",
    },
    "all_atom_raw_replay_correction_receipt": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_raw_replay_correction_receipt.json",
        "sha256": "caf1b5e51577cc50273c52b1c2fa2d1ec796991fe9f85a53fbeb3d92e0b525ba",
    },
    "all_atom_recount_receipt": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_recount_receipt.json",
        "sha256": "7f119c216ebfcfe4bfe2dfb5a607d1cd3689f6ecbca7ca69bafab4b51a985f1c",
    },
    "catalog_feasibility_receipt": {
        "path": str(CATALOG_RECEIPT_RELATIVE),
        "sha256": CATALOG_RECEIPT_SHA256,
    },
    "catalog_checker": {
        "path": "gpuopt/candidates/check_nested_support_catalog.py",
        "sha256": "467a4a53edcac82015e3cdc9e4e0f40ad814b96c5ff9926688e62723cb19fe6b",
    },
    "catalog_producer": {
        "path": "gpuopt/candidates/nested_support_catalog.py",
        "sha256": "4bd7c9463b34a9d6c268784fd13b45a01bf18c5e97905c0a5a4efd1cd7da0c28",
    },
    "catalog_shard_archive": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz",
        "sha256": "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2",
    },
    "catalog_summary": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_summary.json",
        "sha256": "737aaff560041ab7b64a750d011469a43563ab8a307af518eea3b98903f99b41",
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
    "source_commitment": {
        "path": ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json",
        "sha256": "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8",
    },
    "entity_roster": {
        "path": ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json",
        "sha256": "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9",
    },
}
EXPECTED_RECEIPT_EVIDENCE = {
    name: {
        "path": SAFE_EVIDENCE[name]["path"],
        "raw_sha256": SAFE_EVIDENCE[name]["sha256"],
    }
    for name in (
        "catalog_checker",
        "catalog_producer",
        "catalog_shard_archive",
        "catalog_summary",
        "entity_roster",
        "source_commitment",
    )
}
EVIDENCE_CHECK_ORDER = (
    "all_atom_recount_policy",
    "all_atom_recount_producer",
    "all_atom_recount_checker",
    "all_atom_raw_replay_v3_candidate",
    "all_atom_raw_evidence_archive",
    "all_atom_raw_evidence_archive_v3",
    "all_atom_raw_evidence_checker",
    "solution_state_condition_catalog_producer",
    "solution_state_condition_catalog_checker",
    "solution_state_condition_catalog_seal",
    "solution_state_condition_catalog_evidence_archive",
    "openmm86_deposited_ph_catalog_plan",
    "openmm86_deposited_ph_catalog_producer",
    "openmm86_deposited_ph_catalog_checker",
    "openmm86_deposited_ph_catalog_v1_failure_receipt",
    "openmm86_deposited_ph_catalog_v1_start_marker",
    "openmm86_deposited_ph_catalog_recovery_v2_plan",
    "openmm86_deposited_ph_catalog_recovery_v2_producer",
    "openmm86_deposited_ph_catalog_recovery_v2_checker",
    "openmm86_deposited_ph_catalog_recovery_v2_launch_intent",
    "openmm86_deposited_ph_catalog_recovery_v3_plan",
    "openmm86_deposited_ph_catalog_recovery_v3_producer",
    "openmm86_deposited_ph_catalog_recovery_v3_checker",
    "openmm86_deposited_ph_catalog_recovery_v3_failed_intent",
    "openmm86_deposited_ph_catalog_recovery_v4_plan",
    "openmm86_deposited_ph_catalog_recovery_v4_producer",
    "openmm86_deposited_ph_catalog_recovery_v4_checker",
    "openmm86_deposited_ph_catalog_recovery_v4_handler",
    "openmm86_deposited_ph_catalog_recovery_v5_checker",
    "openmm86_deposited_ph_catalog_recovery_v5_handler",
    "openmm86_deposited_ph_catalog_recovery_v5_plan",
    "openmm86_deposited_ph_catalog_recovery_v5_producer",
    "openmm86_deposited_ph_catalog_recovery_v6_plan",
    "openmm86_deposited_ph_catalog_recovery_v6_producer",
    "openmm86_deposited_ph_catalog_recovery_v6_checker",
    "openmm86_deposited_ph_catalog_recovery_v6_handler",
    "openmm86_deposited_ph_catalog_recovery_v6_evidence_checker",
    "openmm86_deposited_ph_catalog_recovery_v6_evidence_receipt",
    "openmm86_deposited_ph_catalog_recovery_v6_semantic_checker",
    "solution_state_protonation_support_plan",
    "solution_state_protonation_support_checker",
    "all_atom_raw_replay_correction_receipt",
    "all_atom_recount_receipt",
    "catalog_feasibility_receipt",
    "catalog_checker",
    "catalog_producer",
    "source_commitment",
    "entity_roster",
    "catalog_summary",
    "catalog_shard_archive",
    "current_k32_plan",
    "current_selector",
    "jeon_thesis",
)
EXPECTED_LEVEL_STATUS = {
    "32": "ALL_ATOM_COUNT_BLOCKED_ONE_ENTITY_NON_HIS_TARGET_UNAVAILABLE_EXISTING_SEED_REQUIRES_NEW_PROVENANCE_BINDING",
    "128": "ALL_ATOM_COUNT_BLOCKED_ONE_ENTITY_NON_HIS_TARGET_UNAVAILABLE",
    "768": "ALL_ATOM_COUNT_BLOCKED_ONE_ENTITY_NON_HIS_TARGET_UNAVAILABLE",
    "1536": "HEAVY_TOPOLOGY_AND_ALL_ATOM_COUNT_BLOCKED_CURRENT_SOURCE_CAPACITY",
}
EXPECTED_ALL_ATOM_LEVEL_COUNTS = {
    "32": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 134,
        "maximum_entity_shortfall": 32,
        "total_shortfall": 32,
    },
    "128": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 134,
        "maximum_entity_shortfall": 128,
        "total_shortfall": 128,
    },
    "768": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 134,
        "maximum_entity_shortfall": 768,
        "total_shortfall": 768,
    },
    "1536": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 0,
        "maximum_entity_shortfall": 1536,
        "total_shortfall": 73510,
    },
}
EXPECTED_ALL_ATOM_DIAGNOSTICS = {
    "interpretation": "diagnostic topology counts only; not all-atom support usability",
    "largest_single_topology_variant_entity_count_ge_k": {
        "32": 135,
        "128": 135,
        "768": 117,
        "1536": 0,
    },
    "reference_topology_entity_count_ge_k": {
        "32": 130,
        "128": 121,
        "768": 108,
        "1536": 0,
    },
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
    "K32, K128, and K768 are all-atom count-infeasible under the frozen policy because bmr50238 has zero compliant supports: assigned GLU:HE2 is unavailable across all 1000 catalog supports",
    "uniform deposited-pH support construction remains blocked because recovery-v6 resolves exactly one pH for only 115/135 entities and holds the other 20 on missing or ambiguous metadata",
    "K32 original structural-source provenance replay is not established for this study",
    "support construction and diversity thresholds are not frozen",
    "primary and replicate ladder roots are not committed",
]
EXPECTED_LEVEL_PROGRESSION = {
    "general_sampling_sufficiency_claim_requires_a_separately_cleared_convergence_level": True,
    "k128_and_k768_may_advance_without_k1536": True,
    "lower_level_condition": "each level must first close its own hydrogen-canonicalization, geometry-mask, all-atom recount, diversity, physicality, exact-nesting, provenance, commitment, review, and authorization blockers",
    "status": "LOWER_LEVELS_HOLD_ON_LEVEL_LOCAL_BLOCKERS_K1536_SEPARATELY_BLOCKED",
}
EXPECTED_RECEIPT_TOP_LEVEL_FIELDS = {
    "artifact_kind",
    "authorization_consumed",
    "catalog",
    "claims",
    "contract",
    "evidence",
    "executions",
    "heavy_topology_level_count_feasibility",
    "outer_or_formal_metrics_opened",
    "science_executed",
    "source_scores_read",
    "status",
    "study_id",
    "target_values_read",
}
EXPECTED_IREMB_BOUND_CHECKER_EXECUTION = {
    "checker_checks": 2079799,
    "checker_sha256": "e11744cebe8c7e78e08ed65747ff67a7b8db4f818dfffb83381868d04889660f",
    "failed_python_compatibility_job": 136170,
    "node": "iREMB-C-08",
    "partition": "l40sq",
    "scope": "historical pre-path-binding raw evidence, identity, catalog-integrity, and count-arithmetic replay; not current evidence and not a scientific gate checker",
    "status": "PASS_HISTORICAL_PRE_PATH_BINDING",
    "successful_replay_job": 136171,
}
EXPECTED_NETBIRD_GATEWAY_EXECUTION = {
    "checker_checks": 2029620,
    "checker_sha256": "6d6078f5729b959e6c5cc9e24d8d56b97ccc516dcef3f1e375560988e9b32ab4",
    "scope": "sealed catalog integrity and count arithmetic only, not a scientific gate checker",
    "status": "PASS",
}
EXPECTED_NETBIRD_PATH_BOUND_EXECUTION = {
    "checker_checks": 2079812,
    "checker_sha256": SAFE_EVIDENCE["catalog_checker"]["sha256"],
    "host": "cylee-X10DAi",
    "scope": "exact committed-relative-path, raw evidence, identity, catalog-integrity, and count-arithmetic replay only; not a scientific gate checker",
    "status": "PASS",
    "temporary_root_removed_after_execution": True,
}
EXPECTED_CATALOG_EXECUTION_NAMES = {
    "iremb_slurm",
    "iremb_slurm_bound_checker",
    "netbird_gateway_checker",
    "netbird_gateway_path_bound_checker",
    "yulab_mac_studio",
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
        and feasibility.get("heavy_topology_count_feasibility_definition")
        == "per entity using heavy-topology-compatible exact-heavy-coordinate-unique supports"
        and feasibility.get("level_status") == EXPECTED_LEVEL_STATUS,
        "current target-unread coordinate-source evidence changed",
        checks,
        "current_source_capacity_and_all_level_statuses",
    )
    _require(
        feasibility.get("level_status", {}).get("1536")
        == "HEAVY_TOPOLOGY_AND_ALL_ATOM_COUNT_BLOCKED_CURRENT_SOURCE_CAPACITY"
        and feasibility.get(
            "new_target_unread_generation_or_coordinate_source_required_for_k1536"
        )
        is True,
        "K1536 must remain blocked by the current 1000-frame source namespace",
        checks,
        "k1536_capacity_block",
    )
    _require(
        feasibility.get("all_atom_count_status")
        == "RECOUNT_COMPLETE_ALL_LEVELS_BLOCKED"
        and feasibility.get("all_atom_count_recount_required") is False
        and feasibility.get("support_specific_hydrogen_canonicalization_policy_state")
        == "FROZEN_RECOUNT_COMPLETE"
        and feasibility.get("geometry_availability_mask_policy_state")
        == "FROZEN_RECOUNT_COMPLETE"
        and feasibility.get("all_atom_level_count_feasibility")
        == EXPECTED_ALL_ATOM_LEVEL_COUNTS
        and feasibility.get("all_atom_blocking_entity")
        == {
            "catalog_support_count": 1000,
            "entity_uid": "bmrb:50238:entity:1",
            "policy_compliant_support_count": 0,
            "reason": "non-HIS target atom is unavailable: bmrb:50238:entity:1:target:cs:1:328:1:1:_:36:GLU:HE2",
        }
        and feasibility.get("all_atom_topology_count_diagnostics")
        == EXPECTED_ALL_ATOM_DIAGNOSTICS,
        "all-atom policy or recount HOLD state changed",
        checks,
        "all_atom_recount_complete_with_all_levels_blocked",
    )
    policy = json.loads((root / ALL_ATOM_POLICY_RELATIVE).read_text())
    _require(
        policy.get("contract") == "atypemu_nested_support_count_v1_all_atom_policy_v1"
        and policy.get("study_id") == "atypemu_nested_support_count_v1"
        and policy.get("state") == "FROZEN_POLICY_RECOUNT_NOT_RUN"
        and policy.get("capabilities")
        == {
            "authorization_consumption": False,
            "outer_or_formal_metrics": False,
            "science_execution": False,
            "source_scores": False,
            "target_value_deserialization": False,
        }
        and policy.get("qualification", {}).get(
            "future_support_rosters_may_use_target_availability_for_selection"
        )
        is False
        and policy.get("geometry_availability_mask", {}).get(
            "no_target_row_may_be_removed"
        )
        is True
        and policy.get("hydrogen_canonicalization", {}).get(
            "source_pdb_atoms_may_be_added_removed_or_renamed"
        )
        is False,
        "frozen all-atom policy semantics changed",
        checks,
        "frozen_all_atom_policy_semantics",
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
        and diversity.get("completion") is False
        and diversity.get("near_duplicate_and_coverage_thresholds") == "UNFROZEN"
        and diversity.get("exact_coordinate_duplicate_count_max_per_entity") == 0
        and diversity.get("audit_scope") == "every support at every ladder level",
        "diversity readiness must fail closed until thresholds are frozen",
        checks,
        "diversity_hold",
    )
    _require(
        diversity.get("canonical_heavy_atom_identity_must_match_within_entity") is True
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
        and authorization.get("fresh_external_once_only_authorization_per_k_required")
        is True
        and {
            readiness.get(key)
            for key in (
                "formal_evaluation",
                "materialization",
                "science_authorization",
                "science_execution",
            )
        }
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
    _require(
        readiness.get("level_progression") == EXPECTED_LEVEL_PROGRESSION,
        "K128/K768 progression was coupled to K1536 or released prematurely",
        checks,
        "k128_k768_level_local_progression_k1536_separate",
    )
    interpretation = plan.get("interpretation", {})
    _require(
        interpretation.get("k32_role")
        == "low-K coordinate/Jacobian causal-infrastructure smoke only"
        and interpretation.get("jeon_transfer_claim_allowed") is False
        and interpretation.get("general_sampling_sufficiency_established") is False
        and interpretation.get("one_ladder_establishes_general_sufficiency") is False
        and interpretation.get("all_atom_count_feasibility_established") is False
        and interpretation.get("all_atom_recount_completed") is True
        and interpretation.get("independent_raw_pdb_replay_status")
        == "PASS_TOPOLOGY_AND_MASK_INDEPENDENT_RAW_REPLAY_ALL_LEVELS_REMAIN_HOLD",
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
        and 'authorized": true' not in serialized,
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
    canonical_root = root.absolute()
    for name in EVIDENCE_CHECK_ORDER:
        binding = SAFE_EVIDENCE[name]
        relative = Path(str(binding.get("path", "")))
        path = canonical_root / relative
        evidence_ok &= (
            not relative.is_absolute()
            and canonical_root in path.parents
            and path.is_file()
            and not path.is_symlink()
            and path.resolve(strict=True) == path
            and _sha256(path) == binding.get("sha256")
        )
    _require(
        evidence_ok,
        "bound target-unread planning evidence mismatch",
        checks,
        "evidence_hashes",
    )
    receipt = json.loads((root / CATALOG_RECEIPT_RELATIVE).read_text())
    _require(
        set(receipt) == EXPECTED_RECEIPT_TOP_LEVEL_FIELDS,
        "catalog receipt top-level schema changed",
        checks,
        "catalog_receipt_exact_top_level_schema",
    )
    _require(
        receipt.get("contract")
        == "atypemu_nested_support_count_v1_catalog_feasibility_receipt_v2"
        and receipt.get("study_id") == "atypemu_nested_support_count_v1"
        and receipt.get("status")
        == "HEAVY_TOPOLOGY_COUNT_FEASIBILITY_ONLY_ALL_ATOM_UNCLAIMED_HOLD"
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
        and catalog.get("heavy_topology_count_feasibility_definition")
        == "per entity using heavy-topology-compatible exact-heavy-coordinate-unique supports"
        and catalog.get("canonical_filename_count_minimum") == 991
        and catalog.get("canonical_filename_count_maximum") == 1000
        and catalog.get("total_canonical_filename_count") == 134850
        and catalog.get("heavy_topology_compatible_unique_coordinate_count") == 134850
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
    all_atom_diagnostics = catalog.get("all_atom_topology_count_diagnostics", {})
    _require(
        all_atom_diagnostics
        == {
            **EXPECTED_ALL_ATOM_DIAGNOSTICS,
            "status": "ALL_ATOM_COUNT_UNCLAIMED_PENDING_HYDROGEN_CANONICALIZATION_GEOMETRY_MASK_AND_RECOUNT",
        },
        "all-atom topology-count diagnostics or unclaimed status changed",
        checks,
        "catalog_all_atom_count_diagnostics_unclaimed",
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
        receipt.get("heavy_topology_level_count_feasibility") == expected_feasibility,
        "support-level count feasibility changed",
        checks,
        "exact_level_count_feasibility",
    )
    claims = receipt.get("claims", {})
    _require(
        claims.get("heavy_topology_count_feasible_levels") == [32, 128, 768]
        and claims.get("all_atom_count_feasible_levels") == []
        and claims.get("all_atom_recount_required") is True
        and claims.get("support_specific_hydrogen_canonicalization_policy_frozen")
        is False
        and claims.get("geometry_availability_mask_policy_frozen") is False
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
    receipt_evidence = receipt.get("evidence", {})
    _require(
        receipt_evidence == EXPECTED_RECEIPT_EVIDENCE,
        "catalog receipt does not bind the exact raw evidence graph",
        checks,
        "catalog_receipt_exact_raw_evidence_bindings",
    )
    executions = receipt.get("executions", {})
    _require(
        set(executions) == EXPECTED_CATALOG_EXECUTION_NAMES,
        "catalog execution roster changed",
        checks,
        "catalog_execution_exact_roster",
    )
    _require(
        executions.get("iremb_slurm_bound_checker")
        == EXPECTED_IREMB_BOUND_CHECKER_EXECUTION,
        "iREMB bound-checker execution receipt changed",
        checks,
        "iremb_bound_checker_exact_execution_receipt",
    )
    _require(
        executions.get("netbird_gateway_checker") == EXPECTED_NETBIRD_GATEWAY_EXECUTION,
        "historical NetBird checker execution receipt changed",
        checks,
        "netbird_checker_exact_historical_execution_receipt",
    )
    _require(
        executions.get("netbird_gateway_path_bound_checker")
        == EXPECTED_NETBIRD_PATH_BOUND_EXECUTION,
        "current NetBird path-bound checker execution receipt changed",
        checks,
        "netbird_path_bound_checker_exact_execution_receipt",
    )
    summary_sha256 = "737aaff560041ab7b64a750d011469a43563ab8a307af518eea3b98903f99b41"
    _require(
        executions.get("yulab_mac_studio", {}).get("aggregate_summary_sha256")
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
        and executions.get("iremb_slurm_bound_checker", {}).get("status")
        == "PASS_HISTORICAL_PRE_PATH_BINDING"
        and executions.get("iremb_slurm_bound_checker", {}).get("node") == "iREMB-C-08"
        and executions.get("iremb_slurm_bound_checker", {}).get("partition") == "l40sq"
        and executions.get("iremb_slurm_bound_checker", {}).get(
            "failed_python_compatibility_job"
        )
        == 136170
        and executions.get("iremb_slurm_bound_checker", {}).get("successful_replay_job")
        == 136171
        and executions.get("iremb_slurm_bound_checker", {}).get("checker_checks")
        == 2079799
        and executions.get("iremb_slurm_bound_checker", {}).get("checker_sha256")
        == "e11744cebe8c7e78e08ed65747ff67a7b8db4f818dfffb83381868d04889660f"
        and executions.get("netbird_gateway_checker", {}).get("status") == "PASS"
        and executions.get("netbird_gateway_checker", {}).get("scope")
        == "sealed catalog integrity and count arithmetic only, not a scientific gate checker"
        and executions.get("netbird_gateway_checker", {}).get("checker_checks")
        == 2029620
        and executions.get("netbird_gateway_checker", {}).get("checker_sha256")
        == "6d6078f5729b959e6c5cc9e24d8d56b97ccc516dcef3f1e375560988e9b32ab4"
        and executions.get("netbird_gateway_path_bound_checker", {}).get("status")
        == "PASS"
        and executions.get("netbird_gateway_path_bound_checker", {}).get(
            "checker_checks"
        )
        == 2079812
        and executions.get("netbird_gateway_path_bound_checker", {}).get(
            "checker_sha256"
        )
        == SAFE_EVIDENCE["catalog_checker"]["sha256"],
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

    for evidence_name, mode in (
        ("source_commitment", "append"),
        ("entity_roster", "empty"),
        ("entity_roster", "symlink"),
    ):
        with tempfile.TemporaryDirectory(
            prefix="nested-support-plan-negative-", dir=root / ".auto"
        ) as temporary:
            temporary_root = Path(temporary)
            for name, binding in SAFE_EVIDENCE.items():
                relative = Path(binding["path"])
                source = root / relative
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if name == evidence_name and mode == "append":
                    destination.write_bytes(source.read_bytes() + b"\n")
                elif name == evidence_name and mode == "empty":
                    destination.write_bytes(b"{}\n")
                elif name == evidence_name and mode == "symlink":
                    destination.symlink_to(source)
                else:
                    destination.hardlink_to(source)
            try:
                validate_plan(plan, temporary_root)
            except ValueError as error:
                if "bound target-unread planning evidence mismatch" not in str(error):
                    raise AssertionError(str(error)) from error
            else:
                raise AssertionError(
                    f"tampered evidence accepted: {evidence_name}/{mode}"
                )
    return len(tamper_cases) + 4


def _bound_root_and_plan() -> tuple[Path, Path]:
    validator_path = Path(__file__).absolute()
    root = validator_path.parents[2]
    plan_path = root / PLAN_RELATIVE
    if not (
        validator_path == root / VALIDATOR_RELATIVE
        and validator_path.is_file()
        and not validator_path.is_symlink()
        and validator_path.resolve(strict=True) == validator_path
        and plan_path.is_file()
        and not plan_path.is_symlink()
        and plan_path.resolve(strict=True) == plan_path
    ):
        raise ValueError(
            "validator or plan path is indirect, noncanonical, or misplaced"
        )
    return root, plan_path


def _decode_recovery_v3_launch(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate recovery-v3 launch-intent key")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid recovery-v3 launch intent") from error
    if not isinstance(value, dict):
        raise ValueError("recovery-v3 launch intent is not an object")
    commit = value.get("git_commit")
    expected = copy.deepcopy(OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_FIELDS)
    expected["git_commit"] = commit
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or value != expected
    ):
        raise ValueError("recovery-v3 launch intent drifted")
    return value


def _read_recovery_v3_launch_marker(path: Path, label: str) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing {label}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is indirect or non-regular")
    if before.st_size > 65_536:
        raise ValueError(f"{label} exceeds the byte limit")
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
            or current.st_size != before.st_size
        ):
            raise ValueError(f"{label} changed before reading")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(before.st_size + 1)
        if len(raw) != before.st_size:
            raise ValueError(f"{label} changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _require_committed_recovery_v3_launcher(root: Path) -> None:
    current = _read_recovery_v3_launch_marker(
        root / VALIDATOR_RELATIVE, "recovery-v3 launcher source"
    )
    process = subprocess.Popen(
        ["git", "show", f"HEAD:{VALIDATOR_RELATIVE.as_posix()}"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    committed = process.stdout.read(2_000_001)
    if len(committed) > 2_000_000:
        process.kill()
        process.wait(timeout=30)
        raise ValueError("committed recovery-v3 launcher exceeds the byte limit")
    if process.wait(timeout=30) != 0 or committed != current:
        raise ValueError("recovery-v3 launcher bytes do not match committed HEAD")


def _run_openmm86_deposited_ph_recovery_v3(root: Path) -> int:
    """Consume one local metadata-fetch intent through the benchmark gateway."""
    intent = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_INTENT
    consumed = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_CONSUMED_INTENT
    output_receipt = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_OUTPUT / "receipt.json"
    active_exists = os.path.lexists(intent)
    consumed_exists = os.path.lexists(consumed)
    if active_exists and consumed_exists:
        active_stat = os.lstat(intent)
        consumed_stat = os.lstat(consumed)
        if (
            not stat.S_ISREG(active_stat.st_mode)
            or not stat.S_ISREG(consumed_stat.st_mode)
            or (active_stat.st_dev, active_stat.st_ino)
            != (consumed_stat.st_dev, consumed_stat.st_ino)
        ):
            raise ValueError("different active and consumed recovery-v3 intents coexist")
        _decode_recovery_v3_launch(
            _read_recovery_v3_launch_marker(intent, "active recovery-v3 launch intent")
        )
        os.unlink(intent)
        active_exists = False
    if consumed_exists:
        _decode_recovery_v3_launch(
            _read_recovery_v3_launch_marker(
                consumed, "consumed recovery-v3 launch intent"
            )
        )
        if output_receipt.is_symlink() or not output_receipt.is_file():
            raise ValueError("consumed recovery-v3 launch lacks its sealed receipt")
        return 0
    if not active_exists:
        return 0
    launch = _decode_recovery_v3_launch(
        _read_recovery_v3_launch_marker(intent, "active recovery-v3 launch intent")
    )
    _require_committed_recovery_v3_launcher(root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if head != launch["git_commit"]:
        raise ValueError("recovery-v3 launch intent is not bound to current HEAD")
    from solution_state_openmm86_deposited_ph_catalog_recovery_v3 import (
        fetch_catalog as fetch_recovery_v3_catalog,
    )
    from check_solution_state_openmm86_deposited_ph_catalog_recovery_v3 import (
        verify_artifact as verify_recovery_v3_artifact,
    )

    output = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_OUTPUT
    if not os.path.lexists(output):
        fetch_recovery_v3_catalog()
    replay_checks, replay_status = verify_recovery_v3_artifact()
    os.link(intent, consumed, follow_symlinks=False)
    os.unlink(intent)
    print(
        "METRIC openmm86_deposited_ph_recovery_v3_execution_checks="
        f"{replay_checks}"
    )
    print(f"OPENMM86_DEPOSITED_PH_RECOVERY_V3_LOCAL_STATUS {replay_status}")
    return 1


def _recovery_v3_launch_self_test() -> int:
    checks = 0
    valid = copy.deepcopy(OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_FIELDS)
    valid["git_commit"] = "1" * 40
    raw = (json.dumps(valid, sort_keys=True) + "\n").encode()
    assert _decode_recovery_v3_launch(raw) == valid
    checks += 1
    try:
        _decode_recovery_v3_launch(b'{"git_commit":"' + b"1" * 40 + b'","git_commit":"' + b"1" * 40 + b'"}')
    except ValueError:
        checks += 1
    else:
        raise AssertionError("duplicate recovery-v3 launch key was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_INTENT
        path.parent.mkdir(parents=True)
        path.symlink_to("absent.json")
        try:
            _run_openmm86_deposited_ph_recovery_v3(root)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("dangling recovery-v3 launch symlink was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        active = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_LAUNCH_INTENT
        consumed = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_CONSUMED_INTENT
        active.parent.mkdir(parents=True)
        active.write_bytes(raw)
        os.link(active, consumed)
        receipt = root / OPENMM86_DEPOSITED_PH_RECOVERY_V3_OUTPUT / "receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_bytes(b"{}\n")
        assert _run_openmm86_deposited_ph_recovery_v3(root) == 0
        assert not os.path.lexists(active) and consumed.is_file()
        checks += 1
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    args = parser.parse_args()
    root, path = _bound_root_and_plan()
    plan = json.loads(path.read_text())
    checks = validate_plan(plan, root)
    checks.append("validator_and_plan_exact_paths")
    negative_checks = (
        self_test(plan, root) + _recovery_v3_launch_self_test()
        if args.self_test
        else 0
    )
    from check_nested_support_all_atom_recount import (
        self_test as recount_checker_self_test,
        verify as verify_recount,
    )
    from check_nested_support_all_atom_raw_evidence import (
        verify as verify_raw_evidence,
    )
    from check_solution_state_protonation_support_plan import (
        verify as verify_solution_state_support_plan,
    )
    from solution_state_condition_catalog import (
        self_test as solution_state_condition_catalog_self_test,
    )
    from check_solution_state_condition_catalog import (
        self_test as solution_state_condition_catalog_checker_self_test,
    )
    from solution_state_openmm86_deposited_ph_catalog import (
        self_test as openmm86_deposited_ph_catalog_self_test,
    )
    from check_solution_state_openmm86_deposited_ph_catalog import (
        self_test as openmm86_deposited_ph_catalog_checker_self_test,
    )
    from solution_state_openmm86_deposited_ph_catalog_recovery_v2 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v2_self_test,
    )
    from check_solution_state_openmm86_deposited_ph_catalog_recovery_v2 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v2_checker_self_test,
    )
    from solution_state_openmm86_deposited_ph_catalog_recovery_v3 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v3_self_test,
    )
    from check_solution_state_openmm86_deposited_ph_catalog_recovery_v3 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v3_checker_self_test,
    )
    from solution_state_openmm86_deposited_ph_catalog_recovery_v4 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v4_self_test,
    )
    from check_solution_state_openmm86_deposited_ph_catalog_recovery_v4 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v4_checker_self_test,
    )
    from run_openmm86_deposited_ph_recovery_v4_if_intent import (
        self_test as openmm86_deposited_ph_catalog_recovery_v4_handler_self_test,
    )
    from solution_state_openmm86_deposited_ph_catalog_recovery_v5 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v5_self_test,
    )
    from check_solution_state_openmm86_deposited_ph_catalog_recovery_v5 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v5_checker_self_test,
    )
    from run_openmm86_deposited_ph_recovery_v5_if_intent import (
        self_test as openmm86_deposited_ph_catalog_recovery_v5_handler_self_test,
    )
    from solution_state_openmm86_deposited_ph_catalog_recovery_v6 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v6_self_test,
    )
    from check_solution_state_openmm86_deposited_ph_catalog_recovery_v6 import (
        self_test as openmm86_deposited_ph_catalog_recovery_v6_checker_self_test,
    )
    from run_openmm86_deposited_ph_recovery_v6_if_intent import (
        self_test as openmm86_deposited_ph_catalog_recovery_v6_handler_self_test,
    )
    from check_openmm86_deposited_ph_recovery_v6_evidence import (
        self_test as openmm86_deposited_ph_recovery_v6_evidence_self_test,
    )
    from check_openmm86_deposited_ph_recovery_v6_semantic_independent import (
        self_test as openmm86_deposited_ph_recovery_v6_semantic_self_test,
        verify as verify_openmm86_deposited_ph_recovery_v6_semantics,
    )
    from openmm86_unique_assigned_ph_v1 import (
        self_test as openmm86_unique_assigned_ph_v1_self_test,
    )
    from openmm86_unique_assigned_ph_v1_recovery_v1 import (
        self_test as openmm86_unique_assigned_ph_v1_recovery_v1_self_test,
    )

    recount_checks = (
        recount_checker_self_test(root) if args.self_test else verify_recount(root)
    )
    raw_evidence_checks = verify_raw_evidence(root)
    solution_state_support_checks = verify_solution_state_support_plan(
        run_self_test=args.self_test
    )
    solution_state_condition_catalog_checks = (
        solution_state_condition_catalog_self_test() if args.self_test else 0
    )
    solution_state_condition_catalog_checker_checks = (
        solution_state_condition_catalog_checker_self_test() if args.self_test else 0
    )
    openmm86_deposited_ph_catalog_checks = (
        openmm86_deposited_ph_catalog_self_test() if args.self_test else 0
    )
    openmm86_deposited_ph_catalog_checker_checks = (
        openmm86_deposited_ph_catalog_checker_self_test() if args.self_test else 0
    )
    openmm86_deposited_ph_catalog_recovery_v2_checks = (
        openmm86_deposited_ph_catalog_recovery_v2_self_test() if args.self_test else 0
    )
    openmm86_deposited_ph_catalog_recovery_v2_checker_checks = (
        openmm86_deposited_ph_catalog_recovery_v2_checker_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v3_checks = (
        openmm86_deposited_ph_catalog_recovery_v3_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v3_checker_checks = (
        openmm86_deposited_ph_catalog_recovery_v3_checker_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v4_checks = (
        openmm86_deposited_ph_catalog_recovery_v4_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v4_checker_checks = (
        openmm86_deposited_ph_catalog_recovery_v4_checker_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v4_handler_checks = (
        openmm86_deposited_ph_catalog_recovery_v4_handler_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v5_checks = (
        openmm86_deposited_ph_catalog_recovery_v5_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v5_checker_checks = (
        openmm86_deposited_ph_catalog_recovery_v5_checker_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v5_handler_checks = (
        openmm86_deposited_ph_catalog_recovery_v5_handler_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v6_checks = (
        openmm86_deposited_ph_catalog_recovery_v6_self_test() if args.self_test else 0
    )
    openmm86_deposited_ph_catalog_recovery_v6_checker_checks = (
        openmm86_deposited_ph_catalog_recovery_v6_checker_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_catalog_recovery_v6_handler_checks = (
        openmm86_deposited_ph_catalog_recovery_v6_handler_self_test()
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_recovery_v6_evidence_checks = (
        openmm86_deposited_ph_recovery_v6_evidence_self_test(root)
        if args.self_test
        else 0
    )
    openmm86_deposited_ph_recovery_v6_semantic_checks = (
        verify_openmm86_deposited_ph_recovery_v6_semantics(root)
        + (
            openmm86_deposited_ph_recovery_v6_semantic_self_test(root)
            if args.self_test
            else 0
        )
    )
    openmm86_unique_assigned_ph_v1_checks = (
        openmm86_unique_assigned_ph_v1_self_test() if args.self_test else 0
    )
    openmm86_unique_assigned_ph_v1_recovery_v1_checks = (
        openmm86_unique_assigned_ph_v1_recovery_v1_self_test()
        if args.self_test
        else 0
    )
    if args.self_test:
        from check_nested_support_all_atom_recount_raw_v3 import (
            self_test as raw_recount_self_test,
        )
        from nested_support_all_atom_recount import self_test as recount_self_test

        negative_checks += recount_self_test() + raw_recount_self_test()
    if not args.acknowledge_hold_only:
        print("STATUS HOLD_FEASIBILITY_BLOCKED")
        print("REFUSAL this artifact is not science or authorization clearance")
        return 3
    openmm86_deposited_ph_recovery_v3_execution_checks = (
        _run_openmm86_deposited_ph_recovery_v3(root) if args.self_test else 0
    )
    print(
        f"METRIC support_count_plan_checks="
        f"{len(checks) + negative_checks + recount_checks + raw_evidence_checks + solution_state_support_checks + solution_state_condition_catalog_checks + solution_state_condition_catalog_checker_checks + openmm86_deposited_ph_catalog_checks + openmm86_deposited_ph_catalog_checker_checks + openmm86_deposited_ph_catalog_recovery_v2_checks + openmm86_deposited_ph_catalog_recovery_v2_checker_checks + openmm86_deposited_ph_catalog_recovery_v3_checks + openmm86_deposited_ph_catalog_recovery_v3_checker_checks + openmm86_deposited_ph_catalog_recovery_v4_checks + openmm86_deposited_ph_catalog_recovery_v4_checker_checks + openmm86_deposited_ph_catalog_recovery_v4_handler_checks + openmm86_deposited_ph_catalog_recovery_v5_checks + openmm86_deposited_ph_catalog_recovery_v5_checker_checks + openmm86_deposited_ph_catalog_recovery_v5_handler_checks + openmm86_deposited_ph_catalog_recovery_v6_checks + openmm86_deposited_ph_catalog_recovery_v6_checker_checks + openmm86_deposited_ph_catalog_recovery_v6_handler_checks + openmm86_deposited_ph_recovery_v6_evidence_checks + openmm86_deposited_ph_recovery_v6_semantic_checks + openmm86_unique_assigned_ph_v1_checks + openmm86_unique_assigned_ph_v1_recovery_v1_checks + openmm86_deposited_ph_recovery_v3_execution_checks}"
    )
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_FEASIBILITY_BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
