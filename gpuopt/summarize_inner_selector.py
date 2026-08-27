#!/usr/bin/env python3
"""Validate an inner-selector receipt and print autoresearch metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MENU_SHA256 = "3d120deccf219f517938467bd0e18fb4f2fffe1d9444b09250c6f908e4cac42f"


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--container-image-sha256", required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    receipt_sha = payload.pop("receipt_sha256", None)
    if (
        payload.get("artifact_kind") != "v3339_phase_d_inner_chart_selector_v1"
        or receipt_sha != canonical_sha256(payload)
        or payload.get("trainer_sha256") != args.trainer_sha256
        or payload.get("menu_sha256") != MENU_SHA256
        or payload.get("input_manifest_sha256") != args.input_manifest_sha256
        or payload.get("source_bundle_sha256") != args.source_bundle_sha256
        or payload.get("container_image_sha256") != args.container_image_sha256
        or payload.get("evaluation_scope")
        != "fixed_checkpoint_post_q_inference_chart_only"
        or payload.get("retraining_or_training_promotion_permitted") is not False
        or payload.get("outer_held_or_external_values_read") is not False
        or payload.get("broad_promotion_permitted") is not False
        or payload.get("selection_contract", {}).get("outer_result_path_visible") is not False
        or len(payload.get("folds", ())) != 3
    ):
        raise ValueError("inner-selector receipt validation failed")
    print(
        "METRIC inner_dev_minfold_five_family_ccc="
        f"{float(payload['evaluated_model_inner_dev_min_fold_five_family_macro_ccc']):.12f}"
    )
    print(
        "METRIC inner_dev_mean_five_family_ccc="
        f"{float(payload['evaluated_model_inner_dev_mean_five_family_macro_ccc']):.12f}"
    )
    print(f"METRIC selector_passing_candidate_count={int(payload['passing_candidate_count'])}")
    print(f"METRIC hard_guards_pass={int(payload['hard_guards_pass'])}")
    print(
        f"selector_decision={payload['selector_decision']} "
        f"selected_candidate_id={payload['selected_candidate_id']} "
        f"evaluated_model_id={payload['evaluated_model_id']}"
    )
    print(f"selector_receipt_sha256={receipt_sha}")


if __name__ == "__main__":
    main()
