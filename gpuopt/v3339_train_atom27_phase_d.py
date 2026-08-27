#!/usr/bin/env python3
"""Train or preflight the frozen Phase-D atom27 model without changing its surface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROUTES = ("legacy5", "all_exact_atoms")
D2_SOURCE_SHA256 = "938b3f50f36665fa2ca6989a51edd02ae1da907613a76023eccde01a31f69655"
MATERIALIZED_ATOM27_SOURCE_SHA256 = "938b3f50f36665fa2ca6989a51edd02ae1da907613a76023eccde01a31f69655"
MATERIALIZER_SOURCE_SHA256 = "6f8d4b3f34a2370d8082a80f40c0dcb2e5ed182e9422bdd589abd8de81c10e0a"
RESELECTION_SOURCE_SHA256 = "141737fb8ad643e44650309f39981c1bbeca33dbceaa3015d5a8304c7d13f8d0"
V6_TRAINER_SHA256 = "da18b84a3787a3a5d7b2828d821e9ef7cdb00fb682925f1cc05f9db721d7a672"
Q_TRANSFER_RECEIPT_SHA256 = "dc152898745191ad262407a9b1e41d89dbfa7b058163fb2b25e759bca5f695b9"
COMPARISON_CONTRACT_SHA256 = "705a58f38ba84da311d5e5303934631268fa8965c940d68865e4f7f0c5e0216e"
FIXED_SUPPORT_SHA256 = "e60561799a751e1afc047d724cdc58f0852a802d3081a8bbdf4a696b97986f72"
STAGE_A_SOURCE_SHA256 = "dba72786b35fbfadf99f425252e394eb318487e570b8e25f00889a3073051416"
EXPECTED_STAGE_A_SHA256 = {
    0: "b3e2d78c468db2875a87061dc83a4bf702c6835c6f75a1a2cfb70f9fa8445990",
    1: "5e6e810476379f03412eba028ffdd62d0406cc71f4415dd4b93f62b8d7d1ada8",
    2: "58035a3559ea152bd1564eb210230f67fbe5278985b18d9a9cbde92112e82b69",
}
EXPECTED_ROLE_COUNTS = {
    0: {"inner_train": 116, "inner_dev": 19, "outer_held": 100},
    1: {"inner_train": 140, "inner_dev": 32, "outer_held": 63},
    2: {"inner_train": 131, "inner_dev": 32, "outer_held": 72},
}
FOLD_SURFACE = {
    0: (24_825, "3ed901e22f84c379542ef53ea2bd09684a9586f73f57d993da8d075a9ff49257"),
    1: (14_586, "7515667c9f4350bae269de7dc6db6ab63ad66f72b6ba1a37076376c4eb587e50"),
    2: (15_221, "fbaa5c09ec527282611f333f77cc7f1e80b188920d563d38a379d5f3b6300958"),
}
TARGET_KEYS = (
    "residue_index", "atom_index", "nucleus_index", "stratum_index",
    "structure_slot", "bonded_parent_slot", "equivalent_structure_slots",
    "direct_literal_support_mask", "target_values", "target_centers",
    "target_scales", "conditioning_mask", "scoring_mask", "target_weights",
)
TARGET_METADATA_KEYS = ("target_ids", "atom_names", "atom_strata", "legacy_families")
RESULT_FILES = ("checkpoint.pt", "history.json", "predictions.tsv", "summary.json")
TRANSFER_PREFIXES = ("evidence.", "support.", "measure.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(filename: str, digest: str, name: str) -> Any:
    path = HERE / filename
    if sha256_file(path) != digest:
        raise ValueError(f"pinned source SHA256 drift: {filename}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import pinned source: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D2 = _load_source("v3339_atom27_explicit_h.py", D2_SOURCE_SHA256, "v3339_atom27_train_model")
V6 = _load_source("v3339_train_stage_bc.py", V6_TRAINER_SHA256, "v3339_atom27_v6_train_helpers")


def set_private(path: Path) -> None:
    path.chmod(0o600)


def binding_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def state_sha256(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> str:
    selected = [
        (key, state[key]) for key in sorted(state)
        if any(key.startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        raise ValueError("state selection is empty")
    digest = hashlib.sha256()
    for key, value in selected:
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode())
        digest.update(b"\n")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    if not path.parent.is_dir() or path.is_dir():
        raise ValueError("resume path must be a file under an existing directory")
    pending = path.with_name(f".{path.name}.pending.{os.getpid()}")
    if pending.exists():
        raise ValueError("stale pending resume path exists")
    try:
        torch.save(payload, pending)
        set_private(pending)
        os.replace(pending, path)
        set_private(path)
    except BaseException:
        if pending.exists():
            pending.unlink()
        raise


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    set_private(path)


def read_manifest(path: Path, role: str, fold: int, support_k: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "entity_uid", "role", "outer_fold", "batch_path", "batch_sha256",
        "parent_v6_batch_sha256", "stage_a_checkpoint_sha256",
        "support_bundle_sha256", "support_count", "target_count",
        "supported_entity_uid_roster_sha256", "coordinate_scope",
        "literal_all_hydrogen_conformational_model", "atom27_source_sha256",
        "materializer_source_sha256", "support_reselection_source_sha256",
        "atom27_support_roster_sha256", "scoring_surface_sha256",
    }
    if not rows or required - rows[0].keys():
        raise ValueError("atom27 manifest schema drift")
    if len(rows) != EXPECTED_ROLE_COUNTS[fold][role]:
        raise ValueError("atom27 manifest role count drift")
    if len({row["entity_uid"] for row in rows}) != len(rows):
        raise ValueError("atom27 manifest duplicates an entity")
    for row in rows:
        if (
            row["role"] != role
            or int(row["support_count"]) != support_k
            or row["stage_a_checkpoint_sha256"] != EXPECTED_STAGE_A_SHA256[fold]
            or row["supported_entity_uid_roster_sha256"] != D2.PINNED_SUPPORT_ROSTER_SHA256
            or row["coordinate_scope"] != D2.COORDINATE_SCOPE
            or row["literal_all_hydrogen_conformational_model"] != "true"
            or row["atom27_source_sha256"] != MATERIALIZED_ATOM27_SOURCE_SHA256
            or row["materializer_source_sha256"] != MATERIALIZER_SOURCE_SHA256
            or row["support_reselection_source_sha256"] != RESELECTION_SOURCE_SHA256
            or len(row["atom27_support_roster_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["atom27_support_roster_sha256"])
            or row["scoring_surface_sha256"] != D2.PINNED_SCORING_SURFACE_SHA256
        ):
            raise ValueError("atom27 manifest lineage/role contract drift")
        batch_path = Path(row["batch_path"])
        if not batch_path.is_file() or batch_path.is_symlink():
            raise ValueError("atom27 batch path is absent or symbolic")
        if sha256_file(batch_path) != row["batch_sha256"]:
            raise ValueError("atom27 batch SHA256 drift")
    return sorted(rows, key=lambda row: row["entity_uid"])


def load_batch(row: dict[str, str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(row["batch_path"])
    if sha256_file(path) != row["batch_sha256"]:
        raise ValueError("atom27 batch changed after manifest validation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("artifact_kind") != "v3339_phase_d7_atom27_entity_batch_v1"
        or payload.get("parent_v6_batch_sha256") != row["parent_v6_batch_sha256"]
        or payload.get("stage_a_checkpoint_sha256") != row["stage_a_checkpoint_sha256"]
        or payload.get("support_bundle_sha256") != row["support_bundle_sha256"]
        or payload.get("atom27_source_sha256") != MATERIALIZED_ATOM27_SOURCE_SHA256
        or payload.get("materializer_source_sha256") != MATERIALIZER_SOURCE_SHA256
        or payload.get("support_reselection_source_sha256") != RESELECTION_SOURCE_SHA256
        or payload.get("atom27_support_roster_sha256") != row["atom27_support_roster_sha256"]
    ):
        raise ValueError("atom27 batch artifact lineage drift")
    batch, metadata = payload.get("batch"), payload.get("metadata")
    if not isinstance(batch, dict) or not isinstance(metadata, dict):
        raise ValueError("atom27 batch tensors/metadata are absent")
    if (
        metadata.get("entity_uid") != row["entity_uid"]
        or metadata.get("role") != row["role"]
        or metadata.get("coordinate_scope") != D2.COORDINATE_SCOPE
        or metadata.get("literal_all_hydrogen_conformational_model") is not True
        or int(metadata.get("atom_slots", -1)) != D2.ATOM_COUNT
        or metadata.get("atom27_aware_support_reselection") is not True
        or metadata.get("atom27_support_roster_sha256") != row["atom27_support_roster_sha256"]
        or len(metadata.get("atom27_support_ids", ())) != int(row["support_count"])
        or int(batch["atom27_positions"].shape[1]) != int(row["support_count"])
        or int(batch["target_values"].shape[1]) != int(row["target_count"])
    ):
        raise ValueError("atom27 batch shape/scope drift")
    return batch, metadata


def route_batch(
    batch: dict[str, torch.Tensor], metadata: dict[str, Any], route: str,
    *, evaluation: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if route not in ROUTES:
        raise ValueError("unknown Phase-D route")
    target_count = int(batch["target_values"].shape[1])
    values = {key: list(metadata.get(key, ())) for key in TARGET_METADATA_KEYS}
    if any(len(value) != target_count for value in values.values()):
        raise ValueError("target metadata length drift")
    legacy = torch.tensor([
        family in {"C'", "CA", "CB", "HN", "N"}
        for family in values["legacy_families"]
    ])
    keep = torch.ones(target_count, dtype=torch.bool) if evaluation or route == "all_exact_atoms" else legacy
    if not bool(keep.any()):
        raise ValueError("route would silently drop an entity")
    output = dict(batch)
    for key in TARGET_KEYS:
        value = batch[key]
        if key == "direct_literal_support_mask":
            output[key] = value[:, :, keep]
        else:
            output[key] = value[:, keep]
    if route == "legacy5" and evaluation:
        output["conditioning_mask"] &= legacy[None]
    if not bool(output["scoring_mask"].any()):
        raise ValueError("route has no scoring rows")
    output["target_weights"] = D2.V6_MODEL.balanced_target_weights(
        output["stratum_index"], output["atom_index"], output["scoring_mask"]
    )
    routed = dict(metadata)
    for key, value in values.items():
        routed[key] = [item for item, selected in zip(value, keep.tolist(), strict=True) if selected]
    return output, routed


def load_comparison_contract(path: Path) -> dict[str, Any]:
    if sha256_file(path) != COMPARISON_CONTRACT_SHA256:
        raise ValueError("fixed v6 comparison surface contract SHA256 drift")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        len(contract.get("evaluation_atom_names", ())) != 69
        or len(contract.get("primary_evaluation_atom_names", ())) != 29
        or contract.get("target_only_eligibility", {}).get("scoring_surface_sha256")
        != D2.PINNED_SCORING_SURFACE_SHA256
        or int(contract.get("target_only_eligibility", {}).get("scoring_record_count", -1))
        != D2.EXPECTED_SCORING_ROW_COUNT
    ):
        raise ValueError("fixed 69-atom/29-primary scoring contract drift")
    return contract


def initialize_model(
    q_checkpoint_path: Path,
    q_receipt_path: Path,
    fixed_support_source: Path,
    *, route: str,
    fold: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if sha256_file(q_receipt_path) != Q_TRANSFER_RECEIPT_SHA256:
        raise ValueError("Q initialization-transfer receipt SHA256 drift")
    if sha256_file(fixed_support_source) != FIXED_SUPPORT_SHA256:
        raise ValueError("fixed support generator SHA256 drift")
    receipt = json.loads(q_receipt_path.read_text(encoding="utf-8"))
    matches = [
        row for row in receipt.get("checkpoints", ())
        if row.get("route") == route and int(row.get("outer_fold", -1)) == fold
    ]
    if len(matches) != 1:
        raise ValueError("Q initialization receipt lacks the route/fold")
    declared = matches[0]
    if sha256_file(q_checkpoint_path) != declared["checkpoint_sha256"]:
        raise ValueError("Q initialization checkpoint SHA256 drift")
    checkpoint = torch.load(q_checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("stage") != "stage_b_shared_q"
        or checkpoint.get("route") != route
        or int(checkpoint.get("outer_fold", -1)) != fold
        or checkpoint.get("model_source_sha256") != receipt["old_v6_model_source_sha256"]
    ):
        raise ValueError("Q initialization checkpoint identity drift")
    transfer = {
        key: value for key, value in checkpoint["model_state_dict"].items()
        if any(key.startswith(prefix) for prefix in TRANSFER_PREFIXES)
    }
    if (
        sorted(transfer) != receipt["transferable_keys"]
        or len(transfer) != 16
        or state_sha256(transfer, TRANSFER_PREFIXES) != declared["transfer_state_sha256"]
    ):
        raise ValueError("Q initialization key/shape/value receipt drift")
    config = dict(checkpoint["model_config"])
    config["fixed_support_source"] = fixed_support_source
    config["actuator_enabled"] = True
    torch.manual_seed(3339)
    model = D2.GlobalAllAtomSharedQ(**config)
    fresh = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = model.load_state_dict(transfer, strict=False)
    expected_missing = sorted(
        key for key in model.state_dict()
        if key.startswith("observer.") or key.startswith("actuator.")
    )
    if result.unexpected_keys or sorted(result.missing_keys) != expected_missing:
        raise ValueError("atom27 initialization transfer architecture drift")
    loaded = model.state_dict()
    if any(not torch.equal(loaded[key], fresh[key]) for key in expected_missing):
        raise ValueError("fresh atom27 observer/actuator was overwritten")
    if any(not torch.equal(loaded[key], value) for key, value in transfer.items()):
        raise ValueError("Q initialization did not transfer exactly")
    model.to(device)
    return model, checkpoint, {
        "classification": "initialization_transfer_not_lineage_continuation",
        "q_checkpoint_sha256": declared["checkpoint_sha256"],
        "transfer_state_sha256": declared["transfer_state_sha256"],
        "transfer_key_count": 16,
        "old_atom14_observer_reused": False,
        "fresh_components": ["observer", "actuator"],
    }


def validate_resource_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required_true = (
        "global_worst_shape_selected", "k8_forward_backward_pass",
        "k32_forward_no_grad_pass", "k32_backward_forbidden_and_not_run",
        "complete_train_only_epoch_pass", "deterministic_resume_replay_pass",
        "failed_scale_scan_no_grad", "selected_scale_replayed_once_with_gradient",
        "projection_implementation_atom27_native",
        "observer_common_rbf_once_per_target_chunk",
        "observer_full_target_chunk_checkpoint_use_reentrant_false",
    )
    if (
        receipt.get("artifact_kind") != "v3339_phase_d_atom27_h200_preflight_v1"
        or receipt.get("atom27_source_sha256") != D2_SOURCE_SHA256
        or any(receipt.get(key) is not True for key in required_true)
        or receipt.get("gpu_name") != "NVIDIA H200"
        or receipt.get("outer_predictions_or_metrics_computed") is not False
        or receipt.get("global_worst_k8_shape", [])[:2] != [406, 2346]
        or receipt.get("global_worst_k32_shape", [])[:2] != [406, 2346]
    ):
        raise ValueError("H200 atom27 resource/train-only receipt is incomplete")
    return receipt


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_entity(
    model: torch.nn.Module,
    stage_a: torch.nn.Module,
    row: dict[str, str],
    *, route: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    evaluation: bool = False,
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    raw, metadata = load_batch(row)
    routed, routed_metadata = route_batch(raw, metadata, route, evaluation=evaluation)
    batch = V6.attach_frozen_stage_a(batch_to_device(routed, device), stage_a)
    training = optimizer is not None
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        output = model(batch)
        if not torch.isfinite(output["loss"]):
            raise ValueError("atom27 loss became nonfinite")
        gradient_l2: dict[str, float] = defaultdict(float)
        if training:
            output["loss"].backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient_l2[name.split(".", 1)[0]] += float(
                        parameter.grad.detach().float().square().sum().cpu()
                    )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    metrics = {
        "loss": float(output["loss"].detach().cpu()),
        "data_loss": float(output["data_loss"].detach().cpu()),
        "prior_kl": float(output["prior_kl"].detach().cpu()),
        "actuated_clash_penalty": float(output["actuated_clash_penalty"].detach().cpu()),
        "actuated_feasibility_scale": float(output["actuated_feasibility_scale"].detach().cpu()),
        "observer_gradient_square": gradient_l2.get("observer", 0.0),
        "actuator_gradient_square": gradient_l2.get("actuator", 0.0),
        "measure_gradient_square": gradient_l2.get("measure", 0.0),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("atom27 training metric became nonfinite")
    return metrics, output, routed_metadata, batch


def one_epoch(
    model: torch.nn.Module,
    stage_a: torch.nn.Module,
    rows: list[dict[str, str]],
    *, route: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, Any]:
    model.train(optimizer is not None)
    order = list(rows)
    random.Random(seed).shuffle(order)
    records: list[dict[str, float]] = []
    for row in order:
        metrics, _, _, _ = run_entity(
            model, stage_a, row, route=route, device=device, optimizer=optimizer
        )
        records.append(metrics)
    summary = {
        key: float(np.mean([record[key] for record in records]))
        for key in ("loss", "data_loss", "prior_kl", "actuated_clash_penalty")
    }
    summary.update({
        "minimum_actuated_feasibility_scale": min(
            record["actuated_feasibility_scale"] for record in records
        ),
        "observer_gradient_l2": math.sqrt(sum(
            record["observer_gradient_square"] for record in records
        )),
        "actuator_gradient_l2": math.sqrt(sum(
            record["actuator_gradient_square"] for record in records
        )),
        "measure_gradient_l2": math.sqrt(sum(
            record["measure_gradient_square"] for record in records
        )),
        "entity_count": len(records),
    })
    if optimizer is not None and any(
        summary[key] <= 0 for key in (
            "observer_gradient_l2", "actuator_gradient_l2", "measure_gradient_l2"
        )
    ):
        raise ValueError("complete atom27 epoch did not reach observer/actuator/shared-q")
    return summary


@torch.no_grad()
def score_outer_once(
    model: torch.nn.Module,
    stage_a: torch.nn.Module,
    rows: list[dict[str, str]],
    *, route: str,
    fold: int,
    device: torch.device,
    evaluation_atoms: tuple[str, ...],
    primary_atoms: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for row in rows:
        _, output, metadata, batch = run_entity(
            model, stage_a, row, route=route, device=device,
            optimizer=None, evaluation=True,
        )
        predictions.extend(V6.prediction_records(
            metadata, batch, output["ensemble_mean"], output["ensemble_scale"]
        ))
    surface = V6.scoring_surface_receipt(predictions)
    expected_count, expected_sha = FOLD_SURFACE[fold]
    if surface != {
        "scoring_record_count": expected_count,
        "scoring_surface_sha256": expected_sha,
        "target_value_dtype": "float32",
    }:
        raise ValueError("atom27 outer scoring surface changed")
    metrics = V6.prediction_metrics(
        predictions, evaluation_atoms, primary_atoms, require_all_primary=False
    )
    return predictions, metrics


def validate_split(
    train: list[dict[str, str]], dev: list[dict[str, str]], held: list[dict[str, str]], fold: int
) -> None:
    groups = [{row["entity_uid"] for row in rows} for rows in (train, dev, held)]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("atom27 train/dev/held entity leakage")
    union = set().union(*groups)
    roster = hashlib.sha256("".join(f"{uid}\n" for uid in sorted(union)).encode()).hexdigest()
    if len(union) != 235 or roster != D2.PINNED_SUPPORT_ROSTER_SHA256:
        raise ValueError("atom27 split does not preserve the exact 235 cohort")
    if any(int(row["outer_fold"]) == fold for row in train + dev):
        raise ValueError("outer fold entered train/inner-dev")
    if {int(row["outer_fold"]) for row in held} != {fold}:
        raise ValueError("outer-held fold identity drift")


def verify_existing_result(output_dir: Path, binding_sha: str) -> dict[str, Any]:
    names = set(RESULT_FILES) | {"checksums.sha256"}
    if output_dir.is_symlink() or {path.name for path in output_dir.iterdir()} != names:
        raise ValueError("completed atom27 result inventory drift")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("binding_sha256") != binding_sha:
        raise ValueError("completed atom27 result binding drift")
    expected: dict[str, str] = {}
    for line in (output_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    if set(expected) != set(RESULT_FILES):
        raise ValueError("completed atom27 checksum inventory drift")
    for name, digest in expected.items():
        path = output_dir / name
        if path.is_symlink() or (path.stat().st_mode & 0o777) != 0o600 or sha256_file(path) != digest:
            raise ValueError("completed atom27 result checksum/mode drift")
    return summary


def run_production(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs != 40 or args.seed != 3339 or args.learning_rate != 3.0e-4:
        raise ValueError("Phase-D production schedule must be exact 40/3339/0.0003")
    resource = validate_resource_receipt(args.resource_receipt)
    contract = load_comparison_contract(args.comparison_contract)
    train = read_manifest(args.train_manifest, "inner_train", args.outer_fold, 8)
    dev = read_manifest(args.dev_manifest, "inner_dev", args.outer_fold, 8)
    held = read_manifest(args.outer_held_manifest, "outer_held", args.outer_fold, 32)
    validate_split(train, dev, held, args.outer_fold)
    device = torch.device(args.device)
    if device.type == "cuda" and (
        torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
    ):
        raise ValueError("container must expose exactly CUDA ordinal 0")
    runtime = V6.configure_deterministic_runtime(device)
    V6.seed_runtime(args.seed, device)
    stage_a, _ = V6.load_frozen_stage_a(
        args.stage_a_source, STAGE_A_SOURCE_SHA256,
        args.stage_a_checkpoint, EXPECTED_STAGE_A_SHA256[args.outer_fold],
        outer_fold=args.outer_fold, device=device,
    )
    model, _, initialization = initialize_model(
        args.q_checkpoint, args.q_transfer_receipt, args.fixed_support_source,
        route=args.route, fold=args.outer_fold, device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    binding = {
        "artifact_kind": "v3339_phase_d_atom27_production_binding_v1",
        "route": args.route, "outer_fold": args.outer_fold,
        "atom27_source_sha256": D2_SOURCE_SHA256,
        "q_initialization": initialization,
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "dev_manifest_sha256": sha256_file(args.dev_manifest),
        "outer_held_manifest_sha256": sha256_file(args.outer_held_manifest),
        "comparison_contract_sha256": COMPARISON_CONTRACT_SHA256,
        "resource_receipt_sha256": sha256_file(args.resource_receipt),
        "stage_a_checkpoint_sha256": EXPECTED_STAGE_A_SHA256[args.outer_fold],
        "schedule": {"epochs": 40, "seed": 3339, "learning_rate": 3.0e-4},
        "runtime": runtime,
    }
    binding_sha = binding_sha256(binding)
    if args.output_dir.exists():
        return verify_existing_result(args.output_dir, binding_sha)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 0
    if args.resume_state.exists():
        resume = torch.load(args.resume_state, map_location=device, weights_only=True)
        if resume.get("binding") != binding or resume.get("binding_sha256") != binding_sha:
            raise ValueError("atom27 resume binding drift")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume["history"])
        best_loss = float(resume["best_loss"])
        best_state = resume["best_state_dict"]
        start_epoch = int(resume["next_epoch"])
        if start_epoch != len(history):
            raise ValueError("atom27 resume epoch/history drift")
    for epoch in range(start_epoch, args.epochs):
        V6.seed_runtime(args.seed + epoch, device)
        train_metrics = one_epoch(
            model, stage_a, train, route=args.route, device=device,
            optimizer=optimizer, seed=args.seed + epoch,
        )
        V6.seed_runtime(args.seed + 1_000_000 + epoch, device)
        dev_metrics = one_epoch(
            model, stage_a, dev, route=args.route, device=device,
            optimizer=None, seed=args.seed + epoch,
        )
        history.append({"epoch": epoch + 1, "inner_train": train_metrics, "inner_dev": dev_metrics})
        if dev_metrics["loss"] < best_loss:
            best_loss = dev_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        atomic_torch_save({
            "artifact_kind": "v3339_phase_d_atom27_epoch_resume_v1",
            "binding": binding, "binding_sha256": binding_sha,
            "next_epoch": epoch + 1, "history": history,
            "best_loss": best_loss, "best_state_dict": best_state,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, args.resume_state)
    if best_state is None:
        raise RuntimeError("atom27 training produced no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    predictions, metrics = score_outer_once(
        model, stage_a, held, route=args.route, fold=args.outer_fold, device=device,
        evaluation_atoms=tuple(contract["evaluation_atom_names"]),
        primary_atoms=tuple(contract["primary_evaluation_atom_names"]),
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.stage.", dir=args.output_dir.parent))
    staging.chmod(0o700)
    try:
        checkpoint = {
            "artifact_kind": "v3339_phase_d_atom27_checkpoint_v1",
            "binding": binding, "binding_sha256": binding_sha,
            "model_state_dict": best_state,
            "selected_inner_dev_loss": best_loss,
            "selection_used_outer_held": False,
            "q_initialization_classification": initialization["classification"],
        }
        torch.save(checkpoint, staging / "checkpoint.pt")
        set_private(staging / "checkpoint.pt")
        write_json(staging / "history.json", history)
        V6.write_predictions(staging / "predictions.tsv", predictions)
        summary = {
            "artifact_kind": "v3339_phase_d_atom27_training_result_v1",
            "binding": binding, "binding_sha256": binding_sha, "route": args.route,
            "outer_fold": args.outer_fold, "epochs": 40,
            "selected_inner_dev_loss": best_loss,
            "outer_held_scored_once_after_selection": True,
            "outer_held_used_for_tuning_or_selection": False,
            "scoring_surface_sha256": FOLD_SURFACE[args.outer_fold][1],
            "scoring_row_count": FOLD_SURFACE[args.outer_fold][0],
            "fixed_primary_atom_count": 29,
            "all_exact_atom_diagnostic_count": 69,
            "metrics": metrics,
            "initialization": initialization,
            "resource_receipt_sha256": sha256_file(args.resource_receipt),
            "checkpoint_sha256": sha256_file(staging / "checkpoint.pt"),
            "predictions_sha256": sha256_file(staging / "predictions.tsv"),
        }
        write_json(staging / "summary.json", summary)
        (staging / "checksums.sha256").write_text("".join(
            f"{sha256_file(staging / name)}  {name}\n" for name in RESULT_FILES
        ), encoding="utf-8")
        set_private(staging / "checksums.sha256")
        os.replace(staging, args.output_dir)
        return summary
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("production",))
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--outer-fold", choices=(0, 1, 2), required=True, type=int)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--outer-held-manifest", required=True, type=Path)
    parser.add_argument("--comparison-contract", required=True, type=Path)
    parser.add_argument("--q-checkpoint", required=True, type=Path)
    parser.add_argument("--q-transfer-receipt", required=True, type=Path)
    parser.add_argument("--fixed-support-source", required=True, type=Path)
    parser.add_argument("--stage-a-source", required=True, type=Path)
    parser.add_argument("--stage-a-checkpoint", required=True, type=Path)
    parser.add_argument("--resource-receipt", required=True, type=Path)
    parser.add_argument("--resume-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--seed", default=3339, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.output_dir.parent.is_dir() or not args.resume_state.parent.is_dir():
        raise ValueError("output/resume parents must exist before launch")
    summary = run_production(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
