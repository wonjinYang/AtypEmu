"""PyTorch student training loop for the first AtypEmu density bridge."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from atypemu.training.benchmarking import render_benchmark_artifacts
from atypemu.training.config import BenchmarkRenderConfig, StudentTrainingConfig
from atypemu.training.ccc_geometry import build_ccc_geometry_artifacts
from atypemu.training.data import (
    AA3_VOCAB,
    CHEMICAL_SHIFT_FAMILIES,
    PreparedObservableChannel,
    ccc_proxy_feature_index,
    evidence_log_likelihood_feature_index,
    load_materialized_examples,
)
from atypemu.training.metrics import (
    aggregate_epoch_metrics,
    collect_chemical_shift_prediction_rows,
    evaluate_prepared_example,
    metric_rows_from_loss_summary,
    select_checkpoint_metric,
)
from atypemu.training.model import StudentDensityModel
from atypemu.training.moment_diffusion import build_moment_diffusion_artifacts
from atypemu.training.objectives import (
    build_support_ceiling_report,
    posterior_evidence_kl_loss,
    support_entropy,
    teacher_kl_loss,
)
from atypemu.training.observable_adapters import (
    build_multi_observable_artifacts,
    observable_adapter_oracle_kl_loss,
)
from atypemu.training.reporting import (
    build_benchmark_report,
    build_chemical_shift_baseline_report,
    build_chemical_shift_calibration_artifacts,
)
from atypemu.training.secondary_shift import (
    random_coil_baselines_for_target_ids,
    random_coil_reference_for_target_id,
)
from atypemu.training.tasks import resolve_training_task
from atypemu.training.tasks import (
    POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
    POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
    POSTERIOR_NMR_CANDIDATE_FREE_V1,
    POSTERIOR_NMR_X0_ENSEMBLE_V1,
)


def run_student_training(
    data_root: str | Path,
    integrated_root: str | Path,
    output_dir: str | Path,
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Train the first density student on materialized offline teachers."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    task_spec = resolve_training_task(config.training_task, config.training_input_mode)
    if task_spec.task_name in {
        POSTERIOR_NMR_CANDIDATE_FREE_V1,
        POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
    }:
        from atypemu.training.candidate_free import run_candidate_free_training

        return run_candidate_free_training(
            data_root=data_root,
            integrated_root=integrated_root,
            output_dir=output_dir_path,
            config=config,
        )
    if task_spec.task_name == POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1:
        from atypemu.training.bioemu_latent_nmr import run_bioemu_latent_nmr_training

        return run_bioemu_latent_nmr_training(
            data_root=data_root,
            integrated_root=integrated_root,
            output_dir=output_dir_path,
            config=config,
        )
    if task_spec.task_name == POSTERIOR_NMR_X0_ENSEMBLE_V1:
        from atypemu.training.x0_posterior_training import (
            run_x0_posterior_ensemble_training,
        )

        return run_x0_posterior_ensemble_training(
            data_root=data_root,
            integrated_root=integrated_root,
            output_dir=output_dir_path,
            config=config,
        )

    train_examples, val_examples, source_vocab = load_materialized_examples(
        data_root=data_root,
        integrated_root=integrated_root,
        config=config,
    )
    if not train_examples:
        raise ValueError(
            "No materialized teacher examples were available for the selected "
            "training splits. Run teacher materialization first."
        )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    candidate_feature_dim = int(train_examples[0].candidate_numeric_features.shape[1])
    evidence_feature_index = evidence_log_likelihood_feature_index(
        candidate_feature_dim=candidate_feature_dim,
        input_mode=task_spec.input_feature_mode,
    )
    ccc_proxy_index = ccc_proxy_feature_index(
        candidate_feature_dim=candidate_feature_dim,
        input_mode=task_spec.input_feature_mode,
    )

    model = StudentDensityModel(
        vocab_size=21,
        num_sources=max(source_vocab.values(), default=-1) + 1,
        sequence_embedding_dim=config.sequence_embedding_dim,
        source_embedding_dim=config.source_embedding_dim,
        example_feature_dim=5,
        candidate_feature_dim=candidate_feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        use_candidate_set_encoder=config.use_candidate_set_encoder,
        set_encoder_layers=config.set_encoder_layers,
        set_encoder_heads=config.set_encoder_heads,
        logit_temperature=config.logit_temperature,
        use_bayesian_posterior=task_spec.is_posterior
        and (
            evidence_feature_index is not None
            or float(config.nmr_energy_guidance_weight) > 0.0
        ),
        evidence_feature_index=evidence_feature_index,
        evidence_temperature=config.evidence_temperature,
        enable_forward_residual_head=config.enable_forward_residual_head,
        enable_residue_atom_residual_head=config.enable_residue_atom_residual_head,
        residue_atom_max_residue_index=config.residue_atom_max_residue_index,
        enable_state_mixture=config.enable_state_mixture,
        state_mixture_count=config.state_mixture_count,
        enable_moment_head=config.enable_moment_head,
        enable_posterior_state_tokens=config.enable_posterior_state_tokens,
        posterior_state_count=config.posterior_state_count,
        enable_latent_posterior_flow=config.enable_latent_posterior_flow,
        latent_flow_steps=config.latent_flow_steps,
        enable_family_specific_moment_head=config.enable_family_specific_moment_head,
        enable_nmr_structural_features=config.enable_nmr_structural_features,
        enable_local_evidence_context=config.enable_local_evidence_context,
        enable_same_residue_evidence_context=(
            config.enable_same_residue_evidence_context
        ),
        same_residue_evidence_target_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.same_residue_evidence_target_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        enable_target_set_evidence_encoder=config.enable_target_set_evidence_encoder,
        enable_target_evidence_token=config.enable_target_evidence_token,
        target_set_encoder_layers=config.target_set_encoder_layers,
        target_set_encoder_heads=config.target_set_encoder_heads,
        target_set_encoder_max_targets=config.target_set_encoder_max_targets,
        target_set_context_gate_init=config.target_set_context_gate_init,
        target_set_encoder_masked_only=config.target_set_encoder_masked_only,
        target_set_encoder_target_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.target_set_encoder_target_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        enable_residue_grid_evidence_encoder=(
            config.enable_residue_grid_evidence_encoder
        ),
        residue_grid_encoder_layers=config.residue_grid_encoder_layers,
        residue_grid_encoder_heads=config.residue_grid_encoder_heads,
        residue_grid_context_gate_init=config.residue_grid_context_gate_init,
        enable_residue_grid_family_gates=config.enable_residue_grid_family_gates,
        residue_grid_encoder_masked_only=config.residue_grid_encoder_masked_only,
        residue_grid_encoder_target_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.residue_grid_encoder_target_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        local_evidence_window=config.local_evidence_window,
        enable_residue_anchor_evidence_context=(
            config.enable_residue_anchor_evidence_context
        ),
        residue_anchor_evidence_offsets=tuple(
            int(offset) for offset in config.residue_anchor_evidence_offsets
        ),
        residue_anchor_evidence_target_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.residue_anchor_evidence_target_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        enable_backbone_evidence_context=config.enable_backbone_evidence_context,
        backbone_evidence_target_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.backbone_evidence_target_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        enable_all_family_evidence_affine_calibration=(
            config.enable_all_family_evidence_affine_calibration
        ),
        evidence_affine_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.evidence_affine_calibration_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        enable_hn_variance_calibration=config.enable_hn_variance_calibration,
        enable_cprime_robust_likelihood=config.enable_cprime_robust_likelihood,
        enable_pair_evidence_mean_blend=config.enable_pair_evidence_mean_blend,
        pair_evidence_mean_blend_weight=config.pair_evidence_mean_blend_weight,
        enable_same_family_evidence_interpolation=(
            config.enable_same_family_evidence_interpolation
        ),
        same_family_evidence_interpolation_weight=(
            config.same_family_evidence_interpolation_weight
        ),
        same_family_evidence_interpolation_sigma=(
            config.same_family_evidence_interpolation_sigma
        ),
        same_family_evidence_interpolation_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.same_family_evidence_interpolation_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        enable_candidate_observable_context=(
            config.enable_candidate_observable_context
        ),
        candidate_observable_context_gate_init=(
            config.candidate_observable_context_gate_init
        ),
        candidate_observable_context_target_family_indices=tuple(
            CHEMICAL_SHIFT_FAMILIES.index(family)
            for family in config.candidate_observable_context_target_families
            if family in CHEMICAL_SHIFT_FAMILIES
        ),
        nmr_energy_guidance_weight=config.nmr_energy_guidance_weight,
        mirror_descent_steps=config.mirror_descent_steps,
        mirror_descent_step_size=config.mirror_descent_step_size,
        ccc_proxy_feature_index=ccc_proxy_index,
        robust_likelihood_mode=config.robust_likelihood_mode,
        student_t_degrees_of_freedom=config.student_t_degrees_of_freedom,
    ).to(device)
    initial_checkpoint_report = maybe_load_initial_checkpoint(
        model=model,
        device=device,
        config=config,
    )
    trainable_parameter_report = apply_trainable_parameter_filter(
        model=model,
        patterns=config.trainable_parameter_name_patterns,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_checkpoint: dict[str, Any] | None = None
    best_comparison_key: tuple[float, ...] | None = None
    history_jsonl_path = output_dir_path / "history.jsonl"
    if history_jsonl_path.exists():
        history_jsonl_path.unlink()

    for epoch in range(1, config.epochs + 1):
        learning_rate = learning_rate_for_epoch(config, epoch)
        set_optimizer_learning_rate(optimizer, learning_rate)
        train_epoch = run_epoch(
            model=model,
            examples=train_examples,
            optimizer=optimizer,
            device=device,
            config=config,
            training=True,
            split_name="train",
        )
        val_epoch = (
            run_epoch(
                model=model,
                examples=val_examples,
                optimizer=None,
                device=device,
                config=config,
                training=False,
                split_name="val",
            )
            if val_examples
            else _empty_epoch_payload(split_name="val")
        )
        checkpoint_split = "val" if val_examples else "train"
        checkpoint_record = select_checkpoint_metric(
            metric_summary=(
                val_epoch["metric_summary"]
                if checkpoint_split == "val"
                else train_epoch["metric_summary"]
            ),
            split=checkpoint_split,
            config=config,
        )
        comparison_key = tuple(checkpoint_record["comparison_key"])

        epoch_record = {
            "epoch": epoch,
            "train": train_epoch,
            "val": val_epoch,
            "checkpoint_metric": checkpoint_record,
            "learning_rate": learning_rate,
        }
        append_learning_rate_metric(train_epoch, learning_rate)
        append_learning_rate_metric(val_epoch, learning_rate)
        history.append(epoch_record)
        append_jsonl(history_jsonl_path, epoch_record)
        save_json(output_dir_path / "history.json", history_window(history, config))

        should_write_last = (
            epoch == config.epochs
            or config.last_checkpoint_interval <= 1
            or epoch % config.last_checkpoint_interval == 0
        )
        if should_write_last:
            save_checkpoint(
                path=output_dir_path / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                source_vocab=source_vocab,
                config=config.as_dict(),
                metrics=epoch_record,
            )
        if best_comparison_key is None or comparison_key < best_comparison_key:
            best_comparison_key = comparison_key
            best_checkpoint = checkpoint_record
            best_epoch = epoch
            save_checkpoint(
                path=output_dir_path / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                source_vocab=source_vocab,
                config=config.as_dict(),
                metrics=epoch_record,
            )
            best_preview_report = maybe_render_best_preview(
                model=model,
                train_examples=train_examples,
                val_examples=val_examples,
                device=device,
                config=config,
                output_dir=output_dir_path,
                epoch=epoch,
                checkpoint_record=checkpoint_record,
            )
            if best_preview_report is not None:
                epoch_record["best_preview"] = best_preview_report
                save_json(
                    output_dir_path / "history.json", history_window(history, config)
                )

    if best_epoch == 0:
        raise RuntimeError("Training completed without selecting a best checkpoint.")
    (output_dir_path / "reports" / "metrics").mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir_path / "reports" / "metrics" / "history.json",
        history_window(history, config),
    )

    best_checkpoint_state = torch.load(
        output_dir_path / "best.pt",
        map_location=device,
    )
    model.load_state_dict(best_checkpoint_state["model_state_dict"])

    best_train_epoch = run_epoch(
        model=model,
        examples=train_examples,
        optimizer=None,
        device=device,
        config=config,
        training=False,
        split_name="train",
        collect_prediction_rows=True,
    )
    best_val_epoch = (
        run_epoch(
            model=model,
            examples=val_examples,
            optimizer=None,
            device=device,
            config=config,
            training=False,
            split_name="val",
            collect_prediction_rows=True,
        )
        if val_examples
        else _empty_epoch_payload(split_name="val")
    )
    learnability_report = build_learnability_report(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        device=device,
        config=config,
    )
    support_ceiling_report = build_support_ceiling_report(
        learnability_report,
        task_contract=task_spec.as_dict(),
    )
    ccc_geometry_report = build_ccc_geometry_artifacts(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        device=device,
        config=config,
        output_dir=output_dir_path,
    )
    support_ceiling_report["ccc_geometry"] = ccc_geometry_report

    benchmark_report = build_benchmark_report(
        data_root=data_root,
        integrated_root=integrated_root,
        train_epoch=best_train_epoch,
        val_epoch=best_val_epoch,
        checkpoint_metric=best_checkpoint or {},
    )
    render_outputs = render_benchmark_artifacts(
        data_root=data_root,
        integrated_root=integrated_root,
        output_dir=output_dir_path,
        epoch_payload=best_val_epoch,
        render_config=BenchmarkRenderConfig.from_dict(config.benchmark_render_config),
    )
    benchmark_report["ensemble_fidelity"] = render_outputs["benchmark_overlay"][
        "ensemble_fidelity"
    ]
    benchmark_report["uncertainty_calibration"] = render_outputs["benchmark_overlay"][
        "uncertainty_calibration"
    ]
    benchmark_report["representative_entity_uid"] = render_outputs["benchmark_overlay"][
        "representative_entity_uid"
    ]
    chemical_shift_baseline_report = build_chemical_shift_baseline_report(
        prediction_rows=best_val_epoch.get("prediction_rows", []),
        config=config,
    )
    calibration_report = build_chemical_shift_calibration_artifacts(
        train_prediction_rows=best_train_epoch.get("prediction_rows", []),
        val_prediction_rows=best_val_epoch.get("prediction_rows", []),
        output_dir=output_dir_path,
    )
    write_forward_residual_predictions(
        output_dir=output_dir_path,
        train_rows=best_train_epoch.get("forward_residual_rows", []),
        val_rows=best_val_epoch.get("forward_residual_rows", []),
    )
    physics_shift_report = build_physics_shift_artifacts(
        output_dir=output_dir_path,
        train_rows=best_train_epoch.get("prediction_rows", []),
        val_rows=best_val_epoch.get("prediction_rows", []),
        support_ceiling_report=support_ceiling_report,
    )
    observable_adapter_report = build_multi_observable_artifacts(
        output_dir=output_dir_path,
        train_examples=train_examples,
        val_examples=val_examples,
        train_epoch=best_train_epoch,
        val_epoch=best_val_epoch,
        config=config,
    )
    moment_diffusion_report = build_moment_diffusion_artifacts(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        device=device,
        config=config,
        output_dir=output_dir_path,
    )
    benchmark_report["learnability"] = learnability_report
    benchmark_report["support_ceiling"] = support_ceiling_report
    benchmark_report["ccc_geometry"] = ccc_geometry_report
    benchmark_report["physics_shift"] = physics_shift_report
    benchmark_report["observable_adapters"] = observable_adapter_report
    benchmark_report["moment_diffusion"] = moment_diffusion_report
    benchmark_report["chemical_shift_affine_calibration"] = calibration_report
    save_json(output_dir_path / "benchmark_report.json", benchmark_report)
    save_json(output_dir_path / "chemical_shift_calibration.json", calibration_report)
    save_json(
        output_dir_path / "reports" / "metrics" / "learnability_report.json",
        learnability_report,
    )
    save_json(output_dir_path / "support_ceiling_report.json", support_ceiling_report)
    save_json(
        output_dir_path / "physics_support_ceiling_report.json",
        physics_shift_report.get("support_ceiling", {}),
    )
    save_json(
        output_dir_path / "raw_vs_secondary_ccc_report.json",
        physics_shift_report,
    )
    save_json(
        output_dir_path / "reports" / "metrics" / "support_ceiling_report.json",
        support_ceiling_report,
    )
    save_json(
        output_dir_path / "reports" / "metrics" / "chemical_shift_calibration.json",
        calibration_report,
    )
    save_json(
        output_dir_path / "chemical_shift_baseline_report.json",
        chemical_shift_baseline_report,
    )
    save_json(
        output_dir_path / "uncertainty_report.json",
        render_outputs["uncertainty_report"],
    )
    save_json(
        output_dir_path / "reports" / "metrics" / "benchmark_report.json",
        benchmark_report,
    )
    save_json(
        output_dir_path / "reports" / "metrics" / "chemical_shift_baseline_report.json",
        chemical_shift_baseline_report,
    )

    summary = {
        "device": str(device),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_epoch": best_epoch,
        "best_metric": float(
            (best_checkpoint or {}).get("comparison_key", [math.inf])[0]
        ),
        "best_checkpoint_metric": best_checkpoint,
        "eligible_counts": {
            "train": best_train_epoch["eligible_counts"],
            "val": best_val_epoch["eligible_counts"],
        },
        "source_scorecards": benchmark_report["source_scorecards"],
        "source_vocab": source_vocab,
        "task_contract": task_spec.as_dict(),
        "training_input_mode": config.training_input_mode,
        "initial_checkpoint": initial_checkpoint_report,
        "trainable_parameters": trainable_parameter_report,
        "teacher_kl_guardrail": guardrail_status(
            best_val_epoch.get("metric_summary", {}),
            config,
        ),
        "observable_adapters": {
            "status": observable_adapter_report.get("status"),
            "channels": sorted(
                (observable_adapter_report.get("channels") or {}).keys()
            ),
        },
        "moment_diffusion": {
            "status": moment_diffusion_report.get("status"),
            "ccc_095_status": (
                moment_diffusion_report.get("oracle", {}).get("ccc_095_status")
                if isinstance(moment_diffusion_report.get("oracle"), dict)
                else None
            ),
        },
    }
    save_json(output_dir_path / "summary.json", summary)
    save_json(output_dir_path / "reports" / "metrics" / "summary.json", summary)
    save_json(output_dir_path / "student_training_config.json", config.as_dict())
    save_json(output_dir_path / "source_vocab.json", source_vocab)
    return summary


def maybe_load_initial_checkpoint(
    model: StudentDensityModel,
    device: torch.device,
    config: StudentTrainingConfig,
) -> dict[str, Any] | None:
    """Load model weights from a checkpoint without restoring optimizer state."""
    if not config.initial_checkpoint_path:
        return None

    checkpoint_path = Path(config.initial_checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    incompatible = model.load_state_dict(
        model_state,
        strict=config.initial_checkpoint_strict,
    )
    return {
        "path": str(checkpoint_path),
        "strict": bool(config.initial_checkpoint_strict),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }


def apply_trainable_parameter_filter(
    *,
    model: StudentDensityModel,
    patterns: list[str],
) -> dict[str, Any]:
    """Optionally freeze parameters unless their names match configured patterns."""
    normalized = [str(pattern) for pattern in patterns if str(pattern)]
    if not normalized:
        total = sum(parameter.numel() for parameter in model.parameters())
        return {
            "status": "all_trainable",
            "patterns": [],
            "trainable_tensors": sum(1 for _ in model.parameters()),
            "trainable_parameters": int(total),
            "frozen_parameters": 0,
        }
    trainable_tensors = 0
    trainable_parameters = 0
    frozen_parameters = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        is_trainable = any(pattern in name for pattern in normalized)
        parameter.requires_grad = bool(is_trainable)
        if is_trainable:
            trainable_tensors += 1
            trainable_parameters += int(parameter.numel())
            trainable_names.append(name)
        else:
            frozen_parameters += int(parameter.numel())
    if trainable_tensors == 0:
        raise ValueError(
            "trainable_parameter_name_patterns matched no model parameters: "
            + ", ".join(normalized)
        )
    return {
        "status": "filtered",
        "patterns": normalized,
        "trainable_tensors": int(trainable_tensors),
        "trainable_parameters": int(trainable_parameters),
        "frozen_parameters": int(frozen_parameters),
        "trainable_names": trainable_names[:50],
    }


def maybe_render_best_preview(
    *,
    model: StudentDensityModel,
    train_examples,
    val_examples,
    device: torch.device,
    config: StudentTrainingConfig,
    output_dir: Path,
    epoch: int,
    checkpoint_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Render lightweight best-checkpoint analysis artifacts while training runs."""
    if not config.render_best_preview:
        return None

    train_limit = max(int(config.best_preview_max_train_examples), 0)
    val_limit = max(int(config.best_preview_max_val_examples), 0)
    preview_train_examples = list(train_examples[:train_limit])
    preview_val_examples = list(val_examples[:val_limit])
    if not preview_train_examples and not preview_val_examples:
        report = {
            "status": "skipped_no_examples",
            "epoch": int(epoch),
            "checkpoint_metric": checkpoint_record,
            "max_train_examples": train_limit,
            "max_val_examples": val_limit,
        }
        _write_best_preview_report(output_dir, report)
        return report

    preview_train_epoch = (
        run_epoch(
            model=model,
            examples=preview_train_examples,
            optimizer=None,
            device=device,
            config=config,
            training=False,
            split_name="train",
            collect_prediction_rows=True,
        )
        if preview_train_examples
        else _empty_epoch_payload(split_name="train")
    )
    preview_val_epoch = (
        run_epoch(
            model=model,
            examples=preview_val_examples,
            optimizer=None,
            device=device,
            config=config,
            training=False,
            split_name="val",
            collect_prediction_rows=True,
        )
        if preview_val_examples
        else _empty_epoch_payload(split_name="val")
    )
    report: dict[str, Any] = {
        "status": "ok",
        "epoch": int(epoch),
        "checkpoint_metric": checkpoint_record,
        "preview_scope": {
            "train_examples": len(preview_train_examples),
            "val_examples": len(preview_val_examples),
            "max_train_examples": train_limit,
            "max_val_examples": val_limit,
        },
        "train": _best_preview_split_summary(preview_train_epoch),
        "val": _best_preview_split_summary(preview_val_epoch),
    }
    if config.enable_moment_head:
        report["moment_diffusion"] = build_moment_diffusion_artifacts(
            model=model,
            train_examples=preview_train_examples,
            val_examples=preview_val_examples,
            device=device,
            config=config,
            output_dir=output_dir,
        )
    _write_best_preview_report(output_dir, report)
    return report


def _best_preview_split_summary(epoch_payload: dict[str, Any]) -> dict[str, Any]:
    """Return compact monitor-friendly metrics for one best-preview split."""
    metric_summary = epoch_payload.get("metric_summary") or {}
    interesting_names = [
        "cs_masked_holdout_family_ccc_macro",
        "cs_reconstruction_family_ccc_macro",
        "val_cs_masked_holdout_family_ccc_macro",
        "val_cs_reconstruction_family_ccc_macro",
        "cs_family_ccc_macro",
        "cs_rmse_z_macro",
        "teacher_kl_macro",
        "teacher_js_macro",
    ]
    selected = {
        name: metric_summary[name]
        for name in interesting_names
        if name in metric_summary
    }
    return {
        "loss_summary": epoch_payload.get("loss_summary") or {},
        "metrics": selected,
        "eligible_counts": epoch_payload.get("eligible_counts") or {},
        "prediction_rows": len(epoch_payload.get("prediction_rows") or []),
    }


def _write_best_preview_report(output_dir: Path, report: dict[str, Any]) -> None:
    """Write the current best-preview report in root and metrics locations."""
    save_json(output_dir / "best_preview_report.json", report)
    save_json(
        output_dir / "reports" / "metrics" / "best_preview_report.json",
        report,
    )


def run_epoch(
    model: StudentDensityModel,
    examples,
    optimizer,
    device: torch.device,
    config: StudentTrainingConfig,
    training: bool,
    split_name: str,
    collect_prediction_rows: bool = False,
) -> dict[str, Any]:
    """Run one train or validation epoch."""
    ordered_examples = list(examples)
    if training:
        random.shuffle(ordered_examples)
        model.train()
    else:
        model.eval()

    totals: dict[str, float] = {}
    example_count = 0
    evaluated_examples = []
    for example in ordered_examples:
        tensors = to_tensors(example, device)
        with torch.set_grad_enabled(training):
            evidence_mask = chemical_shift_evidence_mask(
                tensors["channels"].get("chemical_shifts"),
                config=config,
                training=training,
                device=device,
                example_index=example_count,
            )
            outputs = model_outputs_for_example(
                model=model,
                tensors=tensors,
                evidence_mask=evidence_mask,
                config=config,
            )
            losses = compute_losses(
                predicted_weights=outputs["weights"],
                teacher_weights=tensors["teacher_weights"],
                channels=tensors["channels"],
                config=config,
                candidate_numeric_features=tensors["candidate_numeric_features"],
                forward_residual_mu=outputs.get("forward_residual_mu"),
                forward_residual_sigma=outputs.get("forward_residual_sigma"),
                evidence_log_likelihood=outputs.get("evidence_log_likelihood"),
                chemical_shift_evidence_mask=evidence_mask,
                moment_mu=outputs.get("moment_mu"),
                moment_sigma=outputs.get("moment_sigma"),
                moment_sigma_scale=outputs.get("moment_sigma_scale"),
                moment_outlier_score=outputs.get("moment_outlier_score"),
                posterior_state_weights=outputs.get("posterior_state_weights"),
                posterior_state_assignments=outputs.get("posterior_state_assignments"),
                posterior_state_tokens=outputs.get("posterior_state_tokens"),
            )
            reconstruction_outputs = None
            if config.enable_residue_atom_residual_head:
                reconstruction_outputs = model_outputs_for_example(
                    model=model,
                    tensors=tensors,
                    evidence_mask=None,
                    config=config,
                )
                reconstruction_ccc_loss = chemical_shift_concordance_loss(
                    reconstruction_outputs["weights"],
                    tensors["channels"].get("chemical_shifts"),
                    forward_residual_mu=reconstruction_outputs.get(
                        "forward_residual_mu"
                    ),
                    family_balance=config.chemical_shift_family_balance,
                    family_weights=config.chemical_shift_family_weights,
                )
                masked_holdout_ccc_loss = chemical_shift_concordance_loss(
                    outputs["weights"],
                    tensors["channels"].get("chemical_shifts"),
                    forward_residual_mu=outputs.get("forward_residual_mu"),
                    target_row_mask=(
                        ~evidence_mask if evidence_mask is not None else None
                    ),
                    family_balance=config.chemical_shift_family_balance,
                    family_weights=config.chemical_shift_family_weights,
                )
                losses["reconstruction_family_ccc_loss"] = reconstruction_ccc_loss
                losses["masked_holdout_family_ccc_loss"] = masked_holdout_ccc_loss
                losses["total_loss"] = (
                    losses["total_loss"]
                    + config.reconstruction_ccc_loss_weight * reconstruction_ccc_loss
                    + config.masked_holdout_ccc_loss_weight * masked_holdout_ccc_loss
                )
                whitened_reconstruction_loss = (
                    affine_calibrated_concordance_loss(
                        reconstruction_outputs["weights"],
                        tensors["channels"].get("chemical_shifts"),
                        forward_residual_mu=reconstruction_outputs.get(
                            "forward_residual_mu"
                        ),
                        calibration_row_mask=None,
                        target_row_mask=None,
                        family_weights=config.chemical_shift_family_weights,
                    )
                    if config.whitened_reconstruction_ccc_loss_weight > 0.0
                    else losses["total_loss"].new_tensor(0.0)
                )
                whitened_masked_loss = (
                    affine_calibrated_concordance_loss(
                        outputs["weights"],
                        tensors["channels"].get("chemical_shifts"),
                        forward_residual_mu=outputs.get("forward_residual_mu"),
                        calibration_row_mask=evidence_mask,
                        target_row_mask=(
                            ~evidence_mask if evidence_mask is not None else None
                        ),
                        family_weights=config.chemical_shift_family_weights,
                    )
                    if config.whitened_masked_ccc_loss_weight > 0.0
                    else losses["total_loss"].new_tensor(0.0)
                )
                losses["whitened_reconstruction_family_ccc_loss"] = (
                    whitened_reconstruction_loss
                )
                losses["whitened_masked_holdout_family_ccc_loss"] = whitened_masked_loss
                losses["total_loss"] = (
                    losses["total_loss"]
                    + config.whitened_reconstruction_ccc_loss_weight
                    * whitened_reconstruction_loss
                    + config.whitened_masked_ccc_loss_weight * whitened_masked_loss
                )
                if config.enable_secondary_shift_targets:
                    secondary_reconstruction_loss = secondary_shift_concordance_loss(
                        reconstruction_outputs["weights"],
                        tensors["channels"].get("chemical_shifts"),
                        forward_residual_mu=reconstruction_outputs.get(
                            "forward_residual_mu"
                        ),
                        family_balance=True,
                        family_weights=config.chemical_shift_family_weights,
                    )
                    secondary_masked_loss = secondary_shift_concordance_loss(
                        outputs["weights"],
                        tensors["channels"].get("chemical_shifts"),
                        forward_residual_mu=outputs.get("forward_residual_mu"),
                        target_row_mask=(
                            ~evidence_mask if evidence_mask is not None else None
                        ),
                        family_balance=True,
                        family_weights=config.chemical_shift_family_weights,
                    )
                    losses["secondary_reconstruction_family_ccc_loss"] = (
                        secondary_reconstruction_loss
                    )
                    losses["secondary_masked_holdout_family_ccc_loss"] = (
                        secondary_masked_loss
                    )
                    losses["total_loss"] = (
                        losses["total_loss"]
                        + config.secondary_reconstruction_ccc_loss_weight
                        * secondary_reconstruction_loss
                        + config.secondary_masked_holdout_ccc_loss_weight
                        * secondary_masked_loss
                    )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
                optimizer.step()

        metric_outputs = (
            reconstruction_outputs if reconstruction_outputs is not None else outputs
        )
        evaluated_examples.append(
            evaluate_prepared_example(
                example=example,
                predicted_weights=metric_outputs["weights"].detach().cpu().numpy(),
                chemical_shift_residual_mu=(
                    metric_outputs["forward_residual_mu"].detach().cpu().numpy()
                    if (
                        config.enable_forward_residual_head
                        or config.enable_residue_atom_residual_head
                    )
                    and "forward_residual_mu" in metric_outputs
                    else None
                ),
                chemical_shift_moment_mu=(
                    metric_outputs["moment_mu"].detach().cpu().numpy()
                    if config.enable_moment_head
                    and "moment_mu" in metric_outputs
                    and metric_outputs["moment_mu"].numel() > 0
                    else None
                ),
            )
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
        example_count += 1

    loss_summary = {
        key: value / max(example_count, 1) for key, value in sorted(totals.items())
    }
    metric_payload = aggregate_epoch_metrics(
        payloads=evaluated_examples,
        split=split_name,
        report_micro_metrics=config.report_micro_metrics,
    )
    append_reconstruction_alias_metrics(metric_payload, split_name)
    if config.enable_residue_atom_residual_head or config.enable_moment_head:
        append_masked_holdout_metrics(
            metric_payload=metric_payload,
            model=model,
            examples=ordered_examples,
            device=device,
            config=config,
            split_name=split_name,
        )
    return {
        "loss_summary": loss_summary,
        "loss_rows": metric_rows_from_loss_summary(
            loss_summary=loss_summary,
            split=split_name,
            example_count=example_count,
        ),
        "metric_rows": metric_payload["metric_rows"],
        "metric_summary": metric_payload["metric_summary"],
        "eligible_counts": metric_payload["eligible_counts"],
        "prediction_rows": (
            collect_chemical_shift_prediction_rows(evaluated_examples, split_name)
            if collect_prediction_rows
            else []
        ),
        "forward_residual_rows": (
            _collect_forward_residual_rows_from_epoch(
                examples=ordered_examples,
                model=model,
                device=device,
                config=config,
                split_name=split_name,
            )
            if collect_prediction_rows
            and (
                config.enable_forward_residual_head
                or config.enable_residue_atom_residual_head
            )
            else []
        ),
        "evaluated_examples": evaluated_examples if collect_prediction_rows else [],
        "split": split_name,
    }


def model_outputs_for_example(
    model: StudentDensityModel,
    tensors: dict[str, Any],
    evidence_mask: torch.Tensor | None,
    config: StudentTrainingConfig,
) -> dict[str, torch.Tensor]:
    """Run the model with optional residue-atom chemical-shift evidence."""
    chemical_shift_channel = tensors["channels"].get("chemical_shifts")
    use_chemical_shift_targets = chemical_shift_channel is not None and (
        config.enable_residue_atom_residual_head or config.enable_moment_head
    )
    return model(
        sequence_tokens=tensors["sequence_tokens"],
        example_features=tensors["example_features"],
        candidate_source_indices=tensors["candidate_source_indices"],
        candidate_numeric_features=tensors["candidate_numeric_features"],
        chemical_shift_target_features=(
            tensors.get("chemical_shift_target_features")
            if use_chemical_shift_targets
            else None
        ),
        chemical_shift_values=(
            chemical_shift_channel.values if use_chemical_shift_targets else None
        ),
        chemical_shift_mask=(
            chemical_shift_channel.mask if use_chemical_shift_targets else None
        ),
        chemical_shift_targets=(
            chemical_shift_channel.target_values if use_chemical_shift_targets else None
        ),
        chemical_shift_sigmas=(
            chemical_shift_channel.target_sigmas if use_chemical_shift_targets else None
        ),
        chemical_shift_evidence_mask=(
            evidence_mask if use_chemical_shift_targets else None
        ),
    )


def chemical_shift_evidence_mask(
    channel: PreparedObservableChannel | None,
    *,
    config: StudentTrainingConfig,
    training: bool,
    device: torch.device,
    example_index: int,
) -> torch.Tensor | None:
    """Return the target rows available as evidence for one forward pass."""
    if channel is None or not (
        config.enable_residue_atom_residual_head or config.enable_moment_head
    ):
        return None
    target_count = int(channel.values.shape[0])
    if target_count == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    mask_fraction = (
        config.residue_atom_mask_fraction
        if training or config.residue_atom_eval_mask_fraction is None
        else config.residue_atom_eval_mask_fraction
    )
    block_fraction_config = (
        config.residue_atom_block_mask_fraction
        if training or config.residue_atom_eval_block_mask_fraction is None
        else config.residue_atom_eval_block_mask_fraction
    )
    fraction = min(max(float(mask_fraction), 0.0), 0.95)
    if fraction <= 0.0:
        return torch.ones(target_count, dtype=torch.bool, device=device)
    generator = None
    if not training:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config.seed) + int(example_index) * 7919)
    if block_fraction_config > 0.0:
        evidence_mask = torch.ones(target_count, dtype=torch.bool, device=device)
        holdout_count = max(1, int(round(float(target_count) * fraction)))
        block_fraction = min(max(float(block_fraction_config), 0.0), 1.0)
        block_count = min(
            target_count, int(round(float(holdout_count) * block_fraction))
        )
        if block_count > 0:
            high = max(target_count - block_count + 1, 1)
            if generator is None:
                start = int(torch.randint(high, (1,), device=device).item())
            else:
                start = int(
                    torch.randint(high, (1,), generator=generator, device=device).item()
                )
            evidence_mask[start : start + block_count] = False
        random_count = max(holdout_count - block_count, 0)
        if random_count > 0:
            scores = (
                torch.rand(target_count, generator=generator, device=device)
                if generator is not None
                else torch.rand(target_count, device=device)
            )
            scores = scores.masked_fill(~evidence_mask, 2.0)
            indices = torch.topk(
                scores, k=min(random_count, target_count), largest=False
            ).indices
            evidence_mask[indices] = False
    elif training:
        evidence_mask = torch.rand(target_count, device=device) >= fraction
    else:
        assert generator is not None
        evidence_mask = (
            torch.rand(target_count, generator=generator, device=device) >= fraction
        )
    if not torch.any(evidence_mask):
        evidence_mask[0] = True
    return evidence_mask


def append_reconstruction_alias_metrics(
    metric_payload: dict[str, Any],
    split_name: str,
) -> None:
    """Alias chemical-shift CCC metrics as reconstruction metrics."""
    metric_rows = metric_payload["metric_rows"]
    metric_summary = metric_payload["metric_summary"]
    eligible_counts = metric_payload["eligible_counts"]
    new_rows = []
    for row in metric_rows:
        metric_name = str(row.get("metric_name", ""))
        if not metric_name.startswith("cs_") or "ccc" not in metric_name:
            continue
        aliased_name = metric_name.replace("cs_", "cs_reconstruction_", 1)
        aliased = dict(row)
        aliased["metric_name"] = aliased_name
        new_rows.append(aliased)
        source_key = f"{split_name}_{metric_name}_{row.get('aggregation', 'macro')}"
        target_key = f"{split_name}_{aliased_name}_{row.get('aggregation', 'macro')}"
        if source_key in metric_summary:
            metric_summary[target_key] = metric_summary[source_key]
        if source_key in eligible_counts:
            eligible_counts[target_key] = dict(eligible_counts[source_key])
    metric_rows.extend(new_rows)


def append_masked_holdout_metrics(
    metric_payload: dict[str, Any],
    model: StudentDensityModel,
    examples: list[Any],
    device: torch.device,
    config: StudentTrainingConfig,
    split_name: str,
) -> None:
    """Append CCC metrics on target rows hidden from posterior evidence."""
    per_example: list[float] = []
    per_example_loss: list[float] = []
    per_example_counts: list[int] = []
    family_values: dict[str, list[float]] = {
        name: [] for name in CHEMICAL_SHIFT_FAMILIES
    }
    family_counts: dict[str, list[int]] = {name: [] for name in CHEMICAL_SHIFT_FAMILIES}
    whitened_per_example: list[float] = []
    whitened_per_example_counts: list[int] = []
    whitened_family_values: dict[str, list[float]] = {
        name: [] for name in CHEMICAL_SHIFT_FAMILIES
    }
    whitened_family_counts: dict[str, list[int]] = {
        name: [] for name in CHEMICAL_SHIFT_FAMILIES
    }
    secondary_per_example: list[float] = []
    secondary_per_example_loss: list[float] = []
    secondary_per_example_counts: list[int] = []
    secondary_family_values: dict[str, list[float]] = {
        name: [] for name in CHEMICAL_SHIFT_FAMILIES
    }
    secondary_family_counts: dict[str, list[int]] = {
        name: [] for name in CHEMICAL_SHIFT_FAMILIES
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for example_index, example in enumerate(examples):
            tensors = to_tensors(example, device)
            channel = tensors["channels"].get("chemical_shifts")
            if channel is None:
                continue
            evidence_mask = chemical_shift_evidence_mask(
                channel,
                config=config,
                training=False,
                device=device,
                example_index=example_index,
            )
            if evidence_mask is None:
                continue
            holdout_mask = ~evidence_mask.detach().cpu().numpy().astype(bool)
            if np.count_nonzero(holdout_mask) < 2:
                continue
            outputs = model_outputs_for_example(
                model=model,
                tensors=tensors,
                evidence_mask=evidence_mask,
                config=config,
            )
            evaluated = evaluate_prepared_example(
                example=example,
                predicted_weights=outputs["weights"].detach().cpu().numpy(),
                chemical_shift_residual_mu=outputs["forward_residual_mu"]
                .detach()
                .cpu()
                .numpy(),
                chemical_shift_moment_mu=(
                    outputs["moment_mu"].detach().cpu().numpy()
                    if config.enable_moment_head
                    and outputs.get("moment_mu") is not None
                    and outputs["moment_mu"].numel() > 0
                    else None
                ),
            )
            evaluated_channel = evaluated.channels.get("chemical_shifts")
            if evaluated_channel is None:
                continue
            holdout_ids = {
                target_id
                for target_id, is_holdout in zip(
                    example.channels["chemical_shifts"].target_ids,
                    holdout_mask,
                    strict=True,
                )
                if is_holdout
            }
            row_mask = np.asarray(
                [
                    target_id in holdout_ids
                    for target_id in evaluated_channel.target_ids
                ],
                dtype=bool,
            )
            if np.count_nonzero(row_mask) < 2:
                continue
            example_family_ccc: list[float] = []
            example_whitened_ccc: list[float] = []
            example_secondary_ccc: list[float] = []
            baselines = random_coil_baselines_for_target_ids(
                evaluated_channel.target_ids
            )
            for family_name in CHEMICAL_SHIFT_FAMILIES:
                family_mask = row_mask & np.asarray(
                    [
                        _atom_family_from_target_id(target_id) == family_name
                        for target_id in evaluated_channel.target_ids
                    ],
                    dtype=bool,
                )
                if np.count_nonzero(family_mask) < 2:
                    continue
                ccc = _safe_ccc_numpy(
                    evaluated_channel.predictions[family_mask],
                    evaluated_channel.targets[family_mask],
                )
                if not math.isfinite(ccc):
                    continue
                example_family_ccc.append(ccc)
                family_values[family_name].append(ccc)
                family_counts[family_name].append(int(np.count_nonzero(family_mask)))
                calibration_mask = (~row_mask) & np.asarray(
                    [
                        _atom_family_from_target_id(target_id) == family_name
                        for target_id in evaluated_channel.target_ids
                    ],
                    dtype=bool,
                )
                calibrated_predictions = _affine_calibrated_predictions_numpy(
                    predictions=evaluated_channel.predictions,
                    targets=evaluated_channel.targets,
                    calibration_mask=calibration_mask,
                )
                whitened_ccc = _safe_ccc_numpy(
                    calibrated_predictions[family_mask],
                    evaluated_channel.targets[family_mask],
                )
                if math.isfinite(whitened_ccc):
                    example_whitened_ccc.append(whitened_ccc)
                    whitened_family_values[family_name].append(whitened_ccc)
                    whitened_family_counts[family_name].append(
                        int(np.count_nonzero(family_mask))
                    )
                secondary_mask = family_mask & np.isfinite(baselines)
                if np.count_nonzero(secondary_mask) < 2:
                    continue
                secondary_ccc = _safe_ccc_numpy(
                    evaluated_channel.predictions[secondary_mask]
                    - baselines[secondary_mask],
                    evaluated_channel.targets[secondary_mask]
                    - baselines[secondary_mask],
                )
                if not math.isfinite(secondary_ccc):
                    continue
                example_secondary_ccc.append(secondary_ccc)
                secondary_family_values[family_name].append(secondary_ccc)
                secondary_family_counts[family_name].append(
                    int(np.count_nonzero(secondary_mask))
                )
            if example_family_ccc:
                value = float(np.mean(example_family_ccc))
                per_example.append(value)
                per_example_loss.append(1.0 - value)
                per_example_counts.append(int(np.count_nonzero(row_mask)))
            if example_whitened_ccc:
                value = float(np.mean(example_whitened_ccc))
                whitened_per_example.append(value)
                whitened_per_example_counts.append(int(np.count_nonzero(row_mask)))
            if example_secondary_ccc:
                value = float(np.mean(example_secondary_ccc))
                secondary_per_example.append(value)
                secondary_per_example_loss.append(1.0 - value)
                secondary_per_example_counts.append(int(np.count_nonzero(row_mask)))
    if was_training:
        model.train()
    _append_local_metric(
        metric_payload,
        split_name,
        "cs_masked_holdout_family_ccc",
        per_example,
        per_example_counts,
    )
    _append_local_metric(
        metric_payload,
        split_name,
        "cs_masked_holdout_family_ccc_loss",
        per_example_loss,
        per_example_counts,
    )
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        _append_local_metric(
            metric_payload,
            split_name,
            f"cs_masked_holdout_{family_name}_ccc",
            family_values[family_name],
            family_counts[family_name],
        )
    _append_local_metric(
        metric_payload,
        split_name,
        "cs_masked_holdout_whitened_family_ccc",
        whitened_per_example,
        whitened_per_example_counts,
    )
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        _append_local_metric(
            metric_payload,
            split_name,
            f"cs_masked_holdout_whitened_{family_name}_ccc",
            whitened_family_values[family_name],
            whitened_family_counts[family_name],
        )
    _append_local_metric(
        metric_payload,
        split_name,
        "cs_masked_holdout_secondary_family_ccc",
        secondary_per_example,
        secondary_per_example_counts,
    )
    _append_local_metric(
        metric_payload,
        split_name,
        "cs_masked_holdout_secondary_family_ccc_loss",
        secondary_per_example_loss,
        secondary_per_example_counts,
    )
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        _append_local_metric(
            metric_payload,
            split_name,
            f"cs_masked_holdout_secondary_{family_name}_ccc",
            secondary_family_values[family_name],
            secondary_family_counts[family_name],
        )


def _append_local_metric(
    metric_payload: dict[str, Any],
    split_name: str,
    metric_name: str,
    values: list[float],
    counts: list[int],
) -> None:
    """Append one local macro metric row to an aggregate payload."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return
    value = float(np.mean(finite))
    key = f"{split_name}_{metric_name}_macro"
    count = int(sum(counts))
    row = {
        "metric_name": metric_name,
        "split": split_name,
        "aggregation": "macro",
        "value": value,
        "eligible_examples": len(finite),
        "eligible_measurements": count,
        "tier": "observable_quality",
    }
    metric_payload["metric_rows"].append(row)
    metric_payload["metric_summary"][key] = value
    metric_payload["eligible_counts"][key] = {
        "eligible_examples": len(finite),
        "eligible_measurements": count,
    }


def compute_losses(
    predicted_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    channels: dict[str, PreparedObservableChannel],
    config: StudentTrainingConfig,
    candidate_numeric_features: torch.Tensor | None = None,
    forward_residual_mu: torch.Tensor | None = None,
    forward_residual_sigma: torch.Tensor | None = None,
    evidence_log_likelihood: torch.Tensor | None = None,
    chemical_shift_evidence_mask: torch.Tensor | None = None,
    moment_mu: torch.Tensor | None = None,
    moment_sigma: torch.Tensor | None = None,
    moment_sigma_scale: torch.Tensor | None = None,
    moment_outlier_score: torch.Tensor | None = None,
    posterior_state_weights: torch.Tensor | None = None,
    posterior_state_assignments: torch.Tensor | None = None,
    posterior_state_tokens: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the staged density and observable losses."""
    task_spec = resolve_training_task(config.training_task, config.training_input_mode)
    weight_kl = teacher_kl_loss(predicted_weights, teacher_weights)
    entropy = support_entropy(predicted_weights)
    chemical_shift_loss = observable_loss(
        predicted_weights,
        channels.get("chemical_shifts"),
        forward_residual_mu=forward_residual_mu,
        family_balance=config.chemical_shift_family_balance,
        family_weights=config.chemical_shift_family_weights,
        loss_mode=config.chemical_shift_loss_mode,
        huber_delta=config.chemical_shift_huber_delta,
    )
    chemical_shift_ccc_loss = chemical_shift_concordance_loss(
        predicted_weights,
        channels.get("chemical_shifts"),
        forward_residual_mu=forward_residual_mu,
        family_balance=config.chemical_shift_family_balance,
        family_weights=config.chemical_shift_family_weights,
    )
    ccc_oracle_kl = ccc_oracle_kl_loss(
        predicted_weights=predicted_weights,
        candidate_numeric_features=candidate_numeric_features,
        config=config,
    )
    posterior_evidence_kl = posterior_evidence_kl_loss(
        predicted_weights=predicted_weights,
        evidence_log_likelihood=(
            evidence_log_likelihood
            if evidence_log_likelihood is not None
            else _posterior_evidence_vector(
                candidate_numeric_features,
                task_spec.input_feature_mode,
            )
        ),
        evidence_temperature=config.evidence_temperature,
    )
    evidence_likelihood_nll = evidence_likelihood_nll_loss(
        predicted_weights=predicted_weights,
        evidence_log_likelihood=evidence_log_likelihood,
    )
    normalized_evidence_nll = normalized_chemical_shift_evidence_nll_loss(
        predicted_weights=predicted_weights,
        channel=channels.get("chemical_shifts"),
        forward_residual_mu=forward_residual_mu,
        evidence_mask=chemical_shift_evidence_mask,
    )
    ccc_geometry = ccc_geometry_alignment_loss(
        predicted_weights=predicted_weights,
        candidate_numeric_features=candidate_numeric_features,
        config=config,
        task_input_mode=task_spec.input_feature_mode,
    )
    observable_oracle_kl = observable_adapter_oracle_kl_loss(
        predicted_weights=predicted_weights,
        channels=channels,
        config=config,
        allows_target_conditioning=task_spec.allows_target_conditioning,
    )
    moment_reconstruction_ccc = moment_concordance_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        family_balance=config.chemical_shift_family_balance,
        family_weights=config.chemical_shift_family_weights,
    )
    moment_masked_holdout_ccc = moment_concordance_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
        family_balance=config.chemical_shift_family_balance,
        family_weights=config.chemical_shift_family_weights,
    )
    moment_oracle_distillation = moment_oracle_distillation_loss(
        moment_mu=moment_mu,
        moment_sigma=moment_sigma,
        channel=channels.get("chemical_shifts"),
    )
    hn_variance = hn_variance_calibration_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
    )
    cprime_robust = cprime_robust_likelihood_loss(
        moment_mu=moment_mu,
        moment_sigma=moment_sigma,
        outlier_score=moment_outlier_score,
        channel=channels.get("chemical_shifts"),
    )
    tail_calibration = family_tail_calibration_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
        family_names=config.tail_calibration_families,
        tail_fraction=config.tail_calibration_fraction,
    )
    residual_trend = family_residual_trend_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
        family_names=config.residual_trend_families,
    )
    ccc_decomposition = ccc_decomposition_losses(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
        family_weights=config.chemical_shift_family_weights,
        measure_aware=config.enable_measure_aware_row_weights,
    )
    student_t_nll = student_t_moment_nll_loss(
        moment_mu=moment_mu,
        moment_sigma=moment_sigma,
        channel=channels.get("chemical_shifts"),
        degrees_of_freedom=config.student_t_degrees_of_freedom,
        measure_aware=config.enable_measure_aware_row_weights,
    )
    crps_calibration = gaussian_crps_calibration_loss(
        moment_mu=moment_mu,
        moment_sigma=moment_sigma,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
    )
    joint_nmr = joint_nmr_posterior_loss(
        moment_mu=moment_mu,
        moment_sigma=moment_sigma,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
    )
    paired_evidence_imputation = paired_evidence_imputation_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        evidence_row_mask=chemical_shift_evidence_mask,
        target_families=config.paired_evidence_imputation_targets,
    )
    hn_context = family_tail_calibration_loss(
        moment_mu=moment_mu,
        channel=channels.get("chemical_shifts"),
        target_row_mask=(
            ~chemical_shift_evidence_mask
            if chemical_shift_evidence_mask is not None
            else None
        ),
        family_names=("HN",),
        tail_fraction=0.25,
    )
    moment_sample_consistency = moment_sample_consistency_loss(
        moment_mu=moment_mu,
        predicted_weights=predicted_weights,
        channel=channels.get("chemical_shifts"),
        forward_residual_mu=forward_residual_mu,
    )
    state_occupancy_distillation = state_occupancy_distillation_loss(
        state_weights=posterior_state_weights,
        state_assignments=posterior_state_assignments,
        teacher_weights=teacher_weights,
    )
    state_diversity = state_diversity_loss(posterior_state_weights)
    state_entropy_floor = state_entropy_floor_loss(
        posterior_state_weights,
        floor=config.state_entropy_floor,
    )
    state_repulsion = state_repulsion_loss(posterior_state_tokens)
    basin_coverage = basin_coverage_loss(
        posterior_state_assignments,
        floor=config.state_entropy_floor,
    )
    forward_residual_loss = forward_residual_supervision_loss(
        residual_mu=forward_residual_mu,
        residual_sigma=forward_residual_sigma,
        channel=channels.get("chemical_shifts"),
    )
    forward_residual_regularization = forward_residual_regularization_loss(
        residual_mu=forward_residual_mu,
    )
    sequence_smoothness = residue_atom_sequence_smoothness_loss(
        residual_mu=forward_residual_mu,
        channel=channels.get("chemical_shifts"),
    )
    j_coupling_loss = observable_loss(
        predicted_weights,
        channels.get("j_couplings"),
    )
    noe_loss = observable_loss(
        predicted_weights,
        channels.get("noe_restraints"),
    )

    total = config.weight_kl_weight * weight_kl
    total = total + config.chemical_shift_loss_weight * chemical_shift_loss
    total = (
        total
        + (
            config.chemical_shift_ccc_loss_weight
            + config.raw_family_ccc_guardrail_weight
        )
        * chemical_shift_ccc_loss
    )
    total = (
        total
        + (config.ccc_oracle_kl_weight + config.ccc_oracle_distillation_weight)
        * ccc_oracle_kl
    )
    total = total + config.ccc_geometry_loss_weight * ccc_geometry
    total = total + config.observable_oracle_distillation_weight * observable_oracle_kl
    total = total + config.reconstruction_ccc_loss_weight * moment_reconstruction_ccc
    total = total + config.masked_holdout_ccc_loss_weight * moment_masked_holdout_ccc
    total = (
        total + config.moment_oracle_distillation_weight * moment_oracle_distillation
    )
    total = total + config.hn_variance_loss_weight * hn_variance
    total = total + config.cprime_outlier_loss_weight * cprime_robust
    total = total + config.tail_calibration_loss_weight * tail_calibration
    total = total + config.residual_trend_loss_weight * residual_trend
    total = total + config.family_corr_loss_weight * ccc_decomposition["corr"]
    total = total + config.family_scale_loss_weight * ccc_decomposition["scale"]
    total = total + config.family_bias_loss_weight * ccc_decomposition["bias"]
    total = total + config.student_t_nll_weight * student_t_nll
    total = total + config.crps_calibration_loss_weight * crps_calibration
    total = total + config.joint_nmr_loss_weight * joint_nmr
    total = (
        total
        + config.paired_evidence_imputation_loss_weight * paired_evidence_imputation
    )
    total = total + config.hn_context_loss_weight * hn_context
    total = total + config.moment_sample_consistency_weight * moment_sample_consistency
    total = (
        total
        + config.state_occupancy_distillation_weight * state_occupancy_distillation
    )
    total = total + config.state_diversity_weight * state_diversity
    total = total + config.state_diversity_weight * state_entropy_floor
    total = total + config.state_repulsion_weight * state_repulsion
    total = total + config.basin_coverage_weight * basin_coverage
    residual_loss_weight = float(config.forward_residual_loss_weight)
    residual_regularization_weight = float(
        config.forward_residual_regularization_weight
    )
    if config.enable_residue_atom_residual_head:
        residual_loss_weight += float(config.residue_atom_residual_loss_weight)
        residual_regularization_weight += float(
            config.residue_atom_residual_regularization_weight
        )
    total = total + residual_loss_weight * forward_residual_loss
    total = total + residual_regularization_weight * forward_residual_regularization
    total = total + config.posterior_evidence_kl_weight * posterior_evidence_kl
    total = total + config.evidence_likelihood_nll_weight * evidence_likelihood_nll
    total = total + config.normalized_evidence_nll_weight * normalized_evidence_nll
    total = total + config.sequence_smoothness_loss_weight * sequence_smoothness
    total = total + config.j_coupling_loss_weight * j_coupling_loss
    total = total + config.noe_loss_weight * noe_loss
    total = total - config.entropy_regularization_weight * entropy

    return {
        "total_loss": total,
        "weight_kl": weight_kl,
        "chemical_shift_loss": chemical_shift_loss,
        "chemical_shift_ccc_loss": chemical_shift_ccc_loss,
        "ccc_oracle_kl": ccc_oracle_kl,
        "ccc_geometry": ccc_geometry,
        "observable_oracle_kl": observable_oracle_kl,
        "moment_reconstruction_family_ccc_loss": moment_reconstruction_ccc,
        "moment_masked_holdout_family_ccc_loss": moment_masked_holdout_ccc,
        "moment_oracle_distillation": moment_oracle_distillation,
        "hn_variance_calibration": hn_variance,
        "cprime_robust_likelihood": cprime_robust,
        "family_tail_calibration": tail_calibration,
        "family_residual_trend": residual_trend,
        "family_corr_loss": ccc_decomposition["corr"],
        "family_scale_loss": ccc_decomposition["scale"],
        "family_bias_loss": ccc_decomposition["bias"],
        "student_t_nll": student_t_nll,
        "crps_calibration_loss": crps_calibration,
        "joint_nmr_loss": joint_nmr,
        "paired_evidence_imputation_loss": paired_evidence_imputation,
        "hn_context_loss": hn_context,
        "moment_sample_consistency": moment_sample_consistency,
        "state_occupancy_distillation": state_occupancy_distillation,
        "state_diversity": state_diversity,
        "state_entropy_floor": state_entropy_floor,
        "state_repulsion": state_repulsion,
        "basin_coverage": basin_coverage,
        "posterior_evidence_kl": posterior_evidence_kl,
        "evidence_likelihood_nll": evidence_likelihood_nll,
        "normalized_evidence_nll": normalized_evidence_nll,
        "forward_residual_loss": forward_residual_loss,
        "forward_residual_regularization": forward_residual_regularization,
        "sequence_smoothness": sequence_smoothness,
        "j_coupling_loss": j_coupling_loss,
        "noe_loss": noe_loss,
        "entropy": entropy,
    }


def _posterior_evidence_vector(
    candidate_numeric_features: torch.Tensor | None,
    input_mode: str,
) -> torch.Tensor | None:
    """Return explicit evidence log-likelihood values from candidate features."""
    if candidate_numeric_features is None or candidate_numeric_features.ndim != 2:
        return None
    index = evidence_log_likelihood_feature_index(
        candidate_feature_dim=int(candidate_numeric_features.shape[1]),
        input_mode=input_mode,
    )
    if index is None:
        return None
    return candidate_numeric_features[:, index]


def observable_loss(
    weights: torch.Tensor,
    channel: PreparedObservableChannel | None,
    *,
    forward_residual_mu: torch.Tensor | None = None,
    family_balance: bool = False,
    family_weights: dict[str, float] | None = None,
    loss_mode: str = "mse",
    huber_delta: float = 5.0,
) -> torch.Tensor:
    """Return one channel-level weighted expectation loss."""
    if channel is None:
        return weights.new_tensor(0.0)

    predictions, usable = channel_expectations(
        weights,
        channel,
        forward_residual_mu=forward_residual_mu,
    )
    if not torch.any(usable):
        return weights.new_tensor(0.0)
    target_values = channel.target_values
    target_sigmas = channel.target_sigmas.clamp_min(1e-6)

    residual = (predictions[usable] - target_values[usable]) / target_sigmas[usable]
    per_target_loss = residual_loss(
        residual=residual,
        loss_mode=loss_mode,
        huber_delta=huber_delta,
    )
    if family_balance and channel.target_ids:
        return family_balanced_mean(
            per_target_values=per_target_loss,
            usable=usable,
            target_ids=channel.target_ids,
            family_weights=family_weights or {},
        )
    return torch.mean(per_target_loss)


def residual_loss(
    residual: torch.Tensor,
    loss_mode: str,
    huber_delta: float,
) -> torch.Tensor:
    """Return per-target residual loss values for one residual vector."""
    mode = loss_mode.lower()
    if mode == "mse":
        return torch.square(residual)
    if mode == "huber":
        delta = max(float(huber_delta), 1e-6)
        absolute = torch.abs(residual)
        quadratic = torch.minimum(absolute, residual.new_tensor(delta))
        linear = absolute - quadratic
        return 0.5 * torch.square(quadratic) + delta * linear
    raise ValueError(f"Unsupported observable loss mode: {loss_mode}")


def chemical_shift_concordance_loss(
    weights: torch.Tensor,
    channel: PreparedObservableChannel | None,
    *,
    forward_residual_mu: torch.Tensor | None = None,
    target_row_mask: torch.Tensor | None = None,
    family_balance: bool = False,
    family_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return a differentiable CCC loss for chemical-shift posterior means."""
    if channel is None:
        return weights.new_tensor(0.0)
    predictions, usable = channel_expectations(
        weights,
        channel,
        forward_residual_mu=forward_residual_mu,
    )
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(usable.device)
    if not torch.any(usable):
        return weights.new_tensor(0.0)
    if family_balance and channel.target_ids:
        losses: list[torch.Tensor] = []
        weights_for_average: list[float] = []
        families = sorted(
            {
                family
                for family in (
                    _atom_family_from_target_id(t) for t in channel.target_ids
                )
                if family is not None
            }
        )
        for family_name in families:
            family_mask = torch.tensor(
                [
                    _atom_family_from_target_id(target_id) == family_name
                    for target_id in channel.target_ids
                ],
                dtype=torch.bool,
                device=predictions.device,
            )
            combined = usable & family_mask
            if torch.count_nonzero(combined) < 2:
                continue
            losses.append(
                concordance_loss(predictions[combined], channel.target_values[combined])
            )
            weights_for_average.append(
                float((family_weights or {}).get(family_name, 1.0))
            )
        if losses:
            weight_tensor = torch.tensor(
                weights_for_average,
                dtype=predictions.dtype,
                device=predictions.device,
            ).clamp_min(0.0)
            stacked = torch.stack(losses)
            if torch.sum(weight_tensor) > 0:
                return torch.sum(stacked * weight_tensor) / torch.sum(weight_tensor)
            return torch.mean(stacked)
    if torch.count_nonzero(usable) < 2:
        return weights.new_tensor(0.0)
    return concordance_loss(predictions[usable], channel.target_values[usable])


def moment_concordance_loss(
    *,
    moment_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
    family_balance: bool = False,
    family_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return CCC loss for direct fast posterior moment predictions."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0:
        return moment_mu.new_tensor(0.0)
    if moment_mu.shape[0] != channel.target_values.shape[0]:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(usable.device)
    if not torch.any(usable):
        return moment_mu.new_tensor(0.0)
    if family_balance and channel.target_ids:
        losses: list[torch.Tensor] = []
        weights_for_average: list[float] = []
        for family_name in CHEMICAL_SHIFT_FAMILIES:
            family_mask = torch.tensor(
                [
                    _atom_family_from_target_id(target_id) == family_name
                    for target_id in channel.target_ids
                ],
                dtype=torch.bool,
                device=moment_mu.device,
            )
            combined = usable & family_mask
            if torch.count_nonzero(combined) < 2:
                continue
            losses.append(
                concordance_loss(moment_mu[combined], channel.target_values[combined])
            )
            weights_for_average.append(
                float((family_weights or {}).get(family_name, 1.0))
            )
        if losses:
            weight_tensor = torch.tensor(
                weights_for_average,
                dtype=moment_mu.dtype,
                device=moment_mu.device,
            ).clamp_min(0.0)
            stacked = torch.stack(losses)
            if torch.sum(weight_tensor) > 0:
                return torch.sum(stacked * weight_tensor) / torch.sum(weight_tensor)
            return torch.mean(stacked)
    if torch.count_nonzero(usable) < 2:
        return moment_mu.new_tensor(0.0)
    return concordance_loss(moment_mu[usable], channel.target_values[usable])


def moment_oracle_distillation_loss(
    *,
    moment_mu: torch.Tensor | None,
    moment_sigma: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
) -> torch.Tensor:
    """Distill the fast moment toward the desired oracle expectation."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0:
        return moment_mu.new_tensor(0.0)
    if moment_mu.shape[0] != channel.target_values.shape[0]:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if torch.count_nonzero(usable) == 0:
        return moment_mu.new_tensor(0.0)
    sigma = (
        moment_sigma.clamp_min(1e-6)
        if moment_sigma is not None and moment_sigma.shape == moment_mu.shape
        else channel.target_sigmas.clamp_min(1e-6)
    )
    residual = (moment_mu[usable] - channel.target_values[usable]) / sigma[usable]
    nll = 0.5 * torch.square(residual) + torch.log(sigma[usable])
    return torch.mean(nll)


def hn_variance_calibration_loss(
    *,
    moment_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match HN prediction spread to observed HN spread on eligible rows."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(usable.device)
    hn_mask = torch.tensor(
        [
            _atom_family_from_target_id(target_id) == "HN"
            for target_id in channel.target_ids
        ],
        dtype=torch.bool,
        device=moment_mu.device,
    )
    combined = usable & hn_mask
    if torch.count_nonzero(combined) < 3:
        return moment_mu.new_tensor(0.0)
    prediction_std = torch.std(moment_mu[combined], unbiased=False).clamp_min(1e-4)
    target_std = torch.std(
        channel.target_values[combined],
        unbiased=False,
    ).clamp_min(1e-4)
    variance_loss = torch.nn.functional.smooth_l1_loss(
        torch.log(prediction_std),
        torch.log(target_std),
    )
    prediction_z = (moment_mu[combined] - torch.mean(moment_mu[combined])) / (
        prediction_std + 1e-6
    )
    target_z = (
        channel.target_values[combined] - torch.mean(channel.target_values[combined])
    ) / (target_std + 1e-6)
    shape_loss = torch.nn.functional.smooth_l1_loss(prediction_z, target_z)
    slope = covariance_slope(moment_mu[combined], channel.target_values[combined])
    slope_loss = torch.square(slope - slope.new_tensor(1.0))
    return variance_loss + 0.25 * shape_loss + 0.10 * slope_loss


def cprime_robust_likelihood_loss(
    *,
    moment_mu: torch.Tensor | None,
    moment_sigma: torch.Tensor | None,
    outlier_score: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
) -> torch.Tensor:
    """Return robust C' likelihood loss with optional outlier confidence penalty."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    cprime_mask = torch.tensor(
        [
            _atom_family_from_target_id(target_id) == "C'"
            for target_id in channel.target_ids
        ],
        dtype=torch.bool,
        device=moment_mu.device,
    )
    combined = usable & cprime_mask
    if torch.count_nonzero(combined) == 0:
        return moment_mu.new_tensor(0.0)
    sigma = (
        moment_sigma.clamp(1e-3, 12.0)
        if moment_sigma is not None and moment_sigma.shape == moment_mu.shape
        else channel.target_sigmas.clamp_min(1e-3)
    )
    residual = (moment_mu[combined] - channel.target_values[combined]) / sigma[combined]
    robust = torch.log1p(torch.square(residual) / 4.0)
    if outlier_score is not None and outlier_score.shape == moment_mu.shape:
        # Encourage high outlier scores only for genuinely large robust residuals.
        target_outlier = (torch.abs(residual).detach() > 3.0).to(moment_mu.dtype)
        outlier_loss = torch.nn.functional.binary_cross_entropy(
            outlier_score[combined].clamp(1e-6, 1.0 - 1e-6),
            target_outlier,
        )
        return (
            torch.mean(robust)
            + 0.1 * outlier_loss
            + _sigma_anchor_loss(
                sigma[combined],
                channel.target_sigmas[combined],
            )
        )
    return torch.mean(robust) + _sigma_anchor_loss(
        sigma[combined],
        channel.target_sigmas[combined],
    )


def family_tail_calibration_loss(
    *,
    moment_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
    family_names: Sequence[str] | None = None,
    tail_fraction: float = 0.2,
) -> torch.Tensor:
    """Penalize family-wise tail compression on held-out chemical-shift rows."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return moment_mu.new_tensor(0.0)
    families = tuple(family_names or ())
    if not families:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(moment_mu.device)
    family_labels = [
        _atom_family_from_target_id(target_id) for target_id in channel.target_ids
    ]
    losses: list[torch.Tensor] = []
    fraction = min(max(float(tail_fraction), 0.05), 0.45)
    for family_name in families:
        family_mask = torch.tensor(
            [label == family_name for label in family_labels],
            dtype=torch.bool,
            device=moment_mu.device,
        )
        combined = usable & family_mask
        if torch.count_nonzero(combined) < 8:
            continue
        predictions = moment_mu[combined]
        targets = channel.target_values[combined].to(moment_mu.device)
        prediction_std = torch.std(predictions, unbiased=False).clamp_min(1e-4)
        target_std = torch.std(targets, unbiased=False).clamp_min(1e-4)
        prediction_z = (predictions - torch.mean(predictions)) / prediction_std
        target_z = (targets - torch.mean(targets)) / target_std
        slope = covariance_slope(predictions, targets)
        slope_loss = torch.square(slope - slope.new_tensor(1.0))
        threshold = torch.quantile(torch.abs(target_z).detach(), 1.0 - fraction)
        tail_mask = torch.abs(target_z) >= threshold
        if torch.count_nonzero(tail_mask) < 3:
            tail_loss = torch.nn.functional.smooth_l1_loss(prediction_z, target_z)
        else:
            tail_loss = torch.nn.functional.smooth_l1_loss(
                prediction_z[tail_mask],
                target_z[tail_mask],
            )
        spread_loss = torch.nn.functional.smooth_l1_loss(
            torch.log(prediction_std),
            torch.log(target_std),
        )
        losses.append(tail_loss + 0.25 * slope_loss + 0.25 * spread_loss)
    if not losses:
        return moment_mu.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def family_residual_trend_loss(
    *,
    moment_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
    family_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Penalize residual fan-shape trends in selected atom families."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return moment_mu.new_tensor(0.0)
    families = tuple(family_names or ())
    if not families:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(moment_mu.device)
    family_labels = [
        _atom_family_from_target_id(target_id) for target_id in channel.target_ids
    ]
    losses: list[torch.Tensor] = []
    for family_name in families:
        family_mask = torch.tensor(
            [label == family_name for label in family_labels],
            dtype=torch.bool,
            device=moment_mu.device,
        )
        combined = usable & family_mask
        if torch.count_nonzero(combined) < 8:
            continue
        predictions = moment_mu[combined]
        targets = channel.target_values[combined].to(moment_mu.device)
        residuals = predictions - targets
        residual_slope = covariance_slope(residuals, targets)
        residual_center = torch.mean(residuals)
        residual_std = torch.std(residuals, unbiased=False).clamp_min(1e-4)
        target_std = torch.std(targets, unbiased=False).clamp_min(1e-4)
        residual_z = (residuals - residual_center) / residual_std
        target_z = (targets - torch.mean(targets)) / target_std
        residual_corr = torch.mean(residual_z * target_z)
        losses.append(torch.square(residual_slope) + 0.25 * torch.square(residual_corr))
    if not losses:
        return moment_mu.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def ccc_decomposition_losses(
    *,
    moment_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
    family_weights: dict[str, float] | None = None,
    measure_aware: bool = False,
) -> dict[str, torch.Tensor]:
    """Return family-balanced correlation, scale, and bias CCC components."""
    if moment_mu is None:
        zero = torch.tensor(0.0)
        return {"corr": zero, "scale": zero, "bias": zero}
    zero = moment_mu.new_tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return {"corr": zero, "scale": zero, "bias": zero}
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(moment_mu.device)
    family_labels = [
        _atom_family_from_target_id(target_id) for target_id in channel.target_ids
    ]
    corr_losses: list[torch.Tensor] = []
    scale_losses: list[torch.Tensor] = []
    bias_losses: list[torch.Tensor] = []
    average_weights: list[float] = []
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        family_mask = torch.tensor(
            [label == family_name for label in family_labels],
            dtype=torch.bool,
            device=moment_mu.device,
        )
        combined = usable & family_mask
        if torch.count_nonzero(combined) < 3:
            continue
        predictions = moment_mu[combined]
        targets = channel.target_values[combined].to(moment_mu.device)
        residuals = predictions - targets
        row_weights = (
            _measure_aware_row_weights(
                target_ids=[
                    target_id
                    for target_id, keep in zip(
                        channel.target_ids, combined.detach().cpu().tolist(), strict=False
                    )
                    if keep
                ],
                family_name=family_name,
                residuals=residuals.detach(),
                dtype=predictions.dtype,
                device=predictions.device,
            )
            if measure_aware
            else torch.ones_like(predictions)
        )
        corr, pred_std, target_std, pred_mean, target_mean = _weighted_corr_stats(
            predictions,
            targets,
            row_weights,
        )
        corr_losses.append(torch.relu(corr.new_tensor(1.0) - corr))
        scale_losses.append(torch.square(torch.log(pred_std / target_std)))
        bias_losses.append(torch.square((pred_mean - target_mean) / target_std))
        average_weights.append(float((family_weights or {}).get(family_name, 1.0)))
    if not corr_losses:
        return {"corr": zero, "scale": zero, "bias": zero}
    weight_tensor = torch.tensor(
        average_weights,
        dtype=moment_mu.dtype,
        device=moment_mu.device,
    ).clamp_min(0.0)
    return {
        "corr": _weighted_stack_average(corr_losses, weight_tensor),
        "scale": _weighted_stack_average(scale_losses, weight_tensor),
        "bias": _weighted_stack_average(bias_losses, weight_tensor),
    }


def student_t_moment_nll_loss(
    *,
    moment_mu: torch.Tensor | None,
    moment_sigma: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    degrees_of_freedom: float,
    target_row_mask: torch.Tensor | None = None,
    measure_aware: bool = False,
) -> torch.Tensor:
    """Return robust Student-t likelihood for moment predictions."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or moment_mu.shape != channel.target_values.shape:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(moment_mu.device)
    if torch.count_nonzero(usable) == 0:
        return moment_mu.new_tensor(0.0)
    sigma = (
        moment_sigma.to(moment_mu.device).clamp(1e-3, 20.0)
        if moment_sigma is not None and moment_sigma.shape == moment_mu.shape
        else channel.target_sigmas.to(moment_mu.device).clamp(1e-3, 20.0)
    )
    residuals = (moment_mu[usable] - channel.target_values[usable]) / sigma[usable]
    nu = max(float(degrees_of_freedom), 1.1)
    loss = 0.5 * (nu + 1.0) * torch.log1p(torch.square(residuals) / nu) + torch.log(
        sigma[usable]
    )
    if measure_aware and channel.target_ids:
        target_ids = [
            target_id
            for target_id, keep in zip(
                channel.target_ids, usable.detach().cpu().tolist(), strict=False
            )
            if keep
        ]
        weights = torch.ones_like(loss)
        for family_name in CHEMICAL_SHIFT_FAMILIES:
            family_mask = torch.tensor(
                [
                    _atom_family_from_target_id(target_id) == family_name
                    for target_id in target_ids
                ],
                dtype=torch.bool,
                device=moment_mu.device,
            )
            if torch.any(family_mask):
                weights[family_mask] = _measure_aware_row_weights(
                    target_ids=[
                        target_id
                        for target_id, keep in zip(
                            target_ids,
                            family_mask.detach().cpu().tolist(),
                            strict=False,
                        )
                        if keep
                    ],
                    family_name=family_name,
                    residuals=residuals[family_mask].detach(),
                    dtype=moment_mu.dtype,
                    device=moment_mu.device,
                )
        return torch.sum(weights * loss) / torch.sum(weights).clamp_min(1e-6)
    return torch.mean(loss)


def gaussian_crps_calibration_loss(
    *,
    moment_mu: torch.Tensor | None,
    moment_sigma: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Gaussian CRPS calibration loss for moment posteriors."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or moment_mu.shape != channel.target_values.shape:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(moment_mu.device)
    if torch.count_nonzero(usable) == 0:
        return moment_mu.new_tensor(0.0)
    sigma = (
        moment_sigma.to(moment_mu.device).clamp(1e-3, 20.0)
        if moment_sigma is not None and moment_sigma.shape == moment_mu.shape
        else channel.target_sigmas.to(moment_mu.device).clamp(1e-3, 20.0)
    )
    z = (channel.target_values[usable] - moment_mu[usable]) / sigma[usable]
    inv_sqrt_two = z.new_tensor(1.0 / math.sqrt(2.0))
    normal_cdf = 0.5 * (1.0 + torch.erf(z * inv_sqrt_two))
    normal_pdf = torch.exp(-0.5 * torch.square(z)) / math.sqrt(2.0 * math.pi)
    crps = sigma[usable] * (
        z * (2.0 * normal_cdf - 1.0)
        + 2.0 * normal_pdf
        - z.new_tensor(1.0 / math.sqrt(math.pi))
    )
    return torch.mean(torch.relu(crps))


def joint_nmr_posterior_loss(
    *,
    moment_mu: torch.Tensor | None,
    moment_sigma: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    target_row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a diagonal-Gaussian joint loss for common NMR atom sets."""
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(moment_mu.device)
    sigma = (
        moment_sigma.to(moment_mu.device).clamp(1e-3, 20.0)
        if moment_sigma is not None and moment_sigma.shape == moment_mu.shape
        else channel.target_sigmas.to(moment_mu.device).clamp(1e-3, 20.0)
    )
    residue_rows: dict[tuple[str, int, str], dict[str, int]] = {}
    parsed_rows = [parse_chemical_shift_target_id(target_id) for target_id in channel.target_ids]
    for index, parsed in enumerate(parsed_rows):
        if not bool(usable[index]):
            continue
        family_name = parsed.get("atom_family")
        if family_name not in CHEMICAL_SHIFT_FAMILIES:
            continue
        key = (
            str(parsed.get("chain_id", "_")),
            int(parsed.get("residue_index", 0)),
            str(parsed.get("residue_name", "")),
        )
        residue_rows.setdefault(key, {})[str(family_name)] = index
    atom_sets = (
        ("HN", "N"),
        ("CA", "CB"),
        ("CA", "C'"),
        ("HN", "N", "CA"),
        ("CA", "CB", "C'"),
    )
    losses: list[torch.Tensor] = []
    for row_map in residue_rows.values():
        for atom_set in atom_sets:
            if not all(atom in row_map for atom in atom_set):
                continue
            indices = torch.tensor(
                [row_map[atom] for atom in atom_set],
                dtype=torch.long,
                device=moment_mu.device,
            )
            residual = (
                moment_mu[indices] - channel.target_values[indices]
            ) / sigma[indices]
            losses.append(torch.mean(torch.log1p(torch.square(residual))))
    if not losses:
        return moment_mu.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def paired_evidence_imputation_loss(
    *,
    moment_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
    evidence_row_mask: torch.Tensor | None = None,
    target_families: Sequence[str] | None = None,
) -> torch.Tensor:
    """Teach masked HN/C' rows from same-residue observed NMR evidence.

    This loss deliberately fits the imputation map only on evidence rows. It
    then applies the map to masked rows, so the held-out target value is used
    only as supervision and never as posterior evidence.
    """
    if moment_mu is None:
        return torch.tensor(0.0)
    if channel is None or moment_mu.numel() == 0 or not channel.target_ids:
        return moment_mu.new_tensor(0.0)
    if evidence_row_mask is None or evidence_row_mask.numel() != moment_mu.numel():
        return moment_mu.new_tensor(0.0)
    enabled = set(target_families or ())
    if not enabled:
        return moment_mu.new_tensor(0.0)
    usable = torch.any(channel.mask.bool(), dim=1) & torch.isfinite(moment_mu)
    evidence = usable & evidence_row_mask.bool().to(moment_mu.device)
    holdout = usable & (~evidence_row_mask.bool().to(moment_mu.device))
    parsed_rows = [parse_chemical_shift_target_id(target_id) for target_id in channel.target_ids]
    residue_maps: dict[tuple[str, int, str], dict[str, int]] = {}
    for index, parsed in enumerate(parsed_rows):
        family_name = parsed.get("atom_family")
        if family_name not in CHEMICAL_SHIFT_FAMILIES:
            continue
        key = (
            str(parsed.get("chain_id", "_")),
            int(parsed.get("residue_index", 0)),
            str(parsed.get("residue_name", "")),
        )
        residue_maps.setdefault(key, {})[str(family_name)] = index

    losses: list[torch.Tensor] = []
    pair_specs = {
        "HN": ("N",),
        "C'": ("CA", "CB"),
    }
    for target_family, source_families in pair_specs.items():
        if target_family not in enabled:
            continue
        for source_family in source_families:
            evidence_pairs: list[tuple[int, int]] = []
            holdout_pairs: list[tuple[int, int]] = []
            for row_map in residue_maps.values():
                if target_family not in row_map or source_family not in row_map:
                    continue
                target_index = row_map[target_family]
                source_index = row_map[source_family]
                if bool(evidence[target_index]) and bool(evidence[source_index]):
                    evidence_pairs.append((target_index, source_index))
                if bool(holdout[target_index]) and bool(evidence[source_index]):
                    holdout_pairs.append((target_index, source_index))
            if len(evidence_pairs) < 4 or len(holdout_pairs) < 2:
                continue
            evidence_target_indices = torch.tensor(
                [target for target, _source in evidence_pairs],
                dtype=torch.long,
                device=moment_mu.device,
            )
            evidence_source_indices = torch.tensor(
                [source for _target, source in evidence_pairs],
                dtype=torch.long,
                device=moment_mu.device,
            )
            holdout_target_indices = torch.tensor(
                [target for target, _source in holdout_pairs],
                dtype=torch.long,
                device=moment_mu.device,
            )
            holdout_source_indices = torch.tensor(
                [source for _target, source in holdout_pairs],
                dtype=torch.long,
                device=moment_mu.device,
            )
            source_values = channel.target_values[evidence_source_indices].detach()
            target_values = channel.target_values[evidence_target_indices].detach()
            source_mean = torch.mean(source_values)
            target_mean = torch.mean(target_values)
            source_centered = source_values - source_mean
            target_centered = target_values - target_mean
            slope = torch.sum(source_centered * target_centered) / torch.sum(
                torch.square(source_centered)
            ).clamp_min(1e-6)
            intercept = target_mean - slope * source_mean
            imputed_targets = (
                slope * channel.target_values[holdout_source_indices].detach()
                + intercept
            )
            predicted = moment_mu[holdout_target_indices]
            target_scale = torch.std(target_values, unbiased=False).clamp_min(1e-3)
            residual = (predicted - imputed_targets) / target_scale
            direct = (
                moment_mu[holdout_target_indices]
                - channel.target_values[holdout_target_indices]
            ) / target_scale
            # The imputation term teaches use of paired evidence; the small
            # direct term keeps the target CCC objective anchored.
            losses.append(
                torch.mean(torch.log1p(torch.square(residual)))
                + 0.25 * torch.mean(torch.log1p(torch.square(direct)))
            )
    if not losses:
        return moment_mu.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def _measure_aware_row_weights(
    *,
    target_ids: Sequence[str],
    family_name: str,
    residuals: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return simple NMR-measure reliability weights for one atom family."""
    weights = torch.ones(len(target_ids), dtype=dtype, device=device)
    if len(target_ids) == 0:
        return weights
    parsed_rows = [parse_chemical_shift_target_id(target_id) for target_id in target_ids]
    if family_name == "HN":
        penalties = []
        for parsed in parsed_rows:
            residue_index = int(parsed.get("residue_index", 0))
            residue_name = str(parsed.get("residue_name", ""))
            penalty = 1.0
            if residue_index <= 2:
                penalty *= 0.75
            if residue_name == "PRO":
                penalty *= 0.60
            penalties.append(penalty)
        weights = weights * torch.tensor(penalties, dtype=dtype, device=device)
    elif family_name == "C'" and residuals.numel() == weights.numel():
        residual_std = torch.std(residuals, unbiased=False).clamp_min(1e-3)
        robust = 1.0 / (1.0 + torch.abs(residuals / residual_std) / 4.0)
        weights = weights * robust.clamp(0.25, 1.0)
    return weights.clamp_min(0.05)


def _weighted_corr_stats(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted correlation statistics used by CCC decomposition."""
    weights = weights.to(predictions.device, predictions.dtype).clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-6)
    pred_mean = torch.sum(weights * predictions)
    target_mean = torch.sum(weights * targets)
    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean
    pred_var = torch.sum(weights * torch.square(pred_centered)).clamp_min(1e-6)
    target_var = torch.sum(weights * torch.square(target_centered)).clamp_min(1e-6)
    pred_std = torch.sqrt(pred_var)
    target_std = torch.sqrt(target_var)
    corr = torch.sum(weights * pred_centered * target_centered) / (
        pred_std * target_std
    ).clamp_min(1e-6)
    return corr.clamp(-1.0, 1.0), pred_std, target_std, pred_mean, target_mean


def _weighted_stack_average(
    values: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Average a sequence of scalar tensors with nonnegative weights."""
    stacked = torch.stack(list(values))
    weights = weights.to(stacked.device, stacked.dtype)
    if torch.sum(weights) <= 0:
        return torch.mean(stacked)
    return torch.sum(stacked * weights) / torch.sum(weights).clamp_min(1e-6)


def covariance_slope(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return least-squares slope of predictions against targets."""

    centered_pred = predictions - torch.mean(predictions)
    centered_target = targets - torch.mean(targets)
    variance = torch.mean(torch.square(centered_target)).clamp_min(1e-6)
    covariance = torch.mean(centered_pred * centered_target)
    return covariance / variance


def _sigma_anchor_loss(sigma: torch.Tensor, target_sigma: torch.Tensor) -> torch.Tensor:
    """Keep learned sigma near measurement-scale floors to avoid sigma escape."""

    anchor = target_sigma.clamp(0.2, 4.0)
    return 0.05 * torch.mean(torch.square(torch.log(sigma / anchor)))


def moment_sample_consistency_loss(
    *,
    moment_mu: torch.Tensor | None,
    predicted_weights: torch.Tensor,
    channel: PreparedObservableChannel | None,
    forward_residual_mu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep direct fast moments consistent with weighted sample expectations."""
    if moment_mu is None:
        return predicted_weights.new_tensor(0.0)
    if channel is None or moment_mu.numel() == 0:
        return moment_mu.new_tensor(0.0)
    sample_mean, usable = channel_expectations(
        predicted_weights,
        channel,
        forward_residual_mu=forward_residual_mu,
    )
    if moment_mu.shape != sample_mean.shape:
        return moment_mu.new_tensor(0.0)
    usable = usable & torch.isfinite(moment_mu)
    if torch.count_nonzero(usable) == 0:
        return moment_mu.new_tensor(0.0)
    return torch.nn.functional.smooth_l1_loss(moment_mu[usable], sample_mean[usable])


def state_occupancy_distillation_loss(
    *,
    state_weights: torch.Tensor | None,
    state_assignments: torch.Tensor | None,
    teacher_weights: torch.Tensor,
) -> torch.Tensor:
    """Distill posterior state occupancy from teacher/candidate mass."""
    if state_weights is None or state_assignments is None:
        return teacher_weights.new_tensor(0.0)
    if state_weights.numel() == 0 or state_assignments.numel() == 0:
        return teacher_weights.new_tensor(0.0)
    if state_assignments.shape[0] != teacher_weights.shape[0]:
        return teacher_weights.new_tensor(0.0)
    target = state_assignments.t() @ teacher_weights
    target = target / target.sum().clamp_min(1e-8)
    predicted = state_weights / state_weights.sum().clamp_min(1e-8)
    return torch.sum(
        target.clamp_min(1e-8)
        * (torch.log(target.clamp_min(1e-8)) - torch.log(predicted.clamp_min(1e-8)))
    )


def state_diversity_loss(state_weights: torch.Tensor | None) -> torch.Tensor:
    """Return a negative entropy penalty to discourage state collapse."""
    if state_weights is None or state_weights.numel() == 0:
        return torch.tensor(0.0)
    weights = state_weights / state_weights.sum().clamp_min(1e-8)
    entropy = -torch.sum(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8)))
    return -entropy


def state_entropy_floor_loss(
    state_weights: torch.Tensor | None,
    *,
    floor: float,
) -> torch.Tensor:
    """Penalize posterior state entropy below one normalized floor."""
    if state_weights is None or state_weights.numel() <= 1 or floor <= 0.0:
        return torch.tensor(0.0)
    weights = state_weights / state_weights.sum().clamp_min(1e-8)
    entropy = -torch.sum(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8)))
    normalized = entropy / torch.log(weights.new_tensor(float(weights.numel())))
    return torch.square(torch.relu(weights.new_tensor(float(floor)) - normalized))


def state_repulsion_loss(state_tokens: torch.Tensor | None) -> torch.Tensor:
    """Discourage posterior state-token collapse in latent space."""
    if state_tokens is None or state_tokens.numel() == 0 or state_tokens.shape[0] <= 1:
        return torch.tensor(0.0)
    normalized = torch.nn.functional.normalize(state_tokens, dim=-1)
    similarity = normalized @ normalized.t()
    off_diagonal = ~torch.eye(
        similarity.shape[0],
        dtype=torch.bool,
        device=similarity.device,
    )
    return torch.mean(torch.square(torch.relu(similarity[off_diagonal])))


def basin_coverage_loss(
    state_assignments: torch.Tensor | None,
    *,
    floor: float,
) -> torch.Tensor:
    """Encourage candidate basins to cover more than one state token."""
    if state_assignments is None or state_assignments.numel() == 0:
        return torch.tensor(0.0)
    coverage = state_assignments.mean(dim=0)
    return state_entropy_floor_loss(coverage, floor=floor)


def secondary_shift_concordance_loss(
    weights: torch.Tensor,
    channel: PreparedObservableChannel | None,
    *,
    forward_residual_mu: torch.Tensor | None = None,
    target_row_mask: torch.Tensor | None = None,
    family_balance: bool = True,
    family_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return CCC loss after random-coil centering of chemical shifts."""
    if channel is None or not channel.target_ids:
        return weights.new_tensor(0.0)
    predictions, usable = channel_expectations(
        weights,
        channel,
        forward_residual_mu=forward_residual_mu,
    )
    baselines = torch.tensor(
        random_coil_baselines_for_target_ids(channel.target_ids),
        dtype=predictions.dtype,
        device=predictions.device,
    )
    usable = usable & torch.isfinite(baselines)
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        usable = usable & target_row_mask.bool().to(usable.device)
    if not torch.any(usable):
        return weights.new_tensor(0.0)
    secondary_predictions = predictions - baselines
    secondary_targets = channel.target_values - baselines
    if family_balance:
        losses: list[torch.Tensor] = []
        weights_for_average: list[float] = []
        for family_name in CHEMICAL_SHIFT_FAMILIES:
            family_mask = torch.tensor(
                [
                    _atom_family_from_target_id(target_id) == family_name
                    for target_id in channel.target_ids
                ],
                dtype=torch.bool,
                device=predictions.device,
            )
            combined = usable & family_mask
            if torch.count_nonzero(combined) < 2:
                continue
            losses.append(
                concordance_loss(
                    secondary_predictions[combined],
                    secondary_targets[combined],
                )
            )
            weights_for_average.append(
                float((family_weights or {}).get(family_name, 1.0))
            )
        if losses:
            weight_tensor = torch.tensor(
                weights_for_average,
                dtype=predictions.dtype,
                device=predictions.device,
            ).clamp_min(0.0)
            stacked = torch.stack(losses)
            if torch.sum(weight_tensor) > 0:
                return torch.sum(stacked * weight_tensor) / torch.sum(weight_tensor)
            return torch.mean(stacked)
    if torch.count_nonzero(usable) < 2:
        return weights.new_tensor(0.0)
    return concordance_loss(secondary_predictions[usable], secondary_targets[usable])


def affine_calibrated_concordance_loss(
    weights: torch.Tensor,
    channel: PreparedObservableChannel | None,
    *,
    forward_residual_mu: torch.Tensor | None = None,
    calibration_row_mask: torch.Tensor | None = None,
    target_row_mask: torch.Tensor | None = None,
    family_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return CCC loss after evidence-only family affine calibration.

    The calibration rows fit ``target ~= a * prediction + b``. Masked holdout
    rows can then be evaluated without letting their target values influence the
    affine parameters.
    """
    if channel is None or not channel.target_ids:
        return weights.new_tensor(0.0)
    predictions, usable = channel_expectations(
        weights,
        channel,
        forward_residual_mu=forward_residual_mu,
    )
    if target_row_mask is not None and target_row_mask.numel() == usable.numel():
        evaluation_mask = usable & target_row_mask.bool().to(usable.device)
    else:
        evaluation_mask = usable
    if (
        calibration_row_mask is not None
        and calibration_row_mask.numel() == usable.numel()
    ):
        calibration_mask = usable & calibration_row_mask.bool().to(usable.device)
    else:
        calibration_mask = usable
    losses: list[torch.Tensor] = []
    weights_for_average: list[float] = []
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        family_mask = torch.tensor(
            [
                _atom_family_from_target_id(target_id) == family_name
                for target_id in channel.target_ids
            ],
            dtype=torch.bool,
            device=predictions.device,
        )
        family_eval = evaluation_mask & family_mask
        if torch.count_nonzero(family_eval) < 2:
            continue
        family_cal = calibration_mask & family_mask
        calibrated = _affine_calibrated_predictions(
            predictions=predictions,
            targets=channel.target_values,
            calibration_mask=family_cal,
        )
        losses.append(
            concordance_loss(
                calibrated[family_eval], channel.target_values[family_eval]
            )
        )
        weights_for_average.append(float((family_weights or {}).get(family_name, 1.0)))
    if not losses:
        return weights.new_tensor(0.0)
    weight_tensor = torch.tensor(
        weights_for_average,
        dtype=predictions.dtype,
        device=predictions.device,
    ).clamp_min(0.0)
    stacked = torch.stack(losses)
    if torch.sum(weight_tensor) > 0:
        return torch.sum(stacked * weight_tensor) / torch.sum(weight_tensor)
    return torch.mean(stacked)


def _affine_calibrated_predictions(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    calibration_mask: torch.Tensor,
) -> torch.Tensor:
    """Fit a differentiable one-dimensional affine calibration."""
    if torch.count_nonzero(calibration_mask) < 2:
        return predictions
    x = predictions[calibration_mask]
    y = targets[calibration_mask]
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    slope = torch.sum(x_centered * y_centered) / torch.sum(
        torch.square(x_centered)
    ).clamp_min(1e-6)
    intercept = y_mean - slope * x_mean
    return slope * predictions + intercept


def ccc_oracle_kl_loss(
    *,
    predicted_weights: torch.Tensor,
    candidate_numeric_features: torch.Tensor | None,
    config: StudentTrainingConfig,
) -> torch.Tensor:
    """Distill a target-conditioned CCC-oracle candidate ranking."""
    if (
        (config.ccc_oracle_kl_weight + config.ccc_oracle_distillation_weight) <= 0.0
        or candidate_numeric_features is None
        or candidate_numeric_features.ndim != 2
    ):
        return predicted_weights.new_tensor(0.0)
    task_spec = resolve_training_task(config.training_task, config.training_input_mode)
    proxy_index = ccc_proxy_feature_index(
        candidate_feature_dim=int(candidate_numeric_features.shape[1]),
        input_mode=task_spec.input_feature_mode,
    )
    if proxy_index is None:
        return predicted_weights.new_tensor(0.0)
    proxy_scores = candidate_numeric_features[:, proxy_index].to(
        predicted_weights.dtype
    )
    temperature = max(float(config.ccc_oracle_temperature), 1e-6)
    oracle_weights = torch.softmax(proxy_scores / temperature, dim=0).detach()
    clipped_oracle = oracle_weights.clamp_min(1e-8)
    clipped_predicted = predicted_weights.clamp_min(1e-8)
    return torch.sum(
        clipped_oracle * (torch.log(clipped_oracle) - torch.log(clipped_predicted))
    )


def evidence_likelihood_nll_loss(
    *,
    predicted_weights: torch.Tensor,
    evidence_log_likelihood: torch.Tensor | None,
) -> torch.Tensor:
    """Return expected negative evidence log-likelihood under posterior weights."""
    if evidence_log_likelihood is None:
        return predicted_weights.new_tensor(0.0)
    evidence = torch.nan_to_num(
        evidence_log_likelihood.to(predicted_weights.dtype),
        nan=-100.0,
        neginf=-100.0,
        posinf=10.0,
    )
    return -torch.sum(predicted_weights * evidence)


def normalized_chemical_shift_evidence_nll_loss(
    *,
    predicted_weights: torch.Tensor,
    channel: PreparedObservableChannel | None,
    forward_residual_mu: torch.Tensor | None,
    evidence_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return family-balanced evidence NLL independent of target count."""
    if channel is None or forward_residual_mu is None or not channel.target_ids:
        return predicted_weights.new_tensor(0.0)
    values = _values_with_forward_residual(channel, forward_residual_mu)
    if values.shape != channel.values.shape:
        return predicted_weights.new_tensor(0.0)
    usable = channel.mask.bool()
    if evidence_mask is not None and evidence_mask.numel() == usable.shape[0]:
        usable = usable & evidence_mask.bool().to(usable.device).unsqueeze(1)
    target = channel.target_values.to(values.dtype)
    sigma = channel.target_sigmas.to(values.dtype).clamp_min(1e-6)
    residual = (values - target.unsqueeze(1)) / sigma.unsqueeze(1)
    nll = (
        0.5 * torch.square(residual)
        + torch.log(sigma.unsqueeze(1))
        + 0.5 * torch.log(values.new_tensor(2.0 * torch.pi))
    )
    family_losses: list[torch.Tensor] = []
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        family_mask = torch.tensor(
            [
                _atom_family_from_target_id(target_id) == family_name
                for target_id in channel.target_ids
            ],
            dtype=torch.bool,
            device=values.device,
        )
        combined = usable & family_mask.unsqueeze(1)
        counts = combined.sum(dim=0).to(values.dtype)
        valid_candidates = counts > 0
        if torch.count_nonzero(valid_candidates) == 0:
            continue
        candidate_nll = torch.where(combined, nll, torch.zeros_like(nll)).sum(
            dim=0
        ) / counts.clamp_min(1.0)
        family_weight = predicted_weights[valid_candidates]
        family_weight = family_weight / family_weight.sum().clamp_min(1e-8)
        family_losses.append(torch.sum(family_weight * candidate_nll[valid_candidates]))
    if not family_losses:
        return predicted_weights.new_tensor(0.0)
    return torch.mean(torch.stack(family_losses))


def residue_atom_sequence_smoothness_loss(
    residual_mu: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
) -> torch.Tensor:
    """Penalize sharp adjacent-residue residual jumps within each atom family."""
    if (
        residual_mu is None
        or channel is None
        or residual_mu.shape != channel.values.shape
    ):
        if residual_mu is not None:
            return residual_mu.new_tensor(0.0)
        return torch.tensor(0.0)
    pair_losses: list[torch.Tensor] = []
    parsed_rows = [
        parse_chemical_shift_target_id(target_id) for target_id in channel.target_ids
    ]
    for family_name in CHEMICAL_SHIFT_FAMILIES:
        family_indices = [
            index
            for index, parsed in enumerate(parsed_rows)
            if parsed["atom_family"] == family_name
        ]
        family_indices.sort(
            key=lambda index: (
                str(parsed_rows[index]["chain_id"]),
                int(parsed_rows[index]["residue_index"]),
            )
        )
        for left, right in zip(family_indices, family_indices[1:], strict=False):
            left_row = parsed_rows[left]
            right_row = parsed_rows[right]
            if left_row["chain_id"] != right_row["chain_id"]:
                continue
            if int(right_row["residue_index"]) - int(left_row["residue_index"]) != 1:
                continue
            pair_losses.append(
                torch.mean(torch.square(residual_mu[right] - residual_mu[left]))
            )
    if not pair_losses:
        return residual_mu.new_tensor(0.0)
    return torch.mean(torch.stack(pair_losses))


def channel_expectations(
    weights: torch.Tensor,
    channel: PreparedObservableChannel,
    forward_residual_mu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted expectation predictions and a usable-target mask."""
    values = _values_with_forward_residual(channel, forward_residual_mu)
    validity = channel.mask.float()
    denominator = (validity * weights.unsqueeze(0)).sum(dim=1)
    usable = denominator > 1e-8
    averaged = (validity * values * weights.unsqueeze(0)).sum(
        dim=1
    ) / denominator.clamp_min(1e-8)
    if channel.transform == "inverse_sixth":
        predictions = averaged.clamp_min(1e-8).pow(-1.0 / 6.0)
    else:
        predictions = averaged
    return predictions, usable


def ccc_geometry_alignment_loss(
    predicted_weights: torch.Tensor,
    candidate_numeric_features: torch.Tensor | None,
    config: StudentTrainingConfig,
    task_input_mode: str,
) -> torch.Tensor:
    """Reward posterior mass assigned to high-CCC geometry proxy candidates."""
    if (
        config.ccc_geometry_loss_weight <= 0.0
        or candidate_numeric_features is None
        or candidate_numeric_features.ndim != 2
    ):
        return predicted_weights.new_tensor(0.0)
    proxy_index = ccc_proxy_feature_index(
        candidate_feature_dim=int(candidate_numeric_features.shape[1]),
        input_mode=task_input_mode,
    )
    if proxy_index is None:
        return predicted_weights.new_tensor(0.0)
    scores = candidate_numeric_features[:, proxy_index].to(predicted_weights.dtype)
    valid = torch.isfinite(scores)
    if torch.count_nonzero(valid) < 2:
        return predicted_weights.new_tensor(0.0)
    centered = scores[valid] - torch.mean(scores[valid])
    scaled = centered / torch.std(centered, unbiased=False).clamp_min(1e-6)
    weights = predicted_weights[valid] / predicted_weights[valid].sum().clamp_min(1e-8)
    return -torch.sum(weights * scaled)


def forward_residual_supervision_loss(
    residual_mu: torch.Tensor | None,
    residual_sigma: torch.Tensor | None,
    channel: PreparedObservableChannel | None,
) -> torch.Tensor:
    """Supervise atom-family residual heads against candidate shift residuals."""
    if residual_mu is None or channel is None:
        if residual_sigma is not None:
            return residual_sigma.new_tensor(0.0)
        return torch.tensor(0.0)
    if residual_mu.numel() == 0:
        return residual_mu.new_tensor(0.0)
    if residual_mu.shape == channel.values.shape:
        target_residual = channel.target_values.unsqueeze(1) - channel.values
        validity = channel.mask.float()
        if torch.count_nonzero(validity) == 0:
            return residual_mu.new_tensor(0.0)
        per_target_candidate = torch.nn.functional.smooth_l1_loss(
            residual_mu,
            target_residual.to(residual_mu.dtype),
            reduction="none",
        )
        residual_loss = (
            per_target_candidate * validity
        ).sum() / validity.sum().clamp_min(1.0)
        sigma_loss = residual_mu.new_tensor(0.0)
        if residual_sigma is not None and residual_sigma.shape == residual_mu.shape:
            sigma_loss = torch.mean(torch.log(residual_sigma.clamp_min(1e-6)))
        return residual_loss + 0.01 * sigma_loss
    losses: list[torch.Tensor] = []
    for family_index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
        row_mask = torch.tensor(
            [
                _atom_family_from_target_id(target_id) == family_name
                for target_id in channel.target_ids
            ],
            dtype=torch.bool,
            device=residual_mu.device,
        )
        if torch.count_nonzero(row_mask) == 0:
            continue
        validity = channel.mask[row_mask].float()
        if torch.count_nonzero(validity) == 0:
            continue
        target_residual = (
            channel.target_values[row_mask].unsqueeze(1) - channel.values[row_mask]
        )
        denominator = validity.sum(dim=0).clamp_min(1.0)
        candidate_target = (target_residual * validity).sum(dim=0) / denominator
        candidate_valid = denominator > 0
        if torch.count_nonzero(candidate_valid) == 0:
            continue
        prediction = residual_mu[:, family_index][candidate_valid]
        target = candidate_target[candidate_valid].to(prediction.dtype)
        losses.append(torch.nn.functional.smooth_l1_loss(prediction, target))
    if not losses:
        return residual_mu.new_tensor(0.0)
    sigma_loss = residual_mu.new_tensor(0.0)
    if residual_sigma is not None:
        sigma_loss = torch.mean(torch.log(residual_sigma.clamp_min(1e-6)))
    return torch.mean(torch.stack(losses)) + 0.01 * sigma_loss


def forward_residual_regularization_loss(
    residual_mu: torch.Tensor | None,
) -> torch.Tensor:
    """Keep learned chemical-shift residuals conservative."""
    if residual_mu is None:
        return torch.tensor(0.0)
    return torch.mean(torch.square(residual_mu))


def _values_with_forward_residual(
    channel: PreparedObservableChannel,
    residual_mu: torch.Tensor | None,
) -> torch.Tensor:
    """Return channel values after optional atom-family residual correction."""
    if residual_mu is None or not channel.target_ids:
        return channel.values
    if residual_mu.shape == channel.values.shape:
        return channel.values + residual_mu
    adjusted = channel.values
    for family_index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
        row_mask = torch.tensor(
            [
                _atom_family_from_target_id(target_id) == family_name
                for target_id in channel.target_ids
            ],
            dtype=torch.bool,
            device=channel.values.device,
        )
        if torch.count_nonzero(row_mask) == 0:
            continue
        adjusted = adjusted.clone()
        adjusted[row_mask] = adjusted[row_mask] + residual_mu[
            :, family_index
        ].unsqueeze(0)
    return adjusted


def concordance_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return ``1 - Lin's CCC`` for one paired vector."""
    if predictions.numel() < 2:
        return predictions.new_tensor(0.0)
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)
    pred_var = torch.mean(torch.square(predictions - pred_mean))
    target_var = torch.mean(torch.square(targets - target_mean))
    covariance = torch.mean((predictions - pred_mean) * (targets - target_mean))
    denominator = pred_var + target_var + torch.square(pred_mean - target_mean)
    ccc = (2.0 * covariance) / denominator.clamp_min(1e-8)
    return 1.0 - ccc


def family_balanced_mean(
    per_target_values: torch.Tensor,
    usable: torch.Tensor,
    target_ids: list[str],
    family_weights: dict[str, float],
) -> torch.Tensor:
    """Average target losses by atom family before averaging across families."""
    family_losses: list[torch.Tensor] = []
    weights_for_average: list[float] = []
    usable_indices = torch.nonzero(usable, as_tuple=False).flatten()
    usable_values = per_target_values
    for family_name in sorted(
        {
            family
            for family in (_atom_family_from_target_id(t) for t in target_ids)
            if family is not None
        }
    ):
        family_mask = torch.tensor(
            [
                _atom_family_from_target_id(target_ids[int(index)]) == family_name
                for index in usable_indices.detach().cpu().tolist()
            ],
            dtype=torch.bool,
            device=per_target_values.device,
        )
        if not torch.any(family_mask):
            continue
        family_losses.append(torch.mean(usable_values[family_mask]))
        weights_for_average.append(float(family_weights.get(family_name, 1.0)))
    if not family_losses:
        return torch.mean(per_target_values)
    weight_tensor = torch.tensor(
        weights_for_average,
        dtype=per_target_values.dtype,
        device=per_target_values.device,
    ).clamp_min(0.0)
    stacked = torch.stack(family_losses)
    if torch.sum(weight_tensor) <= 0:
        return torch.mean(stacked)
    return torch.sum(stacked * weight_tensor) / torch.sum(weight_tensor)


def _atom_family_from_target_id(target_id: str) -> str | None:
    """Return one canonical chemical-shift atom-family label."""
    if not str(target_id).startswith("cs:"):
        return None
    atom_name = str(target_id).rsplit(":", 1)[-1]
    return {
        "H": "HN",
        "HN": "HN",
        "N": "N",
        "CA": "CA",
        "CB": "CB",
        "C": "C'",
    }.get(atom_name)


def to_tensors(example, device: torch.device) -> dict[str, Any]:
    """Move one prepared example to torch tensors on the target device."""
    channels = {
        name: PreparedObservableChannel(
            values=torch.tensor(channel.values, dtype=torch.float32, device=device),
            mask=torch.tensor(channel.mask, dtype=torch.bool, device=device),
            target_values=torch.tensor(
                channel.target_values,
                dtype=torch.float32,
                device=device,
            ),
            target_sigmas=torch.tensor(
                channel.target_sigmas,
                dtype=torch.float32,
                device=device,
            ),
            transform=channel.transform,
            target_ids=list(channel.target_ids),
            lower_bounds=(
                torch.tensor(channel.lower_bounds, dtype=torch.float32, device=device)
                if channel.lower_bounds is not None
                else None
            ),
            upper_bounds=(
                torch.tensor(channel.upper_bounds, dtype=torch.float32, device=device)
                if channel.upper_bounds is not None
                else None
            ),
        )
        for name, channel in example.channels.items()
    }
    chemical_shift_channel = example.channels.get("chemical_shifts")
    return {
        "sequence_tokens": torch.tensor(
            example.sequence_tokens,
            dtype=torch.long,
            device=device,
        ),
        "example_features": torch.tensor(
            example.example_features,
            dtype=torch.float32,
            device=device,
        ),
        "candidate_source_indices": torch.tensor(
            example.candidate_source_indices,
            dtype=torch.long,
            device=device,
        ),
        "candidate_numeric_features": torch.tensor(
            example.candidate_numeric_features,
            dtype=torch.float32,
            device=device,
        ),
        "teacher_weights": torch.tensor(
            example.teacher_weights,
            dtype=torch.float32,
            device=device,
        ),
        "channels": channels,
        "chemical_shift_target_features": torch.tensor(
            chemical_shift_target_features(
                target_ids=(
                    chemical_shift_channel.target_ids
                    if chemical_shift_channel is not None
                    else []
                ),
                sequence_length=len(example.sequence_tokens),
            ),
            dtype=torch.float32,
            device=device,
        ),
    }


def chemical_shift_target_features(
    target_ids: list[str],
    sequence_length: int,
) -> np.ndarray:
    """Return residue/atom metadata features for chemical-shift targets."""
    rows = []
    denominator = max(int(sequence_length), 1)
    for target_id in target_ids:
        parsed = parse_chemical_shift_target_id(target_id)
        residue_index = parsed["residue_index"]
        rows.append(
            [
                float(residue_index),
                float(AA3_VOCAB.get(parsed["residue_name"], 0)),
                float(_atom_family_index(parsed["atom_family"])),
                float(residue_index) / float(denominator),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def parse_chemical_shift_target_id(target_id: str) -> dict[str, Any]:
    """Parse ``cs:{chain}:{seq_id}:{comp_id}:{atom_id}`` target ids."""
    parts = str(target_id).split(":")
    if len(parts) != 5 or parts[0] != "cs":
        return {
            "chain_id": "_",
            "residue_index": 0,
            "residue_name": "",
            "atom_name": "",
            "atom_family": None,
        }
    residue_index = 0
    try:
        residue_index = int(parts[2])
    except ValueError:
        residue_index = 0
    return {
        "chain_id": parts[1],
        "residue_index": max(residue_index, 0),
        "residue_name": parts[3].upper(),
        "atom_name": parts[4].upper(),
        "atom_family": _atom_family_from_target_id(target_id),
    }


def _atom_family_index(atom_family: str | None) -> int:
    """Return the canonical atom-family index used by residual heads."""
    try:
        return CHEMICAL_SHIFT_FAMILIES.index(atom_family or "")
    except ValueError:
        return 0


def build_learnability_report(
    model: StudentDensityModel,
    train_examples: list[Any],
    val_examples: list[Any],
    device: torch.device,
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Compare simple support baselines, oracle proxies, and model gradients."""
    train_sample = list(train_examples[:50])
    val_sample = list(val_examples)
    variants = {
        "uniform": lambda example: _uniform_weights(len(example.teacher_weights)),
        "teacher": lambda example: _normalized_np(example.teacher_weights),
        "best_csfit_onehot": _best_csfit_onehot_weights,
        "softmax_csfit": lambda example: _softmax_csfit_weights(example, tau=0.25),
        "best_family_ccc_onehot": _best_family_ccc_onehot_weights,
        "softmax_family_ccc": lambda example: _softmax_family_ccc_weights(
            example,
            tau=config.ccc_oracle_temperature,
        ),
        "per_family_best_candidate": lambda example: _per_family_best_candidate_weights(
            example,
            config.chemical_shift_family_weights,
        ),
        "family_weighted_oracle": lambda example: _family_weighted_oracle_weights(
            example,
            config.chemical_shift_family_weights,
            tau=config.ccc_oracle_temperature,
        ),
        "current_model": lambda example: _model_weights(
            model,
            example,
            device,
            config,
        ),
    }
    split_payloads = {}
    for split_name, examples in [
        ("train_head50", train_sample),
        ("val", val_sample),
    ]:
        split_payloads[split_name] = {
            variant_name: _learnability_variant_summary(
                examples=examples,
                split_name=split_name,
                weight_fn=weight_fn,
            )
            for variant_name, weight_fn in variants.items()
        }
    return {
        "status": "ok",
        "task_contract": resolve_training_task(
            config.training_task,
            config.training_input_mode,
        ).as_dict(),
        "training_input_mode": config.training_input_mode,
        "examples": {
            "train_head50": len(train_sample),
            "val": len(val_sample),
        },
        "variant_metrics": split_payloads,
        "gradient_summary": _component_gradient_summary(
            model=model,
            examples=list(train_examples[:40]),
            device=device,
            config=config,
        ),
    }


def guardrail_status(
    metric_summary: dict[str, float],
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Return the configured teacher-KL guardrail state for one best epoch."""
    if config.teacher_kl_guardrail is None:
        return {"status": "not_configured"}
    value = metric_summary.get("val_teacher_kl_macro")
    if value is None or not math.isfinite(float(value)):
        return {
            "status": "missing_metric",
            "threshold": float(config.teacher_kl_guardrail),
        }
    return {
        "status": "ok" if float(value) <= config.teacher_kl_guardrail else "failed",
        "value": float(value),
        "threshold": float(config.teacher_kl_guardrail),
    }


def _learnability_variant_summary(
    examples: list[Any],
    split_name: str,
    weight_fn,
) -> dict[str, float]:
    """Aggregate one learnability baseline or model variant."""
    if not examples:
        return {}
    payloads = [
        evaluate_prepared_example(example, weight_fn(example)) for example in examples
    ]
    summary = aggregate_epoch_metrics(
        payloads=payloads,
        split=split_name,
        report_micro_metrics=True,
    )["metric_summary"]
    wanted = [
        "teacher_kl_macro",
        "teacher_js_macro",
        "teacher_top10_mass_overlap_macro",
        "teacher_ess_abs_error_macro",
        "cs_rmse_z_macro",
        "cs_rmse_ppm_macro",
        "cs_mae_ppm_macro",
        "cs_ccc_macro",
        "cs_family_ccc_macro",
        "cs_family_ccc_loss_macro",
        "cs_HN_ccc_macro",
        "cs_N_ccc_macro",
        "cs_CA_ccc_macro",
        "cs_CB_ccc_macro",
        "cs_C'_ccc_macro",
    ]
    return {
        name: _json_float(summary.get(f"{split_name}_{name}"))
        for name in wanted
        if f"{split_name}_{name}" in summary
    }


def _component_gradient_summary(
    model: StudentDensityModel,
    examples: list[Any],
    device: torch.device,
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Summarize component gradient norms and cosine agreement on logits."""
    if not examples:
        return {"status": "no_examples"}
    model.eval()
    rows = []
    for example in examples:
        tensors = to_tensors(example, device)
        evidence_mask = chemical_shift_evidence_mask(
            tensors["channels"].get("chemical_shifts"),
            config=config,
            training=False,
            device=device,
            example_index=0,
        )
        outputs = model_outputs_for_example(
            model=model,
            tensors=tensors,
            evidence_mask=evidence_mask,
            config=config,
        )
        logits = outputs["logits"]
        weights = torch.softmax(logits, dim=0)
        losses = compute_losses(
            predicted_weights=weights,
            teacher_weights=tensors["teacher_weights"],
            channels=tensors["channels"],
            config=config,
            candidate_numeric_features=tensors["candidate_numeric_features"],
            forward_residual_mu=outputs.get("forward_residual_mu"),
            forward_residual_sigma=outputs.get("forward_residual_sigma"),
            evidence_log_likelihood=outputs.get("evidence_log_likelihood"),
            chemical_shift_evidence_mask=evidence_mask,
            moment_mu=outputs.get("moment_mu"),
            moment_sigma=outputs.get("moment_sigma"),
            posterior_state_weights=outputs.get("posterior_state_weights"),
            posterior_state_assignments=outputs.get("posterior_state_assignments"),
        )
        gradients = {
            "kl": _grad_for(config.weight_kl_weight * losses["weight_kl"], logits),
            "cs": _grad_for(
                config.chemical_shift_loss_weight * losses["chemical_shift_loss"],
                logits,
            ),
            "ccc": _grad_for(
                config.chemical_shift_ccc_loss_weight
                * losses["chemical_shift_ccc_loss"],
                logits,
            ),
            "ccc_oracle": _grad_for(
                config.ccc_oracle_kl_weight * losses["ccc_oracle_kl"],
                logits,
            ),
            "posterior_evidence": _grad_for(
                config.posterior_evidence_kl_weight * losses["posterior_evidence_kl"],
                logits,
            ),
            "observable_oracle": _grad_for(
                config.observable_oracle_distillation_weight
                * losses["observable_oracle_kl"],
                logits,
            ),
            "total": _grad_for(losses["total_loss"], logits),
        }
        rows.append(
            {
                "kl_norm": _tensor_norm(gradients["kl"]),
                "cs_norm": _tensor_norm(gradients["cs"]),
                "ccc_norm": _tensor_norm(gradients["ccc"]),
                "ccc_oracle_norm": _tensor_norm(gradients["ccc_oracle"]),
                "posterior_evidence_norm": _tensor_norm(
                    gradients["posterior_evidence"]
                ),
                "observable_oracle_norm": _tensor_norm(gradients["observable_oracle"]),
                "total_norm": _tensor_norm(gradients["total"]),
                "cos_kl_cs": _tensor_cosine(gradients["kl"], gradients["cs"]),
                "cos_kl_ccc": _tensor_cosine(gradients["kl"], gradients["ccc"]),
                "cos_kl_ccc_oracle": _tensor_cosine(
                    gradients["kl"],
                    gradients["ccc_oracle"],
                ),
                "cos_cs_ccc": _tensor_cosine(gradients["cs"], gradients["ccc"]),
                "cos_ccc_oracle_ccc": _tensor_cosine(
                    gradients["ccc_oracle"],
                    gradients["ccc"],
                ),
                "cos_posterior_evidence_ccc": _tensor_cosine(
                    gradients["posterior_evidence"],
                    gradients["ccc"],
                ),
            }
        )
    return {
        "status": "ok",
        "examples": len(rows),
        "metrics": {
            key: _distribution_summary([row[key] for row in rows]) for key in rows[0]
        },
    }


def _grad_for(loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Return a detached gradient for one scalar loss with zero fallback."""
    if not loss.requires_grad:
        return torch.zeros_like(logits).detach()
    gradient = torch.autograd.grad(
        loss,
        logits,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(logits).detach()
    return gradient.detach()


def _model_weights(
    model: StudentDensityModel,
    example: Any,
    device: torch.device,
    config: StudentTrainingConfig,
) -> np.ndarray:
    """Return current model weights for one prepared example."""
    model.eval()
    with torch.no_grad():
        tensors = to_tensors(example, device)
        outputs = model_outputs_for_example(
            model=model,
            tensors=tensors,
            evidence_mask=None,
            config=config,
        )
    return outputs["weights"].detach().cpu().numpy()


def _uniform_weights(candidate_count: int) -> np.ndarray:
    """Return one uniform candidate-weight vector."""
    return np.full(candidate_count, 1.0 / max(candidate_count, 1), dtype=np.float64)


def _normalized_np(values: Any) -> np.ndarray:
    """Return one normalized numpy vector."""
    array = np.asarray(values, dtype=np.float64)
    return array / max(float(array.sum()), 1e-12)


def _best_csfit_onehot_weights(example: Any) -> np.ndarray:
    """Return one-hot weight on the candidate with lowest CS fit RMSE feature."""
    features = np.asarray(example.candidate_numeric_features, dtype=np.float64)
    candidate_count = len(example.teacher_weights)
    weights = np.zeros(candidate_count, dtype=np.float64)
    score = features[:, 6] if features.shape[1] > 6 else np.zeros(candidate_count)
    weights[int(np.nanargmin(score))] = 1.0
    return weights


def _softmax_csfit_weights(example: Any, tau: float) -> np.ndarray:
    """Return a softmax over the negative CS fit RMSE feature."""
    features = np.asarray(example.candidate_numeric_features, dtype=np.float64)
    candidate_count = len(example.teacher_weights)
    score = features[:, 6] if features.shape[1] > 6 else np.zeros(candidate_count)
    logits = -score / max(float(tau), 1e-6)
    logits = logits - float(np.nanmax(logits))
    weights = np.exp(logits)
    return weights / max(float(weights.sum()), 1e-12)


def _best_family_ccc_onehot_weights(example: Any) -> np.ndarray:
    """Return one-hot weight on the candidate with highest family CCC proxy."""
    scores = _candidate_family_ccc_proxy_scores(example)
    weights = np.zeros(len(example.teacher_weights), dtype=np.float64)
    weights[int(np.nanargmax(scores))] = 1.0
    return weights


def _softmax_family_ccc_weights(example: Any, tau: float) -> np.ndarray:
    """Return a softmax over candidate family CCC proxy scores."""
    scores = _candidate_family_ccc_proxy_scores(example)
    logits = scores / max(float(tau), 1e-6)
    logits = logits - float(np.nanmax(logits))
    weights = np.exp(logits)
    return weights / max(float(weights.sum()), 1e-12)


def _per_family_best_candidate_weights(
    example: Any,
    family_weights: dict[str, float],
) -> np.ndarray:
    """Place mass on the best candidate for each atom family."""
    family_scores = _candidate_family_ccc_matrix(example)
    weights = np.zeros(len(example.teacher_weights), dtype=np.float64)
    for family_index, family_name in enumerate(["HN", "N", "CA", "CB", "C'"]):
        scores = family_scores[:, family_index]
        finite = np.isfinite(scores)
        if not np.any(finite):
            continue
        best_index = int(np.nanargmax(scores))
        weights[best_index] += float(family_weights.get(family_name, 1.0))
    if weights.sum() <= 0.0:
        return _uniform_weights(len(example.teacher_weights))
    return weights / float(weights.sum())


def _family_weighted_oracle_weights(
    example: Any,
    family_weights: dict[str, float],
    tau: float,
) -> np.ndarray:
    """Return softmax weights from a family-weighted CCC proxy."""
    family_scores = _candidate_family_ccc_matrix(example)
    weights = np.asarray(
        [
            float(family_weights.get(family, 1.0))
            for family in ["HN", "N", "CA", "CB", "C'"]
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(family_scores)
    weighted_scores = np.where(finite, family_scores, 0.0) * weights.reshape(1, -1)
    denominators = np.where(finite, weights.reshape(1, -1), 0.0).sum(axis=1)
    proxy = np.divide(
        weighted_scores.sum(axis=1),
        np.clip(denominators, 1e-12, None),
    )
    proxy = np.where(denominators > 0.0, proxy, -1.0)
    logits = proxy / max(float(tau), 1e-6)
    logits = logits - float(np.nanmax(logits))
    oracle = np.exp(logits)
    return oracle / max(float(oracle.sum()), 1e-12)


def _candidate_family_ccc_proxy_scores(example: Any) -> np.ndarray:
    """Return one scalar family CCC proxy per candidate."""
    features = np.asarray(example.candidate_numeric_features, dtype=np.float64)
    if features.shape[1] >= 33:
        return np.where(np.isfinite(features[:, -4]), features[:, -4], -1.0)
    family_scores = _candidate_family_ccc_matrix(example)
    proxy = np.nanmean(family_scores, axis=1)
    return np.where(np.isfinite(proxy), proxy, -1.0)


def _candidate_family_ccc_matrix(example: Any) -> np.ndarray:
    """Return candidate-by-family CCC scores for one prepared example."""
    channel = example.channels.get("chemical_shifts")
    candidate_count = len(example.teacher_weights)
    family_names = ["HN", "N", "CA", "CB", "C'"]
    scores = np.full((candidate_count, len(family_names)), np.nan, dtype=np.float64)
    if channel is None:
        return scores
    values = np.asarray(channel.values, dtype=np.float64)
    mask = np.asarray(channel.mask, dtype=bool)
    targets = np.asarray(channel.target_values, dtype=np.float64)
    target_families = np.asarray(
        [_atom_family_from_target_id(target_id) for target_id in channel.target_ids],
        dtype=object,
    )
    for candidate_index in range(candidate_count):
        valid = mask[:, candidate_index]
        for family_index, family_name in enumerate(family_names):
            family_valid = valid & (target_families == family_name)
            if np.count_nonzero(family_valid) < 2:
                continue
            score = _safe_ccc_numpy(
                values[family_valid, candidate_index],
                targets[family_valid],
            )
            if math.isfinite(score):
                scores[candidate_index, family_index] = score
    return scores


def _safe_ccc_numpy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Lin's CCC for numpy arrays with NaN fallback."""
    if predictions.size < 2:
        return float("nan")
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_var = float(np.mean(np.square(predictions - pred_mean)))
    target_var = float(np.mean(np.square(targets - target_mean)))
    covariance = float(np.mean((predictions - pred_mean) * (targets - target_mean)))
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    if denominator <= 1e-12:
        return float("nan")
    return float((2.0 * covariance) / denominator)


def _affine_calibrated_predictions_numpy(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    calibration_mask: np.ndarray,
) -> np.ndarray:
    """Return predictions after evidence-only affine calibration."""
    mask = np.asarray(calibration_mask, dtype=bool)
    values = np.asarray(predictions, dtype=float)
    if np.count_nonzero(mask) < 2:
        return values
    x = values[mask]
    y = np.asarray(targets, dtype=float)[mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denominator = float(np.sum(np.square(x - x_mean)))
    if denominator <= 1e-12:
        return values
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    intercept = y_mean - slope * x_mean
    return slope * values + intercept


def _tensor_norm(value: torch.Tensor) -> float:
    """Return a finite tensor norm as JSON-safe float."""
    return _json_float(float(torch.linalg.vector_norm(value).detach().cpu()))


def _tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Return cosine similarity between two flattened tensors."""
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.detach().cpu()) <= 1e-20:
        return None
    value = torch.dot(left.flatten(), right.flatten()) / denominator
    return _json_float(float(value.detach().cpu()))


def _distribution_summary(values: list[float | None]) -> dict[str, float | None]:
    """Return compact finite-value distribution summaries."""
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(value)
        ],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": _json_float(float(np.mean(finite))),
        "median": _json_float(float(np.median(finite))),
        "p10": _json_float(float(np.quantile(finite, 0.1))),
        "p90": _json_float(float(np.quantile(finite, 0.9))),
    }


def _json_float(value: Any) -> float | None:
    """Convert one value to a finite JSON float or ``None``."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def resolve_device(device_name: str) -> torch.device:
    """Resolve one configured training device string."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: StudentDensityModel,
    optimizer: torch.optim.Optimizer,
    source_vocab: dict[str, int],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Write one training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "source_vocab": source_vocab,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def save_json(path: Path, payload: Any) -> None:
    """Write one JSON file with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one epoch payload to a full-history JSONL stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _collect_forward_residual_rows_from_epoch(
    examples: list[Any],
    model: StudentDensityModel,
    device: torch.device,
    config: StudentTrainingConfig,
    split_name: str,
) -> list[dict[str, Any]]:
    """Collect candidate-level forward residual predictions for reports."""
    if not (
        config.enable_forward_residual_head or config.enable_residue_atom_residual_head
    ):
        return []
    rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for example in examples:
            tensors = to_tensors(example, device)
            outputs = model_outputs_for_example(
                model=model,
                tensors=tensors,
                evidence_mask=None,
                config=config,
            )
            residual_mu = outputs["forward_residual_mu"].detach().cpu().numpy()
            residual_sigma = outputs["forward_residual_sigma"].detach().cpu().numpy()
            weights = outputs["weights"].detach().cpu().numpy()
            channel = tensors["channels"].get("chemical_shifts")
            if channel is not None and residual_mu.shape == tuple(channel.values.shape):
                candidate_order = np.argsort(-weights)[
                    : max(int(config.residue_atom_artifact_top_k), 1)
                ]
                target_ids = list(example.channels["chemical_shifts"].target_ids)
                for target_index, target_id in enumerate(target_ids):
                    parsed = parse_chemical_shift_target_id(target_id)
                    for candidate_index in candidate_order:
                        candidate_id = (
                            example.candidate_ids[int(candidate_index)]
                            if int(candidate_index) < len(example.candidate_ids)
                            else str(int(candidate_index))
                        )
                        rows.append(
                            {
                                "entity_uid": example.entity_uid,
                                "bmrb_id": example.metadata.get("bmrb_id", ""),
                                "split": split_name,
                                "candidate_index": int(candidate_index),
                                "candidate_id": candidate_id,
                                "target_index": int(target_index),
                                "target_id": target_id,
                                "chain_id": parsed["chain_id"],
                                "residue_index": int(parsed["residue_index"]),
                                "residue_name": parsed["residue_name"],
                                "atom_family": parsed["atom_family"],
                                "posterior_weight": float(
                                    weights[int(candidate_index)]
                                ),
                                "residual_mu": float(
                                    residual_mu[target_index, int(candidate_index)]
                                ),
                                "residual_sigma": float(
                                    residual_sigma[target_index, int(candidate_index)]
                                ),
                            }
                        )
                continue
            for candidate_index in range(residual_mu.shape[0]):
                candidate_id = (
                    example.candidate_ids[candidate_index]
                    if candidate_index < len(example.candidate_ids)
                    else str(candidate_index)
                )
                for family_index, family_name in enumerate(CHEMICAL_SHIFT_FAMILIES):
                    rows.append(
                        {
                            "entity_uid": example.entity_uid,
                            "bmrb_id": example.metadata.get("bmrb_id", ""),
                            "split": split_name,
                            "candidate_index": int(candidate_index),
                            "candidate_id": candidate_id,
                            "atom_family": family_name,
                            "posterior_weight": float(weights[candidate_index]),
                            "residual_mu": float(
                                residual_mu[candidate_index, family_index]
                            ),
                            "residual_sigma": float(
                                residual_sigma[candidate_index, family_index]
                            ),
                        }
                    )
    if was_training:
        model.train()
    return rows


def write_forward_residual_predictions(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    """Write final forward residual predictions as an optional Parquet artifact."""
    arrays_dir = output_dir / "reports" / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    rows = [*train_rows, *val_rows]
    frame = pd.DataFrame(rows)
    if "target_id" in frame.columns:
        frame.to_parquet(
            arrays_dir / "residue_atom_forward_residuals.parquet",
            index=False,
        )
        render_residue_atom_residual_heatmap(frame, output_dir / "reports" / "figures")
    else:
        frame.to_parquet(
            arrays_dir / "forward_residual_predictions.parquet",
            index=False,
        )


def build_physics_shift_artifacts(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    support_ceiling_report: dict[str, Any],
) -> dict[str, Any]:
    """Write secondary-shift arrays and raw-vs-secondary CCC summaries."""
    arrays_dir = output_dir / "reports" / "arrays"
    metrics_dir = output_dir / "reports" / "metrics"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([*train_rows, *val_rows])
    if frame.empty or "target_id" not in frame.columns:
        empty_report = {
            "status": "missing_prediction_rows",
            "support_ceiling": dict(support_ceiling_report),
        }
        pd.DataFrame().to_parquet(arrays_dir / "secondary_shift_targets.parquet")
        pd.DataFrame().to_parquet(arrays_dir / "secondary_shift_posteriors.parquet")
        save_json(metrics_dir / "raw_vs_secondary_ccc_report.json", empty_report)
        return empty_report
    frame = frame.copy()
    frame["random_coil_reference"] = frame["target_id"].map(
        random_coil_reference_for_target_id
    )
    frame["predicted_secondary_shift"] = (
        frame["predicted_value"].astype(float) - frame["random_coil_reference"]
    )
    frame["target_secondary_shift"] = (
        frame["target_value"].astype(float) - frame["random_coil_reference"]
    )
    frame.to_parquet(arrays_dir / "secondary_shift_posteriors.parquet", index=False)
    target_columns = [
        column
        for column in [
            "entity_uid",
            "bmrb_id",
            "split",
            "target_id",
            "atom_family",
            "target_value",
            "random_coil_reference",
            "target_secondary_shift",
        ]
        if column in frame.columns
    ]
    frame[target_columns].drop_duplicates().to_parquet(
        arrays_dir / "secondary_shift_targets.parquet",
        index=False,
    )
    report = {
        "status": "ok",
        "rows": int(len(frame)),
        "support_ceiling": dict(support_ceiling_report),
        "split_family_metrics": _raw_vs_secondary_ccc_summary(frame),
    }
    save_json(metrics_dir / "raw_vs_secondary_ccc_report.json", report)
    return report


def _raw_vs_secondary_ccc_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return raw and secondary CCC summaries by split and atom family."""
    rows: list[dict[str, Any]] = []
    for (split, family), subset in frame.groupby(["split", "atom_family"], dropna=True):
        if len(subset) < 2:
            continue
        raw_ccc = _safe_ccc_numpy(
            subset["predicted_value"].to_numpy(dtype=float),
            subset["target_value"].to_numpy(dtype=float),
        )
        secondary_ccc = _safe_ccc_numpy(
            subset["predicted_secondary_shift"].to_numpy(dtype=float),
            subset["target_secondary_shift"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "split": str(split),
                "atom_family": str(family),
                "rows": int(len(subset)),
                "raw_ccc": _json_float(raw_ccc),
                "secondary_ccc": _json_float(secondary_ccc),
            }
        )
    for split, subset in frame.groupby("split", dropna=True):
        if len(subset) < 2:
            continue
        rows.append(
            {
                "split": str(split),
                "atom_family": "ALL_POOLED_SANITY",
                "rows": int(len(subset)),
                "raw_ccc": _json_float(
                    _safe_ccc_numpy(
                        subset["predicted_value"].to_numpy(dtype=float),
                        subset["target_value"].to_numpy(dtype=float),
                    )
                ),
                "secondary_ccc": _json_float(
                    _safe_ccc_numpy(
                        subset["predicted_secondary_shift"].to_numpy(dtype=float),
                        subset["target_secondary_shift"].to_numpy(dtype=float),
                    )
                ),
            }
        )
    return rows


def render_residue_atom_residual_heatmap(
    frame: pd.DataFrame, figures_dir: Path
) -> None:
    """Render a compact heatmap of residue-atom residuals for the first entity."""
    if frame.empty or "target_id" not in frame.columns:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    first_entity = str(frame["entity_uid"].iloc[0])
    subset = frame.loc[frame["entity_uid"].astype(str) == first_entity].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(
        index="target_index",
        columns="candidate_index",
        values="residual_mu",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    axis.set_title(f"Residue-atom residuals: {first_entity}")
    axis.set_xlabel("Top posterior candidate")
    axis.set_ylabel("Chemical-shift target index")
    figure.colorbar(image, ax=axis, label="residual ppm")
    figure.tight_layout()
    figure.savefig(figures_dir / "residue_atom_residual_heatmap.png", dpi=200)
    plt.close(figure)


def history_window(
    history: list[dict[str, Any]],
    config: StudentTrainingConfig,
) -> list[dict[str, Any]]:
    """Return the live JSON history window for monitor-friendly long runs."""
    if config.history_window_epochs is None or config.history_window_epochs <= 0:
        return history
    return history[-config.history_window_epochs :]


def learning_rate_for_epoch(config: StudentTrainingConfig, epoch: int) -> float:
    """Return the configured learning rate for one 1-based epoch."""
    schedule = config.learning_rate_schedule.lower()
    base_lr = float(config.learning_rate)
    min_lr = min(max(float(config.min_learning_rate), 0.0), base_lr)
    warmup_epochs = max(int(config.warmup_epochs), 0)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(warmup_epochs)
    if schedule == "constant":
        return base_lr
    progress_denominator = max(config.epochs - warmup_epochs, 1)
    progress = min(
        max(float(epoch - warmup_epochs) / float(progress_denominator), 0.0),
        1.0,
    )
    if schedule == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine
    raise ValueError(
        f"Unsupported learning_rate_schedule: {config.learning_rate_schedule}"
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Set all optimizer parameter groups to one learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def append_learning_rate_metric(
    epoch_payload: dict[str, Any],
    learning_rate: float,
) -> None:
    """Expose learning rate as a monitor-friendly scalar metric."""
    epoch_payload.setdefault("loss_summary", {})["learning_rate"] = float(learning_rate)
    epoch_payload.setdefault("loss_rows", []).append(
        {
            "metric_name": "learning_rate",
            "split": epoch_payload.get("split", ""),
            "aggregation": "macro",
            "value": float(learning_rate),
            "eligible_examples": 0,
            "eligible_measurements": 0,
            "tier": "optimizer",
        }
    )


def _empty_epoch_payload(split_name: str) -> dict[str, Any]:
    """Return one empty epoch payload for missing validation examples."""
    return {
        "loss_summary": {},
        "loss_rows": [],
        "metric_rows": [],
        "metric_summary": {},
        "eligible_counts": {},
        "prediction_rows": [],
        "split": split_name,
    }
