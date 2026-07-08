"""Catalog of frozen protein embedding models for AtypEmu feature builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SequenceEmbeddingModelSpec:
    """Static decision record for a residue/context embedding model."""

    provider: str
    model_name: str
    feature_scope: str
    runtime_tier: str
    resource_fit: str
    utility_score: float
    recommended_role: str
    implementation_status: str
    problem_md_coverage: tuple[str, ...]
    main_risk: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["problem_md_coverage"] = list(self.problem_md_coverage)
        return payload


SEQUENCE_EMBEDDING_MODEL_CATALOG: tuple[SequenceEmbeddingModelSpec, ...] = (
    SequenceEmbeddingModelSpec(
        provider="learned",
        model_name="token_embedding",
        feature_scope="residue_identity_baseline",
        runtime_tier="native",
        resource_fit="excellent",
        utility_score=0.25,
        recommended_role="negative_control",
        implementation_status="implemented",
        problem_md_coverage=(
            "baseline_for_handcrafted_feature_ablation",
            "does_not_reduce_index_local_context_risk",
        ),
        main_risk="too little context beyond residue identity",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm2",
        model_name="esm2_t6_8M_UR50D",
        feature_scope="cheap_residue_language_context",
        runtime_tier="online_smoke",
        resource_fit="excellent",
        utility_score=0.55,
        recommended_role="smoke_or_cache_warmup",
        implementation_status="implemented",
        problem_md_coverage=(
            "replaces_some_onehot_local_context",
            "low_cost_runtime_probe",
        ),
        main_risk="too small for difficult HN/N/Cprime local context",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm2",
        model_name="esm2_t12_35M_UR50D",
        feature_scope="fast_residue_language_context",
        runtime_tier="online_triage",
        resource_fit="very_good",
        utility_score=0.68,
        recommended_role="fast_ablation",
        implementation_status="implemented",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "handcrafted_feature_downweight_control",
            "python311_compatible_plm_context",
        ),
        main_risk="may underfit long-range residue context",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm2",
        model_name="esm2_t30_150M_UR50D",
        feature_scope="balanced_residue_language_context",
        runtime_tier="online_cached_triage",
        resource_fit="good",
        utility_score=0.82,
        recommended_role="default_current_plm_feature",
        implementation_status="implemented",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "reduces_handcrafted_static_context_dependency",
            "reduces_index_local_context_dependency",
            "keeps_residue_local_conditioning_field",
        ),
        main_risk="sequence-only context cannot prove generated conformer quality",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm2",
        model_name="esm2_t33_650M_UR50D",
        feature_scope="richer_residue_language_context",
        runtime_tier="cached_offline_ablation",
        resource_fit="moderate",
        utility_score=0.76,
        recommended_role="second_pass_cached_ablation",
        implementation_status="implemented_provider_not_default",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "richer_sequence_context_than_t30",
            "handcrafted_feature_downweight_candidate",
        ),
        main_risk="higher memory and embedding latency for limited active-subset gain",
    ),
    SequenceEmbeddingModelSpec(
        provider="esmc",
        model_name="biohub/ESMC-600M",
        feature_scope="modern_residue_language_context",
        runtime_tier="offline_precompute_only",
        resource_fit="conditional",
        utility_score=0.74,
        recommended_role="external_npz_feature_table_ablation_after_esm2_t30",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "richer_sequence_context_than_esm2_t30",
            "modern_plm_context_ablation",
        ),
        main_risk="new dependency and cache contract before the support-prior bottleneck is fixed",
    ),
    SequenceEmbeddingModelSpec(
        provider="esmc",
        model_name="biohub/ESMC-6B",
        feature_scope="large_modern_residue_language_context",
        runtime_tier="offline_precompute_only",
        resource_fit="poor_for_online",
        utility_score=0.69,
        recommended_role="late_high_capacity_ablation_only",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "long_range_sequence_context",
            "modern_plm_context_ablation",
        ),
        main_risk="too expensive for online training and unlikely to fix diffuse generated-prior mass alone",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm3",
        model_name="esm3_sm_open_v1",
        feature_scope="modern_sequence_context",
        runtime_tier="runtime_gated",
        resource_fit="conditional",
        utility_score=0.70,
        recommended_role="use_only_when_runtime_container_supports_esm3",
        implementation_status="implemented_runtime_fragile",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "plm_context_probe",
            "handcrafted_feature_downweight_candidate",
        ),
        main_risk="current Python/runtime package mismatch can silently become fallback",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm_msa",
        model_name="esm_msa1b_t12_100M_UR50S",
        feature_scope="msa_context",
        runtime_tier="offline_precompute_only",
        resource_fit="conditional",
        utility_score=0.58,
        recommended_role="only_if_reliable_msa_pipeline_exists",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "long_range_sequence_context",
            "evolutionary_context_ablation",
        ),
        main_risk="requires MSA provenance and can add dataset availability bias",
    ),
    SequenceEmbeddingModelSpec(
        provider="prot_t5",
        model_name="Rostlab/prot_t5_xl_uniref50",
        feature_scope="large_sequence_language_context",
        runtime_tier="offline_precompute_only",
        resource_fit="poor_for_online",
        utility_score=0.66,
        recommended_role="external_npz_feature_table_ablation",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "sequence_context_ablation_against_esm2",
            "handcrafted_feature_downweight_candidate",
        ),
        main_risk="large runtime cost and no current online provider",
    ),
    SequenceEmbeddingModelSpec(
        provider="ankh",
        model_name="ankh-base-or-large",
        feature_scope="sequence_language_context",
        runtime_tier="offline_precompute_only",
        resource_fit="moderate",
        utility_score=0.60,
        recommended_role="external_npz_feature_table_ablation",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "handcrafted_feature_dependency",
            "sequence_context_ablation_against_esm2",
            "handcrafted_feature_downweight_candidate",
        ),
        main_risk="extra dependency surface without a proven AtypEmu gain",
    ),
    SequenceEmbeddingModelSpec(
        provider="esm_if1",
        model_name="esm_if1_gvp4_t16_142M_UR50",
        feature_scope="backbone_structure_context",
        runtime_tier="offline_structure_conditioned_ablation",
        resource_fit="conditional",
        utility_score=0.64,
        recommended_role="backbone-conditioned_control_for_local_environment_features",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "local_3d_context_beyond_index_features",
            "index_local_context_replacement_candidate",
            "structure_context_ablation",
        ),
        main_risk="uses generated/backbone structure context and must be audited for support artifact leakage",
    ),
    SequenceEmbeddingModelSpec(
        provider="saprot",
        model_name="SaProt_sequence_structure",
        feature_scope="sequence_plus_structure_context",
        runtime_tier="offline_structure_conditioned_ablation",
        resource_fit="conditional",
        utility_score=0.72,
        recommended_role="later_local_environment_ablation",
        implementation_status="catalog_only",
        problem_md_coverage=(
            "local_3d_context_beyond_index_features",
            "hbond_ring_electrostatic_proxy_replacement_candidate",
        ),
        main_risk="requires reliable structure tokens and can leak support artifacts",
    ),
)


def sequence_embedding_model_catalog() -> tuple[SequenceEmbeddingModelSpec, ...]:
    """Return the immutable embedding model decision catalog."""

    return SEQUENCE_EMBEDDING_MODEL_CATALOG


def sequence_embedding_model_catalog_rows() -> list[dict[str, Any]]:
    """Return catalog rows suitable for reports, JSON, or contract tests."""

    return [spec.as_dict() for spec in SEQUENCE_EMBEDDING_MODEL_CATALOG]


def recommended_online_sequence_embedding_model() -> SequenceEmbeddingModelSpec:
    """Return the current online-safe default for iremb-c-08/l40sq triage."""

    implemented = [
        spec
        for spec in SEQUENCE_EMBEDDING_MODEL_CATALOG
        if spec.implementation_status.startswith("implemented")
        and spec.runtime_tier in {"online_triage", "online_cached_triage"}
    ]
    return max(implemented, key=lambda spec: spec.utility_score)


def embedding_models_by_problem_axis(axis: str) -> list[SequenceEmbeddingModelSpec]:
    """Rank embedding candidates that explicitly cover a problem.md axis."""

    needle = str(axis).strip().lower()
    matches = [
        spec
        for spec in SEQUENCE_EMBEDDING_MODEL_CATALOG
        if any(needle in item.lower() for item in spec.problem_md_coverage)
    ]
    return sorted(matches, key=lambda spec: spec.utility_score, reverse=True)
