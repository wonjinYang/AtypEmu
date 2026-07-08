"""Teacher materialization and student-training helpers for AtypEmu."""

from atypemu.training.config import (
    BenchmarkRenderConfig,
    StudentTrainingConfig,
    TeacherMaterializationConfig,
    TrainingReadinessConfig,
    UCBShiftGenerationConfig,
)
from atypemu.training.bioemu_score_adapter_lora import (
    BioEmuScoreAdapterLoRAConfig,
    BioEmuScoreAdapterLoRACheckpoint,
    BioEmuScoreAdapterLoRATrainer,
    load_checkpoint as load_bioemu_score_adapter_lora_checkpoint,
    save_checkpoint as save_bioemu_score_adapter_lora_checkpoint,
    score_adapter_runtime_contract,
    structural_generator_metadata as bioemu_score_adapter_structural_generator_metadata,
)
from atypemu.training.chemical_shift_predictor import (
    ChemicalShiftPredictorConfig,
    FamilyConditionedChemicalShiftPredictor,
    load_predictor_bundle as load_chemical_shift_predictor_bundle,
    save_predictor_bundle as save_chemical_shift_predictor_bundle,
    train_and_evaluate_chemical_shift_predictor,
    train_family_conditioned_chemical_shift_predictor,
)
from atypemu.training.chemical_shift_shared_q import (
    ChemicalShiftSharedQConfig,
    ChemicalShiftSharedQSweepConfig,
    infer_chemical_shift_shared_q_posterior,
    infer_shared_q_posterior_from_paths,
    sweep_chemical_shift_shared_q_posterior,
    sweep_shared_q_posterior_from_paths,
)
from atypemu.training.cs_reweighting_teacher_export import (
    export_cs_reweighting_teacher_for_bioemu,
)
from atypemu.training.materialize import materialize_teacher_examples
from atypemu.training.tasks import (
    BIOEMU_LATENT_NMR_V1,
    FORWARD_OBSERVABLE,
    POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
    POSTERIOR_NMR_CONDITIONED,
    POSTERIOR_NMR_CONDITIONED_V2,
    POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
    POSTERIOR_NMR_CANDIDATE_FREE_V1,
    POSTERIOR_NMR_MEASURE_AWARE_V3,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    POSTERIOR_NMR_PHYSICS_V1,
    POSTERIOR_NMR_RESIDUE_ATOM_V1,
    POSTERIOR_NMR_X0_ENSEMBLE_V1,
    PRIOR_SEQUENCE_ONLY,
    TrainingTaskSpec,
    resolve_training_task,
)

__all__ = [
    "BenchmarkRenderConfig",
    "BioEmuScoreAdapterLoRAConfig",
    "BioEmuScoreAdapterLoRACheckpoint",
    "BioEmuScoreAdapterLoRATrainer",
    "ChemicalShiftPredictorConfig",
    "ChemicalShiftSharedQConfig",
    "ChemicalShiftSharedQSweepConfig",
    "FamilyConditionedChemicalShiftPredictor",
    "StudentTrainingConfig",
    "TeacherMaterializationConfig",
    "TrainingReadinessConfig",
    "TrainingTaskSpec",
    "UCBShiftGenerationConfig",
    "PRIOR_SEQUENCE_ONLY",
    "BIOEMU_LATENT_NMR_V1",
    "POSTERIOR_NMR_CONDITIONED",
    "POSTERIOR_NMR_CONDITIONED_V2",
    "POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1",
    "POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1",
    "POSTERIOR_NMR_CANDIDATE_FREE_V1",
    "POSTERIOR_NMR_RESIDUE_ATOM_V1",
    "POSTERIOR_NMR_PHYSICS_V1",
    "POSTERIOR_NMR_MOMENT_DIFFUSION_V1",
    "POSTERIOR_NMR_MOMENT_DIFFUSION_V2",
    "POSTERIOR_NMR_MEASURE_AWARE_V3",
    "POSTERIOR_NMR_X0_ENSEMBLE_V1",
    "FORWARD_OBSERVABLE",
    "materialize_teacher_examples",
    "export_cs_reweighting_teacher_for_bioemu",
    "bioemu_score_adapter_structural_generator_metadata",
    "load_chemical_shift_predictor_bundle",
    "load_bioemu_score_adapter_lora_checkpoint",
    "save_chemical_shift_predictor_bundle",
    "save_bioemu_score_adapter_lora_checkpoint",
    "score_adapter_runtime_contract",
    "infer_chemical_shift_shared_q_posterior",
    "infer_shared_q_posterior_from_paths",
    "sweep_chemical_shift_shared_q_posterior",
    "sweep_shared_q_posterior_from_paths",
    "train_and_evaluate_chemical_shift_predictor",
    "train_family_conditioned_chemical_shift_predictor",
    "audit_training_readiness",
    "resolve_training_task",
    "run_student_training",
]


def __getattr__(name: str):
    """Lazily expose torch-dependent training helpers."""
    if name == "audit_training_readiness":
        from atypemu.training.readiness import audit_training_readiness

        return audit_training_readiness
    if name == "run_student_training":
        from atypemu.training.trainer import run_student_training

        return run_student_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
