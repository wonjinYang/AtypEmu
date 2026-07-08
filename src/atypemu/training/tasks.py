"""Training-task contracts for the staged AtypEmu redesign."""

from __future__ import annotations

from dataclasses import dataclass


PRIOR_SEQUENCE_ONLY = "prior_sequence_only"
POSTERIOR_NMR_CONDITIONED = "posterior_nmr_conditioned"
POSTERIOR_NMR_CONDITIONED_V2 = "posterior_nmr_conditioned_v2"
POSTERIOR_NMR_RESIDUE_ATOM_V1 = "posterior_nmr_residue_atom_v1"
POSTERIOR_NMR_PHYSICS_V1 = "posterior_nmr_physics_v1"
POSTERIOR_NMR_CCC_GEOMETRY_V2 = "posterior_nmr_ccc_geometry_v2"
POSTERIOR_NMR_MOMENT_DIFFUSION_V1 = "posterior_nmr_moment_diffusion_v1"
POSTERIOR_NMR_MOMENT_DIFFUSION_V2 = "posterior_nmr_moment_diffusion_v2"
POSTERIOR_NMR_MEASURE_AWARE_V3 = "posterior_nmr_measure_aware_v3"
POSTERIOR_NMR_CANDIDATE_FREE_V1 = "posterior_nmr_candidate_free_v1"
POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1 = "posterior_nmr_candidate_free_hn95_v1"
POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1 = "posterior_nmr_bioemu_latent_atlas_v1"
POSTERIOR_NMR_X0_ENSEMBLE_V1 = "posterior_nmr_x0_ensemble_v1"
FORWARD_OBSERVABLE = "forward_observable"

LEGACY_CS_FIT = "legacy_cs_fit"
TARGET_CONDITIONED_CS_FIT = "target_conditioned_cs_fit"
TARGET_CONDITIONED_CCC_FIT = "target_conditioned_ccc_fit"
BIOEMU_LATENT_NMR_V1 = "bioemu_latent_nmr_v1"

TRAINING_TASKS = {
    PRIOR_SEQUENCE_ONLY,
    POSTERIOR_NMR_CONDITIONED,
    POSTERIOR_NMR_CONDITIONED_V2,
    POSTERIOR_NMR_RESIDUE_ATOM_V1,
    POSTERIOR_NMR_PHYSICS_V1,
    POSTERIOR_NMR_CCC_GEOMETRY_V2,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    POSTERIOR_NMR_MEASURE_AWARE_V3,
    POSTERIOR_NMR_CANDIDATE_FREE_V1,
    POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
    POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
    POSTERIOR_NMR_X0_ENSEMBLE_V1,
    FORWARD_OBSERVABLE,
}
LEGACY_POSTERIOR_MODES = {
    LEGACY_CS_FIT,
    TARGET_CONDITIONED_CS_FIT,
    TARGET_CONDITIONED_CCC_FIT,
}
TARGET_DERIVED_MODES = {
    TARGET_CONDITIONED_CS_FIT,
    TARGET_CONDITIONED_CCC_FIT,
    POSTERIOR_NMR_CONDITIONED,
    POSTERIOR_NMR_CONDITIONED_V2,
    POSTERIOR_NMR_RESIDUE_ATOM_V1,
    POSTERIOR_NMR_PHYSICS_V1,
    POSTERIOR_NMR_CCC_GEOMETRY_V2,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
    POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
    POSTERIOR_NMR_MEASURE_AWARE_V3,
    POSTERIOR_NMR_CANDIDATE_FREE_V1,
    POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
    POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
    POSTERIOR_NMR_X0_ENSEMBLE_V1,
    BIOEMU_LATENT_NMR_V1,
}


@dataclass(frozen=True, slots=True)
class TrainingTaskSpec:
    """Resolved public training-task contract."""

    task_name: str
    input_feature_mode: str
    scientific_claim: str
    allows_target_conditioning: bool
    is_legacy_alias: bool = False

    @property
    def is_prior(self) -> bool:
        """Return whether this task is the sequence-only ensemble prior."""
        return self.task_name == PRIOR_SEQUENCE_ONLY

    @property
    def is_posterior(self) -> bool:
        """Return whether this task is NMR-evidence-conditioned inference."""
        return self.task_name in {
            POSTERIOR_NMR_CONDITIONED,
            POSTERIOR_NMR_CONDITIONED_V2,
            POSTERIOR_NMR_RESIDUE_ATOM_V1,
            POSTERIOR_NMR_PHYSICS_V1,
            POSTERIOR_NMR_CCC_GEOMETRY_V2,
            POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
            POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            POSTERIOR_NMR_MEASURE_AWARE_V3,
            POSTERIOR_NMR_CANDIDATE_FREE_V1,
            POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
            POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
            POSTERIOR_NMR_X0_ENSEMBLE_V1,
        }

    @property
    def is_forward(self) -> bool:
        """Return whether this task is observable forward-head training."""
        return self.task_name == FORWARD_OBSERVABLE

    def as_dict(self) -> dict[str, object]:
        """Serialize the task contract into run metadata."""
        return {
            "task_name": self.task_name,
            "input_feature_mode": self.input_feature_mode,
            "scientific_claim": self.scientific_claim,
            "allows_target_conditioning": self.allows_target_conditioning,
            "is_legacy_alias": self.is_legacy_alias,
        }


def resolve_training_task(
    training_task: str | None,
    training_input_mode: str,
) -> TrainingTaskSpec:
    """Resolve legacy input modes into the explicit staged task contract.

    Args:
        training_task: Optional explicit public task name.
        training_input_mode: Legacy or canonical feature mode from config.

    Returns:
        Resolved task specification.

    Raises:
        ValueError: If the task/mode combination would leak target-derived
            information into the sequence-only prior.
    """
    mode = str(training_input_mode or LEGACY_CS_FIT)
    task = None if training_task is None else str(training_task)
    if task is not None and task not in TRAINING_TASKS:
        allowed = ", ".join(sorted(TRAINING_TASKS))
        raise ValueError(f"Unsupported training_task={task!r}; use {allowed}.")

    if task == PRIOR_SEQUENCE_ONLY:
        if mode in TARGET_DERIVED_MODES:
            raise ValueError(
                "prior_sequence_only forbids target-derived training_input_mode="
                f"{mode!r}."
            )
        return TrainingTaskSpec(
            task_name=PRIOR_SEQUENCE_ONLY,
            input_feature_mode=PRIOR_SEQUENCE_ONLY,
            scientific_claim="sequence-conditioned ensemble prior",
            allows_target_conditioning=False,
        )

    if task == POSTERIOR_NMR_CONDITIONED:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CONDITIONED,
            input_feature_mode=POSTERIOR_NMR_CONDITIONED,
            scientific_claim="NMR-guided ensemble refinement",
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_CONDITIONED_V2:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CONDITIONED_V2,
            input_feature_mode=POSTERIOR_NMR_CONDITIONED_V2,
            scientific_claim="CCC-first NMR-guided ensemble refinement",
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_RESIDUE_ATOM_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_RESIDUE_ATOM_V1,
            input_feature_mode=POSTERIOR_NMR_RESIDUE_ATOM_V1,
            scientific_claim="residue-atom NMR posterior reconstruction",
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_PHYSICS_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_PHYSICS_V1,
            input_feature_mode=POSTERIOR_NMR_PHYSICS_V1,
            scientific_claim="physics-aware secondary-shift NMR posterior",
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_CCC_GEOMETRY_V2:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CCC_GEOMETRY_V2,
            input_feature_mode=POSTERIOR_NMR_RESIDUE_ATOM_V1,
            scientific_claim="CCC-geometry NMR posterior reconstruction",
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_MOMENT_DIFFUSION_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
            input_feature_mode=POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
            scientific_claim=(
                "NMR-conditioned fast posterior moment reconstruction with "
                "BioEmu-style ensemble diagnostics"
            ),
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_MOMENT_DIFFUSION_V2:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            input_feature_mode=POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            scientific_claim=(
                "NMR-physics-guided fast posterior moment reconstruction with "
                "family-specific adapters and BioEmu-style ensemble diagnostics"
            ),
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_MEASURE_AWARE_V3:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_MEASURE_AWARE_V3,
            input_feature_mode=POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            scientific_claim=(
                "measure-aware NMR-conditioned fast posterior moment "
                "reconstruction with entry/family calibration diagnostics"
            ),
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_CANDIDATE_FREE_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CANDIDATE_FREE_V1,
            input_feature_mode=POSTERIOR_NMR_CANDIDATE_FREE_V1,
            scientific_claim=(
                "candidate-free NMR-conditioned posterior moment reconstruction"
            ),
            allows_target_conditioning=True,
        )
    if task == POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
            input_feature_mode=POSTERIOR_NMR_CANDIDATE_FREE_V1,
            scientific_claim=(
                "candidate-free NMR-conditioned HN/C' 0.95 posterior reconstruction"
            ),
            allows_target_conditioning=True,
        )
    if task == POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
            input_feature_mode=BIOEMU_LATENT_NMR_V1,
            scientific_claim=(
                "BioEmu latent ensemble atlas NMR posterior mean reconstruction"
            ),
            allows_target_conditioning=True,
        )

    if task == POSTERIOR_NMR_X0_ENSEMBLE_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_X0_ENSEMBLE_V1,
            input_feature_mode=BIOEMU_LATENT_NMR_V1,
            scientific_claim=(
                "BioEmu x0-support posterior ensemble with conformer-level "
                "chemical-shift decoder"
            ),
            allows_target_conditioning=True,
        )

    if task == FORWARD_OBSERVABLE:
        return TrainingTaskSpec(
            task_name=FORWARD_OBSERVABLE,
            input_feature_mode=PRIOR_SEQUENCE_ONLY,
            scientific_claim="observable back-calculation model",
            allows_target_conditioning=False,
        )

    if mode == PRIOR_SEQUENCE_ONLY:
        return TrainingTaskSpec(
            task_name=PRIOR_SEQUENCE_ONLY,
            input_feature_mode=PRIOR_SEQUENCE_ONLY,
            scientific_claim="sequence-conditioned ensemble prior",
            allows_target_conditioning=False,
        )
    if mode == POSTERIOR_NMR_CONDITIONED:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CONDITIONED,
            input_feature_mode=POSTERIOR_NMR_CONDITIONED,
            scientific_claim="NMR-guided ensemble refinement",
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_CONDITIONED_V2:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CONDITIONED_V2,
            input_feature_mode=POSTERIOR_NMR_CONDITIONED_V2,
            scientific_claim="CCC-first NMR-guided ensemble refinement",
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_RESIDUE_ATOM_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_RESIDUE_ATOM_V1,
            input_feature_mode=POSTERIOR_NMR_RESIDUE_ATOM_V1,
            scientific_claim="residue-atom NMR posterior reconstruction",
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_PHYSICS_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_PHYSICS_V1,
            input_feature_mode=POSTERIOR_NMR_PHYSICS_V1,
            scientific_claim="physics-aware secondary-shift NMR posterior",
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_CCC_GEOMETRY_V2:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CCC_GEOMETRY_V2,
            input_feature_mode=POSTERIOR_NMR_RESIDUE_ATOM_V1,
            scientific_claim="CCC-geometry NMR posterior reconstruction",
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_MOMENT_DIFFUSION_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
            input_feature_mode=POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
            scientific_claim=(
                "NMR-conditioned fast posterior moment reconstruction with "
                "BioEmu-style ensemble diagnostics"
            ),
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_MOMENT_DIFFUSION_V2:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            input_feature_mode=POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            scientific_claim=(
                "NMR-physics-guided fast posterior moment reconstruction with "
                "family-specific adapters and BioEmu-style ensemble diagnostics"
            ),
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_MEASURE_AWARE_V3:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_MEASURE_AWARE_V3,
            input_feature_mode=POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            scientific_claim=(
                "measure-aware NMR-conditioned fast posterior moment "
                "reconstruction with entry/family calibration diagnostics"
            ),
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_CANDIDATE_FREE_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CANDIDATE_FREE_V1,
            input_feature_mode=POSTERIOR_NMR_CANDIDATE_FREE_V1,
            scientific_claim=(
                "candidate-free NMR-conditioned posterior moment reconstruction"
            ),
            allows_target_conditioning=True,
        )
    if mode == POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
            input_feature_mode=POSTERIOR_NMR_CANDIDATE_FREE_V1,
            scientific_claim=(
                "candidate-free NMR-conditioned HN/C' 0.95 posterior reconstruction"
            ),
            allows_target_conditioning=True,
        )
    if mode in {POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1, BIOEMU_LATENT_NMR_V1}:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
            input_feature_mode=BIOEMU_LATENT_NMR_V1,
            scientific_claim=(
                "BioEmu latent ensemble atlas NMR posterior mean reconstruction"
            ),
            allows_target_conditioning=True,
        )
    if mode in LEGACY_POSTERIOR_MODES:
        return TrainingTaskSpec(
            task_name=POSTERIOR_NMR_CONDITIONED,
            input_feature_mode=mode,
            scientific_claim="legacy NMR-conditioned candidate reweighting",
            allows_target_conditioning=True,
            is_legacy_alias=True,
        )

    allowed_modes = sorted(
        {
            PRIOR_SEQUENCE_ONLY,
            POSTERIOR_NMR_CONDITIONED,
            POSTERIOR_NMR_CONDITIONED_V2,
            POSTERIOR_NMR_RESIDUE_ATOM_V1,
            POSTERIOR_NMR_PHYSICS_V1,
            POSTERIOR_NMR_CCC_GEOMETRY_V2,
            POSTERIOR_NMR_MOMENT_DIFFUSION_V1,
            POSTERIOR_NMR_MOMENT_DIFFUSION_V2,
            POSTERIOR_NMR_MEASURE_AWARE_V3,
            POSTERIOR_NMR_CANDIDATE_FREE_V1,
            POSTERIOR_NMR_CANDIDATE_FREE_HN95_V1,
            POSTERIOR_NMR_BIOEMU_LATENT_ATLAS_V1,
            BIOEMU_LATENT_NMR_V1,
            *LEGACY_POSTERIOR_MODES,
        }
    )
    raise ValueError(
        f"Unsupported training_input_mode={mode!r}; use {', '.join(allowed_modes)}."
    )
