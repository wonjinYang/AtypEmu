"""Backward-compatible model aliases for AtypEmu staged training."""

from atypemu.training.models import (
    BayesianPosteriorHead,
    CandidateSetScorer,
    ConformerEncoder,
    FamilyAffineCalibrationLayer,
    MirrorDescentPosteriorRefiner,
    ObservableForwardHead,
    LatentPosteriorFlow,
    PosteriorMomentHead,
    PosteriorStateTokenHead,
    ResidueAtomResidualHead,
    StatePosteriorScorer,
    StudentDensityModel,
)

__all__ = [
    "BayesianPosteriorHead",
    "CandidateSetScorer",
    "ConformerEncoder",
    "FamilyAffineCalibrationLayer",
    "MirrorDescentPosteriorRefiner",
    "ObservableForwardHead",
    "LatentPosteriorFlow",
    "PosteriorMomentHead",
    "PosteriorStateTokenHead",
    "ResidueAtomResidualHead",
    "StatePosteriorScorer",
    "StudentDensityModel",
]
