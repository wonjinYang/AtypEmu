"""Core public dataclasses used across the AtypEmu offline core."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def canonical_atom_name(atom_id: str) -> str:
    """Return a canonical atom name for target matching.

    Args:
        atom_id: Raw atom identifier read from a file.

    Returns:
        A normalized atom identifier.
    """
    normalized = atom_id.strip().upper()
    if normalized == "HN":
        return "H"
    return normalized


@dataclass(slots=True)
class ChemicalShiftTarget:
    """Observed chemical shift for one residue and atom."""

    seq_id: int
    comp_id: str
    atom_id: str
    value: float
    uncertainty: float | None = None
    chain_id: str | None = None

    def target_id(self) -> str:
        """Return a stable identifier for this target."""
        chain_id = self.chain_id or "_"
        return f"cs:{chain_id}:{self.seq_id}:{self.comp_id}:{self.atom_id}"


@dataclass(slots=True)
class JCouplingTarget:
    """Observed scalar coupling for one residue-local interaction."""

    seq_id: int
    comp_id: str
    atom_id_1: str
    atom_id_2: str
    value: float
    coupling_type: str = "3J_HNHA"
    uncertainty: float | None = None
    chain_id: str | None = None

    def target_id(self) -> str:
        """Return a stable identifier for this target."""
        chain_id = self.chain_id or "_"
        return (
            f"jc:{chain_id}:{self.seq_id}:{self.comp_id}:"
            f"{self.atom_id_1}-{self.atom_id_2}:{self.coupling_type}"
        )


@dataclass(slots=True)
class NOERestraint:
    """Simple unique NOE distance restraint between two atoms."""

    seq_id_1: int
    comp_id_1: str
    atom_id_1: str
    seq_id_2: int
    comp_id_2: str
    atom_id_2: str
    target_value: float
    uncertainty: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    chain_id_1: str | None = None
    chain_id_2: str | None = None

    def target_id(self) -> str:
        """Return a stable identifier for this target."""
        chain_id_1 = self.chain_id_1 or "_"
        chain_id_2 = self.chain_id_2 or "_"
        left = f"{chain_id_1}:{self.seq_id_1}:{self.comp_id_1}:{self.atom_id_1}"
        right = f"{chain_id_2}:{self.seq_id_2}:{self.comp_id_2}:{self.atom_id_2}"
        return f"noe:{left}--{right}"


@dataclass(slots=True)
class NMRTargetBundle:
    """Container for all experimental observables used by AtypEmu."""

    chemical_shifts: list[ChemicalShiftTarget] = field(default_factory=list)
    j_couplings: list[JCouplingTarget] = field(default_factory=list)
    noe_restraints: list[NOERestraint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the bundle to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the bundle to disk as JSON.

        Args:
            path: Output JSON path.
        """
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NMRTargetBundle":
        """Construct a target bundle from a dictionary."""
        return cls(
            chemical_shifts=[
                ChemicalShiftTarget(**item)
                for item in payload.get("chemical_shifts", [])
            ],
            j_couplings=[
                JCouplingTarget(**item) for item in payload.get("j_couplings", [])
            ],
            noe_restraints=[
                NOERestraint(**item) for item in payload.get("noe_restraints", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "NMRTargetBundle":
        """Read a target bundle from JSON."""
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)


@dataclass(slots=True)
class CandidateRecord:
    """One conformer entry inside a candidate pool."""

    candidate_id: str
    structure_path: str
    source: str
    chemical_shift_path: str | None = None
    chemical_shift_format: str | None = None
    is_protonated: bool | None = None
    residue_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the record to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class CandidatePool:
    """Collection of candidate conformers with optional sidecar metadata."""

    records: list[CandidateRecord]

    def to_jsonl(self, path: str | Path) -> None:
        """Write the pool as JSON Lines.

        Args:
            path: Output JSONL path.
        """
        lines = [json.dumps(record.as_dict()) for record in self.records]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CandidatePool":
        """Read a pool from JSON Lines."""
        records: list[CandidateRecord] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            records.append(CandidateRecord(**json.loads(line)))
        return cls(records=records)

    @property
    def candidate_ids(self) -> list[str]:
        """Return candidate identifiers in pool order."""
        return [record.candidate_id for record in self.records]


@dataclass(slots=True)
class ObservableMatrix:
    """Predicted observable values for one channel."""

    label: str
    values: np.ndarray
    mask: np.ndarray
    target_ids: list[str]
    candidate_ids: list[str]
    transform: str = "identity"

    def __post_init__(self) -> None:
        """Validate matrix dimensions after construction."""
        if self.values.shape != self.mask.shape:
            raise ValueError("Observable values and masks must have the same shape.")
        if self.values.shape[0] != len(self.target_ids):
            raise ValueError("Target ID count must match the number of rows.")
        if self.values.shape[1] != len(self.candidate_ids):
            raise ValueError("Candidate ID count must match the number of columns.")


@dataclass(slots=True)
class ObservableBundle:
    """Bundle of observable matrices sharing one candidate ordering."""

    chemical_shifts: ObservableMatrix | None = None
    j_couplings: ObservableMatrix | None = None
    noe_restraints: ObservableMatrix | None = None

    def iter_channels(self) -> list[tuple[str, ObservableMatrix]]:
        """Return non-empty channels in a stable order."""
        channels: list[tuple[str, ObservableMatrix]] = []
        if self.chemical_shifts is not None:
            channels.append(("chemical_shifts", self.chemical_shifts))
        if self.j_couplings is not None:
            channels.append(("j_couplings", self.j_couplings))
        if self.noe_restraints is not None:
            channels.append(("noe_restraints", self.noe_restraints))
        return channels

    def candidate_ids(self) -> list[str]:
        """Return shared candidate identifiers."""
        channels = self.iter_channels()
        if not channels:
            return []
        candidate_ids = channels[0][1].candidate_ids
        for _, matrix in channels[1:]:
            if matrix.candidate_ids != candidate_ids:
                raise ValueError(
                    "Observable matrices must share the same candidate order."
                )
        return candidate_ids

    def to_npz(self, path: str | Path) -> None:
        """Write the observable bundle to an NPZ archive."""
        payload: dict[str, Any] = {}
        for name, matrix in self.iter_channels():
            payload[f"{name}_values"] = matrix.values
            payload[f"{name}_mask"] = matrix.mask.astype(bool)
            payload[f"{name}_target_ids"] = np.asarray(matrix.target_ids, dtype=object)
            payload[f"{name}_candidate_ids"] = np.asarray(
                matrix.candidate_ids, dtype=object
            )
            payload[f"{name}_transform"] = np.asarray([matrix.transform], dtype=object)
        np.savez(path, **payload)

    @classmethod
    def from_npz(cls, path: str | Path) -> "ObservableBundle":
        """Read an observable bundle from an NPZ archive."""
        payload = np.load(path, allow_pickle=True)

        def load_matrix(name: str) -> ObservableMatrix | None:
            values_key = f"{name}_values"
            if values_key not in payload:
                return None
            return ObservableMatrix(
                label=name,
                values=payload[values_key],
                mask=payload[f"{name}_mask"].astype(bool),
                target_ids=payload[f"{name}_target_ids"].tolist(),
                candidate_ids=payload[f"{name}_candidate_ids"].tolist(),
                transform=payload[f"{name}_transform"].tolist()[0],
            )

        return cls(
            chemical_shifts=load_matrix("chemical_shifts"),
            j_couplings=load_matrix("j_couplings"),
            noe_restraints=load_matrix("noe_restraints"),
        )


@dataclass(slots=True)
class WeightSolution:
    """Result of a simplex-constrained optimization."""

    method: str
    weights: np.ndarray
    energy: float
    converged: bool
    iterations: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the solution to a JSON-compatible dictionary."""
        return {
            "method": self.method,
            "weights": self.weights.tolist(),
            "energy": self.energy,
            "converged": self.converged,
            "iterations": self.iterations,
            "diagnostics": self.diagnostics,
        }

    def to_json(self, path: str | Path) -> None:
        """Write the solution to disk as JSON."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "WeightSolution":
        """Read a weight solution from JSON."""
        payload = json.loads(Path(path).read_text())
        return cls(
            method=payload["method"],
            weights=np.asarray(payload["weights"], dtype=float),
            energy=float(payload["energy"]),
            converged=bool(payload["converged"]),
            iterations=int(payload["iterations"]),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


@dataclass(slots=True)
class EnergyBreakdown:
    """Posterior energy and its diagnostic decomposition."""

    energy: float
    weights: np.ndarray
    channel_scores: dict[str, float]
    ess: float
    entropy: float
    iterations: int
    converged: bool
    diagnostics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the breakdown to a JSON-compatible dictionary."""
        return {
            "energy": self.energy,
            "weights": self.weights.tolist(),
            "channel_scores": self.channel_scores,
            "ess": self.ess,
            "entropy": self.entropy,
            "iterations": self.iterations,
            "converged": self.converged,
            "diagnostics": self.diagnostics,
        }

    def to_json(self, path: str | Path) -> None:
        """Write the breakdown to disk as JSON."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "EnergyBreakdown":
        """Read an energy breakdown from JSON."""
        payload = json.loads(Path(path).read_text())
        return cls(
            energy=float(payload["energy"]),
            weights=np.asarray(payload["weights"], dtype=float),
            channel_scores=dict(payload["channel_scores"]),
            ess=float(payload["ess"]),
            entropy=float(payload["entropy"]),
            iterations=int(payload["iterations"]),
            converged=bool(payload["converged"]),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


@dataclass(slots=True)
class EvaluationReport:
    """User-facing evaluation metrics for a fitted solution."""

    metrics: dict[str, float]
    channel_metrics: dict[str, dict[str, float]]
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        """Write the evaluation report to disk as JSON."""
        payload = {
            "metrics": self.metrics,
            "channel_metrics": self.channel_metrics,
            "diagnostics": self.diagnostics,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
