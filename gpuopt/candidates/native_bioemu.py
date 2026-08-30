"""Differentiable complete-coordinate adapter for native BioEmu frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .torsion_assimilator import AA1_TO_3, SIDECHAIN_CHI_BONDS, TORSION_COLUMNS


def dihedral(points: tuple[torch.Tensor, ...]) -> torch.Tensor:
    p0, p1, p2, p3 = points
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def frame(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x = ca - n
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    y = c - ca
    y = y - (x * y).sum(-1, keepdim=True) * x
    y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


def read_template(path: Path, sequence: str, device: torch.device) -> dict[str, object]:
    records = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            records.append(
                (
                    int(line[22:26]),
                    line[12:16].strip(),
                    np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                )
            )
    residue = torch.tensor([row[0] - 1 for row in records], device=device)
    coordinates = torch.tensor(
        np.stack([row[2] for row in records]), device=device, dtype=torch.float32
    )
    names = [row[1] for row in records]
    local = torch.empty_like(coordinates)
    distal: list[list[tuple[torch.Tensor, int, int]]] = [[] for _ in sequence]
    backbone = []
    for seq_index in range(len(sequence)):
        indices = torch.where(residue == seq_index)[0]
        lookup = {names[index]: int(index) for index in indices.tolist()}
        if not {"N", "CA", "C"}.issubset(lookup):
            raise ValueError(f"incomplete template backbone at residue {seq_index + 1}")
        backbone.append(lookup)
        rotation = frame(
            coordinates[lookup["N"]], coordinates[lookup["CA"]], coordinates[lookup["C"]]
        )
        local[indices] = torch.einsum(
            "ij,nj->ni",
            rotation.transpose(0, 1),
            coordinates[indices] - coordinates[lookup["CA"]],
        )

        adjacency = {int(index): set() for index in indices.tolist()}
        for offset, first in enumerate(indices.tolist()):
            for second in indices.tolist()[offset + 1 :]:
                cutoff = (
                    1.25
                    if names[first].startswith("H") or names[second].startswith("H")
                    else 1.95
                )
                if torch.linalg.vector_norm(coordinates[first] - coordinates[second]) <= cutoff:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        for proximal_name, axis_name in SIDECHAIN_CHI_BONDS.get(
            AA1_TO_3[sequence[seq_index]], ()
        ):
            if proximal_name not in lookup or axis_name not in lookup:
                continue
            proximal, axis = lookup[proximal_name], lookup[axis_name]
            component, stack = {axis}, [axis]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if {current, neighbor} == {proximal, axis} or neighbor in component:
                        continue
                    component.add(neighbor)
                    stack.append(neighbor)
            if any(names[index] in {"N", "CA", "C", "O", "OXT"} for index in component):
                continue
            distal[seq_index].append(
                (torch.tensor(sorted(component), device=device), proximal, axis)
            )
    return {
        "residue": residue,
        "local": local,
        "names": names,
        "distal": distal,
        "backbone": backbone,
    }


def complete_coordinates(
    pos_nm: torch.Tensor,
    orientations: torch.Tensor,
    template: dict[str, object],
    chi_delta: torch.Tensor,
) -> torch.Tensor:
    residue = template["residue"]
    coordinates = torch.einsum("nij,nj->ni", orientations[residue], template["local"])
    coordinates = coordinates + 10.0 * pos_nm[residue]
    for seq_index, rotations in enumerate(template["distal"]):
        for chi_index, (indices, proximal, axis) in enumerate(rotations):
            angle = chi_delta[seq_index, chi_index]
            origin = coordinates[proximal]
            unit = coordinates[axis] - origin
            unit = unit / torch.linalg.vector_norm(unit).clamp_min(1e-8)
            shifted = coordinates[indices] - origin
            rotated = (
                shifted * torch.cos(angle)
                + torch.cross(unit.expand_as(shifted), shifted, dim=-1) * torch.sin(angle)
                + (shifted @ unit)[:, None] * unit * (1.0 - torch.cos(angle))
                + origin
            )
            updated = coordinates.clone()
            updated[indices] = rotated
            coordinates = updated
    return coordinates


def observer_torsions(
    coordinates: torch.Tensor,
    template: dict[str, object],
    row_residue: torch.Tensor,
    base: torch.Tensor,
    chi_delta: torch.Tensor,
) -> torch.Tensor:
    backbone = template["backbone"]
    output = base.clone()
    angles = torch.zeros(len(backbone), 3, device=coordinates.device)
    available = torch.zeros_like(angles)
    for index in range(len(backbone)):
        if index > 0:
            angles[index, 0] = dihedral(
                (
                    coordinates[backbone[index - 1]["C"]],
                    coordinates[backbone[index]["N"]],
                    coordinates[backbone[index]["CA"]],
                    coordinates[backbone[index]["C"]],
                )
            )
            available[index, 0] = 1
        if index + 1 < len(backbone):
            angles[index, 1] = dihedral(
                (
                    coordinates[backbone[index]["N"]],
                    coordinates[backbone[index]["CA"]],
                    coordinates[backbone[index]["C"]],
                    coordinates[backbone[index + 1]["N"]],
                )
            )
            angles[index, 2] = dihedral(
                (
                    coordinates[backbone[index]["CA"]],
                    coordinates[backbone[index]["C"]],
                    coordinates[backbone[index + 1]["N"]],
                    coordinates[backbone[index + 1]["CA"]],
                )
            )
            available[index, 1:] = 1
    for dimension, name in enumerate(("phi", "psi", "omega")):
        output[:, TORSION_COLUMNS.index(f"{name}_sin")] = torch.sin(
            angles[row_residue, dimension]
        )
        output[:, TORSION_COLUMNS.index(f"{name}_cos")] = torch.cos(
            angles[row_residue, dimension]
        )
        output[:, TORSION_COLUMNS.index(f"{name}_available")] = available[
            row_residue, dimension
        ]
    for dimension, name in enumerate(("chi1", "chi2", "chi3", "chi4")):
        sin_index = TORSION_COLUMNS.index(f"{name}_sin")
        cos_index = TORSION_COLUMNS.index(f"{name}_cos")
        angle = chi_delta[row_residue, dimension]
        sin_value, cos_value = output[:, sin_index].clone(), output[:, cos_index].clone()
        output[:, sin_index] = sin_value * torch.cos(angle) + cos_value * torch.sin(angle)
        output[:, cos_index] = cos_value * torch.cos(angle) - sin_value * torch.sin(angle)
    return output


def severe_clash_stats(
    coordinates: torch.Tensor, template: dict[str, object]
) -> tuple[torch.Tensor, torch.Tensor]:
    residue = template["residue"]
    distances = torch.cdist(coordinates, coordinates)
    audited = distances[torch.abs(residue[:, None] - residue[None, :]) > 1]
    return audited.min(), (audited < 0.5).sum()
