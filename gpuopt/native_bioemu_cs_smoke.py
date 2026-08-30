#!/usr/bin/env python3
"""Source-only smoke test for CS gradients through native BioEmu sampling."""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
from bioemu.sample import get_context_chemgraph
from bioemu.training.loss import _rollout
from torch_geometric.data import Batch

from candidates.torsion_assimilator import (
    CoordinateObserver,
    TORSION_COLUMNS,
    atom_weights,
    fold_copy,
    normalization,
    tensor,
    train_observer,
)


ROOT = Path(__file__).resolve().parents[1]
ENTITY_UID = "bmrb:25218:entity:1"
SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
CHI_BOUNDS = torch.tensor((0.10, 0.10, 0.10, 0.10))


def dihedral(points: tuple[torch.Tensor, ...]) -> torch.Tensor:
    p0, p1, p2, p3 = points
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
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


def read_template(path: Path, device: torch.device) -> dict[str, object]:
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
    distal: list[list[tuple[torch.Tensor, int, int]]] = [[] for _ in SEQUENCE]
    for seq_index in range(len(SEQUENCE)):
        indices = torch.where(residue == seq_index)[0]
        lookup = {names[index]: int(index) for index in indices.tolist()}
        if not {"N", "CA", "C"}.issubset(lookup):
            raise ValueError(f"incomplete template backbone at residue {seq_index + 1}")
        rotation = frame(
            coordinates[lookup["N"]], coordinates[lookup["CA"]], coordinates[lookup["C"]]
        )
        local[indices] = torch.einsum(
            "ij,nj->ni", rotation.transpose(0, 1), coordinates[indices] - coordinates[lookup["CA"]]
        )

        adjacency = {int(index): set() for index in indices.tolist()}
        for offset, first in enumerate(indices.tolist()):
            for second in indices.tolist()[offset + 1 :]:
                cutoff = 1.25 if names[first].startswith("H") or names[second].startswith("H") else 1.95
                if torch.linalg.vector_norm(coordinates[first] - coordinates[second]) <= cutoff:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        bonds = {
            "R": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE")),
            "N": (("CA", "CB"), ("CB", "CG")), "D": (("CA", "CB"), ("CB", "CG")),
            "C": (("CA", "CB"),), "Q": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
            "E": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")), "H": (("CA", "CB"), ("CB", "CG")),
            "I": (("CA", "CB"), ("CB", "CG1")), "L": (("CA", "CB"), ("CB", "CG")),
            "K": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE")),
            "M": (("CA", "CB"), ("CB", "CG"), ("CG", "SD")), "F": (("CA", "CB"), ("CB", "CG")),
            "S": (("CA", "CB"),), "T": (("CA", "CB"),), "W": (("CA", "CB"), ("CB", "CG")),
            "Y": (("CA", "CB"), ("CB", "CG")), "V": (("CA", "CB"),),
        }.get(SEQUENCE[seq_index], ())
        for proximal_name, axis_name in bonds:
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
            distal[seq_index].append((torch.tensor(sorted(component), device=device), proximal, axis))
    return {"residue": residue, "local": local, "names": names, "distal": distal}


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
    residue = template["residue"]
    names = template["names"]
    backbone = []
    for seq_index in range(len(SEQUENCE)):
        indices = torch.where(residue == seq_index)[0].tolist()
        lookup = {names[index]: index for index in indices}
        backbone.append(lookup)
    output = base.clone()
    angles = torch.zeros(len(SEQUENCE), 3, device=coordinates.device)
    available = torch.zeros_like(angles)
    for index in range(len(SEQUENCE)):
        if index > 0:
            angles[index, 0] = dihedral(
                (coordinates[backbone[index - 1]["C"]], coordinates[backbone[index]["N"]],
                 coordinates[backbone[index]["CA"]], coordinates[backbone[index]["C"]])
            )
            available[index, 0] = 1
        if index + 1 < len(SEQUENCE):
            angles[index, 1] = dihedral(
                (coordinates[backbone[index]["N"]], coordinates[backbone[index]["CA"]],
                 coordinates[backbone[index]["C"]], coordinates[backbone[index + 1]["N"]])
            )
            angles[index, 2] = dihedral(
                (coordinates[backbone[index]["CA"]], coordinates[backbone[index]["C"]],
                 coordinates[backbone[index + 1]["N"]], coordinates[backbone[index + 1]["CA"]])
            )
            available[index, 1:] = 1
    for dimension, name in enumerate(("phi", "psi", "omega")):
        output[:, TORSION_COLUMNS.index(f"{name}_sin")] = torch.sin(angles[row_residue, dimension])
        output[:, TORSION_COLUMNS.index(f"{name}_cos")] = torch.cos(angles[row_residue, dimension])
        output[:, TORSION_COLUMNS.index(f"{name}_available")] = available[row_residue, dimension]
    for dimension, name in enumerate(("chi1", "chi2", "chi3", "chi4")):
        sin_index, cos_index = TORSION_COLUMNS.index(f"{name}_sin"), TORSION_COLUMNS.index(f"{name}_cos")
        angle = chi_delta[row_residue, dimension]
        sin_value, cos_value = output[:, sin_index].clone(), output[:, cos_index].clone()
        output[:, sin_index] = sin_value * torch.cos(angle) + cos_value * torch.sin(angle)
        output[:, cos_index] = cos_value * torch.cos(angle) - sin_value * torch.sin(angle)
    return output


def main() -> int:
    device = torch.device("cuda")
    cached = pickle.load((ROOT / ".auto/cache/torsion_assimilator_folds_v1.pkl").open("rb"))["folds"]
    source = fold_copy(cached["B"])
    if not any(entity["entity_uid"] == ENTITY_UID for entity in source["entities"]):
        raise ValueError("smoke entity is not in source fold B")
    source, _ = normalization(source, fold_copy(cached["B"]))
    atom_levels = int(source["categorical"][:, 1].max()) + 1
    observer_path = ROOT / ".auto/cache/native_smoke_observer_B.pt"
    if observer_path.exists():
        observer = CoordinateObserver(atom_levels).to(device)
        observer.load_state_dict(torch.load(observer_path, map_location=device, weights_only=True))
        observer.eval()
        for parameter in observer.parameters():
            parameter.requires_grad_(False)
    else:
        observer = train_observer(source, atom_levels=atom_levels, device=device, seed=20260830)
        torch.save(observer.state_dict(), observer_path)

    rows_np = source["frame"]["entity_uid"].astype(str).eq(ENTITY_UID).to_numpy()
    rows = torch.from_numpy(np.flatnonzero(rows_np)).to(device)
    global_residue = source["residue_index"][rows_np]
    row_residue_np = np.array([source["residue_keys"][index][1] - 1 for index in global_residue])
    row_residue = torch.tensor(row_residue_np, device=device)
    base_torsion = tensor(source["torsion"][rows_np, 0], device, torch.float32)
    target = tensor(source["normalized_target"][rows_np], device, torch.float32)
    weight = tensor(atom_weights(source["frame"])[rows_np], device, torch.float32)
    esm = tensor(source["esm"][rows_np], device, torch.float32)
    categorical = tensor(source["categorical"][rows_np], device, torch.long)
    numeric = tensor(source["numeric"][rows_np], device, torch.float32)

    checkpoint, config = maybe_download_checkpoint(model_name="bioemu-v1.1")
    generator = load_model(checkpoint, config).to(device).eval()
    for name, parameter in generator.named_parameters():
        parameter.requires_grad_("diff_head" in name)
    sdes = load_sdes(config, cache_so3_dir=ROOT / ".auto/cache/bioemu_so3")
    context = get_context_chemgraph(SEQUENCE, cache_embeds_dir=Path.home() / ".bioemu_embeds_cache")
    context = Batch.from_data_list([context]).to(device)
    torch.manual_seed(20260831)
    fixed_pos_prior = sdes["pos"].prior_sampling(context.pos.shape, device=device)
    fixed_orientation_prior = sdes["node_orientations"].prior_sampling(
        context.node_orientations.shape, device=device
    )
    sdes["pos"].prior_sampling = lambda shape, device=None: fixed_pos_prior.clone().to(device)
    sdes["node_orientations"].prior_sampling = (
        lambda shape, device=None: fixed_orientation_prior.clone().to(device)
    )
    template = read_template(ROOT / "data/BioEmu/bmr25218/bmr25218_BioEmu_1.pdb", device)
    chi_raw = torch.nn.Parameter(torch.zeros(len(SEQUENCE), 4, device=device))
    parameters = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        ({"params": parameters, "lr": 1e-6}, {"params": [chi_raw], "lr": 1e-2})
    )

    def sample_and_loss() -> tuple[torch.Tensor, torch.Tensor]:
        random.seed(20260831)
        np.random.seed(20260831)
        torch.manual_seed(20260831)
        generated = _rollout(
            batch=copy.deepcopy(context),
            sdes=sdes,
            score_model=generator,
            mid_t=0.2,
            N_rollout=4,
            record_grad_steps={1, 2},
            device=device,
        ).get_example(0)
        chi_delta = CHI_BOUNDS.to(device) * torch.tanh(chi_raw)
        coordinates = complete_coordinates(
            generated.pos, generated.node_orientations, template, chi_delta
        )
        torsion = observer_torsions(
            coordinates, template, row_residue, base_torsion, chi_delta
        )
        prediction = observer(esm, categorical, numeric, torsion)
        loss = torch.mean(weight * torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none"))
        return loss, coordinates

    optimizer.zero_grad(set_to_none=True)
    loss_before, coordinates_before = sample_and_loss()
    loss_replay_graph, coordinates_replay_graph = sample_and_loss()
    loss_replay = loss_replay_graph.detach()
    coordinates_replay = coordinates_replay_graph.detach()
    del loss_replay_graph, coordinates_replay_graph
    replay_max_difference = float(
        torch.max(torch.abs(coordinates_replay - coordinates_before)).detach()
    )
    if replay_max_difference > 1e-5 or not torch.allclose(loss_replay, loss_before):
        raise RuntimeError(
            f"fixed-seed native sampling is not reproducible: {replay_max_difference}"
        )
    loss_before.backward()
    generator_gradient = math.sqrt(
        sum(float(torch.sum(parameter.grad.square())) for parameter in parameters if parameter.grad is not None)
    )
    sidechain_gradient = float(torch.linalg.vector_norm(chi_raw.grad))
    optimizer.step()
    with torch.no_grad():
        loss_after, coordinates_after = sample_and_loss()
    displacement = torch.sqrt(torch.mean(torch.sum((coordinates_after - coordinates_before.detach()) ** 2, dim=1)))
    residue = template["residue"]
    nonlocal_mask = torch.abs(residue[:, None] - residue[None, :]) > 1
    distances = torch.cdist(coordinates_after, coordinates_after)
    audited_distances = distances[nonlocal_mask]
    severe_clash_count = int((audited_distances < 0.5).sum())
    result = {
        "contract": "native_bioemu_complete_coordinate_cs_gradient_smoke_v0",
        "claim_scope": "source-only autograd connectivity smoke; not predictive validation or a primary score",
        "source_fold": "B",
        "observer_training_entity_overlap": True,
        "predictive_evidence_allowed": False,
        "source_entity_uid": ENTITY_UID,
        "assigned_row_count": int(rows.numel()),
        "complete_atom_count": int(coordinates_before.shape[0]),
        "fixed_diffusion_prior_control": True,
        "loss_before": float(loss_before.detach()),
        "loss_after_one_step": float(loss_after),
        "generator_diff_head_gradient_norm": generator_gradient,
        "sidechain_torsion_gradient_norm": sidechain_gradient,
        "coordinate_rms_movement_angstrom": float(displacement),
        "fixed_seed_replay_max_difference_angstrom": replay_max_difference,
        "minimum_nonlocal_interatomic_distance_angstrom": float(audited_distances.min()),
        "directed_nonlocal_pairs_below_0_5_angstrom": severe_clash_count,
        "template_decoder_physicality_pass": severe_clash_count == 0,
        "all_finite": bool(
            torch.isfinite(coordinates_after).all()
            and math.isfinite(generator_gradient)
            and math.isfinite(sidechain_gradient)
        ),
        "sealed_evaluation_targets_read": False,
    }
    output = ROOT / ".auto/preunblind/native_bioemu_cs_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not (result["all_finite"] and generator_gradient > 0 and sidechain_gradient > 0 and displacement > 0):
        raise RuntimeError("native BioEmu CS-gradient smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
