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

from candidates.native_bioemu import (
    complete_coordinates,
    observer_torsions,
    read_template,
    severe_clash_stats,
)
from candidates.torsion_assimilator import (
    CoordinateObserver,
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
    template_support_id = 626
    template = read_template(
        ROOT / f"data/BioEmu/bmr25218/bmr25218_BioEmu_{template_support_id}.pdb",
        SEQUENCE,
        device,
    )
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
    minimum_distance, severe_clash_count_tensor = severe_clash_stats(
        coordinates_after, template
    )
    severe_clash_count = int(severe_clash_count_tensor)
    result = {
        "contract": "native_bioemu_complete_coordinate_cs_gradient_smoke_v0",
        "claim_scope": "source-only autograd connectivity smoke; not predictive validation or a primary score",
        "source_fold": "B",
        "observer_training_entity_overlap": True,
        "predictive_evidence_allowed": False,
        "source_entity_uid": ENTITY_UID,
        "assigned_row_count": int(rows.numel()),
        "complete_atom_count": int(coordinates_before.shape[0]),
        "target_free_template_support_id": template_support_id,
        "fixed_diffusion_prior_control": True,
        "loss_before": float(loss_before.detach()),
        "loss_after_one_step": float(loss_after),
        "generator_diff_head_gradient_norm": generator_gradient,
        "sidechain_torsion_gradient_norm": sidechain_gradient,
        "coordinate_rms_movement_angstrom": float(displacement),
        "fixed_seed_replay_max_difference_angstrom": replay_max_difference,
        "minimum_nonlocal_interatomic_distance_angstrom": float(minimum_distance),
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
