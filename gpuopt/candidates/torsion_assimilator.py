"""Small crossfit observer plus NMR-conditioned residue-shared chi1 generator."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


AA3 = (
    "<UNK>",
    "<NTERM>",
    "<CTERM>",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
AA_INDEX = {name: index for index, name in enumerate(AA3)}
AA1_TO_3 = dict(zip("ARNDCQEGHILKMFPSTWYV", AA3[3:], strict=True))
TORSION_NAMES = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4")
TORSION_COLUMNS = tuple(
    f"{name}_{suffix}"
    for name in TORSION_NAMES
    for suffix in ("sin", "cos", "available")
)
ESM_MODEL = "esm2_t30_150M_UR50D"
ESM_DIM = 640
ACTUATOR_SPECS = (
    ("phi", 0.20),
    ("psi", 0.20),
    ("chi1", 0.50),
)
SUPPORT_COUNT = 8
OBSERVER_RESIDUAL_GAIN = 0.1
STRUCTURAL_RESPONSE_GAIN = 1.0
NUMERIC_COLUMNS = (
    "relative_position",
    "log_length",
    "solution_ph",
    "solution_temperature_k",
    "solution_ionic_strength_mm",
    "solution_pressure_atm",
    "solution_ph_available",
    "solution_temperature_k_available",
    "solution_ionic_strength_mm_available",
    "solution_pressure_atm_available",
)
FEATURE_COLUMNS = (
    "entity_uid",
    "target_id",
    "support_id",
    "seq_id",
    "comp_id",
    "atom_id",
    *NUMERIC_COLUMNS[2:],
    *TORSION_COLUMNS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_path(cache: Path, sequence: str) -> Path:
    digest = hashlib.sha1(f"{ESM_MODEL}|{ESM_DIM}|{sequence}".encode()).hexdigest()[:16]
    return cache / f"esm2_{digest}.npz"


def residue_index(value: object) -> int:
    token = str(value or "").strip().upper()
    return AA_INDEX.get(AA1_TO_3.get(token, token), 0)


def load_fold(
    data_root: Path,
    entities: list[dict[str, Any]],
    *,
    atom_index: dict[str, int],
    embedding_cache: Path,
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    esm_arrays: list[np.ndarray] = []
    torsion_arrays: list[np.ndarray] = []
    support_ids_reference: list[str] | None = None
    residue_lookup: dict[tuple[str, int], int] = {}
    residue_keys: list[tuple[str, int]] = []

    for entity_number, entity in enumerate(entities):
        entity_uid = str(entity["entity_uid"])
        bmrb_id = str(entity["bmrb_id"])
        sequence = str(entity["sequence"])
        support_ids = [str(value) for value in entity["support_ids"]]
        if len(support_ids) != SUPPORT_COUNT:
            raise ValueError(f"unexpected support count: {entity_uid}")
        if support_ids_reference is None:
            support_ids_reference = support_ids
        elif support_ids != support_ids_reference:
            raise ValueError("support ordering differs across entities")

        feature = pd.read_parquet(
            data_root / "features" / f"{bmrb_id}.parquet", columns=FEATURE_COLUMNS
        )
        target = pd.read_parquet(
            data_root / "targets" / f"{bmrb_id}.parquet",
            columns=("entity_uid", "target_id", "target_value"),
        )
        feature = feature[feature["entity_uid"].astype(str).eq(entity_uid)].copy()
        target = target[target["entity_uid"].astype(str).eq(entity_uid)].copy()
        target["target_value"] = pd.to_numeric(target["target_value"], errors="coerce")
        target = target[target["target_value"].map(math.isfinite)].copy()
        feature = feature[feature["target_id"].isin(target["target_id"])].copy()
        feature["support_id"] = pd.Categorical(
            feature["support_id"].astype(str), categories=support_ids, ordered=True
        )
        feature = feature.sort_values(["target_id", "support_id"], kind="stable")
        if not (
            feature.groupby("target_id", observed=True)["support_id"].nunique()
            == SUPPORT_COUNT
        ).all():
            raise ValueError(f"incomplete support surface: {entity_uid}")
        first = feature.drop_duplicates("target_id", keep="first").copy()
        target = target[["target_id", "target_value"]]
        first = first.merge(target, on="target_id", validate="one_to_one")
        if len(first) != len(target):
            raise ValueError(f"target/feature mismatch: {entity_uid}")

        sequence_esm = np.load(embedding_path(embedding_cache, sequence))["features"]
        seq_index = first["seq_id"].to_numpy(dtype=np.int64) - 1
        if np.any(seq_index < 0) or np.any(seq_index >= len(sequence_esm)):
            raise ValueError(f"sequence embedding mismatch: {entity_uid}")
        esm_arrays.append(sequence_esm[seq_index].astype(np.float16))

        torsion = (
            feature[list(TORSION_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            .reshape(len(first), SUPPORT_COUNT, len(TORSION_COLUMNS))
        )
        torsion_arrays.append(torsion)

        first["entity_number"] = entity_number
        first["comp_number"] = first["comp_id"].map(residue_index)
        first["atom_number"] = (
            first["atom_id"].astype(str).map(atom_index).fillna(0).astype(int)
        )
        first["previous_number"] = [
            AA_INDEX["<NTERM>"] if seq_id <= 1 else residue_index(sequence[seq_id - 2])
            for seq_id in first["seq_id"].astype(int)
        ]
        first["next_number"] = [
            AA_INDEX["<CTERM>"]
            if seq_id >= len(sequence)
            else residue_index(sequence[seq_id])
            for seq_id in first["seq_id"].astype(int)
        ]
        first["relative_position"] = (first["seq_id"].astype(float) - 1.0) / max(
            len(sequence) - 1, 1
        )
        first["log_length"] = math.log(max(len(sequence), 1)) / 6.0
        for seq_id in first["seq_id"].astype(int).unique():
            key = (entity_uid, int(seq_id))
            if key not in residue_lookup:
                residue_lookup[key] = len(residue_keys)
                residue_keys.append(key)
        first["residue_number"] = [
            residue_lookup[(entity_uid, int(seq_id))]
            for seq_id in first["seq_id"].astype(int)
        ]
        frames.append(first)

    frame = pd.concat(frames, ignore_index=True)
    numeric = (
        frame[list(NUMERIC_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return {
        "frame": frame[
            ["entity_uid", "target_id", "comp_id", "atom_id", "target_value"]
        ].copy(),
        "esm": np.concatenate(esm_arrays),
        "torsion": np.concatenate(torsion_arrays),
        "numeric": numeric,
        "categorical": frame[
            ["comp_number", "atom_number", "previous_number", "next_number"]
        ].to_numpy(dtype=np.int64),
        "entity_index": frame["entity_number"].to_numpy(dtype=np.int64),
        "residue_index": frame["residue_number"].to_numpy(dtype=np.int64),
        "residue_keys": residue_keys,
        "support_ids": support_ids_reference or [],
        "entities": entities,
    }


def normalization(
    train: dict[str, Any], evaluation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    mean = train["numeric"].mean(axis=0)
    scale = train["numeric"].std(axis=0).clip(min=1.0e-5)
    train["numeric"] = ((train["numeric"] - mean) / scale).astype(np.float32)
    evaluation["numeric"] = ((evaluation["numeric"] - mean) / scale).astype(np.float32)

    train_frame = train["frame"]
    cell = train_frame.groupby(["comp_id", "atom_id"])["target_value"].agg(
        ["mean", "std"]
    )
    atom = train_frame.groupby("atom_id")["target_value"].agg(["mean", "std"])
    element_frame = train_frame.assign(
        element=train_frame["atom_id"].astype(str).str[0]
    )
    element = element_frame.groupby("element")["target_value"].agg(["mean", "std"])
    global_mean = float(train_frame["target_value"].mean())
    global_scale = max(float(train_frame["target_value"].std()), 0.1)

    def attach(values: dict[str, Any]) -> None:
        centers = []
        scales = []
        for row in values["frame"][["comp_id", "atom_id"]].itertuples(index=False):
            key = (row.comp_id, row.atom_id)
            if key in cell.index:
                center, width = cell.loc[key, ["mean", "std"]]
            elif row.atom_id in atom.index:
                center, width = atom.loc[row.atom_id, ["mean", "std"]]
            elif str(row.atom_id)[0] in element.index:
                center, width = element.loc[str(row.atom_id)[0], ["mean", "std"]]
            else:
                center, width = global_mean, global_scale
            centers.append(float(center))
            scales.append(
                max(float(width) if math.isfinite(float(width)) else 0.1, 0.1)
            )
        values["center"] = (
            np.asarray(values["anchor"], dtype=np.float32)
            if "anchor" in values
            else np.asarray(centers, dtype=np.float32)
        )
        values["scale"] = np.asarray(scales, dtype=np.float32)
        target = values["frame"]["target_value"].to_numpy(dtype=np.float32)
        values["normalized_target"] = (target - values["center"]) / values["scale"]

    attach(train)
    attach(evaluation)
    return train, evaluation


def fold_copy(values: dict[str, Any]) -> dict[str, Any]:
    """Copy only arrays mutated by direction-specific train normalization."""

    output = dict(values)
    output["numeric"] = values["numeric"].copy()
    return output


class CoordinateObserver(nn.Module):
    def __init__(self, atom_levels: int, width: int = 96) -> None:
        super().__init__()
        self.comp = nn.Embedding(len(AA3), 12)
        self.atom = nn.Embedding(atom_levels, 24)
        self.neighbor = nn.Embedding(len(AA3), 8)
        self.esm = nn.Sequential(
            nn.LayerNorm(ESM_DIM), nn.Linear(ESM_DIM, 48), nn.SiLU()
        )
        self.numeric = nn.Sequential(nn.Linear(10, 16), nn.SiLU())
        self.torsion = nn.Sequential(nn.Linear(len(TORSION_COLUMNS), 24), nn.SiLU())
        total = 12 + 24 + 2 * 8 + 48 + 16 + 24
        self.readout = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        esm: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        torsion: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.cat(
            (
                self.comp(categorical[:, 0]),
                self.atom(categorical[:, 1]),
                self.neighbor(categorical[:, 2]),
                self.neighbor(categorical[:, 3]),
                self.esm(esm),
                self.numeric(numeric),
                self.torsion(torsion),
            ),
            dim=1,
        )
        return self.readout(state).squeeze(1)


def tensor(
    values: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def atom_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("atom_id")["target_id"].transform("count").to_numpy(float)
    weight = 1.0 / np.maximum(counts, 1.0)
    return (weight / weight.mean()).astype(np.float32)


def train_observer(
    data: dict[str, Any],
    *,
    atom_levels: int,
    device: torch.device,
    seed: int,
    epochs: int = 8,
    batch_size: int = 4096,
) -> CoordinateObserver:
    torch.manual_seed(seed)
    model = CoordinateObserver(atom_levels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    esm = tensor(data["esm"], device, torch.float32)
    categorical = tensor(data["categorical"], device, torch.long)
    numeric = tensor(data["numeric"], device, torch.float32)
    torsion = tensor(data["torsion"], device, torch.float32)
    target = tensor(data["normalized_target"], device, torch.float32)
    weight = tensor(atom_weights(data["frame"]), device, torch.float32)
    size = len(target)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(size, generator=generator, device=device)
        model.train()
        for start in range(0, size, batch_size):
            row = order[start : start + batch_size]
            count = len(row)
            prediction = model(
                esm[row].repeat_interleave(SUPPORT_COUNT, dim=0),
                categorical[row].repeat_interleave(SUPPORT_COUNT, dim=0),
                numeric[row].repeat_interleave(SUPPORT_COUNT, dim=0),
                torsion[row].reshape(-1, torsion.shape[-1]),
            ).reshape(count, SUPPORT_COUNT)
            loss = torch.mean(
                weight[row]
                * torch.nn.functional.smooth_l1_loss(
                    prediction.mean(dim=1), target[row], reduction="none"
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def actuator_delta(delta_raw: torch.Tensor) -> torch.Tensor:
    bounds = delta_raw.new_tensor([bound for _name, bound in ACTUATOR_SPECS])
    return bounds * torch.tanh(delta_raw)


def actuate_torsions(
    torsion: torch.Tensor,
    residue_index_tensor: torch.Tensor,
    delta_raw: torch.Tensor,
) -> torch.Tensor:
    delta = actuator_delta(delta_raw[residue_index_tensor])
    output = torsion.clone()
    for dimension, (name, _bound) in enumerate(ACTUATOR_SPECS):
        sin_index = TORSION_COLUMNS.index(f"{name}_sin")
        cos_index = TORSION_COLUMNS.index(f"{name}_cos")
        available_index = TORSION_COLUMNS.index(f"{name}_available")
        angle = delta[..., dimension] * torsion[..., available_index]
        sin_value = torsion[..., sin_index]
        cos_value = torsion[..., cos_index]
        output[..., sin_index] = sin_value * torch.cos(angle) + cos_value * torch.sin(
            angle
        )
        output[..., cos_index] = cos_value * torch.cos(angle) - sin_value * torch.sin(
            angle
        )
    return output


def coordinate_response(
    model: CoordinateObserver,
    esm: torch.Tensor,
    categorical: torch.Tensor,
    numeric: torch.Tensor,
    base_torsion: torch.Tensor,
    active_torsion: torch.Tensor,
) -> torch.Tensor:
    """Shrink only the absolute mean, not calibrated conformer differences."""

    count = len(esm)
    repeated_esm = esm.repeat_interleave(SUPPORT_COUNT, dim=0)
    repeated_categorical = categorical.repeat_interleave(SUPPORT_COUNT, dim=0)
    repeated_numeric = numeric.repeat_interleave(SUPPORT_COUNT, dim=0)
    base = model(
        repeated_esm,
        repeated_categorical,
        repeated_numeric,
        base_torsion.reshape(-1, base_torsion.shape[-1]),
    ).reshape(count, SUPPORT_COUNT)
    active = (
        base
        if active_torsion is base_torsion
        else model(
            repeated_esm,
            repeated_categorical,
            repeated_numeric,
            active_torsion.reshape(-1, active_torsion.shape[-1]),
        ).reshape(count, SUPPORT_COUNT)
    )
    base_mean = base.mean(dim=1, keepdim=True)
    return OBSERVER_RESIDUAL_GAIN * base_mean + STRUCTURAL_RESPONSE_GAIN * (
        active - base_mean
    )


def predict_surface(
    model: CoordinateObserver,
    values: dict[str, Any],
    *,
    device: torch.device,
    delta_raw: torch.Tensor | None,
    chunk_size: int = 2048,
) -> np.ndarray:
    output = []
    for start in range(0, len(values["frame"]), chunk_size):
        stop = min(start + chunk_size, len(values["frame"]))
        esm = tensor(values["esm"][start:stop], device, torch.float32)
        categorical = tensor(values["categorical"][start:stop], device, torch.long)
        numeric = tensor(values["numeric"][start:stop], device, torch.float32)
        base_torsion = tensor(values["torsion"][start:stop], device, torch.float32)
        torsion = base_torsion
        if delta_raw is not None:
            residues = tensor(values["residue_index"][start:stop], device, torch.long)
            torsion = actuate_torsions(torsion, residues, delta_raw)
        flat_prediction = coordinate_response(
            model, esm, categorical, numeric, base_torsion, torsion
        )
        output.append(flat_prediction.detach().cpu().numpy())
    normalized = np.concatenate(output)
    return values["center"][:, None] + values["scale"][:, None] * normalized


def optimize_assimilation(
    model: CoordinateObserver,
    values: dict[str, Any],
    *,
    device: torch.device,
    steps: int = 100,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    residue_count = len(values["residue_keys"])
    entity_count = len(values["entities"])
    delta_raw = nn.Parameter(
        torch.zeros(residue_count, SUPPORT_COUNT, len(ACTUATOR_SPECS), device=device)
    )
    q_logits = nn.Parameter(torch.zeros(entity_count, SUPPORT_COUNT, device=device))
    optimizer = torch.optim.Adam((delta_raw, q_logits), lr=0.08)
    weights = atom_weights(values["frame"])
    row_count = len(values["frame"])

    def backward_objective() -> None:
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            esm = tensor(values["esm"][start:stop], device, torch.float32)
            categorical = tensor(values["categorical"][start:stop], device, torch.long)
            numeric = tensor(values["numeric"][start:stop], device, torch.float32)
            base_torsion = tensor(values["torsion"][start:stop], device, torch.float32)
            residues = tensor(values["residue_index"][start:stop], device, torch.long)
            entities = tensor(values["entity_index"][start:stop], device, torch.long)
            target = tensor(
                values["normalized_target"][start:stop], device, torch.float32
            )
            weight = tensor(weights[start:stop], device, torch.float32)
            torsion = actuate_torsions(base_torsion, residues, delta_raw)
            support_prediction = coordinate_response(
                model, esm, categorical, numeric, base_torsion, torsion
            )
            q = torch.softmax(q_logits[entities], dim=1)
            aggregate = torch.sum(q * support_prediction, dim=1)
            loss = torch.sum(weight * torch.square(aggregate - target)) / row_count
            loss.backward()
        q = torch.softmax(q_logits, dim=1)
        regularizer = 1.0e-3 * torch.mean(torch.square(torch.tanh(delta_raw)))
        regularizer = regularizer + 1.0e-3 * torch.mean(
            torch.sum(q * torch.log((q * SUPPORT_COUNT).clamp_min(1.0e-12)), dim=1)
        )
        regularizer.backward()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        backward_objective()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    backward_objective()
    gradient_norm = float(delta_raw.grad.norm().detach().cpu())
    return delta_raw.detach(), torch.softmax(q_logits.detach(), dim=1), gradient_norm


def surface_frame(values: dict[str, Any], prediction: np.ndarray) -> pd.DataFrame:
    frames = []
    identity = values["frame"][["entity_uid", "target_id"]].reset_index(drop=True)
    for support_number, support_id in enumerate(values["support_ids"]):
        part = identity.copy()
        part["support_id"] = support_id
        part["support_prediction"] = prediction[:, support_number]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def q_frame(values: dict[str, Any], q: np.ndarray) -> pd.DataFrame:
    rows = []
    for entity_number, entity in enumerate(values["entities"]):
        weights = np.asarray(q[entity_number], dtype=np.float64)
        weights = weights / weights.sum()
        weights[-1] = 1.0 - float(weights[:-1].sum())
        for support_number, support_id in enumerate(values["support_ids"]):
            rows.append(
                {
                    "entity_uid": str(entity["entity_uid"]),
                    "support_id": support_id,
                    "posterior_weight": float(weights[support_number]),
                }
            )
    return pd.DataFrame(rows)


def rotate_about_axis(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    shifted = points - origin
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        shifted * cosine
        + np.cross(unit, shifted) * sine
        + np.outer(shifted @ unit, unit) * (1.0 - cosine)
        + origin
    )


def coordinate_audit(
    values: dict[str, Any],
    delta_raw: torch.Tensor,
    *,
    structure_root: Path,
    output: Path,
) -> None:
    delta = actuator_delta(delta_raw).cpu().numpy()
    availability = np.zeros_like(delta)
    for row_number, residue_number in enumerate(values["residue_index"]):
        for dimension, (name, _bound) in enumerate(ACTUATOR_SPECS):
            available_index = TORSION_COLUMNS.index(f"{name}_available")
            availability[residue_number, :, dimension] = np.maximum(
                availability[residue_number, :, dimension],
                values["torsion"][row_number, :, available_index],
            )
    score = np.sum(np.abs(delta) * availability, axis=2)
    residue_number, support_number = np.unravel_index(np.argmax(score), score.shape)
    entity_uid, _ = values["residue_keys"][residue_number]
    entity = next(
        item for item in values["entities"] if str(item["entity_uid"]) == entity_uid
    )
    support_id = values["support_ids"][support_number]
    path = (
        structure_root
        / str(entity["bmrb_id"])
        / f"{entity['bmrb_id']}_{support_id}.pdb"
    )

    records = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        records.append(
            {
                "name": line[12:16].strip(),
                "chain": line[21:22],
                "seq_id": int(line[22:26]),
                "coord": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ),
            }
        )
    base = np.stack([record["coord"] for record in records])
    conditioned = base.copy()
    backbone = {
        "N",
        "CA",
        "C",
        "O",
        "OXT",
        "H",
        "H1",
        "H2",
        "H3",
        "HA",
        "HA2",
        "HA3",
        "CB",
    }
    relevant = {
        seq_id: delta[index, support_number]
        for index, (uid, seq_id) in enumerate(values["residue_keys"])
        if uid == entity_uid
    }
    for seq_id, angles in sorted(relevant.items()):
        indices = [
            index for index, record in enumerate(records) if record["seq_id"] == seq_id
        ]
        if not indices:
            continue
        chain = records[indices[0]]["chain"]
        indices = [index for index in indices if records[index]["chain"] == chain]
        n_atom = next(
            (index for index in indices if records[index]["name"] == "N"), None
        )
        ca = next((index for index in indices if records[index]["name"] == "CA"), None)
        c_atom = next(
            (index for index in indices if records[index]["name"] == "C"), None
        )
        cb = next((index for index in indices if records[index]["name"] == "CB"), None)
        phi, psi, chi1 = (float(value) for value in angles)
        if n_atom is not None and ca is not None and abs(phi) > 1.0e-12:
            n_side = {"N", "H", "H1", "H2", "H3"}
            rotated = [
                index
                for index, record in enumerate(records)
                if record["chain"] == chain
                and (
                    record["seq_id"] > seq_id
                    or (record["seq_id"] == seq_id and record["name"] not in n_side)
                )
            ]
            conditioned[rotated] = rotate_about_axis(
                conditioned[rotated],
                conditioned[n_atom],
                conditioned[ca] - conditioned[n_atom],
                phi,
            )
        if ca is not None and c_atom is not None and abs(psi) > 1.0e-12:
            rotated = [
                index
                for index, record in enumerate(records)
                if record["chain"] == chain
                and (
                    record["seq_id"] > seq_id
                    or (record["seq_id"] == seq_id and record["name"] in {"O", "OXT"})
                )
            ]
            conditioned[rotated] = rotate_about_axis(
                conditioned[rotated],
                conditioned[ca],
                conditioned[c_atom] - conditioned[ca],
                psi,
            )
        distal = [index for index in indices if records[index]["name"] not in backbone]
        if ca is not None and cb is not None and distal and abs(chi1) > 1.0e-12:
            conditioned[distal] = rotate_about_axis(
                conditioned[distal],
                conditioned[ca],
                conditioned[cb] - conditioned[ca],
                chi1,
            )
    np.savez(
        output,
        conditioned_coordinates=conditioned.astype(np.float32),
        no_evidence_coordinates=base.astype(np.float32),
        atom_mask=np.ones(len(base), dtype=bool),
    )
