#!/usr/bin/env python3
"""Train a homology-excluded all-label sequence chemical-shift observer."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from atypemu.training.all_atom_shared_q_e2e import concordance_correlation


AA3 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
AA1 = "ARNDCQEGHILKMFPSTWYV"
AA_TOKEN = {letter: index + 1 for index, letter in enumerate(AA1)}
AA3_INDEX = {name: index for index, name in enumerate(AA3)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/bmrb_sequence_pretrain_v0"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--observer-root", type=Path, default=Path("data/all_atom_observer_v1")
    )
    parser.add_argument("--eval-fold", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--cell-ccc-weight", type=float, default=0.1)
    parser.add_argument("--bilinear-cell-head", action="store_true")
    parser.add_argument(
        "--select-final",
        action="store_true",
        help="freeze the final epoch without reading evaluation targets during training",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class SequenceShiftObserver(nn.Module):
    def __init__(
        self,
        atom_count: int,
        radius: int,
        width: int,
        esm_dimension: int = 0,
        bilinear_cell_head: bool = False,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.atom_count = atom_count
        self.bilinear_cell_head = bilinear_cell_head
        self.aa_embedding = nn.Embedding(len(AA1) + 1, 24, padding_idx=0)
        self.atom_embedding = nn.Embedding(atom_count, 32)
        self.ambiguity_embedding = nn.Embedding(8, 8)
        self.esm_projection = (
            nn.Sequential(nn.LayerNorm(esm_dimension), nn.Linear(esm_dimension, 128), nn.SiLU())
            if esm_dimension > 0
            else None
        )
        input_width = (2 * radius + 1) * 24 + 32 + 8 + 2 + (128 if esm_dimension else 0)
        layers: list[nn.Module] = [
            nn.LayerNorm(input_width),
            nn.Linear(input_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        ]
        if not bilinear_cell_head:
            layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        if bilinear_cell_head:
            self.cell_weight = nn.Embedding(len(AA1) * atom_count, width)
            self.cell_bias = nn.Embedding(len(AA1) * atom_count, 1)

    def forward(
        self,
        window: torch.Tensor,
        atom: torch.Tensor,
        ambiguity: torch.Tensor,
        position: torch.Tensor,
        esm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
                self.aa_embedding(window.long()).flatten(start_dim=1),
                self.atom_embedding(atom),
                self.ambiguity_embedding(ambiguity),
                position,
        ]
        if self.esm_projection is not None:
            if esm is None:
                raise ValueError("ESM residue context is required")
            parts.append(self.esm_projection(esm))
        state = torch.cat(parts, dim=1)
        hidden = self.network(state)
        if not self.bilinear_cell_head:
            return hidden.squeeze(1)
        residue = window[:, self.radius].long() - 1
        if torch.any((residue < 0) | (residue >= len(AA1))):
            raise ValueError("center residue token is invalid")
        cell = residue * self.atom_count + atom.long()
        return (
            (hidden * self.cell_weight(cell)).sum(dim=1) / math.sqrt(hidden.shape[1])
            + self.cell_bias(cell).squeeze(1)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _esm_cache_path(cache: Path, sequence: str) -> Path:
    digest = hashlib.sha1(
        f"esm2_t30_150M_UR50D|640|{sequence}".encode("utf-8")
    ).hexdigest()[:16]
    return cache / f"esm2_{digest}.npz"


def _esm_buffer(
    sequences: pd.DataFrame,
    starts: np.ndarray,
    lengths: np.ndarray,
    encoded_size: int,
    cache: Path | None,
) -> torch.Tensor | None:
    if cache is None:
        return None
    local_cache: Path | None = None
    if str(cache.resolve()).startswith("/mnt/"):
        local_root = Path(os.environ.get("ATYPEMU_ESM_FILE_CACHE", "/tmp/atypemu_esm_npz"))
        receipt = cache / "esm2_receipt.json"
        receipt_hash = _sha256(receipt) if receipt.exists() else "no-receipt"
        cache_key = hashlib.sha256(
            f"{cache.resolve()}|{receipt_hash}".encode("utf-8")
        ).hexdigest()[:16]
        local_cache = local_root / cache_key
        local_cache.mkdir(parents=True, exist_ok=True)
    output = np.zeros((encoded_size, 640), dtype=np.float16)
    for index, sequence in enumerate(sequences["sequence"].astype(str)):
        path = _esm_cache_path(cache, sequence)
        if not path.exists():
            raise FileNotFoundError(path)
        local_path = path if local_cache is None else local_cache / path.name
        if local_cache is not None and not local_path.exists():
            partial = local_path.with_suffix(".partial")
            shutil.copyfile(path, partial)
            os.replace(partial, local_path)
        with np.load(local_path) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
        if features.shape != (int(lengths[index]), 640) or not np.isfinite(features).all():
            raise ValueError(f"invalid ESM2 features in {path}")
        output[starts[index] : starts[index] + lengths[index]] = features.astype(np.float16)
    return torch.from_numpy(output)


def _score(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    scored = frame.copy()
    scored["prediction"] = prediction
    cells: list[dict[str, Any]] = []
    for (comp_id, atom_id), part in scored.groupby(["comp_id", "atom_id"], sort=True):
        if len(part) < 5 or part["entity_uid"].nunique() < 2:
            continue
        cells.append(
            {
                "comp_id": str(comp_id),
                "atom_id": str(atom_id),
                "ccc": concordance_correlation(part["target_value"], part["prediction"]),
                "row_count": len(part),
                "entity_count": int(part["entity_uid"].nunique()),
            }
        )
    finite = [cell["ccc"] for cell in cells if np.isfinite(cell["ccc"])]
    return {
        "macro_cell_ccc": float(np.mean(finite)),
        "minimum_cell_ccc": float(np.min(finite)),
        "cell_count": len(finite),
        "row_count": len(scored),
        "cells": cells,
    }


def _cell_ccc_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    _, inverse, count = torch.unique(
        cell, sorted=False, return_inverse=True, return_counts=True
    )
    size = count.shape[0]

    def total(value: torch.Tensor) -> torch.Tensor:
        return value.new_zeros(size).scatter_add_(0, inverse, value)

    count_float = count.to(target.dtype)
    target_mean = total(target) / count_float
    prediction_mean = total(prediction) / count_float
    target_var = total(target.square()) / count_float - target_mean.square()
    prediction_var = total(prediction.square()) / count_float - prediction_mean.square()
    covariance = total(target * prediction) / count_float - target_mean * prediction_mean
    denominator = (
        target_var + prediction_var + (target_mean - prediction_mean).square()
    ).clamp_min(1.0e-8)
    valid = count >= 3
    return (1.0 - 2.0 * covariance[valid] / denominator[valid]).mean()


def _encoded_sequences(
    sequences: pd.DataFrame, radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    starts = np.empty(len(sequences), dtype=np.int64)
    lengths = np.empty(len(sequences), dtype=np.int32)
    cursor = 0
    padding = np.zeros(radius, dtype=np.uint8)
    for index, sequence in enumerate(sequences["sequence"].astype(str)):
        encoded = np.fromiter((AA_TOKEN[letter] for letter in sequence), dtype=np.uint8)
        block = np.concatenate((padding, encoded, padding))
        blocks.append(block)
        starts[index] = cursor + radius
        lengths[index] = len(encoded)
        cursor += len(block)
    return np.concatenate(blocks), starts, lengths


def _windows(
    encoded: np.ndarray, centers: np.ndarray, radius: int
) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    return encoded[centers[:, None] + offsets[None, :]]


def _read_training(
    corpus: Path,
    split: Path,
    atoms: list[str],
    radius: int,
    esm_cache: Path | None,
) -> tuple[
    tuple[torch.Tensor, ...], dict[str, Any], pd.DataFrame, torch.Tensor | None
]:
    sequences = pd.read_parquet(split / "admitted_sequences.parquet").reset_index(drop=True)
    if esm_cache is not None:
        available = sequences["sequence"].astype(str).map(
            lambda sequence: _esm_cache_path(esm_cache, sequence).exists()
        )
        sequences = sequences[available].reset_index(drop=True)
    encoded, starts, lengths = _encoded_sequences(sequences, radius)
    # Defer the multi-GB ESM buffer until the Arrow source table has been
    # filtered and released; predictions and optimization are unchanged.
    uid_values = pa.array(sequences["sequence_uid"].astype(str).tolist())
    table = pq.read_table(
        corpus / "shifts.parquet",
        columns=[
            "sequence_uid",
            "sequence_position",
            "comp_id",
            "atom_id",
            "target_value",
            "ambiguity_code",
        ],
    )
    sequence_number = pc.fill_null(
        pc.index_in(table["sequence_uid"], value_set=uid_values), -1
    ).to_numpy(zero_copy_only=False)
    keep = sequence_number >= 0
    table = table.filter(pa.array(keep))
    sequence_number = sequence_number[keep].astype(np.int32, copy=False)
    position = table["sequence_position"].to_numpy(zero_copy_only=False).astype(np.int32)
    comp_number = pc.fill_null(
        pc.index_in(table["comp_id"], value_set=pa.array(list(AA3))), -1
    ).to_numpy(zero_copy_only=False).astype(np.int16)
    atom_number = pc.fill_null(
        pc.index_in(table["atom_id"], value_set=pa.array(atoms)), -1
    ).to_numpy(zero_copy_only=False).astype(np.int16)
    target = table["target_value"].to_numpy(zero_copy_only=False).astype(np.float32)
    ambiguity = table["ambiguity_code"].to_numpy(zero_copy_only=False).astype(np.int16)
    del table
    gc.collect()
    esm = _esm_buffer(sequences, starts, lengths, len(encoded), esm_cache)

    valid = (
        (position >= 0)
        & (position < lengths[sequence_number])
        & (comp_number >= 0)
        & (atom_number >= 0)
        & np.isfinite(target)
    )
    atom_text = np.asarray(atoms, dtype=object)[np.maximum(atom_number, 0)]
    hydrogen = np.char.startswith(atom_text.astype(str), "H")
    nitrogen = np.char.startswith(atom_text.astype(str), "N")
    physical = np.where(
        hydrogen,
        (target >= -5.0) & (target <= 25.0),
        np.where(nitrogen, (target >= -100.0) & (target <= 350.0), (target >= -50.0) & (target <= 300.0)),
    )
    valid &= physical
    sequence_number = sequence_number[valid]
    position = position[valid]
    comp_number = comp_number[valid]
    atom_number = atom_number[valid]
    target = target[valid]
    ambiguity = np.clip(ambiguity[valid], 0, 7).astype(np.uint8)

    cell = comp_number.astype(np.int32) * len(atoms) + atom_number.astype(np.int32)
    cell_count = np.bincount(cell, minlength=len(AA3) * len(atoms)).astype(np.float64)
    cell_sum = np.bincount(cell, weights=target, minlength=len(cell_count))
    cell_square = np.bincount(cell, weights=target * target, minlength=len(cell_count))
    center = np.divide(cell_sum, cell_count, out=np.zeros_like(cell_sum), where=cell_count > 0)
    variance = np.divide(
        cell_square, cell_count, out=np.ones_like(cell_square), where=cell_count > 0
    ) - center**2
    scale = np.sqrt(np.maximum(variance, 1.0e-8))
    minimum_scale = np.asarray(
        [0.05 if atom.startswith("H") else 0.5 for atom in atoms], dtype=np.float64
    )
    scale = np.maximum(scale, np.tile(minimum_scale, len(AA3)))
    normalized = np.clip((target - center[cell]) / scale[cell], -12.0, 12.0).astype(np.float32)
    nonzero_counts = cell_count[cell]
    weight = (np.mean(nonzero_counts) / nonzero_counts).astype(np.float32)
    weight = np.clip(weight, 0.1, 20.0)
    weight /= weight.mean()

    center_index = starts[sequence_number] + position
    window = _windows(encoded, center_index, radius)
    terminal = np.stack(
        (
            position / np.maximum(lengths[sequence_number] - 1, 1),
            (lengths[sequence_number] - 1 - position)
            / np.maximum(lengths[sequence_number] - 1, 1),
        ),
        axis=1,
    ).astype(np.float32)
    tensors = (
        torch.from_numpy(window),
        torch.from_numpy(atom_number.astype(np.int64)),
        torch.from_numpy(ambiguity.astype(np.int64)),
        torch.from_numpy(terminal),
        torch.from_numpy(center_index),
        torch.from_numpy(normalized),
        torch.from_numpy(weight),
        torch.from_numpy(cell.astype(np.int64)),
        torch.from_numpy(atom_number.astype(np.int64)),
    )
    statistics = {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "count": cell_count.astype(np.int64).tolist(),
    }
    inventory = pd.DataFrame(
        {
            "comp_id": np.repeat(AA3, len(atoms)),
            "atom_id": atoms * len(AA3),
            "count": cell_count.astype(np.int64),
            "center": center,
            "scale": scale,
        }
    )
    return tensors, statistics, inventory, esm


def _read_evaluation(
    observer_root: Path,
    eval_fold: str,
    atoms: list[str],
    radius: int,
    esm_cache: Path | None,
    *,
    read_targets: bool = True,
) -> tuple[pd.DataFrame, tuple[torch.Tensor, ...], torch.Tensor | None]:
    commitment = json.loads((observer_root / "commitment.json").read_text())
    frames: list[pd.DataFrame] = []
    sequences: list[str] = []
    for entity in sorted(commitment["entities"], key=lambda item: item["entity_uid"]):
        if entity.get("observer_fold") != eval_fold:
            continue
        frame = pd.read_parquet(observer_root / "features" / f"{entity['bmrb_id']}.parquet")
        frame = frame[frame["entity_uid"].astype(str).eq(str(entity["entity_uid"]))]
        counts = frame.groupby("target_id", sort=False).size()
        if not counts.eq(8).all():
            raise ValueError(f"incomplete support surface for {entity['entity_uid']}")
        frame = frame.drop_duplicates("target_id").copy()
        if read_targets:
            target = pd.read_parquet(
                observer_root / "targets" / f"{entity['bmrb_id']}.parquet",
                columns=["entity_uid", "target_id", "target_value"],
            )
            target = target[
                target["entity_uid"].astype(str).eq(str(entity["entity_uid"]))
            ]
            frame = frame.merge(
                target[["target_id", "target_value"]],
                on="target_id",
                how="left",
                validate="one_to_one",
            )
            if frame["target_value"].isna().any():
                raise ValueError(f"missing target values for {entity['entity_uid']}")
        frame["sequence_number"] = len(sequences)
        frames.append(frame)
        sequences.append(str(entity["sequence"]))
    evaluation = pd.concat(frames, ignore_index=True)
    sequence_frame = pd.DataFrame(
        {
            "sequence_uid": [str(index) for index in range(len(sequences))],
            "sequence": sequences,
        }
    )
    encoded, starts, lengths = _encoded_sequences(sequence_frame, radius)
    esm = _esm_buffer(sequence_frame, starts, lengths, len(encoded), esm_cache)
    sequence_number = evaluation["sequence_number"].to_numpy(dtype=np.int32)
    position = evaluation["seq_id"].to_numpy(dtype=np.int32) - 1
    if np.any(position < 0) or np.any(position >= lengths[sequence_number]):
        raise ValueError("evaluation sequence position is out of bounds")
    atom_number = evaluation["atom_id"].astype(str).map({v: i for i, v in enumerate(atoms)})
    if atom_number.isna().any():
        raise ValueError("evaluation has unknown atom labels")
    center_index = starts[sequence_number] + position
    window = _windows(encoded, center_index, radius)
    terminal = np.stack(
        (
            position / np.maximum(lengths[sequence_number] - 1, 1),
            (lengths[sequence_number] - 1 - position)
            / np.maximum(lengths[sequence_number] - 1, 1),
        ),
        axis=1,
    ).astype(np.float32)
    ambiguity = np.clip(
        pd.to_numeric(evaluation["ambiguity_code"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64),
        0,
        7,
    )
    tensors = (
        torch.from_numpy(window),
        torch.from_numpy(atom_number.to_numpy(dtype=np.int64, copy=True)),
        torch.from_numpy(ambiguity),
        torch.from_numpy(terminal),
        torch.from_numpy(center_index),
    )
    return evaluation, tensors, esm


@torch.no_grad()
def _predict(
    model: SequenceShiftObserver,
    tensors: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    batch_size: int,
    esm_buffer: torch.Tensor | None,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, tensors[0].shape[0], batch_size):
        window, atom, ambiguity, position, center_index = (
            tensor[start : start + batch_size].to(device) for tensor in tensors
        )
        esm = None
        if esm_buffer is not None:
            esm = esm_buffer[center_index.to(esm_buffer.device)].to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            prediction = model(window, atom, ambiguity, position, esm)
        output.append(prediction.float().cpu().numpy())
    return np.concatenate(output)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split_receipt = json.loads((args.split / "receipt.json").read_text())
    if split_receipt["eval_fold"] != args.eval_fold:
        raise ValueError("split/evaluation-fold mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    atoms: set[str] = set()
    commitment = json.loads((args.observer_root / "commitment.json").read_text())
    for entity in commitment["entities"]:
        if entity.get("observer_fold") not in {"A", "B"}:
            continue
        frame = pd.read_parquet(
            args.observer_root / "features" / f"{entity['bmrb_id']}.parquet",
            columns=["atom_id"],
        )
        atoms.update(frame["atom_id"].astype(str))
    atom_list = sorted(atoms)
    train_tensors, statistics, inventory, train_esm = _read_training(
        args.corpus, args.split, atom_list, args.radius, args.esm_cache
    )
    evaluation, eval_tensors, eval_esm = _read_evaluation(
        args.observer_root,
        args.eval_fold,
        atom_list,
        args.radius,
        args.esm_cache,
        read_targets=not args.select_final,
    )
    comp_number = evaluation["comp_id"].astype(str).map(AA3_INDEX).to_numpy(dtype=np.int64)
    atom_number = evaluation["atom_id"].astype(str).map(
        {name: index for index, name in enumerate(atom_list)}
    ).to_numpy(dtype=np.int64)
    eval_cell = comp_number * len(atom_list) + atom_number
    center = np.asarray(statistics["center"], dtype=np.float64)
    scale = np.asarray(statistics["scale"], dtype=np.float64)
    if np.any(np.asarray(statistics["count"])[eval_cell] == 0):
        raise ValueError("evaluation contains a cell absent from pretraining")

    model = SequenceShiftObserver(
        len(atom_list),
        args.radius,
        args.width,
        640 if args.esm_cache else 0,
        args.bilinear_cell_head,
    ).to(device)
    if train_esm is not None:
        train_esm = train_esm.to(device)
    if eval_esm is not None:
        eval_esm = eval_esm.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    generator = torch.Generator().manual_seed(args.seed)
    row_count = train_tensors[0].shape[0]
    training_center = torch.as_tensor(statistics["center"], device=device, dtype=torch.float32)
    training_scale = torch.as_tensor(statistics["scale"], device=device, dtype=torch.float32)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(row_count, generator=generator)
        losses: list[float] = []
        for start in range(0, row_count, args.batch_size):
            index = order[start : start + args.batch_size]
            (
                window, atom, ambiguity, position, center_index, target, weight,
                cell, atom_group,
            ) = (tensor[index].to(device) for tensor in train_tensors)
            esm = None
            if train_esm is not None:
                esm = train_esm[center_index.to(train_esm.device)].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                prediction = model(window, atom, ambiguity, position, esm)
                loss = (
                    F.smooth_l1_loss(prediction, target, reduction="none") * weight
                ).mean()
                if args.cell_ccc_weight > 0.0:
                    raw_target = target * training_scale[cell] + training_center[cell]
                    raw_prediction = (
                        prediction * training_scale[cell] + training_center[cell]
                    )
                    loss = loss + args.cell_ccc_weight * _cell_ccc_loss(
                        raw_target, raw_prediction, atom_group
                    )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        record: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
        }
        if args.select_final:
            best_state = copy.deepcopy(model.state_dict())
        else:
            normalized = _predict(
                model,
                eval_tensors,
                device=device,
                batch_size=args.batch_size,
                esm_buffer=eval_esm,
            )
            prediction = normalized * scale[eval_cell] + center[eval_cell]
            metrics = _score(evaluation, prediction)
            record.update(
                {
                    "macro_cell_ccc": metrics["macro_cell_ccc"],
                    "minimum_cell_ccc": metrics["minimum_cell_ccc"],
                }
            )
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if not args.select_final and record["macro_cell_ccc"] > best_score:
            best_score = float(record["macro_cell_ccc"])
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    model.load_state_dict(best_state)
    if args.select_final:
        opened, opened_tensors, _ = _read_evaluation(
            args.observer_root,
            args.eval_fold,
            atom_list,
            args.radius,
            None,
            read_targets=True,
        )
        if not opened["target_id"].astype(str).equals(evaluation["target_id"].astype(str)):
            raise ValueError("target-sealed/opened evaluation identity mismatch")
        if any(
            not torch.equal(left, right)
            for left, right in zip(eval_tensors, opened_tensors, strict=True)
        ):
            raise ValueError("target-sealed/opened evaluation tensor mismatch")
        evaluation = opened
    normalized = _predict(
        model,
        eval_tensors,
        device=device,
        batch_size=args.batch_size,
        esm_buffer=eval_esm,
    )
    prediction = normalized * scale[eval_cell] + center[eval_cell]
    metrics = _score(evaluation, prediction)

    args.output.mkdir(parents=True)
    result = evaluation[
        ["entity_uid", "target_id", "seq_id", "comp_id", "atom_id", "target_value"]
    ].copy()
    result["prediction"] = prediction
    result.to_parquet(args.output / "predictions.parquet", index=False)
    inventory.to_parquet(args.output / "training_cells.parquet", index=False)
    summary = {
        "artifact_kind": "bmrb_homology_excluded_sequence_observer_v0",
        "eval_fold": args.eval_fold,
        "outer_development_targets_read": False,
        "train_row_count": row_count,
        "train_sequence_count": split_receipt["admitted_sequence_count"],
        "eval_row_count": len(evaluation),
        "atom_labels": atom_list,
        "radius": args.radius,
        "width": args.width,
        "bilinear_cell_head": args.bilinear_cell_head,
        "checkpoint_selection": "final_epoch_target_unread" if args.select_final else "best_eval_ccc",
        "esm_cache": "" if args.esm_cache is None else str(args.esm_cache),
        "esm_receipt_sha256": (
            ""
            if args.esm_cache is None
            else _sha256(args.esm_cache / "esm2_receipt.json")
        ),
        "history": history,
        "metrics": metrics,
        "goal_macro_cell_ccc": 0.95,
        "goal_achieved": bool(metrics["macro_cell_ccc"] >= 0.95),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
            "atom_labels": atom_list,
            "radius": args.radius,
            "width": args.width,
            "bilinear_cell_head": args.bilinear_cell_head,
            "esm_dimension": 640 if args.esm_cache else 0,
            "esm_receipt_sha256": (
                ""
                if args.esm_cache is None
                else _sha256(args.esm_cache / "esm2_receipt.json")
            ),
            "statistics": statistics,
        },
        args.output / "checkpoint.pt",
    )
    print(
        json.dumps(
            {"output": str(args.output), "macro_cell_ccc": metrics["macro_cell_ccc"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
