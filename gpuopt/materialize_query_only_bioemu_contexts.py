#!/usr/bin/env python3
"""Materialize target-unread BioEmu contexts from local query-only A3Ms."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.colabfold_inline import model_runner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    a3m_root = args.output_root / "a3m"
    a3m_root.mkdir(exist_ok=True)

    # Loading the frozen model_3 parameters once avoids 135 redundant 355 MB reads.
    model_runner._load_model_and_params = functools.lru_cache(maxsize=1)(
        model_runner._load_model_and_params
    )
    commitment = json.loads(args.commitment.read_text())
    entities = sorted(
        (
            entity
            for entity in commitment["entities"]
            if entity.get("split") == "train" and entity.get("observer_fold") in {"A", "B"}
        ),
        key=lambda entity: (len(entity["sequence"]), entity["entity_uid"]),
    )
    records = []
    for number, entity in enumerate(entities, start=1):
        sequence = entity["sequence"]
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        if sequence_hash != entity["sequence_sha256"]:
            raise ValueError(f"sequence hash mismatch: {entity['entity_uid']}")
        a3m_path = a3m_root / f"{sequence_hash}.a3m"
        a3m_path.write_text(f">{entity['entity_uid']}\n{sequence}\n")
        print(f"[{number}/{len(entities)}] {entity['entity_uid']} L={len(sequence)}", flush=True)
        single_path, pair_path = map(
            Path,
            get_colabfold_embeds(
                sequence,
                args.output_root,
                msa_file=a3m_path,
            ),
        )
        single = np.load(single_path, mmap_mode="r")
        pair = np.load(pair_path, mmap_mode="r")
        if single.shape != (len(sequence), 384) or pair.shape != (
            len(sequence),
            len(sequence),
            128,
        ):
            raise ValueError(f"invalid BioEmu context shape: {entity['entity_uid']}")
        if not np.isfinite(single).all() or not np.isfinite(pair).all():
            raise ValueError(f"nonfinite BioEmu context: {entity['entity_uid']}")
        records.append(
            {
                "entity_uid": entity["entity_uid"],
                "observer_fold": entity["observer_fold"],
                "sequence_sha256": sequence_hash,
                "single_sha256": sha256(single_path),
                "pair_sha256": sha256(pair_path),
            }
        )

    params_path = (
        model_runner.DEFAULT_PARAMS_DIR
        / "params"
        / f"params_model_{model_runner.MODEL_NUMBER}.npz"
    )
    receipt = {
        "artifact_kind": "target_unread_query_only_bioemu_contexts_v0",
        "entity_count": len(records),
        "commitment_sha256": sha256(args.commitment),
        "alphafold_model": f"model_{model_runner.MODEL_NUMBER}",
        "num_recycle": model_runner.NUM_RECYCLE,
        "msa_contract": "local query sequence only; no remote MSA request",
        "params_sha256": sha256(params_path),
        "records": records,
    }
    (args.output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
