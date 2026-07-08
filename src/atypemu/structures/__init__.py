"""Structure and candidate-pool helpers for AtypEmu."""

from atypemu.structures.archives import unpack_zip_archives
from atypemu.structures.geometry import (
    compute_chain_surface_area,
    compute_pair_surface_area,
    compute_residue_contact_graph,
    extract_polymer_residue_nodes,
    load_first_model,
    write_contact_graph_npz,
)
from atypemu.structures.pool import (
    ProtonationValidator,
    ResidueMapper,
    StructureLoader,
    build_candidate_pool,
)
from atypemu.structures.staging import (
    STAGE_MANIFEST_FILENAME,
    infer_candidate_metadata,
    load_stage_manifest,
    parse_candidate_filename,
    stage_bmrb_accession,
)

__all__ = [
    "ProtonationValidator",
    "ResidueMapper",
    "STAGE_MANIFEST_FILENAME",
    "StructureLoader",
    "build_candidate_pool",
    "compute_chain_surface_area",
    "compute_pair_surface_area",
    "compute_residue_contact_graph",
    "extract_polymer_residue_nodes",
    "infer_candidate_metadata",
    "load_stage_manifest",
    "load_first_model",
    "parse_candidate_filename",
    "stage_bmrb_accession",
    "unpack_zip_archives",
    "write_contact_graph_npz",
]
